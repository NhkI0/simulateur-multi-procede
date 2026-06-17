import math
import numpy as np
import logging
from typing import Dict, Any, List

from core.process.process_node import ProcessNode
from core.model.model_registry import ModelRegistry
from core.solver.cstr_solver import CSTRSolver

logger = logging.getLogger(__name__)


class AnaerobicDigesterProcess(ProcessNode):
    """
    Digesteur anaérobie utilisant ADM1.
    Reçoit n'importe quel flowdata (boues, effluent ASM, influent brut).
    Produit un digestat liquide+métriques biogaz.
    """

    def __init__(self, node_id: str, name: str, config: Dict[str, Any]) -> None:
        super().__init__(node_id, name, config)

        self.volume = config.get('volume', 2000.0)
        self.temperature = config.get('temperature', 35.0)
        self.waste_ratio = config.get('waste_ratio', 0.05)
        self.recycle_ratio = config.get('recycle_ratio', 0.0)
        self.k_L_a = config.get('k_L_a', 200.0)

        model_params = config.get('model_parameters') or {}

        self.registry = ModelRegistry.get_instance()
        self.model_instance = self.registry.create_model('ADM1Model', params=model_params)
        self.concentrations = np.zeros(self.model_instance.N_LIQUID)
        self.biogas_ch4_cumul = 0.0

    def initialize(self) -> None:
        """État initial typique de notre digesteur"""
        init = {
            'x_c': 2.0,  # kg COD/m^3
            'x_ac': 0.5,
            'x_h2': 0.3,
            's_ac': 0.1,
            's_ic': 0.06,
            's_in': 0.002,
            's_cat': 0.04,
            's_an': 0.02,
        }
        self.concentrations = self.model_instance.dict_to_concentrations(init)

    def get_required_inputs(self) -> List[str]:
        return ['flowrate', 'temperature']

    def process(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        inputs = self.fractionate_input(inputs, target_model='ADM1')

        q_in = inputs['flowrate']
        c_in = self.model_instance.dict_to_concentrations(inputs['components'])

        hrt_h = self.volume / max(q_in, 1e-6)
        dilution = 1.0 / (hrt_h / 24.0)  # 1/j
        dt_day = dt / 24.0

        c_out = CSTRSolver.solve_step(
            c=self.concentrations.copy(),
            c_in=c_in,
            reaction_func=self.model_instance.derivatives,
            dt=dt_day,
            dilution_rate=dilution,
            method='rk4',
            oxygen_idx=None,  # évidemment pas d'oxygène
            do_setpoint=None,
        )
        c_out = np.maximum(c_out, 0.0)

        # Calcul du transfert gaz-liquid
        # HELP !!:glehtklrùnf
        # Loi de Henry simplifiée: flux_gaz = k_L_a * (S_dissous - S_sat)
        # S_sat_h2 = 0 (H2 presque insoluble), S_sat_ch4 = 0
        S_h2_idx = self.model_instance.COMPONENT_INDICES.get('s_h2', 7)
        S_ch4_idx = self.model_instance.COMPONENT_INDICES.get('s_ch4', 8)

        retention = math.exp(-self.k_L_a * dt_day)
        q_h2_transfer  = c_out[S_h2_idx]  * (1.0 - retention) * self.volume
        q_ch4_transfer = c_out[S_ch4_idx] * (1.0 - retention) * self.volume

        c_out[S_h2_idx]  *= retention
        c_out[S_ch4_idx] *= retention

        ch4_kgCOD_per_day = q_ch4_transfer
        ch4_m3_per_day = ch4_kgCOD_per_day / 0.395  # 1kg COD CH4 = 0.395 m^3 à 35°C
        ch4_energy_kwh = ch4_m3_per_day * 9.97  # 9.97 kWh/m^3 CH4

        self.biogas_ch4_cumul += ch4_m3_per_day * dt_day
        self.concentrations = c_out

        comp_out = self.model_instance.concentrations_to_dict(c_out)

        cod_out = sum(
            c_out[self.model_instance.COMPONENT_INDICES[k]]
            for k in ['s_su', 's_aa', 's_fa', 's_va', 's_bu', 's_pro', 's_ac',
                      's_h2', 's_ch4', 's_i', 'x_c', 'x_ch', 'x_pr', 'x_li',
                      'x_su', 'x_aa', 'x_fa', 'x_c4', 'x_pro', 'x_ac', 'x_h2', 'x_i']
            if k in self.model_instance.COMPONENT_INDICES
        ) * 1000.0  # kg COD/m^3 -> mg/L

        s_in_idx = self.model_instance.COMPONENT_INDICES.get('s_in', 10)
        nh4_out = c_out[s_in_idx] * 14.0 * 1000.0  # kmol N/m^3 -> mg N/L

        results = {
            'flowrate': q_in,
            'temperature': self.temperature,
            'model_type': 'ADM1MODEL',
            'components': comp_out,

            # Paramètres standards afin de se connecter à n'importe quel procédé en aval
            'cod': cod_out,
            'tss': self._compute_tss(c_out) * 1000.0,
            'nh4': nh4_out,
            'no3': 0.0,
            'po4': 0.0,

            # Métriques spécifiques ADM1
            'ch4_m3_per_day': ch4_m3_per_day,
            'ch4_energy_kwh': ch4_energy_kwh,
            'biogas_ch4_cumul': self.biogas_ch4_cumul,
            'hrt_hours': hrt_h,
            'cod_removal_rate': max(0, (inputs.get('cod', cod_out) - cod_out)
                                    / max(inputs.get('cod', 1), 1) * 100),
            'vfa_total': self._compute_vfa(c_out) * 1000.0,
        }

        self.outputs = results
        self.metrics = {
            'ch4_m3_per_day': ch4_m3_per_day,
            'ch4_energy_kwh': ch4_energy_kwh,
            'hrt': hrt_h,
            'vfa_total': results['vfa_total']
        }
        return results

    def update_state(self, outputs: Dict[str, Any]) -> None:
        self.state = outputs.get('components', {}).copy()
        self.outputs = outputs

    def _compute_tss(self, c: np.ndarray) -> float:
        particulate = ['x_c', 'x_ch', 'x_pr', 'x_li', 'x_su', 'x_aa',
                       'x_fa', 'x_c4', 'x_pro', 'x_ac', 'x_h2', 'x_i']
        return sum(c[self.model_instance.COMPONENT_INDICES[k]]
                   for k in particulate
                   if k in self.model_instance.COMPONENT_INDICES)

    def _compute_vfa(self, c: np.ndarray) -> float:
        vfas = ['s_va', 's_bu', 's_pro', 's_ac']
        return sum(c[self.model_instance.COMPONENT_INDICES[k]]
                   for k in vfas
                   if k in self.model_instance.COMPONENT_INDICES)
