import numpy as np
import logging
from typing import Dict, Any, List
from scipy.integrate import solve_ivp

from core.process.process_node import ProcessNode
from core.model.model_registry import ModelRegistry
from models.empyrical.adm1.kinetics import calculate_pH

logger = logging.getLogger(__name__)

# Constante des gaz parfaits en bar·m³/(kmol·K)
_R_BAR = 0.083145


def _KH_T(KH_ref: float, dH: float, T_K: float) -> float:
    """Constante de Henry corrigée en température (Rosen & Jeppsson 2006, Table 18)."""
    return KH_ref * np.exp(dH * (1 / 298.15 - 1 / T_K) / (100.0 * _R_BAR))


class AnaerobicDigesterProcess(ProcessNode):
    """
    Digesteur anaérobie utilisant ADM1 avec phase gazeuse complète.
    Implémentation conforme à Rosen & Jeppsson (2006) :
    - 3 ODEs gaz : S_gas_h2, S_gas_ch4, S_gas_co2 (kmol/m³_gaz)
    - Transfert gaz-liquide avec correction de Henry pour H2, CH4 et CO2
    - CO2 gazeux couplé à S_IC (correction pH et alcalinité)
    """

    def __init__(self, node_id: str, name: str, config: Dict[str, Any]) -> None:
        super().__init__(node_id, name, config)

        self.volume      = config.get('volume', 2000.0)
        self.V_gas       = config.get('V_gas', 300.0)
        self.temperature = config.get('temperature', 35.0)
        self.waste_ratio = config.get('waste_ratio', 0.05)
        self.k_L_a       = config.get('k_L_a', 200.0)
        self.k_p         = config.get('k_p', 5e4)    # m³/(j·bar)
        self.p_atm       = config.get('p_atm', 1.013) # bar

        model_params = config.get('model_parameters') or {}
        self._initial_state = config.get('initial_state') or {}

        self.registry = ModelRegistry.get_instance()
        self.model_instance = self.registry.create_model('ADM1Model', params=model_params)
        self.concentrations = np.zeros(self.model_instance.N_LIQUID)
        self.biogas_ch4_cumul = 0.0

        # Constantes thermodynamiques à la température de fonctionnement
        T_K = self.temperature + 273.15
        self._RT       = _R_BAR * T_K                    # bar·m³/kmol
        self._p_h2o    = 0.0313 * np.exp(5290 * (1/298.15 - 1/T_K) / (100.0 * _R_BAR))
        self._KH_h2    = _KH_T(7.8e-4,  -4180,  T_K)   # kmol/(m³·bar)
        self._KH_ch4   = _KH_T(1.4e-3,  -14240, T_K)   # kmol/(m³·bar)
        self._KH_co2   = _KH_T(0.035,   -19410, T_K)   # kmol/(m³·bar)
        # Ka_co2 à T_K (pour calculer S_CO2 dissous depuis S_IC)
        self._Ka_co2   = (10 ** -6.35) * np.exp(7646 / _R_BAR * (1/298.15 - 1/T_K) / 100.0)

        # État de la phase gazeuse [S_gas_h2, S_gas_ch4, S_gas_co2] en kmol/m³
        self.gas_state = np.array([7.8e-7, 0.0228, 0.0127])

    def initialize(self) -> None:
        """État initial du digesteur — surchargeable via initial_state dans la config."""
        init = {
            'x_c':   2.0,  'x_ac': 0.5,  'x_h2': 0.3,
            's_ac':  0.1,  's_ic': 0.06, 's_in': 0.002,
            's_cat': 0.04, 's_an': 0.02,
        }
        init.update(self._initial_state)
        self.concentrations = self.model_instance.dict_to_concentrations(init)

        # Initialisation phase gaz depuis l'équilibre de Henry avec l'état liquide
        S_h2_idx  = self.model_instance.COMPONENT_INDICES.get('s_h2',  7)
        S_ch4_idx = self.model_instance.COMPONENT_INDICES.get('s_ch4', 8)
        S_IC_idx  = self.model_instance.COMPONENT_INDICES.get('s_ic',  9)

        c = self.concentrations
        _, h, _ = calculate_pH(c, self.model_instance.params)
        S_CO2 = c[S_IC_idx] * h / (h + self._Ka_co2)

        # p_liq = concentration dissoute / KH → pression partielle à l'équilibre
        p_h2  = max(c[S_h2_idx]  / 16 / self._KH_h2,  1e-8)
        p_ch4 = max(c[S_ch4_idx] / 64 / self._KH_ch4, 1e-8)
        p_co2 = max(S_CO2            / self._KH_co2,   1e-8)

        self.gas_state = np.array([
            p_h2  / self._RT,
            p_ch4 / self._RT,
            p_co2 / self._RT,
        ])

    def get_required_inputs(self) -> List[str]:
        return ['flowrate', 'temperature']

    def process(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        inputs = self.fractionate_input(inputs, target_model='ADM1')

        q_in  = inputs['flowrate']
        c_in  = self.model_instance.dict_to_concentrations(inputs['components'])

        hrt_h   = self.volume / max(q_in, 1e-6)
        dilution = 1.0 / (hrt_h / 24.0)   # j⁻¹
        dt_day  = dt / 24.0

        S_h2_idx  = self.model_instance.COMPONENT_INDICES.get('s_h2',  7)
        S_ch4_idx = self.model_instance.COMPONENT_INDICES.get('s_ch4', 8)
        S_IC_idx  = self.model_instance.COMPONENT_INDICES.get('s_ic',  9)

        k_L_a  = self.k_L_a
        RT     = self._RT
        p_h2o  = self._p_h2o
        p_atm  = self.p_atm
        k_p    = self.k_p
        V_liq  = self.volume
        V_gas  = self.V_gas
        KH_h2  = self._KH_h2
        KH_ch4 = self._KH_ch4
        KH_co2 = self._KH_co2
        Ka_co2 = self._Ka_co2

        def dc_dt(t: float, state: np.ndarray) -> np.ndarray:
            c_liq = np.maximum(state[:26], 0.0)
            c_gas = np.maximum(state[26:], 0.0)   # [S_gas_h2, S_gas_ch4, S_gas_co2]

            # Cinétique biologique + dilution
            dxdt = self.model_instance.derivatives(c_liq)
            dxdt += dilution * (c_in - c_liq)

            # pH courant → fraction CO2 dissous
            _, h, _ = calculate_pH(c_liq, self.model_instance.params)
            S_CO2 = c_liq[S_IC_idx] * h / (h + Ka_co2)   # kmol CO2/m³

            # Pressions partielles gaz [bar]
            p_h2  = c_gas[0] * RT
            p_ch4 = c_gas[1] * RT
            p_co2 = c_gas[2] * RT
            p_gas = p_h2 + p_ch4 + p_co2 + p_h2o
            q_gas = max(k_p * (p_gas - p_atm), 0.0)   # m³_gaz/j

            # Taux de transfert gaz-liquide [kmol/(m³_liq·j)]
            rho_T_h2  = k_L_a * (c_liq[S_h2_idx]  / 16 - KH_h2  * p_h2)
            rho_T_ch4 = k_L_a * (c_liq[S_ch4_idx] / 64 - KH_ch4 * p_ch4)
            rho_T_co2 = k_L_a * (S_CO2                  - KH_co2 * p_co2)

            # Phase liquide : soustraction des transferts (liquide → gaz)
            dxdt[S_h2_idx]  -= rho_T_h2  * 16   # kmol/m³/j → kg COD/m³/j
            dxdt[S_ch4_idx] -= rho_T_ch4 * 64
            dxdt[S_IC_idx]  -= rho_T_co2          # S_IC en kmol C/m³

            # Phase gazeuse [kmol/(m³_gaz·j)]
            V_ratio = V_liq / V_gas
            dc_gas = np.array([
                rho_T_h2  * V_ratio - c_gas[0] * q_gas / V_gas,
                rho_T_ch4 * V_ratio - c_gas[1] * q_gas / V_gas,
                rho_T_co2 * V_ratio - c_gas[2] * q_gas / V_gas,
            ])

            return np.concatenate([dxdt, dc_gas])

        # Vecteur d'état étendu : 26 liquide + 3 gaz
        y0 = np.concatenate([self.concentrations.copy(), self.gas_state.copy()])

        sol = solve_ivp(
            dc_dt,
            t_span=(0.0, dt_day),
            y0=y0,
            method='Radau',
            rtol=1e-3,
            atol=1e-5,
        )

        c_out     = np.maximum(sol.y[:26, -1], 0.0)
        gas_out   = np.maximum(sol.y[26:, -1], 0.0)

        self.concentrations = c_out
        self.gas_state      = gas_out

        # Métriques gaz
        p_ch4_bar = gas_out[1] * RT
        p_co2_bar = gas_out[2] * RT
        p_h2_bar  = gas_out[0] * RT
        p_gas_tot = p_h2_bar + p_ch4_bar + p_co2_bar + p_h2o
        q_gas_day = max(k_p * (p_gas_tot - p_atm), 0.0)   # m³/j

        # CH4 produit = débit gaz × fraction CH4 (en volume = fraction molaire)
        ch4_frac      = p_ch4_bar / max(p_gas_tot - p_h2o, 1e-6)
        ch4_m3_per_day = q_gas_day * ch4_frac
        ch4_energy_kwh = ch4_m3_per_day * 9.97

        self.biogas_ch4_cumul += ch4_m3_per_day * dt_day

        # Sorties liquide
        comp_out = self.model_instance.concentrations_to_dict(c_out)
        ph_val, _, _ = calculate_pH(c_out, self.model_instance.params)
        comp_out['pH'] = ph_val

        cod_out = sum(
            c_out[self.model_instance.COMPONENT_INDICES[k]]
            for k in ['s_su', 's_aa', 's_fa', 's_va', 's_bu', 's_pro', 's_ac',
                      's_h2', 's_ch4', 's_i', 'x_c', 'x_ch', 'x_pr', 'x_li',
                      'x_su', 'x_aa', 'x_fa', 'x_c4', 'x_pro', 'x_ac', 'x_h2', 'x_i']
            if k in self.model_instance.COMPONENT_INDICES
        ) * 1000.0

        s_in_idx = self.model_instance.COMPONENT_INDICES.get('s_in', 10)
        nh4_out  = c_out[s_in_idx] * 14.0 * 1000.0

        results = {
            'flowrate':    q_in,
            'temperature': self.temperature,
            'model_type':  'ADM1MODEL',
            'components':  comp_out,
            'cod':         cod_out,
            'tss':         self._compute_tss(c_out) * 1000.0,
            'nh4':         nh4_out,
            'no3':         0.0,
            'po4':         0.0,
            'ch4_m3_per_day':    ch4_m3_per_day,
            'ch4_energy_kwh':    ch4_energy_kwh,
            'biogas_ch4_cumul':  self.biogas_ch4_cumul,
            'p_ch4_bar':         p_ch4_bar,
            'p_co2_bar':         p_co2_bar,
            'hrt_hours':         hrt_h,
            'cod_removal_rate':  max(0, (inputs.get('cod', cod_out) - cod_out)
                                     / max(inputs.get('cod', 1), 1) * 100),
            'vfa_total':         self._compute_vfa(c_out) * 1000.0,
        }

        self.outputs = results
        self.metrics = {
            'ch4_m3_per_day': ch4_m3_per_day,
            'ch4_energy_kwh': ch4_energy_kwh,
            'hrt':            hrt_h,
            'vfa_total':      results['vfa_total'],
        }
        return results

    def update_state(self, outputs: Dict[str, Any]) -> None:
        self.state   = outputs.get('components', {}).copy()
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
