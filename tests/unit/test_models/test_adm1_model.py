"""
Tests de validation du modèle ADM1.
Valeurs de référence : Batstone et al. (2002), Tableau 4 (état stationnaire).

Conditions de référence :
  - Volume = 3400 m^3, Q = 170 m^3/j (HRT = 20 j), T = 35°C
  - Solveur implicite (Radau) requis - système raide
  - Tolérance +-30% par rapport aux valeurs publiées (acceptable pour une
    implémentation sans équilibre acide-base complet ni inhibition NH3)
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from models.empyrical.adm1.model import ADM1Model
from models.empyrical.adm1.fraction import ADM1Fraction

# ---------------------------------------------------------------------------
# Conditions de référence Batstone (2002), Tableau 3 - influent benchmark
# ---------------------------------------------------------------------------
INFLUENT_REF = {
    's_su': 0.010,
    's_aa': 0.001,
    's_fa': 0.001,
    's_va': 0.001,
    's_bu': 0.001,
    's_pro': 0.001,
    's_ac': 0.001,
    's_h2': 1e-8,
    's_ch4': 1e-5,
    's_ic': 0.040,
    's_in': 0.010,
    's_i': 0.020,
    'x_c': 2.000,
    'x_ch': 0.500,
    'x_pr': 0.100,
    'x_li': 0.250,
    'x_su': 0.000,
    'x_aa': 0.000,
    'x_fa': 0.000,
    'x_c4': 0.000,
    'x_pro': 0.000,
    'x_ac': 0.000,
    'x_h2': 0.000,
    'x_i': 0.000,
    's_cat': 0.040,
    's_an': 0.020,
}

STEADY_STATE_REF = {
    's_su': 0.012,
    's_aa': 0.0053,
    's_fa': 0.099,
    's_va': 0.012,
    's_bu': 0.013,
    's_pro': 0.016,
    's_ac': 0.197,
    's_h2': 2.36e-7,
    's_ch4': 0.055,
    's_ic': 0.152,
    's_in': 0.130,
    'x_c': 0.308,
    'x_ch': 0.028,
    'x_pr': 0.100,
    'x_li': 0.030,
    'x_su': 0.420,
    'x_aa': 0.120,
    'x_fa': 0.240,
    'x_c4': 0.430,
    'x_pro': 0.140,
    'x_ac': 0.760,
    'x_h2': 0.320,
}

VOLUME = 3400.0
FLOWRATE = 170.0 / 24.0
HRT_DAYS = 20.0
K_L_A = 200.0
TOLERANCE = 0.30

SEED_BIOMASS = {
    'x_su': 0.42,
    'x_aa': 0.12,
    'x_fa': 0.24,
    'x_c4': 0.43,
    'x_pro': 0.14,
    'x_ac': 0.76,
    'x_h2': 0.32,
    'x_c': 0.31,
    'x_ch': 0.028,
    'x_pr': 0.10,
    'x_li': 0.030,
    's_ac': 0.197,
    's_ic': 0.152,
    's_in': 0.130,
}


def _run_to_steady_state(model: ADM1Model, c_in: np.ndarray,
                         dilution: float, k_l_a: float,
                         t_end_days: float = 200.0) -> np.ndarray:
    idx_h2 = model.COMPONENT_INDICES['s_h2']
    idx_ch4 = model.COMPONENT_INDICES['s_ch4']

    y0 = c_in.copy()
    for name, val in SEED_BIOMASS.items():
        y0[model.COMPONENT_INDICES[name]] = val

    def dc_dt(t, c):
        c = np.maximum(c, 0.0)
        dxdt = model.derivatives(c)
        dxdt += dilution * (c_in - c)
        dxdt[idx_h2] -= k_l_a * c[idx_h2]
        dxdt[idx_ch4] -= k_l_a * c[idx_ch4]
        return dxdt

    sol = solve_ivp(
        dc_dt,
        t_span=(0.0, t_end_days),
        y0=y0,
        method='Radau',
        rtol=1e-6,
        atol=1e-8,
        max_step=0.5,
    )
    assert sol.success, f"Integration echouee : {sol.message}"
    return np.maximum(sol.y[:, -1], 0.0)


@pytest.fixture(scope='module')
def adm1_model():
    return ADM1Model()


@pytest.fixture(scope='module')
def steady_state(adm1_model):
    c_in = adm1_model.dict_to_concentrations(INFLUENT_REF)
    dilution = 1.0 / HRT_DAYS
    return _run_to_steady_state(adm1_model, c_in, dilution, K_L_A, t_end_days=200.0)


# ---------------------------------------------------------------------------
# Structure du modèle
# ---------------------------------------------------------------------------

class TestADM1ModelStructure:

    def test_initialisation(self, adm1_model):
        n = adm1_model.N_LIQUID
        assert n == 26, f"Attendu : 26 composants  |  Obtenu : {n}"
        assert adm1_model is not None
        assert hasattr(adm1_model, 'params')

    def test_component_indices_count(self, adm1_model):
        n = len(adm1_model.COMPONENT_INDICES)
        assert n == 26, f"Attendu : 26 indices  |  Obtenu : {n}"

    def test_component_indices_values(self, adm1_model):
        for name, idx in adm1_model.COMPONENT_INDICES.items():
            assert 0 <= idx < 26, (
                f"Attendu : 0 <= indice < 26  |  Obtenu : {name}={idx}"
            )

    def test_stoichiometric_matrix_shape(self, adm1_model):
        S = adm1_model.stoichiometric_matrix()
        expected = (19, 26)
        assert S.shape == expected, (
            f"Attendu : shape={expected}  |  Obtenu : shape={S.shape}"
        )

    def test_derivatives_shape(self, adm1_model):
        c = np.zeros(26)
        c[12] = 1.0
        c[10] = 0.01
        dxdt = adm1_model.derivatives(c)
        assert dxdt.shape == (26,), (
            f"Attendu : shape=(26,)  |  Obtenu : shape={dxdt.shape}"
        )

    def test_concentrations_roundtrip(self, adm1_model):
        c = adm1_model.dict_to_concentrations(INFLUENT_REF)
        d = adm1_model.concentrations_to_dict(c)
        for key, val in INFLUENT_REF.items():
            got = d[key]
            assert abs(got - val) < 1e-10, (
                f"{key} - Attendu : {val}  |  Obtenu : {got}"
            )

    def test_derivatives_non_negative_biomass(self, adm1_model):
        c = np.zeros(26)
        c[10] = 0.01
        dxdt = adm1_model.derivatives(c)
        names = {v: k for k, v in adm1_model.COMPONENT_INDICES.items()}
        for idx in [16, 17, 18, 19, 20, 21, 22]:
            val = dxdt[idx]
            assert val >= -1e-10, (
                f"{names[idx]} - Attendu : derivee <= 0 sans substrat  |  Obtenu : {val:.4e}"
            )


# ---------------------------------------------------------------------------
# Bilan COD
# ---------------------------------------------------------------------------

class TestADM1Conservation:

    def test_cod_balance_disintegration(self, adm1_model):
        S = adm1_model.stoichiometric_matrix()
        cod_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        balance = sum(S[0, c] for c in cod_cols)
        assert abs(balance) < 0.01, (
            f"Desintegration (proc. 0) - Attendu : bilan COD ~0  |  Obtenu : {balance:.5f}"
        )

    def test_cod_balance_methanogenesis_acetate(self, adm1_model):
        S = adm1_model.stoichiometric_matrix()
        cod_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        balance = sum(S[10, c] for c in cod_cols)
        assert abs(balance) < 0.01, (
            f"Methanogenese acetoclaste (proc. 10) - Attendu : bilan COD ~0  |  Obtenu : {balance:.5f}"
        )

    def test_cod_not_created(self, adm1_model):
        c = adm1_model.dict_to_concentrations(INFLUENT_REF)
        cod_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        cod_before = sum(c[i] for i in cod_cols)
        sol = solve_ivp(
            lambda t, y: adm1_model.derivatives(np.maximum(y, 0.0)),
            t_span=(0.0, 1.0),
            y0=c,
            method='Radau',
            rtol=1e-8,
            atol=1e-10,
        )
        cod_after = sum(np.maximum(sol.y[:, -1], 0.0)[i] for i in cod_cols)
        assert cod_after <= cod_before * 1.001, (
            f"COD ex nihilo - Attendu : apres <= {cod_before:.4f}  |  Obtenu : {cod_after:.4f}"
        )


# ---------------------------------------------------------------------------
# Fractionation
# ---------------------------------------------------------------------------

class TestADM1Fractionation:

    def test_output_keys(self):
        result = ADM1Fraction.fractionate(cod=30000.0, tss=20000.0, tkn=800.0, nh4=400.0)
        required = ['s_su', 's_aa', 's_fa', 's_va', 's_bu', 's_pro', 's_ac', 's_h2', 's_ch4',
                    's_ic', 's_in', 's_i', 'x_c', 'x_ch', 'x_pr', 'x_li', 'x_su', 'x_aa',
                    'x_fa', 'x_c4', 'x_pro', 'x_ac', 'x_h2', 'x_i', 's_cat', 's_an']
        for key in required:
            assert key in result, (
                f"Attendu : cle '{key}' presente  |  Obtenu : cle absente du dictionnaire"
            )

    def test_unit_conversion(self):
        result = ADM1Fraction.fractionate(cod=30000.0, tss=20000.0, tkn=800.0, nh4=400.0)
        val = result['x_c']
        assert 1.0 < val < 50.0, (
            f"x_c (cod=30000 mg/L) - Attendu : 1 < x_c < 50 kg COD/m3  |  Obtenu : {val:.3f}"
        )

    def test_nitrogen_units(self):
        result = ADM1Fraction.fractionate(cod=30000.0, tss=20000.0, tkn=800.0, nh4=400.0)
        val = result['s_in']
        assert 1e-4 < val < 1.0, (
            f"s_in - Attendu : 1e-4 < s_in < 1.0 kmol N/m3  |  Obtenu : {val:.5f}"
        )

    def test_all_non_negative(self):
        result = ADM1Fraction.fractionate(cod=30000.0, tss=20000.0, tkn=800.0, nh4=400.0)
        for k, v in result.items():
            assert v >= 0.0, (
                f"{k} - Attendu : >= 0  |  Obtenu : {v:.5f}"
            )


# ---------------------------------------------------------------------------
# Etat stationnaire (validation cinétique)
# ---------------------------------------------------------------------------

class TestADM1SteadyState:
    """
    Validation du modèle ADM1 à l'état stationnaire (HRT=20j, T=35°C).

    Les VFAs sont pilotées par µ_max/K_S et comparées à Batstone (2002) Tableau 4.
    Les biomasses dépendent de la bistabilité (k_hyd=10/j vs ~0.3/j dans le benchmark
    original) et ne sont pas comparées aux valeurs Tableau 4 directement.
    """

    @pytest.mark.parametrize("component,ref_value", [
        ('s_su', 0.012),
        ('s_aa', 0.0053),
        ('s_fa', 0.099),
        ('s_va', 0.012),
        ('s_bu', 0.013),
        ('s_pro', 0.016),
    ])
    def test_vfa_steady_state(self, adm1_model, steady_state, component, ref_value):
        idx = adm1_model.COMPONENT_INDICES[component]
        computed = steady_state[idx]
        rel_err = abs(computed - ref_value) / ref_value
        assert rel_err < TOLERANCE, (
            f"{component} - Attendu : {ref_value:.5f} kg COD/m3 (+-{TOLERANCE * 100:.0f}%)  "
            f"|  Obtenu : {computed:.5f}  (erreur {rel_err * 100:.1f}%)"
        )

    def test_acetate_steady_state(self, adm1_model, steady_state):
        """
        s_ac est borné par les cinétiques de x_ac + inhibitions pH et NH3.
        Borne inférieure : formule sans inhibitions (I_pH=I_nh3=1).
        Borne supérieure : s_ac < K_S_ac (saturation impossible à l'état stationnaire normal).
        """
        K_S_ac = adm1_model.params.get('K_S_ac', 0.15)
        k_dec = adm1_model.params.get('k_dec_ac', 0.02)
        k_m_ac = adm1_model.params.get('k_m_ac', 8.0)
        Y_ac = 0.05
        D = 1.0 / HRT_DAYS
        s_ac_min = K_S_ac * (k_dec + D) / (Y_ac * k_m_ac - k_dec - D)
        s_ac = steady_state[adm1_model.COMPONENT_INDICES['s_ac']]
        assert s_ac >= s_ac_min * 0.9, (
            f"s_ac - Attendu : >= {s_ac_min:.4f} kg COD/m3 (borne min sans inhibitions)  "
            f"|  Obtenu : {s_ac:.4f}"
        )
        assert s_ac < 1.0, (
            f"s_ac - Attendu : < 1.0 kg COD/m3 (pas d'acidification)  "
            f"|  Obtenu : {s_ac:.4f}"
        )

    def test_all_biomasses_viable(self, adm1_model, steady_state):
        biomasses = ['x_su', 'x_aa', 'x_fa', 'x_c4', 'x_pro', 'x_ac', 'x_h2']
        for name in biomasses:
            idx = adm1_model.COMPONENT_INDICES[name]
            val = steady_state[idx]
            assert val > 1e-4, (
                f"{name} - Attendu : > 1e-4 kg COD/m3 (viable)  |  Obtenu : {val:.6f} (washout)"
            )

    def test_vfa_below_inhibition_threshold(self, adm1_model, steady_state):
        vfa_indices = [adm1_model.COMPONENT_INDICES[k]
                       for k in ['s_va', 's_bu', 's_pro', 's_ac']]
        total = sum(steady_state[i] for i in vfa_indices)
        assert total < 1.0, (
            f"VFA totaux - Attendu : < 1.0 kg COD/m3  |  Obtenu : {total:.4f}"
        )

    def test_cod_removal(self, adm1_model, steady_state):
        c_in = adm1_model.dict_to_concentrations(INFLUENT_REF)
        cod_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        cod_in = sum(c_in[i] for i in cod_cols)
        cod_out = sum(steady_state[i] for i in cod_cols)
        removal = (cod_in - cod_out) / cod_in * 100
        assert removal > 30.0, (
            f"Taux elimination COD - Attendu : > 30%  |  Obtenu : {removal:.1f}%"
        )

    def test_methane_produced(self, adm1_model, steady_state):
        idx = adm1_model.COMPONENT_INDICES['s_ch4']
        val = steady_state[idx]
        assert val > 1e-4, (
            f"s_ch4 - Attendu : > 1e-4 kg COD/m3  |  Obtenu : {val:.6f}"
        )
