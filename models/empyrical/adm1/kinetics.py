import numpy as np


def calculate_process_rates(concentrations: np.ndarray, p: dict) -> np.ndarray:
    """
    Calculation of the 19s processes rates
    Indexes of the concentration vector (26 liquids) :
      0=S_su, 1=S_aa, 2=S_fa, 3=S_va, 4=S_bu, 5=S_pro, 6=S_ac,
      7=S_h2, 8=S_ch4, 9=S_IC, 10=S_IN, 11=S_I,
      12=X_c,  13=X_ch, 14=X_pr, 15=X_li,
      16=X_su, 17=X_aa, 18=X_fa, 19=X_c4, 20=X_pro, 21=X_ac, 22=X_h2,
      23=X_I,  24=S_cat, 25=S_an
    """

    S_su, S_aa, S_fa = concentrations[0], concentrations[1], concentrations[2]
    S_va, S_bu, S_pro = concentrations[3], concentrations[4], concentrations[5]
    S_ac, S_h2 = concentrations[6], concentrations[7]
    S_IN = concentrations[10]
    X_c = concentrations[12]
    X_ch, X_pr, X_li = concentrations[13], concentrations[14], concentrations[15]
    X_su, X_aa, X_fa = concentrations[16], concentrations[17], concentrations[18]
    X_c4, X_pro = concentrations[19], concentrations[20]
    X_ac, X_h2 = concentrations[21], concentrations[22]

    I_h2_fa = p['K_I_h2_fa'] / (p['K_I_h2_fa'] + S_h2)
    I_h2_c4 = p['K_I_h2_c4'] / (p['K_I_h2_c4'] + S_h2)
    I_h2_pro = p['K_I_h2_pro'] / (p['K_I_h2_pro'] + S_h2)

    # Nitrogen limitation: applied to all growth processes (Batstone 2002, eq. 6)
    K_I_IN = p.get('K_I_IN', 1e-4)  # kmol N/m³, default from ADM1 standard
    I_IN = S_IN / (K_I_IN + S_IN)

    I_nh3 = 1.0  # placeholder, NH3 inhibition for acetoclastic methanogenesis

    rho = np.zeros(19)

    # Groupe 1: Désintégration / Hydrolyse (processus 1-4)
    rho[0] = p['k_dis'] * X_c  # désintégration
    rho[1] = p['k_hyd_ch'] * X_ch  # hydrolyse glucides
    rho[2] = p['k_hyd_pr'] * X_pr  # hydrolyse protéines
    rho[3] = p['k_hyd_li'] * X_li  # hydrolyse lipides

    # Groupe 2: Acidogénèse (processus 5-6)
    rho[4] = p['k_m_su'] * (S_su / (p['K_S_su'] + S_su)) * X_su * I_IN
    rho[5] = p['k_m_aa'] * (S_aa / (p['K_S_aa'] + S_aa)) * X_aa * I_IN

    # Groupe 3: Acétogénèse (processus 7-9)
    rho[6] = p['k_m_fa'] * (S_fa / (p['K_S_fa'] + S_fa)) * X_fa * I_h2_fa * I_IN
    rho[7] = p['k_m_c4'] * (S_va / (p['K_S_c4'] + S_va)) * X_c4 \
             * (S_va / (S_va + S_bu + 1e-10)) * I_h2_c4 * I_IN
    rho[8] = p['k_m_c4'] * (S_bu / (p['K_S_c4'] + S_bu)) * X_c4 \
             * (S_bu / (S_va + S_bu + 1e-10)) * I_h2_c4 * I_IN
    rho[9] = p['k_m_pro'] * (S_pro / (p['K_S_pro'] + S_pro)) * X_pro * I_h2_pro * I_IN

    # Groupe 4: Méthanogénèse (processus 10-11)
    rho[10] = p['k_m_ac'] * (S_ac / (p['K_S_ac'] + S_ac)) * X_ac * I_nh3 * I_IN
    rho[11] = p['k_m_h2'] * (S_h2 / (p['K_S_h2'] + S_h2)) * X_h2 * I_IN

    # Décès des 7 groupes de biomasse (processus 12-18)
    rho[12] = p['k_dec_su'] * X_su
    rho[13] = p['k_dec_aa'] * X_aa
    rho[14] = p['k_dec_fa'] * X_fa
    rho[15] = p['k_dec_c4'] * X_c4
    rho[16] = p['k_dec_pro'] * X_pro
    rho[17] = p['k_dec_ac'] * X_ac
    rho[18] = p['k_dec_h2'] * X_h2

    return rho
