import numpy as np


def calculate_pH(concentrations: np.ndarray, p: dict) -> tuple:
    """
    Calcule le pH par bilan de charge ionique (Batstone 2002, Tableau A.4).
    Retourne (pH, h, Ka_IN) où h = [H+] en kmol/m³ = mol/L.
    """
    S_va  = concentrations[3]
    S_bu  = concentrations[4]
    S_pro = concentrations[5]
    S_ac  = concentrations[6]
    S_IC  = concentrations[9]
    S_IN  = concentrations[10]
    S_cat = concentrations[24]
    S_an  = concentrations[25]

    T_K   = p.get('temperature', 35.0) + 273.15
    T_ref = 298.15
    R     = 8.314  # J/(mol·K)

    def Ka(pKa_ref, dH):
        return (10 ** -pKa_ref) * np.exp(dH / R * (1 / T_ref - 1 / T_K))

    Ka_w   = Ka(14.000,  55900)
    Ka_co2 = Ka( 6.350,  -7646)
    Ka_IN  = Ka( 9.250,  51965)
    Ka_ac  = Ka( 4.756,      0)
    Ka_pro = Ka( 4.874,      0)
    Ka_bu  = Ka( 4.820,      0)
    Ka_va  = Ka( 4.863,      0)

    def f(h):
        return (
            h
            + S_cat
            + S_IN * h / (h + Ka_IN)
            - Ka_w / h
            - S_an
            - (S_ac  / 64.0)  * Ka_ac  / (h + Ka_ac)
            - (S_pro / 112.0) * Ka_pro / (h + Ka_pro)
            - (S_bu  / 160.0) * Ka_bu  / (h + Ka_bu)
            - (S_va  / 208.0) * Ka_va  / (h + Ka_va)
            - S_IC * Ka_co2 / (h + Ka_co2)
        )

    # Bisection géométrique entre pH 4 (1e-4) et pH 10 (1e-10)
    h_lo, h_hi = 1e-10, 1e-4
    for _ in range(100):
        h_mid = (h_lo * h_hi) ** 0.5
        if f(h_mid) * f(h_lo) < 0:
            h_hi = h_mid
        else:
            h_lo = h_mid
        if (h_hi - h_lo) / h_mid < 1e-12:
            break

    pH = -np.log10(h_mid)
    return pH, h_mid, Ka_IN


def calculate_process_rates(concentrations: np.ndarray, p: dict) -> np.ndarray:
    """
    Calcul des 19 taux de processus ADM1.
    Indices du vecteur de concentration (26 composants liquides) :
      0=S_su, 1=S_aa, 2=S_fa, 3=S_va, 4=S_bu, 5=S_pro, 6=S_ac,
      7=S_h2, 8=S_ch4, 9=S_IC, 10=S_IN, 11=S_I,
      12=X_c, 13=X_ch, 14=X_pr, 15=X_li,
      16=X_su, 17=X_aa, 18=X_fa, 19=X_c4, 20=X_pro, 21=X_ac, 22=X_h2,
      23=X_I, 24=S_cat, 25=S_an
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

    I_h2_fa  = p['K_I_h2_fa']  / (p['K_I_h2_fa']  + S_h2)
    I_h2_c4  = p['K_I_h2_c4']  / (p['K_I_h2_c4']  + S_h2)
    I_h2_pro = p['K_I_h2_pro'] / (p['K_I_h2_pro'] + S_h2)

    K_I_IN = p.get('K_I_IN', 1e-4)
    I_IN = S_IN / (K_I_IN + S_IN)

    pH, h, Ka_IN = calculate_pH(concentrations, p)

    S_nh3   = S_IN * Ka_IN / (Ka_IN + h)
    K_I_nh3 = p.get('K_I_nh3', 0.0018)
    I_nh3   = K_I_nh3 / (K_I_nh3 + S_nh3)

    def I_pH(pH_LL, pH_UL):
        if pH >= pH_UL:
            return 1.0
        if pH <= pH_LL:
            return 0.0
        return np.exp(-3.0 * ((pH - pH_UL) / (pH_UL - pH_LL)) ** 2)

    I_pH_ac  = I_pH(p.get('pH_LL_ac',  6.0), p.get('pH_UL_ac',  7.0))
    I_pH_h2  = I_pH(p.get('pH_LL_h2',  5.0), p.get('pH_UL_h2',  6.0))
    I_pH_bac = I_pH(p.get('pH_LL_bac', 4.0), p.get('pH_UL_bac', 5.5))

    rho = np.zeros(19)

    # Groupe 1 : Désintégration / Hydrolyse
    rho[0] = p['k_dis']    * X_c
    rho[1] = p['k_hyd_ch'] * X_ch
    rho[2] = p['k_hyd_pr'] * X_pr
    rho[3] = p['k_hyd_li'] * X_li

    # Groupe 2 : Acidogénèse
    rho[4] = p['k_m_su'] * (S_su / (p['K_S_su'] + S_su)) * X_su * I_IN * I_pH_bac
    rho[5] = p['k_m_aa'] * (S_aa / (p['K_S_aa'] + S_aa)) * X_aa * I_IN * I_pH_bac

    # Groupe 3 : Acétogénèse
    rho[6] = p['k_m_fa']  * (S_fa  / (p['K_S_fa']  + S_fa))  * X_fa  * I_h2_fa  * I_IN * I_pH_bac
    rho[7] = p['k_m_c4']  * (S_va  / (p['K_S_c4']  + S_va))  * X_c4  \
             * (S_va / (S_va + S_bu + 1e-10)) * I_h2_c4 * I_IN * I_pH_bac
    rho[8] = p['k_m_c4']  * (S_bu  / (p['K_S_c4']  + S_bu))  * X_c4  \
             * (S_bu / (S_va + S_bu + 1e-10)) * I_h2_c4 * I_IN * I_pH_bac
    rho[9] = p['k_m_pro'] * (S_pro / (p['K_S_pro'] + S_pro)) * X_pro * I_h2_pro * I_IN * I_pH_bac

    # Groupe 4 : Méthanogénèse
    rho[10] = p['k_m_ac'] * (S_ac / (p['K_S_ac'] + S_ac)) * X_ac * I_nh3 * I_IN * I_pH_ac
    rho[11] = p['k_m_h2'] * (S_h2 / (p['K_S_h2'] + S_h2)) * X_h2 * I_IN * I_pH_h2

    # Décès des 7 groupes de biomasse
    rho[12] = p['k_dec_su']  * X_su
    rho[13] = p['k_dec_aa']  * X_aa
    rho[14] = p['k_dec_fa']  * X_fa
    rho[15] = p['k_dec_c4']  * X_c4
    rho[16] = p['k_dec_pro'] * X_pro
    rho[17] = p['k_dec_ac']  * X_ac
    rho[18] = p['k_dec_h2']  * X_h2

    return rho
