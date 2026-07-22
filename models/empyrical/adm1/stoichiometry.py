import numpy as np


def build_stoichiometric_matrix(p: dict) -> np.ndarray:
    """
    Matrice stœchiométrique ADM1 : 19 processus x 26 composants liquide.
    Référence : Batstone et al. (2002), Tableaux A.1 et A.2

    Indices colonnes (composants) :
      0  S_su   monosaccharides          (kg COD/m³)
      1  S_aa   acides aminés            (kg COD/m³)
      2  S_fa   LCFA                     (kg COD/m³)
      3  S_va   valérate                 (kg COD/m³)
      4  S_bu   butyrate                 (kg COD/m³)
      5  S_pro  propionate               (kg COD/m³)
      6  S_ac   acétate                  (kg COD/m³)
      7  S_h2   hydrogène dissous        (kg COD/m³)
      8  S_ch4  méthane dissous          (kg COD/m³)
      9  S_IC   carbone inorganique      (kmol C/m³)
      10 S_IN   azote inorganique        (kmol N/m³)
      11 S_I    substrat inerte soluble  (kg COD/m³)
      12 X_c    composite particulaire   (kg COD/m³)
      13 X_ch   glucides particulaires   (kg COD/m³)
      14 X_pr   protéines particulaires  (kg COD/m³)
      15 X_li   lipides particulaires    (kg COD/m³)
      16 X_su   biomasse sucres          (kg COD/m³)
      17 X_aa   biomasse acides aminés   (kg COD/m³)
      18 X_fa   biomasse LCFA            (kg COD/m³)
      19 X_c4   biomasse C4              (kg COD/m³)
      20 X_pro  biomasse propionate      (kg COD/m³)
      21 X_ac   biomasse acétate         (kg COD/m³)
      22 X_h2   biomasse H2              (kg COD/m³)
      23 X_I    substrat inerte part.    (kg COD/m³)
      24 S_cat  cations (équil. charge)  (kmol/m³)
      25 S_an   anions  (équil. charge)  (kmol/m³)

    Indices lignes (processus) :
      0  Désintégration de X_c
      1  Hydrolyse des glucides
      2  Hydrolyse des protéines
      3  Hydrolyse des lipides
      4  Captation des sucres (acidogénèse)
      5  Captation des acides aminés (acidogénèse)
      6  Captation des LCFA (acétogénèse)
      7  Captation du valérate (acétogénèse)
      8  Captation du butyrate (acétogénèse)
      9  Captation du propionate (acétogénèse)
      10 Captation de l'acétate (méthanogénèse)
      11 Captation de H2 (méthanogénèse)
      12-18  Décès de X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2
    """
    S = np.zeros((19, 26))

    # Fractions de désintégration du composite X_c
    f_sI_xc = p.get('f_sI_xc', 0.10)
    f_xI_xc = p.get('f_xI_xc', 0.20)
    f_ch_xc = p.get('f_ch_xc', 0.20)
    f_pr_xc = p.get('f_pr_xc', 0.20)
    f_li_xc = p.get('f_li_xc', 0.30)

    # Fractions des produits de l'acidogénèse des sucres
    f_bu_su = p.get('f_bu_su', 0.13)
    f_pro_su = p.get('f_pro_su', 0.27)
    f_ac_su = p.get('f_ac_su', 0.41)
    f_h2_su = p.get('f_h2_su', 0.19)

    # Fractions des produits de l'acidogénèse des acides aminés
    f_va_aa = p.get('f_va_aa', 0.23)
    f_bu_aa = p.get('f_bu_aa', 0.26)
    f_pro_aa = p.get('f_pro_aa', 0.05)
    f_ac_aa = p.get('f_ac_aa', 0.40)
    f_h2_aa = p.get('f_h2_aa', 0.06)

    # Fractions des produits de l'acétogénèse des LCFA
    f_ac_fa = p.get('f_ac_fa', 0.70)
    f_h2_fa = p.get('f_h2_fa', 0.30)

    # Fractions des produits de l'acétogénèse du valérate
    f_pro_va = p.get('f_pro_va', 0.54)
    f_ac_va = p.get('f_ac_va', 0.31)
    f_h2_va = p.get('f_h2_va', 0.15)

    # Fractions des produits de l'acétogénèse du butyrate
    f_ac_bu = p.get('f_ac_bu', 0.80)
    f_h2_bu = p.get('f_h2_bu', 0.20)

    # Fractions des produits de l'acétogénèse du propionate
    f_ac_pro = p.get('f_ac_pro', 0.57)
    f_h2_pro = p.get('f_h2_pro', 0.43)

    # Fraction lipides -> LCFA (reste -> S_su / glycérol)
    f_fa_li = p.get('f_fa_li', 0.95)

    # Rendements de croissance (kg COD_biomasse / kg COD_substrat)
    Y_su = p.get('Y_su', 0.10)
    Y_aa = p.get('Y_aa', 0.08)
    Y_fa = p.get('Y_fa', 0.06)
    Y_c4 = p.get('Y_c4', 0.06)
    Y_pro = p.get('Y_pro', 0.04)
    Y_ac = p.get('Y_ac', 0.05)
    Y_h2 = p.get('Y_h2', 0.06)

    # Contenus en carbone (kmol C / kg COD) pour bilan S_IC
    C_su = p.get('C_su', 0.0313)
    C_aa = p.get('C_aa', 0.0300)
    C_fa = p.get('C_fa', 0.0217)
    C_va = p.get('C_va', 0.0240)
    C_bu = p.get('C_bu', 0.0250)
    C_pro = p.get('C_pro', 0.0268)
    C_ac = p.get('C_ac', 0.0313)
    C_ch4 = p.get('C_ch4', 0.0156)
    C_sI = p.get('C_sI', 0.0300)
    C_xI = p.get('C_xI', 0.0300)
    C_ch = p.get('C_ch', 0.0313)
    C_pr = p.get('C_pr', 0.0300)
    C_li = p.get('C_li', 0.0220)
    C_bac = p.get('C_bac', 0.0313)  # toutes les biomasses
    C_xc = p.get('C_xc', 0.0280)  # composite

    # Contenus en azote (kmol N / kg COD) pour bilan S_IN
    N_aa  = p.get('N_aa',  0.007)          # Rosen Table p.16 : 0.007 kmol N/kg COD
    N_bac = p.get('N_bac', 0.08 / 14)     # Rosen Table p.16 : 0.08/14 ≈ 0.00571
    N_I   = p.get('N_I',   0.06 / 14)     # Rosen Table p.16 : 0.06/14 ≈ 0.00429
    N_xc  = p.get('N_xc',  0.0376 / 14)   # Rosen Table p.16 : 0.0376/14 ≈ 0.00269

    # Processus 0: Désintégration de X_c
    # X_c -> f_ch * X_ch + f_pr * X_pr + f_li * X_li + f_sI * S_I + f_xI * X_I
    S[0, 11] = f_sI_xc  # S_I  produit
    S[0, 12] = -1.0  # X_c  consommé
    S[0, 13] = f_ch_xc  # X_ch produit
    S[0, 14] = f_pr_xc  # X_pr produit
    S[0, 15] = f_li_xc  # X_li produit
    S[0, 23] = f_xI_xc  # X_I  produit
    # Bilan carbone -> S_IC
    S[0, 9] = (C_xc
               - f_sI_xc * C_sI
               - f_ch_xc * C_ch
               - f_pr_xc * C_pr
               - f_li_xc * C_li
               - f_xI_xc * C_xI)
    # Bilan azote -> S_IN
    S[0, 10] = (N_xc
                - f_sI_xc * N_I
                - f_xI_xc * N_I
                - f_pr_xc * N_aa)

    # Processus 1: Hydrolyse des glucides
    # X_ch -> S_su
    S[1, 0] = 1.0  # S_su produit
    S[1, 13] = -1.0  # X_ch consommé
    S[1, 9] = C_ch - C_su  # ≈ 0, bilan C

    # Processus 2: Hydrolyse des protéines
    # X_pr -> S_aa
    S[2, 1] = 1.0  # S_aa produit
    S[2, 14] = -1.0  # X_pr consommé
    S[2, 9] = C_pr - C_aa  # ≈ 0, bilan C
    S[2, 10] = N_aa - N_aa  # = 0, bilan N

    # Processus 3: Hydrolyse des lipides
    # X_li -> f_fa_li * S_fa + (1 - f_fa_li) * S_su
    S[3, 0] = 1.0 - f_fa_li  # S_su produit (glycérol)
    S[3, 2] = f_fa_li  # S_fa produit (LCFA)
    S[3, 15] = -1.0  # X_li consommé
    S[3, 9] = C_li - (1.0 - f_fa_li) * C_su - f_fa_li * C_fa

    # Processus 4: Captation des sucres (acidogénèse)
    # S_su -> Y_su * X_su + (1-Y_su) * (f_bu*S_bu + f_pro*S_pro +
    # f_ac*S_ac + f_h2*S_h2)
    S[4, 0] = -1.0  # S_su consommé
    S[4, 4] = (1.0 - Y_su) * f_bu_su  # S_bu produit
    S[4, 5] = (1.0 - Y_su) * f_pro_su  # S_pro produit
    S[4, 6] = (1.0 - Y_su) * f_ac_su  # S_ac produit
    S[4, 7] = (1.0 - Y_su) * f_h2_su  # S_h2 produit
    S[4, 16] = Y_su  # X_su produit
    S[4, 10] = -Y_su * N_bac  # S_IN consommé (biosynthèse)
    S[4, 9] = (C_su
               - Y_su * C_bac
               - (1.0 - Y_su) * (f_bu_su * C_bu
                                 + f_pro_su * C_pro
                                 + f_ac_su * C_ac))

    # Processus 5: Captation des acides aminés (acidogénèse)
    # S_aa -> Y_aa * X_aa + (1-Y_aa) * (f_va*S_va + f_bu*S_bu + ...)
    S[5, 1] = -1.0  # S_aa consommé
    S[5, 3] = (1.0 - Y_aa) * f_va_aa  # S_va produit
    S[5, 4] = (1.0 - Y_aa) * f_bu_aa  # S_bu produit
    S[5, 5] = (1.0 - Y_aa) * f_pro_aa  # S_pro produit
    S[5, 6] = (1.0 - Y_aa) * f_ac_aa  # S_ac produit
    S[5, 7] = (1.0 - Y_aa) * f_h2_aa  # S_h2 produit
    S[5, 17] = Y_aa  # X_aa produit
    S[5, 10] = N_aa - Y_aa * N_bac  # S_IN : libération N (aa), consommation (biomasse)
    S[5, 9] = (C_aa
               - Y_aa * C_bac
               - (1.0 - Y_aa) * (f_va_aa * C_va
                                 + f_bu_aa * C_bu
                                 + f_pro_aa * C_pro
                                 + f_ac_aa * C_ac))

    # Processus 6: Captation des LCFA (acétogénèse)
    # S_fa -> Y_fa * X_fa + (1-Y_fa) * (f_ac*S_ac + f_h2*S_h2)
    S[6, 2] = -1.0  # S_fa consommé
    S[6, 6] = (1.0 - Y_fa) * f_ac_fa  # S_ac produit
    S[6, 7] = (1.0 - Y_fa) * f_h2_fa  # S_h2 produit
    S[6, 18] = Y_fa  # X_fa produit
    S[6, 10] = -Y_fa * N_bac
    S[6, 9] = (C_fa
               - Y_fa * C_bac
               - (1.0 - Y_fa) * f_ac_fa * C_ac)

    # Processus 7: Captation du valérate (acétogénèse)
    # S_va -> Y_c4 * X_c4 + (1-Y_c4) * (f_pro*S_pro + f_ac*S_ac + f_h2*S_h2)
    S[7, 3] = -1.0  # S_va consommé
    S[7, 5] = (1.0 - Y_c4) * f_pro_va  # S_pro produit
    S[7, 6] = (1.0 - Y_c4) * f_ac_va  # S_ac produit
    S[7, 7] = (1.0 - Y_c4) * f_h2_va  # S_h2 produit
    S[7, 19] = Y_c4  # X_c4 produit
    S[7, 10] = -Y_c4 * N_bac
    S[7, 9] = (C_va
               - Y_c4 * C_bac
               - (1.0 - Y_c4) * (f_pro_va * C_pro
                                 + f_ac_va * C_ac))

    # Processus 8: Captation du butyrate (acétogénèse)
    # S_bu -> Y_c4 * X_c4 + (1-Y_c4) * (f_ac*S_ac + f_h2*S_h2)
    S[8, 4] = -1.0  # S_bu consommé
    S[8, 6] = (1.0 - Y_c4) * f_ac_bu  # S_ac produit
    S[8, 7] = (1.0 - Y_c4) * f_h2_bu  # S_h2 produit
    S[8, 19] = Y_c4  # X_c4 produit
    S[8, 10] = -Y_c4 * N_bac
    S[8, 9] = (C_bu
               - Y_c4 * C_bac
               - (1.0 - Y_c4) * f_ac_bu * C_ac)

    # Processus 9: Captation du propionate (acétogénèse)
    # S_pro -> Y_pro * X_pro + (1-Y_pro) * (f_ac*S_ac + f_h2*S_h2) + CO2
    S[9, 5] = -1.0  # S_pro consommé
    S[9, 6] = (1.0 - Y_pro) * f_ac_pro  # S_ac produit
    S[9, 7] = (1.0 - Y_pro) * f_h2_pro  # S_h2 produit
    S[9, 20] = Y_pro  # X_pro produit
    S[9, 10] = -Y_pro * N_bac
    # Le CO2 libéré (propionate = C3, acétate = C2) se retrouve dans S_IC
    S[9, 9] = (C_pro
               - Y_pro * C_bac
               - (1.0 - Y_pro) * f_ac_pro * C_ac)

    # Processus 10: Méthanogénèse acétoclaste
    # S_ac -> Y_ac * X_ac + (1-Y_ac) * S_ch4 + CO2
    S[10, 6] = -1.0  # S_ac consommé
    S[10, 8] = 1.0 - Y_ac  # S_ch4 produit
    S[10, 21] = Y_ac  # X_ac produit
    S[10, 10] = -Y_ac * N_bac
    # L'acétate (C2) donne CH4 (C1) + CO2 (C1)
    S[10, 9] = C_ac - Y_ac * C_bac - (1.0 - Y_ac) * C_ch4

    # Processus 11: Méthanogénèse hydrogénotrophe
    # S_h2 + CO2 -> Y_h2 * X_h2 + (1-Y_h2) * S_ch4
    S[11, 7] = -1.0  # S_h2 consommé
    S[11, 8] = 1.0 - Y_h2  # S_ch4 produit
    S[11, 22] = Y_h2  # X_h2 produit
    S[11, 10] = -Y_h2 * N_bac
    # CO2 consommé (S_h2 n'a pas de carbone -> tout le C du CH4 vient de S_IC)
    S[11, 9] = -(1.0 - Y_h2) * C_ch4 - Y_h2 * C_bac

    # Processus 12-18: Décès des 7 groupes de biomasse -> X_c
    # X_i -> X_c   (biomasse dégradée recyclée comme composite)
    biomass_cols = [16, 17, 18, 19, 20, 21, 22]  # X_su … X_h2
    for k, col in enumerate(biomass_cols):
        row = 12 + k
        S[row, col] = -1.0  # X_i consommé
        S[row, 12] = 1.0  # X_c produit
        # Bilan C : biomasse → composite (différence faible mais incluse)
        S[row, 9] = C_bac - C_xc
        # Bilan N : N_bac = N_xc dans la plupart des implémentations -> 0
        S[row, 10] = N_bac - N_xc

    return S
