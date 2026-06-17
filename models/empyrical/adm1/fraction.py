from typing import Dict, Optional


class ADM1Fraction:
    """"Conversion des paramètres mesurables en composants ADM1"""

    DEFAULT_RATIOS = {
        # fraction SI de la DCO soluble (très petits élements)
        'f_sI': 0.05,
        # fraction XI de la DCO particulaire (plus gros élements en suspension)
        'f_xI': 0.10,
        # glucides
        'f_ch': 0.20,
        # protéines
        'f_pr': 0.20,
        # lipides
        'f_li': 0.25,
    }

    @classmethod
    def fractionate(
            cls,
            cod: float,
            tss: float = 0.0,
            tkn: float = 0.0,
            nh4: float = 0.0,
            no3: float = 0.0,
            po4: float = 0.0,
            alkalinity: Optional[float] = None,
            cod_soluble: Optional[float] = None,
            ratios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        r = {**cls.DEFAULT_RATIOS, **(ratios or {})}

        cod_part = min(cod, 1.48 * tss) if tss > 0 else cod * 0.4
        cod_sol = max(0.0, cod - cod_part)

        c: Dict[str, float] = {}
        c['s_i'] = r['f_sI'] * cod_sol
        c['s_su'] = 0.0  # sucres produits par hydrolyse
        c['s_aa'] = 0.0
        c['s_fa'] = 0.0
        c['s_va'] = c['s_bu'] = c['s_pro'] = c['s_ac'] = c['s_h2'] = c['s_ch4'] = 0.0
        c['x_i'] = r['f_xI'] * cod_part
        cod_deg = cod_part - c['x_i']
        c['x_c'] = cod_deg  # tout le COD dégradable particulaire -> composite
        c['x_ch'] = c['x_pr'] = c['x_li'] = 0.0
        c['x_su'] = c['x_aa'] = c['x_fa'] = 0.0
        c['x_c4'] = c['x_pro'] = c['x_ac'] = c['x_h2'] = 0.0

        # Azote inorganique (kmol N/m^3)
        c['s_in'] = (nh4 if nh4 > 0 else 0.7 * tkn) / (14.0 * 1000.0)
        # Carbone inorganique, approximation depuis alcalinité
        c['s_ic'] = (alkalinity / 1000.0) if alkalinity else 0.04
        # Ions (équilibre de charge avec valeurs typiques)
        c['s_cat'] = 0.04
        c['s_an'] = 0.02

        return c
