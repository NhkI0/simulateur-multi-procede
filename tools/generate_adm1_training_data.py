"""
Générateur de données d'entraînement pour les modèles ML basés sur ADM1.

Principe : simule N scénarios avec ADM1 jusqu'à l'état quasi-stationnaire,
puis enregistre les paires (influent + paramètres réacteur) -> (effluent + biogaz) dans un CSV.

Ce script est ISOLÉ du reste de la simulation, il ne sera plus nécessaire
quand des données réelles de digesteur seront disponibles.

Usage :
    python tools/generate_adm1_training_data.py
    python tools/generate_adm1_training_data.py --n 2000 --output data/processed/custom.csv
    python tools/generate_adm1_training_data.py --seed 123 --n 500
"""
import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from tqdm import tqdm

from models.empyrical.adm1.model import ADM1Model
from models.empyrical.adm1.fraction import ADM1Fraction
from models.empyrical.adm1.kinetics import calculate_pH

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plages de variation des scénarios influent + réacteur
# ---------------------------------------------------------------------------
INFLUENT_RANGES = {
    'cod': (5000.0, 40000.0),  # mg/L DCO totale (influent digesteur = boues)
    'tss': (3000.0, 30000.0),  # mg/L MES
    'tkn': (200.0, 1200.0),  # mg/L NTK
    'nh4': (100.0, 800.0),  # mg/L NH4
    'temperature': (30.0, 42.0),  # °C (plage mésophile/thermophile)
}
REACTOR_RANGES = {
    'volume': (500.0, 5000.0),  # m^3
    'hrt_days': (10.0, 40.0),  # jours — échantillonné directement, débit dérivé
    'k_L_a': (50.0, 300.0),  # 1/j  transfert gaz-liquide
}

STEADY_STATE_DAYS = 200
K_L_A_DEFAULT = 200.0


def _sample_scenarios(n: int, rng: np.random.Generator) -> list:
    """Génère n scénarios par échantillonnage aléatoire uniforme (Latin Hypercube approché)."""
    scenarios = []
    all_ranges = {**INFLUENT_RANGES, **REACTOR_RANGES}
    for _ in range(n):
        s = {}
        for key, (lo, hi) in all_ranges.items():
            s[key] = float(rng.uniform(lo, hi))
        s['nh4'] = min(s['nh4'], s['tkn'] * 0.85)
        # Débit dérivé du TRH et du volume (en m³/h)
        s['flowrate'] = s['volume'] / (s['hrt_days'] * 24.0)
        scenarios.append(s)
    return scenarios


def _run_to_steady_state(
        model: ADM1Model,
        c_in: np.ndarray,
        hrt_days: float,
        k_l_a: float,
        temperature: float,
) -> np.ndarray:
    """
    Intègre le CSTR ADM1 jusqu'à l'état quasi-stationnaire avec le solveur Radau.
    Le réacteur est inoculé avec des valeurs typiques de biomasse pour éviter le
    démarrage à froid (cold start) qui converge vers le point mort bas-biomasse.
    """
    idx_h2 = model.COMPONENT_INDICES['s_h2']
    idx_ch4 = model.COMPONENT_INDICES['s_ch4']
    dilution = 1.0 / hrt_days

    y0 = c_in.copy()
    seed_biomass = {
        'x_c': 0.5, 'x_ch': 0.05, 'x_pr': 0.05, 'x_li': 0.05,
        'x_su': 0.3, 'x_aa': 0.1, 'x_fa': 0.15,
        'x_c4': 0.3, 'x_pro': 0.1, 'x_ac': 0.5, 'x_h2': 0.2,
        's_ic': 0.06, 's_in': 0.005,
    }
    for name, val in seed_biomass.items():
        if name in model.COMPONENT_INDICES:
            y0[model.COMPONENT_INDICES[name]] = max(y0[model.COMPONENT_INDICES[name]], val)

    def dc_dt(t, c):
        c = np.maximum(c, 0.0)
        dxdt = model.derivatives(c)
        dxdt += dilution * (c_in - c)
        dxdt[idx_h2] -= k_l_a * c[idx_h2]
        dxdt[idx_ch4] -= k_l_a * c[idx_ch4]
        return dxdt

    sol = solve_ivp(
        dc_dt,
        t_span=(0.0, STEADY_STATE_DAYS),
        y0=y0,
        method='Radau',
        rtol=1e-4,
        atol=1e-6,
        max_step=1.0,
    )

    if not sol.success:
        return None

    return np.maximum(sol.y[:, -1], 0.0)


def _extract_features_targets(
        scenario: dict,
        c_in: np.ndarray,
        c_out: np.ndarray,
        model: ADM1Model,
        hrt_days: float,
        k_l_a: float,
) -> dict:
    """
    Calcule les features et targets à partir d'un scénario simulé.
    Retourne None si le scénario est physiquement invalide.
    """
    idx = model.COMPONENT_INDICES

    cod_cols = ['s_su', 's_aa', 's_fa', 's_va', 's_bu', 's_pro', 's_ac',
                's_h2', 's_ch4', 's_i', 'x_c', 'x_ch', 'x_pr', 'x_li',
                'x_su', 'x_aa', 'x_fa', 'x_c4', 'x_pro', 'x_ac', 'x_h2', 'x_i']

    cod_in = sum(c_in[idx[k]] for k in cod_cols if k in idx)
    cod_out = sum(c_out[idx[k]] for k in cod_cols if k in idx)

    tss_out = sum(c_out[idx[k]] for k in
                  ['x_c', 'x_ch', 'x_pr', 'x_li', 'x_su', 'x_aa', 'x_fa',
                   'x_c4', 'x_pro', 'x_ac', 'x_h2', 'x_i'] if k in idx)

    vfa_total = sum(c_out[idx[k]] for k in
                    ['s_va', 's_bu', 's_pro', 's_ac'] if k in idx)

    biomass_active = sum(c_out[idx[k]] for k in
                         ['x_su', 'x_aa', 'x_fa', 'x_c4', 'x_pro', 'x_ac', 'x_h2'] if k in idx)

    nh4_out = c_out[idx['s_in']] * 14.0 * 1000.0  # kmol N/m³ → mg N/L

    q_ch4 = k_l_a * c_out[idx['s_ch4']]  # kg COD CH4 / (m³·j)
    ch4_m3_per_m3_per_day = q_ch4 / 0.395  # m³ CH4 / (m³ réacteur·j)

    cod_removal = max(0.0, (cod_in - cod_out) / cod_in * 100.0) if cod_in > 1e-6 else 0.0

    pH, _, _ = calculate_pH(c_out, model.params)

    # Filtres de validité physique
    if biomass_active < 0.01:  # washout total
        return None
    if vfa_total > 3.0:  # acidification sévère
        return None
    if cod_out > cod_in * 1.01:  # instabilité numérique
        return None
    if not (4.0 <= pH <= 9.0):  # pH hors plage physique
        return None

    return {
        # --- Features (entrées du modèle ML) ---
        'flowrate': scenario['flowrate'],
        'temperature': scenario['temperature'],
        'volume': scenario['volume'],
        'hrt_days': hrt_days,
        'k_l_a': k_l_a,
        'cod_in': scenario['cod'],
        'tss_in': scenario['tss'],
        'tkn_in': scenario['tkn'],
        'nh4_in': scenario['nh4'],
        's_in_in': c_in[idx['s_in']],  # kmol N/m^3
        's_ic_in': c_in[idx['s_ic']],  # kmol C/m^3
        'x_c_in': c_in[idx['x_c']],  # kg COD/m^3

        # --- Targets (sorties à prédire) ---
        'cod_out_kg_m3': cod_out,
        'tss_out_kg_m3': tss_out,
        'nh4_out_mg_l': nh4_out,
        'vfa_total_kg_m3': vfa_total,
        'biomass_active_kg_m3': biomass_active,
        's_ac_kg_m3': c_out[idx['s_ac']],
        's_h2_kg_m3': c_out[idx['s_h2']],
        'x_ac_kg_m3': c_out[idx['x_ac']],
        'x_h2_kg_m3': c_out[idx['x_h2']],
        'ch4_m3_per_m3_day': ch4_m3_per_m3_per_day,
        'cod_removal_pct': cod_removal,
        'pH': pH,
    }


def generate(n: int, output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    model = ADM1Model()

    scenarios = _sample_scenarios(n * 3, rng)

    records = []
    skipped = 0

    pbar = tqdm(total=n, desc="Génération ADM1", unit="scén.")

    for i, s in enumerate(scenarios):
        if len(records) >= n:
            break

        hrt_days = s['hrt_days']

        try:
            components = ADM1Fraction.fractionate(
                cod=s['cod'],
                tss=s['tss'],
                tkn=s['tkn'],
                nh4=s['nh4'],
            )
            c_in = model.dict_to_concentrations(components)
            c_in = np.clip(c_in, 0.0, None)

            c_out = _run_to_steady_state(
                model=model,
                c_in=c_in,
                hrt_days=hrt_days,
                k_l_a=s['k_L_a'],
                temperature=s['temperature'],
            )

            if c_out is None:
                skipped += 1
                continue

            row = _extract_features_targets(s, c_in, c_out, model, hrt_days, s['k_L_a'])
            if row is None:
                skipped += 1
                continue

            records.append(row)
            pbar.update(1)
            pbar.set_postfix(filtrés=skipped)

        except Exception as e:
            logger.debug(f"Scénario {i} ignoré : {e}")
            skipped += 1
            continue

    pbar.close()
    df = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nFichier généré : {output}")
    print(f"  Lignes      : {len(df)}")
    print(f"  Filtrés     : {skipped}")
    print(f"  Colonnes    : {list(df.columns)}")
    if not df.empty:
        print(f"\nAperçu statistiques :")
        targets = ['cod_out_kg_m3', 'tss_out_kg_m3', 'nh4_out_mg_l',
                   'vfa_total_kg_m3', 'ch4_m3_per_m3_day', 'cod_removal_pct', 'pH']
        print(df[targets].describe().round(4).to_string())
    else:
        print("\nAucun scénario valide généré — vérifier les filtres et le modèle.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Génère un dataset d'entraînement ML via simulations ADM1"
    )
    parser.add_argument('--n', type=int, default=1000,
                        help='Nombre de scénarios valides à générer (défaut: 1000)')
    parser.add_argument('--output', type=str,
                        default='data/processed/adm1_training_data.csv',
                        help='Chemin de sortie du CSV')
    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire pour la reproductibilité')
    args = parser.parse_args()

    output_path = ROOT / args.output
    generate(n=args.n, output=output_path, seed=args.seed)
