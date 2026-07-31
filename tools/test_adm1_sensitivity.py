"""
Test de sensibilité et d'extrapolation : compare les prédictions ML vs simulations ADM1.

Fait varier un paramètre à la fois en fixant les autres aux valeurs médianes,
puis teste l'extrapolation hors des plages d'entraînement.

Usage :
    python tools/test_adm1_sensitivity.py
    python tools/test_adm1_sensitivity.py --model models/trained/adm1_gb.pkl
    python tools/test_adm1_sensitivity.py --n_points 20
"""
import sys
import argparse
import pickle
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from models.empyrical.adm1.fraction import ADM1Fraction
from processes.digester_process.anaerobic_digester_process import AnaerobicDigesterProcess

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Scénario de base (valeurs médianes des plages d'entraînement)
# ---------------------------------------------------------------------------
BASE = {
    'flowrate':    3400.0 / (20.0 * 24.0),  # volume/hrt
    'temperature': 36.0,
    'volume':      3400.0,
    'hrt_days':    20.0,
    'k_L_a':       175.0,
    'cod':         20000.0,
    'tss':         15000.0,
    'tkn':         700.0,
    'nh4':         400.0,
}

# ---------------------------------------------------------------------------
# Tests de sensibilité (interpolation) et d'extrapolation
# ---------------------------------------------------------------------------
SENSITIVITY_TESTS = [
    {
        'name': 'Température (interpolation 30-42°C)',
        'param': 'temperature',
        'values': np.linspace(30.0, 42.0, 10),
        'extrapolation': False,
    },
    {
        'name': 'Température (extrapolation 25-50°C)',
        'param': 'temperature',
        'values': np.linspace(25.0, 50.0, 10),
        'extrapolation': True,
    },
    {
        'name': 'TRH (interpolation 10-40j)',
        'param': 'hrt_days',
        'values': np.linspace(10.0, 40.0, 10),
        'extrapolation': False,
    },
    {
        'name': 'TRH (extrapolation 5-60j)',
        'param': 'hrt_days',
        'values': np.linspace(5.0, 60.0, 10),
        'extrapolation': True,
    },
    {
        'name': 'Charge DCO (interpolation 5000-40000 mg/L)',
        'param': 'cod',
        'values': np.linspace(5000.0, 40000.0, 10),
        'extrapolation': False,
    },
    {
        'name': 'NH4 influent (interpolation 100-800 mg/L)',
        'param': 'nh4',
        'values': np.linspace(100.0, 800.0, 10),
        'extrapolation': False,
    },
]

STEADY_STATE_HOURS = 200 * 24
TIMESTEP_H = 24.0

SEED_BIOMASS = {
    'x_c': 0.5,  'x_ch': 0.05, 'x_pr': 0.05, 'x_li': 0.05,
    'x_su': 0.3, 'x_aa': 0.8,  'x_fa': 0.15,
    'x_c4': 0.3, 'x_pro': 0.1, 'x_ac': 0.5,  'x_h2': 0.2,
    's_ic': 0.06, 's_in': 0.05,
}


def _simulate_adm1(scenario: dict) -> dict | None:
    """Simule un scénario avec AnaerobicDigesterProcess jusqu'à l'état stationnaire."""
    components = ADM1Fraction.fractionate(
        cod=scenario['cod'],
        tss=scenario['tss'],
        tkn=scenario['tkn'],
        nh4=scenario['nh4'],
    )
    config = {
        'volume':      scenario['volume'],
        'V_gas':       scenario['volume'] * 300.0 / 3400.0,
        'temperature': scenario['temperature'],
        'waste_ratio': 0.0,
        'k_L_a':       scenario['k_L_a'],
        'k_p':         50000.0,
        'initial_state': {**components, **SEED_BIOMASS},
    }
    process = AnaerobicDigesterProcess('sens', 'Sensibilité', config)
    process.initialize()

    inputs = {
        'flowrate':    scenario['flowrate'],
        'temperature': scenario['temperature'],
        'components':  components,
    }
    output = None
    try:
        for _ in range(int(STEADY_STATE_HOURS / TIMESTEP_H)):
            output = process.process(inputs, TIMESTEP_H)
    except Exception:
        return None

    if output is None:
        return None

    comps = output.get('components', {})
    ph = comps.get('pH')
    if ph is None or not (4.0 <= ph <= 9.5):
        return None

    return {
        'ch4_m3_per_day': output.get('ch4_m3_per_day', 0.0),
        'pH':             ph,
        'cod_out_kg_m3':  output.get('cod', 0.0) / 1000.0,
        'nh4_out_mg_l':   output.get('nh4', 0.0),
        'x_ac_kg_m3':     comps.get('x_ac', 0.0),
    }


def _predict_ml(model_bundle: dict, scenario: dict) -> dict:
    """Prédit avec le modèle ML chargé."""
    feature_cols = model_bundle['feature_cols']
    row = {
        'flowrate':    scenario['flowrate'],
        'temperature': scenario['temperature'],
        'volume':      scenario['volume'],
        'hrt_days':    scenario['hrt_days'],
        'k_l_a':       scenario['k_L_a'],
        'cod_in':      scenario['cod'],
        'tss_in':      scenario['tss'],
        'tkn_in':      scenario['tkn'],
        'nh4_in':      scenario['nh4'],
    }
    X = np.array([[row[c] for c in feature_cols]])
    X_s = model_bundle['scaler'].transform(X)
    y = model_bundle['model'].predict(X_s)[0]
    return dict(zip(model_bundle['target_cols'], y))


def run_sensitivity(model_path: Path) -> None:
    print(f"\nChargement du modèle : {model_path}")
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    print(f"  Modèle : {bundle['model_key'].upper()}  |  R² test entraînement : {bundle['r2_test']:.4f}")

    targets = bundle['target_cols']
    all_results = []

    for test in SENSITIVITY_TESTS:
        param  = test['param']
        values = test['values']
        label  = '(EXTRAPOLATION)' if test['extrapolation'] else ''

        print(f"\n{'='*55}")
        print(f"  {test['name']} {label}")
        print(f"{'='*55}")

        rows = []
        for v in values:
            scenario = {**BASE, param: v}
            if param == 'hrt_days':
                scenario['flowrate'] = scenario['volume'] / (v * 24.0)

            adm1 = _simulate_adm1(scenario)
            ml   = _predict_ml(bundle, scenario)

            if adm1 is None:
                print(f"  {v:.2f} -> simulation échouée (acidification probable)")
                continue

            row = {'param': param, 'value': v, 'extrapolation': test['extrapolation']}
            for t in targets:
                a = adm1.get(t, float('nan'))
                m = ml.get(t, float('nan'))
                ecart = (m - a) / max(abs(a), 1e-9) * 100
                row[f'adm1_{t}'] = a
                row[f'ml_{t}']   = m
                row[f'ecart_{t}'] = ecart
            rows.append(row)
            all_results.append(row)

        if rows:
            for t in targets:
                print(f"\n  {t}")
                print(f"  {'Valeur':>8}  {'ADM1':>10}  {'ML':>10}  {'Ecart':>8}")
                print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
                for row in rows:
                    a = row[f'adm1_{t}']
                    m = row[f'ml_{t}']
                    e = row[f'ecart_{t}']
                    print(f"  {row['value']:>8.2f}  {a:>10.4f}  {m:>10.4f}  {e:>+7.1f}%")

    # Export CSV
    out_path = ROOT / 'tools' / 'adm1_sensitivity_results.csv'
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nRésultats exportés : {out_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='models/trained/adm1_gb.pkl',
                        help='Chemin du modèle ML sauvegardé')
    args = parser.parse_args()
    run_sensitivity(ROOT / args.model)
