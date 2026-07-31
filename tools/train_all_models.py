"""
Entraîne tous les modèles ML, affiche une comparaison finale,
puis illustre l'inférence sur plusieurs scénarios types.

Usage :
    python tools/train_all_models.py
    python tools/train_all_models.py --data data/processed/adm1_training_data.csv
"""
import sys
import argparse
import pickle
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from tools.train_adm1_models import MODELS, FEATURE_COLS, TARGET_COLS, load_data

# ---------------------------------------------------------------------------
# Scénarios d'inférence (valeurs en unités brutes, avant scaling)
# ---------------------------------------------------------------------------
# flowrate = volume / (hrt_days * 24)
INFERENCE_SCENARIOS = [
    {
        'label': 'Nominal (BSM2-like)',
        'flowrate': 3400.0 / (20.0 * 24.0), 'temperature': 36.0, 'volume': 3400.0,
        'hrt_days': 20.0, 'k_l_a': 175.0, 'cod_in': 20000.0,
        'tss_in': 15000.0, 'tkn_in': 700.0, 'nh4_in': 400.0,
    },
    {
        'label': 'Charge élevée',
        'flowrate': 3400.0 / (15.0 * 24.0), 'temperature': 37.0, 'volume': 3400.0,
        'hrt_days': 15.0, 'k_l_a': 200.0, 'cod_in': 35000.0,
        'tss_in': 25000.0, 'tkn_in': 1000.0, 'nh4_in': 600.0,
    },
    {
        'label': 'Faible charge / long TRH',
        'flowrate': 3400.0 / (35.0 * 24.0), 'temperature': 34.0, 'volume': 3400.0,
        'hrt_days': 35.0, 'k_l_a': 100.0, 'cod_in': 8000.0,
        'tss_in': 5000.0, 'tkn_in': 300.0, 'nh4_in': 150.0,
    },
    {
        'label': 'Grand digesteur',
        'flowrate': 5000.0 / (25.0 * 24.0), 'temperature': 38.0, 'volume': 5000.0,
        'hrt_days': 25.0, 'k_l_a': 150.0, 'cod_in': 25000.0,
        'tss_in': 18000.0, 'tkn_in': 800.0, 'nh4_in': 500.0,
    },
]


def train_one(model_key, X_train_s, y_train, X_test_s, y_test):
    model = MODELS[model_key]()
    t0 = time.time()
    model.fit(X_train_s, y_train)
    elapsed = time.time() - t0

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    per_target = {}
    for i, col in enumerate(TARGET_COLS):
        per_target[col] = {
            'r2_train': r2_score(y_train[:, i], y_pred_train[:, i]),
            'r2_test': r2_score(y_test[:, i], y_pred_test[:, i]),
            'mae_test': mean_absolute_error(y_test[:, i], y_pred_test[:, i]),
        }

    r2_mean = r2_score(y_test, y_pred_test, multioutput='uniform_average')
    return model, per_target, r2_mean, elapsed


def main(data_path: Path, output_dir: Path):
    print(f"\nChargement : {data_path}")
    X, y = load_data(data_path)
    print(f"  {X.shape[0]} scénarios  |  {X.shape[1]} features  |  {y.shape[1]} targets")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    for key in MODELS:
        print(f"\n  Entraînement {key.upper()} ...", end='', flush=True)
        model, per_target, r2_mean, elapsed = train_one(key, X_train_s, y_train, X_test_s, y_test)
        results[key] = {'model': model, 'per_target': per_target, 'r2_mean': r2_mean, 'elapsed': elapsed}
        print(f" terminé en {elapsed:.1f}s  |  R² moyen = {r2_mean:.4f}")

        out = output_dir / f'adm1_{key}.pkl'
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump({
                'model': model, 'scaler': scaler,
                'feature_cols': FEATURE_COLS, 'target_cols': TARGET_COLS,
                'model_key': key, 'r2_test': r2_mean,
            }, f)

    # ------------------------------------------------------------------
    # Comparaison par target
    # ------------------------------------------------------------------
    col_w = 12
    model_keys = list(MODELS.keys())

    print(f"\n\n{'=' * 75}")
    print("  COMPARAISON R² TEST par target")
    print(f"{'=' * 75}")
    header = f"  {'Target':<22}" + "".join(f"{k.upper():>{col_w}}" for k in model_keys)
    print(header)
    print(f"  {'-' * 22}" + "-" * (col_w * len(model_keys)))
    for col in TARGET_COLS:
        row = f"  {col:<22}"
        best = max(results[k]['per_target'][col]['r2_test'] for k in model_keys)
        for k in model_keys:
            v = results[k]['per_target'][col]['r2_test']
            mark = '*' if abs(v - best) < 1e-9 else ' '
            row += f"{v:>{col_w - 1}.4f}{mark}"
        print(row)
    print(f"  {'-' * 22}" + "-" * (col_w * len(model_keys)))

    # R² moyen
    row = f"  {'R² moyen':<22}"
    best = max(results[k]['r2_mean'] for k in model_keys)
    for k in model_keys:
        v = results[k]['r2_mean']
        mark = '*' if abs(v - best) < 1e-9 else ' '
        row += f"{v:>{col_w - 1}.4f}{mark}"
    print(row)

    # Temps
    row = f"  {'Temps (s)':<22}"
    for k in model_keys:
        row += f"{results[k]['elapsed']:>{col_w-1}.1f} "
    print(row)
    print(f"{'=' * 75}")
    print("  * = meilleur sur cette target\n")

    # Benchmark d'inférence (temps d'appel)
    _benchmark_inference(results, scaler, model_keys)

    # Complexité calculatoire
    _analyze_complexity(results, model_keys, output_dir)


def _benchmark_inference(results: dict, scaler: StandardScaler, model_keys: list,
                         n_repeats: int = 1000) -> None:
    """Mesure le temps moyen d'une prédiction par modèle (1 scénario, n_repeats appels)."""
    X_single = np.array([[s[c] for c in FEATURE_COLS] for s in INFERENCE_SCENARIOS[:1]])
    X_single_s = scaler.transform(X_single)

    W = 75
    print(f"\n{'=' * W}")
    print(f"  BENCHMARK INFÉRENCE  ({n_repeats} appels × 1 scénario)")
    print(f"{'=' * W}")
    print(f"  {'Modèle':<12}  {'Temps/appel':>14}  {'Accélération vs ADM1':>22}")
    print(f"  {'-' * 12}  {'-' * 14}  {'-' * 22}")

    from processes.digester_process.anaerobic_digester_process import AnaerobicDigesterProcess
    from models.empyrical.adm1.fraction import ADM1Fraction

    s0 = INFERENCE_SCENARIOS[0]
    comps = ADM1Fraction.fractionate(cod=s0['cod_in'], tss=s0['tss_in'],
                                     tkn=s0['tkn_in'], nh4=s0['nh4_in'])
    adm1_cfg = {
        'volume': s0['volume'], 'V_gas': s0['volume'] * 300.0 / 3400.0,
        'temperature': s0['temperature'], 'waste_ratio': 0.0,
        'k_L_a': s0['k_l_a'], 'k_p': 50000.0,
        'initial_state': {**comps,
                          'x_c': 0.5, 'x_ch': 0.05, 'x_pr': 0.05, 'x_li': 0.05,
                          'x_su': 0.3, 'x_aa': 0.8, 'x_fa': 0.15,
                          'x_c4': 0.3, 'x_pro': 0.1, 'x_ac': 0.5, 'x_h2': 0.2,
                          's_ic': 0.06, 's_in': 0.05},
    }
    proc = AnaerobicDigesterProcess('bench', 'Bench', adm1_cfg)
    proc.initialize()
    inputs = {'flowrate': s0['flowrate'], 'temperature': s0['temperature'], 'components': comps}
    t0 = time.perf_counter()
    for _ in range(200):
        proc.process(inputs, 24.0)
    adm1_time = time.perf_counter() - t0
    print(f"  {'ADM1 (200j)':<12}  {adm1_time * 1000:>12.1f}ms  {'(référence)':>22}")

    for key in model_keys:
        model = results[key]['model']
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            model.predict(X_single_s)
        elapsed = (time.perf_counter() - t0) / n_repeats
        speedup = adm1_time / elapsed
        print(f"  {key.upper():<12}  {elapsed * 1000:>12.4f}ms  {speedup:>20.0f}×")

    print(f"{'=' * W}\n")


def _analyze_complexity(results: dict, model_keys: list, output_dir: Path) -> None:
    """Estime les FLOPs théoriques par inférence (1 scénario) et la taille des modèles sérialisés."""
    f = len(FEATURE_COLS)
    t = len(TARGET_COLS)

    def _flops_ridge(model):
        # StandardScaler : soustraction + division par feature
        scaler_ops = 2 * f
        # Ridge : produit scalaire (f mul + f add) + biais, par target
        ridge_ops = t * (2 * f + 1)
        return scaler_ops + ridge_ops

    def _flops_rf(model):
        # Traversée de chaque arbre jusqu'à la feuille (profondeur moyenne)
        avg_depth = float(np.mean([e.get_depth() for e in model.estimators_]))
        traversal = model.n_estimators * avg_depth * 2  # comparaison + branchement
        averaging = model.n_estimators * t  # sommation + division par cible
        return int(traversal + averaging)

    def _flops_gb(model):
        # MultiOutputRegressor : un GBR indépendant par cible
        total = 0
        for est in model.estimators_:
            avg_depth = float(np.mean([tree[0].get_depth() for tree in est.estimators_]))
            total += est.n_estimators * avg_depth * 2  # traversée
            total += est.n_estimators  # accumulation résidus + lr
        return int(total)

    def _flops_anfis(model):
        # MultiOutputRegressor : un ANFIS par cible
        r = model.estimators_[0].n_rules
        mf_ops = r * f * 5  # (x-c)/σ=2, carré=1, nég=1, exp=1
        firing_ops = r * (f + 1)  # f additions + 1 division (moyenne)
        norm_ops = r * 2  # somme des r poids + r divisions
        phi_ops = r * (f + 1)  # wbar_i * x pour chaque règle
        out_ops = 2 * r * (f + 1)  # produit scalaire theta
        denorm_ops = 2  # * std + mean
        return t * (mf_ops + firing_ops + norm_ops + phi_ops + out_ops + denorm_ops)

    flops_fn = {'ridge': _flops_ridge, 'rf': _flops_rf, 'gb': _flops_gb, 'anfis': _flops_anfis}

    W = 75
    print(f"\n{'=' * W}")
    print(f"  COMPLEXITÉ CALCULATOIRE  (par inférence, 1 scénario)")
    print(f"{'=' * W}")
    print(f"  {'Modèle':<12}  {'FLOPs estimés':>16}  {'Taille modèle':>14}  {'Paramètres':>12}")
    print(f"  {'-' * 12}  {'-' * 16}  {'-' * 14}  {'-' * 12}")

    for key in model_keys:
        model = results[key]['model']
        flops = flops_fn[key](model)

        pkl_path = output_dir / f'adm1_{key}.pkl'
        if pkl_path.exists():
            kb = pkl_path.stat().st_size / 1024
            size_str = f"{kb:.0f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"
        else:
            size_str = "—"

        # Estimation grossière du nombre de paramètres apprenables
        if key == 'ridge':
            ridge_model = model.named_steps['ridge']
            n_params = sum(e.coef_.size + e.intercept_.size for e in ridge_model.estimators_)
        elif key == 'rf':
            n_params = sum(e.tree_.node_count for e in model.estimators_)
        elif key == 'gb':
            n_params = sum(
                sum(tree[0].tree_.node_count for tree in est.estimators_)
                for est in model.estimators_
            )
        elif key == 'anfis':
            r = model.estimators_[0].n_rules
            n_params = t * (r * f * 2 + r * (f + 1))  # centres+sigmas + theta
        else:
            n_params = 0

        print(f"  {key.upper():<12}  {flops:>16,}  {size_str:>14}  {n_params:>12,}")

    print(f"{'=' * W}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/adm1_training_data.csv')
    parser.add_argument('--output', type=str, default='models/trained')
    args = parser.parse_args()
    main(ROOT / args.data, ROOT / args.output)
