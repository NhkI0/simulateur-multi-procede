"""
Interprétabilité des modèles ML avec SHAP et LIME.

- SHAP : TreeExplainer pour RF et GB, LinearExplainer pour Ridge,
         KernelExplainer pour ANFIS
- LIME : LimeTabularExplainer pour tous les modèles

Usage :
    python tools/explain_models.py
    python tools/explain_models.py --target ch4_m3_per_day
    python tools/explain_models.py --target pH --scenario 5
"""
import sys
import argparse
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import shap
from lime import lime_tabular
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from tools.train_adm1_models import FEATURE_COLS, TARGET_COLS
from models.ml.anfis_sugeno import ANFISSugeno

FIGURES_DIR = ROOT / 'reports' / 'figures'
MODELS_DIR = ROOT / 'models' / 'trained'


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'adm1_training_data.csv')
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COLS].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# Chargement / ré-entraînement des modèles
# ---------------------------------------------------------------------------

def _load_pkl(key):
    path = MODELS_DIR / f'adm1_{key}.pkl'
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)['model']
    except Exception:
        return None


def get_models(X_train_s, y_train):
    """Charge depuis pkl si possible, ré-entraîne sinon (n_jobs=1)."""
    models = {}

    for key in ('rf', 'ridge'):
        m = _load_pkl(key)
        if m is not None:
            models[key] = m
            print(f'  {key.upper():<10} chargé depuis pkl')

    for key, factory in (
            ('gb', lambda: MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, random_state=42),
                n_jobs=1)),
            ('anfis', lambda: MultiOutputRegressor(
                ANFISSugeno(n_rules=15, n_epochs=400, lr=0.01, random_state=42),
                n_jobs=1)),
    ):
        m = _load_pkl(key)
        if m is not None:
            models[key] = m
            print(f'  {key.upper():<10} chargé depuis pkl')
        else:
            print(f'  {key.upper():<10} ré-entraînement ...', end='', flush=True)
            m = factory()
            m.fit(X_train_s, y_train)
            models[key] = m
            print(' OK')

    return models


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _save(fig, path):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    → {path.relative_to(ROOT)}')


def shap_tree(model, X_test_s, target_idx, label, key):
    """TreeExplainer pour RF (multi-output natif) ou GB (MultiOutputRegressor)."""

    if key == 'rf':
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test_s)
        # shap_vals : liste[n_outputs] ou array (n, f, n_outputs)
        if isinstance(shap_vals, list):
            sv = shap_vals[target_idx]
        else:
            sv = shap_vals[:, :, target_idx]
        base = (explainer.expected_value[target_idx]
                if hasattr(explainer.expected_value, '__len__')
                else explainer.expected_value)
    else:
        # GB : MultiOutputRegressor → un GBR par cible
        gbr = model.estimators_[target_idx]
        explainer = shap.TreeExplainer(gbr)
        sv = explainer.shap_values(X_test_s)
        ev = explainer.expected_value
        base = float(ev[0]) if hasattr(ev, '__len__') else float(ev)

    # Beeswarm (importance globale)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(sv, X_test_s, feature_names=FEATURE_COLS,
                      show=False, plot_type='dot')
    ax.set_title(f'SHAP beeswarm — {key.upper()} — {label}', fontsize=11)
    plt.tight_layout()
    _save(fig, FIGURES_DIR / f'shap_{key}_{label}_beeswarm.png')

    # Waterfall (1er scénario de test)
    shap_exp = shap.Explanation(
        values=sv[0],
        base_values=base,
        data=X_test_s[0],
        feature_names=FEATURE_COLS,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(shap_exp, show=False)
    ax.set_title(f'SHAP waterfall — {key.upper()} — {label} (scénario #0)', fontsize=11)
    plt.tight_layout()
    _save(fig, FIGURES_DIR / f'shap_{key}_{label}_waterfall.png')


def shap_linear(model, X_train_s, X_test_s, target_idx, label):
    """LinearExplainer pour Ridge (Pipeline → MultiOutputRegressor)."""

    # Extraire le Ridge sous-jacent pour la cible
    ridge_estimator = model.named_steps['ridge'].estimators_[target_idx]
    scaler_step = model.named_steps['scaler_y']

    # Transformer X dans l'espace du Ridge (X déjà mis à l'échelle features)
    explainer = shap.LinearExplainer(ridge_estimator, X_train_s)
    sv = explainer.shap_values(X_test_s)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(sv, X_test_s, feature_names=FEATURE_COLS,
                      show=False, plot_type='dot')
    ax.set_title(f'SHAP beeswarm — RIDGE — {label}', fontsize=11)
    plt.tight_layout()
    _save(fig, FIGURES_DIR / f'shap_ridge_{label}_beeswarm.png')


def shap_kernel(model, X_train_s, X_test_s, target_idx, label, n_bg=50):
    """KernelExplainer (model-agnostic) pour ANFIS."""

    def predict_fn(X):
        return model.predict(X)[:, target_idx]

    # Résumé du background par k-means pour accélérer
    background = shap.kmeans(X_train_s, n_bg)
    explainer = shap.KernelExplainer(predict_fn, background)
    sv = explainer.shap_values(X_test_s[:50], silent=True)  # sous-échantillon

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(sv, X_test_s[:50], feature_names=FEATURE_COLS,
                      show=False, plot_type='dot')
    ax.set_title(f'SHAP beeswarm — ANFIS — {label}', fontsize=11)
    plt.tight_layout()
    _save(fig, FIGURES_DIR / f'shap_anfis_{label}_beeswarm.png')


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def explain_lime(models, X_train_s, X_test_s, target_idx, label, scenario_idx=0):
    explainer = lime_tabular.LimeTabularExplainer(
        X_train_s,
        feature_names=FEATURE_COLS,
        mode='regression',
        verbose=False,
    )

    x = X_test_s[scenario_idx]

    for key, model in models.items():
        def predict_fn(X, m=model, ti=target_idx):
            return m.predict(X)[:, ti]

        exp = explainer.explain_instance(x, predict_fn,
                                         num_features=len(FEATURE_COLS))
        fig = exp.as_pyplot_figure()
        fig.suptitle(f'LIME — {key.upper()} — {label}  (scénario test #{scenario_idx})',
                     fontsize=10)
        plt.tight_layout()
        _save(fig, FIGURES_DIR / f'lime_{key}_{label}.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(target_name: str, scenario_idx: int):
    print(f'\nChargement des données ...')
    X_train_s, X_test_s, y_train, y_test, scaler = load_data()
    target_idx = TARGET_COLS.index(target_name)
    label = target_name.replace('/', '_')
    print(f'  Target : {target_name}  (index {target_idx})')

    print(f'\nChargement / entraînement des modèles ...')
    models = get_models(X_train_s, y_train)

    # ------------------------------------------------------------------
    print(f'\nSHAP ...')
    for key in ('rf', 'gb'):
        if key in models:
            print(f'  TreeExplainer — {key.upper()}')
            shap_tree(models[key], X_test_s, target_idx, label, key)

    if 'ridge' in models:
        print(f'  LinearExplainer — RIDGE')
        shap_linear(models['ridge'], X_train_s, X_test_s, target_idx, label)

    if 'anfis' in models:
        print(f'  KernelExplainer — ANFIS (peut prendre quelques minutes)')
        shap_kernel(models['anfis'], X_train_s, X_test_s, target_idx, label)

    # ------------------------------------------------------------------
    print(f'\nLIME ...')
    explain_lime(models, X_train_s, X_test_s, target_idx, label, scenario_idx)

    print(f'\nFigures dans : reports/figures/\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='ch4_m3_per_day',
                        choices=TARGET_COLS)
    parser.add_argument('--scenario', type=int, default=0,
                        help='Index du scénario de test pour LIME et SHAP waterfall')
    args = parser.parse_args()
    main(args.target, args.scenario)
