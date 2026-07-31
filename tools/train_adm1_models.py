"""
Entraînement de modèles ML sur les données synthétiques ADM1.

Modifier FEATURE_COLS et TARGET_COLS pour changer ce qui est prédit.

Usage :
    python tools/train_adm1_models.py
    python tools/train_adm1_models.py --data data/processed/adm1_training_data.csv
    python tools/train_adm1_models.py --model rf --output models/saved/adm1_rf.pkl
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from models.ml.anfis_sugeno import ANFISSugeno

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Colonnes à modifier selon ce que l'on veut prédire
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'flowrate',
    'temperature',
    'volume',
    'hrt_days',
    'k_l_a',
    'cod_in',
    'tss_in',
    'tkn_in',
    'nh4_in',
]

TARGET_COLS = [
    'ch4_m3_per_day',    # Production biogaz
    'pH',                # Stabilité du procédé
    'cod_out_kg_m3',     # Efficacité épuration
    'nh4_out_mg_l',      # Azote ammoniacal sortie
    'x_ac_kg_m3',        # Biomasse acétoclaste
]

MODELS = {
    'rf': lambda: RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    'gb': lambda: MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42), n_jobs=-1),
    'ridge': lambda: Pipeline([
        ('scaler_y', StandardScaler()),
        ('ridge',    MultiOutputRegressor(Ridge(alpha=1.0))),
    ]),
    'anfis': lambda: MultiOutputRegressor(
        ANFISSugeno(n_rules=15, n_epochs=400, lr=0.01, random_state=42), n_jobs=-1),
}


def load_data(data_path: Path):
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURE_COLS + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COLS].values.astype(float)
    return X, y


def train(data_path: Path, model_key: str, output: Path, test_size: float = 0.2):
    print(f"\nChargement : {data_path}")
    X, y = load_data(data_path)
    print(f"  {X.shape[0]} scénarios  |  {X.shape[1]} features  |  {y.shape[1]} targets")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MODELS[model_key]()
    print(f"\nEntraînement : {model_key.upper()} ...")
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    print("\n" + "=" * 65)
    print(f"{'Target':<25} {'R² train':>10} {'R² test':>10} {'MAE test':>12}")
    print("-" * 65)
    for i, col in enumerate(TARGET_COLS):
        r2_tr = r2_score(y_train[:, i], y_pred_train[:, i])
        r2_te = r2_score(y_test[:, i], y_pred_test[:, i])
        mae_te = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
        print(f"  {col:<23} {r2_tr:>10.4f} {r2_te:>10.4f} {mae_te:>12.4f}")
    print("=" * 65)

    r2_global = r2_score(y_test, y_pred_test, multioutput='uniform_average')
    print(f"\n  R² moyen (test) : {r2_global:.4f}")

    # MultiOutputRegressor wraps per-target estimators, aggregate importances
    if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
        importances = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
        order = np.argsort(importances)[::-1]
        print("\n  Importance des features (moyenne sur les targets) :")
        for rank, idx in enumerate(order[:8]):
            print(f"    {rank + 1:2d}. {FEATURE_COLS[idx]:<20}  {importances[idx]:.4f}")
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        print("\n  Importance des features :")
        for rank, idx in enumerate(order[:8]):
            print(f"    {rank + 1:2d}. {FEATURE_COLS[idx]:<20}  {importances[idx]:.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': FEATURE_COLS,
            'target_cols': TARGET_COLS,
            'model_key': model_key,
            'r2_test': r2_global,
        }, f)
    print(f"\n  Modèle sauvegardé : {output}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entraîne un modèle ML sur les données ADM1")
    parser.add_argument('--data', type=str, default='data/processed/adm1_training_data.csv')
    parser.add_argument('--model', type=str, default='rf', choices=list(MODELS),
                        help='rf = Random Forest, gb = Gradient Boosting, ridge = Ridge, anfis = ANFIS Sugeno')
    parser.add_argument('--output', type=str, default='models/trained/adm1_{model}.pkl')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    output_path = ROOT / args.output.format(model=args.model)
    train(
        data_path=ROOT / args.data,
        model_key=args.model,
        output=output_path,
        test_size=args.test_size,
    )
