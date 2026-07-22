"""
Générateur de données d'entraînement pour les modèles ML basés sur ADM1.

Utilise directement AnaerobicDigesterProcess pour garantir la cohérence
avec le modèle validé (phase gazeuse, Henry, bilans N/C corrigés).

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
from tqdm import tqdm

from models.empyrical.adm1.model import ADM1Model
from models.empyrical.adm1.fraction import ADM1Fraction
from processes.digester_process.anaerobic_digester_process import AnaerobicDigesterProcess

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plages de variation des scénarios influent + réacteur
# ---------------------------------------------------------------------------
INFLUENT_RANGES = {
    'cod':         (5000.0,  40000.0),  # mg/L DCO totale
    'tss':         (3000.0,  30000.0),  # mg/L MES
    'tkn':         (200.0,   1200.0),   # mg/L NTK
    'nh4':         (100.0,   800.0),    # mg/L NH4
    'temperature': (30.0,    42.0),     # °C
}
REACTOR_RANGES = {
    'volume':   (500.0,  5000.0),   # m³
    'hrt_days': (10.0,   40.0),     # jours
    'k_L_a':    (50.0,   300.0),    # j⁻¹
}

STEADY_STATE_HOURS = 200 * 24   # 200 jours en heures (pas de temps du process)
TIMESTEP_H = 24.0               # pas de temps : 1 jour par appel process

# Inoculation initiale pour éviter le cold-start
SEED_BIOMASS = {
    'x_c': 0.5,  'x_ch': 0.05, 'x_pr': 0.05, 'x_li': 0.05,
    'x_su': 0.3, 'x_aa': 0.8,  'x_fa': 0.15,
    'x_c4': 0.3, 'x_pro': 0.1, 'x_ac': 0.5,  'x_h2': 0.2,
    's_ic': 0.06, 's_in': 0.05,
}


def _sample_scenarios(n: int, rng: np.random.Generator) -> list:
    scenarios = []
    all_ranges = {**INFLUENT_RANGES, **REACTOR_RANGES}
    for _ in range(n):
        s = {}
        for key, (lo, hi) in all_ranges.items():
            s[key] = float(rng.uniform(lo, hi))
        s['nh4'] = min(s['nh4'], s['tkn'] * 0.85)
        s['flowrate'] = s['volume'] / (s['hrt_days'] * 24.0)
        scenarios.append(s)
    return scenarios


def _run_to_steady_state(scenario: dict, components: dict) -> dict | None:
    """
    Intègre le digesteur jusqu'à l'état quasi-stationnaire via AnaerobicDigesterProcess.
    Retourne le dernier output ou None si échec.
    """
    config = {
        'volume':      scenario['volume'],
        'V_gas':       scenario['volume'] * 300.0 / 3400.0,  # V_gas proportionnel
        'temperature': scenario['temperature'],
        'waste_ratio': 0.0,
        'k_L_a':       scenario['k_L_a'],
        'k_p':         50000.0,
        'initial_state': {**components, **SEED_BIOMASS},
    }

    process = AnaerobicDigesterProcess(
        node_id='gen_digesteur',
        name='Générateur',
        config=config,
    )
    process.initialize()

    inputs = {
        'flowrate':    scenario['flowrate'],
        'temperature': scenario['temperature'],
        'components':  components,
    }

    output = None
    n_steps = int(STEADY_STATE_HOURS / TIMESTEP_H)

    try:
        for _ in range(n_steps):
            output = process.process(inputs, TIMESTEP_H)
    except Exception as e:
        logger.debug(f"Échec intégration : {e}")
        return None

    return output


def _extract_row(scenario: dict, output: dict) -> dict | None:
    """Extrait features + targets depuis le dernier output. Retourne None si invalide."""
    comps = output.get('components', {})
    ph    = comps.get('pH', output.get('pH'))

    vfa_total      = output.get('vfa_total', 0.0) / 1000.0   # g/m³ → kg/m³
    biomass_active = sum(comps.get(k, 0.0) for k in
                         ['x_su', 'x_aa', 'x_fa', 'x_c4', 'x_pro', 'x_ac', 'x_h2'])
    cod_out        = output.get('cod', 0.0) / 1000.0          # g/m³ → kg/m³

    # Filtres de validité physique
    if biomass_active < 0.01:
        return None
    if vfa_total > 3.0:
        return None
    if ph is None or not (4.0 <= ph <= 9.0):
        return None

    return {
        # Features
        'flowrate':    scenario['flowrate'],
        'temperature': scenario['temperature'],
        'volume':      scenario['volume'],
        'hrt_days':    scenario['hrt_days'],
        'k_l_a':       scenario['k_L_a'],
        'cod_in':      scenario['cod'],
        'tss_in':      scenario['tss'],
        'tkn_in':      scenario['tkn'],
        'nh4_in':      scenario['nh4'],
        # Targets
        'cod_out_kg_m3':       cod_out,
        'tss_out_kg_m3':       output.get('tss', 0.0) / 1000.0,
        'nh4_out_mg_l':        output.get('nh4', 0.0),
        'vfa_total_kg_m3':     vfa_total,
        'biomass_active_kg_m3': biomass_active,
        's_ac_kg_m3':          comps.get('s_ac', 0.0),
        'x_ac_kg_m3':          comps.get('x_ac', 0.0),
        'x_h2_kg_m3':          comps.get('x_h2', 0.0),
        'ch4_m3_per_day':      output.get('ch4_m3_per_day', 0.0),
        'ch4_energy_kwh':      output.get('ch4_energy_kwh', 0.0),
        'p_ch4_bar':           output.get('p_ch4_bar', 0.0),
        'p_co2_bar':           output.get('p_co2_bar', 0.0),
        'pH':                  ph,
    }


def generate(n: int, output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scenarios = _sample_scenarios(n * 3, rng)

    records = []
    skipped = 0

    with tqdm(total=n, desc="Génération ADM1", unit="scén.") as pbar:
        for i, s in enumerate(scenarios):
            if len(records) >= n:
                break
            try:
                components = ADM1Fraction.fractionate(
                    cod=s['cod'], tss=s['tss'], tkn=s['tkn'], nh4=s['nh4'],
                )
                out = _run_to_steady_state(s, components)
                if out is None:
                    skipped += 1
                    continue

                row = _extract_row(s, out)
                if row is None:
                    skipped += 1
                    continue

                records.append(row)
                pbar.update(1)
                pbar.set_postfix(filtrés=skipped)

            except Exception as e:
                logger.debug(f"Scénario {i} ignoré : {e}")
                skipped += 1

    df = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nFichier généré : {output}")
    print(f"  Lignes    : {len(df)}")
    print(f"  Filtrés   : {skipped}")
    print(f"  Colonnes  : {list(df.columns)}")
    if not df.empty:
        targets = ['cod_out_kg_m3', 'tss_out_kg_m3', 'nh4_out_mg_l',
                   'vfa_total_kg_m3', 'ch4_m3_per_day', 'pH']
        print(f"\nAperçu statistiques :")
        print(df[targets].describe().round(4).to_string())
    else:
        print("\nAucun scénario valide — vérifier les filtres et le modèle.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Génère un dataset d'entraînement ML via simulations ADM1"
    )
    parser.add_argument('--n',      type=int, default=1000,
                        help='Nombre de scénarios valides (défaut: 1000)')
    parser.add_argument('--output', type=str,
                        default='data/processed/adm1_training_data.csv',
                        help='Chemin de sortie CSV')
    parser.add_argument('--seed',   type=int, default=42,
                        help='Graine aléatoire (défaut: 42)')
    args = parser.parse_args()

    generate(n=args.n, output=ROOT / args.output, seed=args.seed)
