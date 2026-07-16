"""
Validation de l'implémentation ADM1.

Lance une simulation de 200 jours avec un influent typique de boues secondaires
et compare les résultats à l'état quasi-stationnaire aux ordres de grandeur
publiés dans la littérature (Batstone et al., 2002 ; Rosen & Jeppsson, 2006).

Usage :
    python tools/validate_adm1.py
    python tools/validate_adm1.py --config config/adm1_validation.json
"""
import json
import argparse
import logging
import sys
from pathlib import Path

# Ajoute la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestrator.simulation_orchestrator import SimulationOrchestrator
from core.process.process_factory import ProcessFactory

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Ordres de grandeur de référence (littérature ADM1)
# Source : Batstone et al. (2002), Rosen & Jeppsson (2006)
# ---------------------------------------------------------------------------
REFERENCE = {
    'pH':              {'min': 7.0,  'max': 7.8,   'unit': '—'},
    'vfa_total':       {'min': 50.0, 'max': 400.0,  'unit': 'mg COD/L'},
    's_ac':            {'min': 30.0, 'max': 250.0,  'unit': 'mg COD/L'},
    'ch4_m3_per_day':  {'min': 50.0, 'max': 2000.0, 'unit': 'm³ CH4/j'},
    'cod_removal_rate':{'min': 30.0, 'max': 70.0,   'unit': '%'},
    'nh4':             {'min': 500.0,'max': 2000.0,  'unit': 'mg N/L'},
}


def run_validation(config_path: str) -> None:
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Validation ADM1 — {config['name']}")
    print(f"{'='*60}\n")

    orchestrator = SimulationOrchestrator(config)
    processes = ProcessFactory.create_from_config(config)
    for p in processes:
        orchestrator.add_process(p)
    orchestrator.initialize()

    print("Simulation en cours (200 jours)...")
    results = orchestrator.run()
    print("Simulation terminée.\n")

    # Récupère le dernier pas de temps du digesteur
    history = results['history'].get('digesteur', [])
    if not history:
        print("ERREUR : aucun historique pour le digesteur.")
        return

    # Moyenne des 7 derniers jours pour lisser les oscillations résiduelles
    n_avg = min(7 * 24, len(history))
    last_steps = history[-n_avg:]

    def avg(key):
        vals = [s.get(key, 0) for s in last_steps if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    # Récupère le pH depuis les composants ADM1
    def avg_ph():
        vals = []
        for s in last_steps:
            comp = s.get('components', {})
            if 'pH' in comp:
                vals.append(comp['pH'])
            elif 'ph' in comp:
                vals.append(comp['ph'])
        return sum(vals) / len(vals) if vals else None

    obtained = {
        'pH':               avg_ph(),
        'vfa_total':        avg('vfa_total'),
        's_ac':             None,  # extrait ci-dessous
        'ch4_m3_per_day':   avg('ch4_m3_per_day'),
        'cod_removal_rate': avg('cod_removal_rate'),
        'nh4':              avg('nh4'),
    }

    # Extraction s_ac depuis components
    s_ac_vals = []
    for s in last_steps:
        comp = s.get('components', {})
        s_ac = comp.get('s_ac', None)
        if s_ac is not None:
            s_ac_vals.append(s_ac * 1000.0)  # kg COD/m³ → mg COD/L
    obtained['s_ac'] = sum(s_ac_vals) / len(s_ac_vals) if s_ac_vals else 0.0

    # ---------------------------------------------------------------------------
    # Tableau de comparaison
    # ---------------------------------------------------------------------------
    print(f"{'Variable':<22} {'Obtenu':>12} {'Min réf.':>10} {'Max réf.':>10} {'Unité':<16} {'Statut'}")
    print('-' * 80)

    all_ok = True
    for var, ref in REFERENCE.items():
        val = obtained.get(var)
        if val is None:
            status = '? N/A'
            row = f"{'—':>12}"
        else:
            in_range = ref['min'] <= val <= ref['max']
            status = 'OK' if in_range else 'HORS PLAGE'
            if not in_range:
                all_ok = False
            row = f"{val:>12.2f}"
        print(f"{var:<22} {row} {ref['min']:>10.1f} {ref['max']:>10.1f} {ref['unit']:<16} {status}")

    print('-' * 80)
    print(f"\nBilan : {'OK - Toutes les variables dans les plages de reference' if all_ok else 'ATTENTION - Certaines variables hors plage'}\n")

    # ---------------------------------------------------------------------------
    # Export JSON des résultats bruts
    # ---------------------------------------------------------------------------
    out = {
        'config': config['name'],
        'n_steps_averaged': n_avg,
        'results': {k: v for k, v in obtained.items() if v is not None},
        'reference': REFERENCE,
    }
    out_path = Path('tools') / 'adm1_validation_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Résultats bruts exportés → {out_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/adm1_validation.json')
    args = parser.parse_args()
    run_validation(args.config)
