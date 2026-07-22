"""
Validation de l'implementation ADM1 contre Rosen & Jeppsson (2006).

Reference : Rosen, C. & Jeppsson, U. (2006). Aspects on ADM1 Implementation
within the BSM2 Framework. Department of Industrial Electrical Engineering
and Automation, Lund University, Sweden.

Usage :
    python tools/validate_adm1.py
    python tools/validate_adm1.py --config config/adm1_validation_rosen.json
"""
import json
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from core.orchestrator.simulation_orchestrator import SimulationOrchestrator
from core.process.process_factory import ProcessFactory

logging.basicConfig(level=logging.WARNING)


def _run_with_progress(orchestrator: SimulationOrchestrator) -> dict:
    """Remplace orchestrator.run() avec une barre de progression tqdm."""
    orchestrator.is_running = True
    total = orchestrator.state.total_steps

    with tqdm(total=total, unit='h', desc='Simulation ADM1') as bar:
        while orchestrator.state.current_time < orchestrator.state.end_time:
            orchestrator._run_timestep()
            orchestrator.state.advance()
            bar.update(1)

    orchestrator.is_running = False
    metadata = {
        'sim_name':       orchestrator.config.get('name', 'simulation'),
        'start_time':     str(orchestrator.state.start_time),
        'end_time':       str(orchestrator.state.end_time),
        'total_hours':    (orchestrator.state.end_time - orchestrator.state.start_time).total_seconds() / 3600,
        'timestep':       orchestrator.state.timestep,
        'steps_completed': orchestrator.state.current_step,
    }
    return orchestrator.result_manager.collect(metadata)

# ---------------------------------------------------------------------------
# Valeurs de reference a l'etat stationnaire — Rosen & Jeppsson (2006)
# ---------------------------------------------------------------------------
REFERENCE = {
    's_su':   {'value': 0.0119548297170,  'unit': 'kg COD/m3'},
    's_aa':   {'value': 0.0053147401716,  'unit': 'kg COD/m3'},
    's_fa':   {'value': 0.0986214009308,  'unit': 'kg COD/m3'},
    's_va':   {'value': 0.0116250064639,  'unit': 'kg COD/m3'},
    's_bu':   {'value': 0.0132507296663,  'unit': 'kg COD/m3'},
    's_pro':  {'value': 0.0157836662845,  'unit': 'kg COD/m3'},
    's_ac':   {'value': 0.1976297169375,  'unit': 'kg COD/m3'},
    's_h2':   {'value': 0.0000002359451,  'unit': 'kg COD/m3'},
    's_ch4':  {'value': 0.0550887764460,  'unit': 'kg COD/m3'},
    's_ic':   {'value': 0.1526778706263,  'unit': 'kmole C/m3'},
    's_in':   {'value': 0.1302298158037,  'unit': 'kmole N/m3'},
    'x_c':    {'value': 0.3086976637215,  'unit': 'kg COD/m3'},
    'x_ch':   {'value': 0.0279472404350,  'unit': 'kg COD/m3'},
    'x_pr':   {'value': 0.1025741061067,  'unit': 'kg COD/m3'},
    'x_li':   {'value': 0.0294830497073,  'unit': 'kg COD/m3'},
    'x_su':   {'value': 0.4201659824546,  'unit': 'kg COD/m3'},
    'x_aa':   {'value': 1.1791717989237,  'unit': 'kg COD/m3'},
    'x_fa':   {'value': 0.2430353447194,  'unit': 'kg COD/m3'},
    'x_c4':   {'value': 0.4319211056360,  'unit': 'kg COD/m3'},
    'x_pro':  {'value': 0.1373059089340,  'unit': 'kg COD/m3'},
    'x_ac':   {'value': 0.7605626583132,  'unit': 'kg COD/m3'},
    'x_h2':   {'value': 0.3170229533613,  'unit': 'kg COD/m3'},
    'x_i':    {'value': 25.6173953274430, 'unit': 'kg COD/m3'},
    'pH':     {'value': 7.4655377698929,  'unit': '—'},
}

TOLERANCE_PCT = 5.0  # ecart acceptable en %


def run_validation(config_path: str) -> None:
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    print(f"\n{'='*70}")
    print(f"  Validation ADM1 — Rosen & Jeppsson (2006)")
    print(f"  Config : {config['name']}")
    print(f"{'='*70}\n")

    orchestrator = SimulationOrchestrator(config)
    processes = ProcessFactory.create_from_config(config)
    for p in processes:
        orchestrator.add_process(p)
    orchestrator.initialize()

    from models.empyrical.adm1.kinetics import calculate_pH
    state0 = next(p for p in orchestrator.process_nodes if p.node_id == 'digesteur').concentrations
    ph0, _, _ = calculate_pH(state0, next(p for p in orchestrator.process_nodes if p.node_id == 'digesteur').model_instance.params)
    print(f"pH initial (t=0) : {ph0:.4f}")

    results = _run_with_progress(orchestrator)

    history = results['history'].get('digesteur', [])
    if not history:
        print("ERREUR : aucun historique pour le digesteur.")
        return

    # Moyenne sur les 7 derniers jours
    n_avg = min(7 * 24, len(history))
    last_steps = history[-n_avg:]

    # Extrait les composants moyens
    comp_avg = {}
    for step in last_steps:
        for k, v in step.get('components', {}).items():
            comp_avg.setdefault(k, []).append(v)
    comp_avg = {k: sum(v) / len(v) for k, v in comp_avg.items()}

    # pH moyen
    ph_vals = [s.get('components', {}).get('pH') or
               s.get('components', {}).get('ph')
               for s in last_steps]
    ph_vals = [v for v in ph_vals if v is not None]
    ph_avg = sum(ph_vals) / len(ph_vals) if ph_vals else None

    # ---------------------------------------------------------------------------
    # Tableau de comparaison
    # ---------------------------------------------------------------------------
    print(f"{'Variable':<10} {'Reference':>16} {'Obtenu':>16} {'Ecart (%)':>10}  {'Statut'}")
    print('-' * 65)

    all_ok = True
    report_rows = []

    for var, ref in REFERENCE.items():
        ref_val = ref['value']
        if var == 'pH':
            obtained = ph_avg
        else:
            obtained = comp_avg.get(var)

        if obtained is None:
            status = 'N/A'
            ecart_str = '—'
            ecart = None
        else:
            ecart = (obtained - ref_val) / ref_val * 100
            ecart_str = f"{ecart:+.2f}%"
            in_tol = abs(ecart) <= TOLERANCE_PCT
            status = 'OK' if in_tol else 'HORS TOLERANCE'
            if not in_tol:
                all_ok = False

        print(f"{var:<10} {ref_val:>16.6f} {obtained or 0:>16.6f} {ecart_str:>10}  {status}")
        report_rows.append({
            'variable': var,
            'reference': ref_val,
            'obtained': obtained,
            'ecart_pct': ecart,
            'unit': ref['unit'],
            'status': status,
        })

    print('-' * 65)
    verdict = 'VALIDATION OK' if all_ok else 'VALIDATION PARTIELLE — voir variables hors tolerance'
    print(f"\nBilan ({TOLERANCE_PCT}% tolerance) : {verdict}\n")

    # Export JSON
    out = {
        'config': config['name'],
        'reference': 'Rosen & Jeppsson (2006)',
        'tolerance_pct': TOLERANCE_PCT,
        'n_steps_averaged': n_avg,
        'verdict': verdict,
        'rows': report_rows,
    }
    out_path = Path('tools') / 'adm1_validation_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Resultats exportes -> {out_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/adm1_validation_rosen.json')
    args = parser.parse_args()
    run_validation(args.config)
