"""IPRO's final fronts and run checks.

For every IPRO run <results_root>/<prefix>/ipro/<env>/s_<seed>, the final front is the non-dominated set of every
oracle answer in oracle_queries.jsonl. Per run the script also reports whether IPRO returned, why it stopped, the oracle
calls, replays, training steps, the pretraining share and the share of answers that do not strictly dominate their
referent; on DST, box_point_learned says whether an oracle answer reached the shallowest treasure.

Outputs (in --output): ipro_fronts.csv (read by build_reference_fronts.py) and ipro_runs.csv (one row per run).

  python check_ipro_runs.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_reference_fronts import nondominated


def newest(run_dir: Path, pattern: str) -> Path | None:
    """Most recently modified file below run_dir that matches pattern."""
    files = sorted(run_dir.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def runtime_config(run_dir: Path) -> dict:
    """Wrapper configuration: the JSON with 'oracle' and 'outer_loop' sections."""
    for path in sorted(run_dir.rglob('*.json')):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and 'oracle' in data and 'outer_loop' in data:
            return data
    return {}


def check(env: str,
          seed: int,
          run_dir: Path,
          atol: float) -> tuple[dict, np.ndarray]:
    """Checks of one IPRO run and its final front."""
    config = runtime_config(run_dir)
    outer = config.get('outer_loop', {})
    method = outer.get('method', 'IPRO')
    tolerance = float(outer.get('tolerance', 1e-6 if method == 'IPRO-2D' else 0.1))
    path = newest(run_dir, 'summary.json')
    summary = json.loads(path.read_text()) if path else None
    calls = []
    if path := newest(run_dir, 'oracle_queries.jsonl'):
        calls = [json.loads(line) for line in path.read_text().split('\n')[:-1] if line]
    answers = np.array([c['returned_vector'] for c in calls], float) if calls else np.empty((0, 0))
    answers = answers[np.isfinite(answers).all(1)]
    calls_made = summary['oracle_calls'] if summary else len(calls)
    if summary is None:
        stop = 'did not return'
    elif calls_made == 0:
        stop = 'initial phase'
    elif 1 - summary['coverage'] <= tolerance:
        stop = 'coverage'
    elif outer.get('max_iterations') is not None and calls_made >= outer['max_iterations']:
        stop = 'iteration cap'
    else:
        stop = 'box queue empty' if method == 'IPRO-2D' else 'unclear'
    front = nondominated(answers) if len(answers) else answers
    row = dict(environment=env, seed=seed, returned=summary is not None, stop=stop, oracle_calls=calls_made,
               failed_share=float(np.mean([not c.get('strict_improvement') for c in calls])) if calls else None,
               front_points=len(front), gamma=config.get('gamma'),
               step=summary['training_steps'] if summary else calls[-1]['training_steps'] if calls else None)
    if summary:
        row.update(replays=summary['replays'], training_steps=summary['training_steps'],
                   online_steps=summary['online_steps'],
                   pretraining_share=summary['pretraining_steps'] / max(summary['training_steps'], 1))
    if config.get('bounds_mode') == 'dst_map':
        shallow = np.asarray(config['maximals'], float)[1]
        row['box_point_learned'] = bool(len(answers) and (np.abs(answers - shallow).max(1) <= atol).any())
    return row | dict(run_dir=str(run_dir)), front


def main() -> None:
    """Checks every IPRO run and writes the run table and the final fronts."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_root', type=Path, default=Path('results'))
    parser.add_argument('--prefix', default='scalarization/baselines',
                        help="Launchers' path prefix below --results_root")
    parser.add_argument('--output', type=Path, default=Path('results/evaluation/scalarization/ipro_checks'))
    parser.add_argument('--envs', nargs='+',
                        default=['dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2', 'minecart', 'reacher'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--atol', type=float, default=1e-6, help='Tolerance for matching the DST box point')
    args = parser.parse_args()

    # Run checks
    runs = []
    fronts = []
    for env in args.envs:
        base = args.results_root / args.prefix / 'ipro' / env
        for seed in args.seeds:
            found = sorted(p for p in base.rglob(f's_{seed:04d}') if p.is_dir())
            if len(found) != 1:
                runs.append(dict(environment=env, seed=seed, stop=f'{len(found)} run folders'))
                continue
            row, front = check(env, seed, found[0], args.atol)
            runs.append(row)
            fronts += [dict(environment=env, seed=seed, step=row['step'], gamma=row['gamma'], point=i)
                       | {f'objective_{j}': v for j, v in enumerate(point)} for i, point in enumerate(front)]

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(runs)
    counts = ['oracle_calls', 'front_points', 'step', 'replays', 'training_steps', 'online_steps']
    table = table.astype({c: 'Int64' for c in counts if c in table})
    table.to_csv(args.output / 'ipro_runs.csv', index=False)
    pd.DataFrame(fronts).to_csv(args.output / 'ipro_fronts.csv', index=False, float_format='%.10g')
    with pd.option_context('display.width', 250, 'display.max_columns', None, 'display.max_rows', None):
        print(table.drop(columns='run_dir').to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    print(f'\nOutput: {args.output}')


if __name__ == '__main__':
    main()