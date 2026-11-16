"""Final active population and final archive of the MO-PPO runs on Minecart, Reacher and MuJoCo, all six scalarizers.

Per run <root>/<environment>/.../<scalarizer>/.../s_<seed>, the last checkpoint of the archive (fronts/*.npz) and of
the current actors (policies/*.npz) is measured against the reference front of build_reference_fronts.py: points,
hv_ratio, igd_plus and hv_retained (actor HV / archive HV). The archive rebuilt without a size limit from all
checkpoints in policies/ is reported as full_*. The script stops if a run lacks a file or the two files end at
different steps, and reports uncovered actor returns and archives that differ from final_fronts.csv.

Outputs (in --output): active_runs.csv (one row per run) and active_summary.csv (mean and std over seeds).

  python compute_metrics_active.py
  python compute_metrics_active.py --envs mujoco
  python compute_metrics_active.py --match H_01000 mu_0p1        if a scalarizer and seed have several runs
"""
import re
import sys
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from build_reference_fronts import nondominated
from compute_metrics_baseline import constructed_reference, hypervolume, igd_plus


# Reading runs

def checkpoints(folder: Path) -> list[tuple[int, np.ndarray]]:
    """Steps and arrays of all checkpoints in the folder's NPZ with t_<step> keys, in order of the steps."""
    found = []
    for path in sorted(folder.glob('*.npz')):
        with np.load(path) as data:
            steps = sorted((int(m.group(1)), key) for key in data.files if (m := re.fullmatch(r't_(\d+)', key)))
            if steps:
                found.append([(step, np.asarray(data[key], dtype=float)) for step, key in steps])
    if len(found) != 1:
        sys.exit(f'STOP: {folder}: expected one NPZ with t_<step> checkpoints, found {len(found)}')
    return found[0]


def discover(root: Path,
             env: str,
             match: list[str],
             scalarizers: tuple[str, ...]) -> dict[tuple[str, int], Path]:
    """Run folders of an environment by (scalarizer, seed); of several runs the one matching most parts of match."""
    runs = {}
    for path in sorted((root / env).rglob('s_*')):
        seed = re.fullmatch(r's_(\d+)', path.name)
        names = [part for part in path.relative_to(root / env).parts[:-1] if part in scalarizers]
        if seed and len(names) == 1 and (path / 'fronts').is_dir() and (path / 'policies').is_dir():
            runs.setdefault((names[0], int(seed.group(1))), []).append(path)
    chosen = {}
    for (scalarizer, seed), paths in sorted(runs.items(), key=lambda item: (scalarizers.index(item[0][0]), item[0][1])):
        hits = [sum(part in str(path) for part in match) for path in paths]
        if hits.count(max(hits)) > 1:
            sys.exit(f'STOP: {env}: several runs for {scalarizer}, seed {seed}; name a part of the path of the one '
                     'to use with --match:\n  ' + '\n  '.join(map(str, paths)))
        chosen[scalarizer, seed] = paths[hits.index(max(hits))]
    return chosen


# Metrics

def tidy(points: np.ndarray) -> np.ndarray:
    """Distinct nondominated returns, rounded as in compute_metrics_baseline.py."""
    return nondominated(np.round(points, 6) + 0.0)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two return sets have the same distinct nondominated points."""
    a = tidy(a)
    b = tidy(b)
    a = a[np.lexsort(a.T[::-1])]
    b = b[np.lexsort(b.T[::-1])]
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0, atol=2e-6))


def evaluate(env: str,
             reference: dict,
             finals: dict,
             runs: dict[tuple[str, int], Path]) -> list[dict]:
    """Archive, active and full metrics of every run of one environment."""
    front = reference['front']
    ref = reference['hv_reference_point']
    reference_hv = hypervolume(front, ref)
    span = np.where(np.ptp(front, 0) > 0, np.ptp(front, 0), 1.0)
    rows = []
    for (scalarizer, seed), path in runs.items():
        stored = checkpoints(path / 'fronts')
        visited = checkpoints(path / 'policies')
        archive_step, archive = stored[-1]
        step, actors = visited[-1]
        if archive_step != step:
            sys.exit(f'STOP: {path}: fronts/ ends at step {archive_step}, policies/ at step {step}')
        full = tidy(np.vstack([a for _, a in visited]))
        slack = 1e-6 * np.maximum(1.0, np.abs(actors))
        covered = (archive[None] >= actors[:, None] - slack[:, None]).all(2).any(1)
        excess = ((actors[:, None] - archive[None]) / span).max(2).min(1)
        final = finals.get((f'MO-PPO {scalarizer}', seed))
        row = dict(environment=env, scalarizer=scalarizer, seed=seed, step=step, actors=len(actors),
                   actors_not_in_archive=int((~covered).sum()),
                   actors_excess=float(max(excess[~covered].max(), 0.0)) if (~covered).any() else 0.0,
                   archive_is_final_front=np.nan if final is None else same(archive, final),
                   archive_is_full=same(archive, full), archive_largest=max(len(a) for _, a in stored),
                   same_checkpoints=[t for t, _ in stored] == [t for t, _ in visited])
        hv = {}
        for name, points in (('archive', tidy(archive)), ('active', tidy(actors)), ('full', full)):
            hv[name] = hypervolume(points, ref)
            row |= {f'{name}_points': len(points), f'{name}_hv_ratio': hv[name] / reference_hv,
                    f'{name}_igd_plus': igd_plus(points, front)}
        row['hv_retained'] = hv['active'] / hv['archive'] if hv['archive'] > 0 else np.nan
        rows.append(row | dict(path=str(path)))
    return rows


# Main

def parse_args() -> argparse.Namespace:
    """Command line arguments; 'mujoco' in --envs expands to the four MuJoCo tasks."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--envs', nargs='+', default=['minecart', 'reacher', 'mujoco'],
                        help="Default: all additional environments; 'mujoco' stands for its variants")
    parser.add_argument('--root', type=Path, default=Path('results/scalarization/baselines/mo_ppo_nl_baselines'),
                        help='Folder with one subfolder of runs per environment')
    parser.add_argument('--match', nargs='*', default=[],
                        help='Where a scalarizer and seed have several runs: parts of the path of the run to use')
    parser.add_argument('--reference_fronts', type=Path, nargs='+',
                        default=[Path('results/evaluation/scalarization/reference_fronts'),
                                 Path('results/evaluation/scalarization/reference_fronts_mujoco')],
                        help='Output folders of build_reference_fronts.py, searched in this order')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--output', type=Path, default=Path('results/evaluation/scalarization/metrics_active'))
    args = parser.parse_args()
    mujoco = ('ant', 'halfcheetah', 'hopper', 'humanoid')
    args.envs = [env for name in args.envs for env in (mujoco if name == 'mujoco' else (name,))]
    return args


def main() -> None:
    """Evaluates the final archive and active population of every run and prints the summary and checks."""
    # Arguments
    args = parse_args()
    scalarizers = ('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm')

    # Metrics
    rows = []
    for env in args.envs:
        reference, finals = constructed_reference(args, env)
        runs = {key: path for key, path in discover(args.root, env, args.match, scalarizers).items()
                if key[1] in args.seeds}
        if not runs:
            sys.exit(f'STOP: {env}: no runs with fronts/ and policies/ below {args.root / env}')
        rows += evaluate(env, reference, {(r['label'], r['seed']): r['front'] for r in finals}, runs)
    runs = pd.DataFrame(rows)
    columns = [f'{c}_{m}' for m in ('points', 'hv_ratio', 'igd_plus') for c in ('archive', 'active', 'full')]
    columns.append('hv_retained')
    summary = runs.groupby(['environment', 'scalarizer'], sort=False).agg(
        runs=('seed', 'size'), **{f'{c}_{s}': (c, s) for c in columns for s in ('mean', 'std')}).reset_index()

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output / 'active_runs.csv', index=False, float_format='%.10g')
    summary.to_csv(args.output / 'active_summary.csv', index=False, float_format='%.10g')
    for env, s in summary.groupby('environment', sort=False):
        actors = ', '.join(map(str, sorted(runs.loc[runs.environment == env, 'actors'].unique())))
        steps = ', '.join(map(str, sorted(runs.loc[runs.environment == env, 'step'].unique())))
        print(f'\n{env}: {actors} actors per run, last checkpoint at step {steps}')
        print(f"  {'':10s} {'runs':>4s}   {'archive HV ratio':>17s}   {'active HV ratio':>17s}   "
              f"{'HV retained':>15s}   {'active IGD+':>17s}   {'active points':>13s}")
        for m in s.itertuples():
            print(f'  {m.scalarizer:10s} {m.runs:4d}   '
                  f'{m.archive_hv_ratio_mean:8.4f} ± {m.archive_hv_ratio_std:6.4f}   '
                  f'{m.active_hv_ratio_mean:8.4f} ± {m.active_hv_ratio_std:6.4f}   '
                  f'{m.hv_retained_mean:6.3f} ± {m.hv_retained_std:6.3f}   '
                  f'{m.active_igd_plus_mean:8.4f} ± {m.active_igd_plus_std:6.4f}   {m.active_points_mean:13.1f}')

    # Checks
    for env, found in runs.groupby('environment', sort=False).scalarizer.unique().items():
        if missing := [name for name in scalarizers if name not in found]:
            print(f"\nNote: {env} has no runs for {', '.join(missing)} below {args.root / env}.")
    print('\nStored archive (fronts/) against the archive rebuilt without a size limit from all checkpoints in '
          'policies/:')
    for env, r in runs.groupby('environment', sort=False):
        print(f'  {env:12s} identical in {int(r.archive_is_full.sum()):3d} of {len(r):3d} runs   '
              f'largest stored archive {r.archive_largest.max():4d}   '
              f'points {r.archive_points.mean():7.1f} -> {r.full_points.mean():7.1f}   '
              f'HV ratio {r.archive_hv_ratio.mean():.4f} -> {r.full_hv_ratio.mean():.4f}   '
              f'IGD+ {r.archive_igd_plus.mean():.4f} -> {r.full_igd_plus.mean():.4f}')
    if not runs.same_checkpoints.all():
        print(f'\nWARNING: in {int((~runs.same_checkpoints).sum())} of {len(runs)} runs, fronts/ and policies/ hold '
              'different checkpoints (column same_checkpoints). The rebuilt archive may then miss returns.')
    uncovered = runs[runs.actors_not_in_archive > 0]
    differing = runs[runs.archive_is_final_front.eq(False)]
    unmatched = runs[runs.archive_is_final_front.isna()]
    if len(uncovered):
        where = ', '.join(f'{env} {n}' for env, n in uncovered.groupby('environment', sort=False).size().items())
        print(f'\nWARNING: in {len(uncovered)} of {len(runs)} runs, actor returns are not covered by the archive of '
              f'the run ({where}). They exceed it by at most {uncovered.actors_excess.max():.3%} of an objective\'s '
              f'range, in the median affected run by {uncovered.actors_excess.median():.3%} (columns '
              'actors_not_in_archive, actors_excess). This is expected where the stored archive reached its size '
              'limit and dropped returns; elsewhere, policies/ may hold returns of another evaluation.')
    if len(differing):
        print(f'\nWARNING: in {len(differing)} of {len(runs)} runs, the archive differs from the final front in '
              'final_fronts.csv (column archive_is_final_front). The runs may belong to another configuration.')
    if len(unmatched):
        print(f"\nNote: {len(unmatched)} of {len(runs)} runs have no final front labelled 'MO-PPO <scalarizer>' in "
              'final_fronts.csv, so their archive was not compared with it.')
    print(f'\nOutput: {args.output}')


if __name__ == '__main__':
    main()