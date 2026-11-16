"""Reference fronts for Minecart, Reacher and the MuJoCo tasks: the non-dominated union of the final front of every run.

Sources (seeds 1-10): MO-PPO and MO-SAC (archive store outside policies/), GPI-LS, PCN and IPRO (--ipro_fronts) for
Minecart and Reacher, C-MORL, MORL/D and GPI-LS for MuJoCo. The script stops if a run is missing or unfinished, or if
the fronts of one environment use different discount factors. With --calibration, only runs without ideal/nadir
references enter, and references.json holds the frozen 'ideal' and 'nadir' for configs/environment_configs.json.

  python build_reference_fronts.py                               HV reference points for all environments
  python build_reference_fronts.py --calibration                 Ideal/nadir points for all environments
  python build_reference_fronts.py --envs mujoco                 HV reference points for the MuJoCo environments
  python build_reference_fronts.py --envs mujoco --calibration   Ideal/nadir points for the MuJoCo environments
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Reading runs

def run_dirs(args: argparse.Namespace,
             directory: str,
             env: str,
             variant: str,
             seed: int) -> list[Path]:
    """Run folders s_<seed> of one algorithm, environment and variant."""
    base = args.results_root / args.prefix / directory / env
    found = [p for p in base.rglob(f's_{seed:04d}') if p.is_dir()]
    return sorted(p for p in found if not variant or variant in p.relative_to(base).parts)


def step_keys(data: np.lib.npyio.NpzFile) -> list[tuple[int, str]]:
    """Sorted (step, key) pairs of the t_<step> arrays."""
    return sorted((int(m.group(1)), k) for k in data.files if (m := re.fullmatch(r't_(\d+)', k)))


def front_stores(run_dir: Path, archive: bool) -> list[Path]:
    """NPZ files with t_<step> arrays; for MO-PPO/MO-SAC only those outside policies/."""
    paths = sorted(run_dir.rglob('*.npz'))
    outside = [p for p in paths if 'policies' not in p.relative_to(run_dir).parts[:-1]]
    stores = []
    for path in outside if outside or archive else paths:
        with np.load(path) as data:
            if step_keys(data):
                stores.append(path)
    return stores


def last_front(path: Path) -> dict:
    """Front with the largest step (non-finite rows dropped), its step and, if stored, its iteration."""
    with np.load(path) as data:
        step, key = step_keys(data)[-1]
        front = np.atleast_2d(np.asarray(data[key], float)) if data[key].size else np.empty((0, 0))
        iteration = int(np.ravel(data[f'{key}_iteration'])[0]) if f'{key}_iteration' in data.files else None
    return dict(front=front[np.isfinite(front).all(1)], step=step, iteration=iteration)


def run_config(run_dir: Path) -> dict:
    """Saved run configuration: a JSON in the run folder or its config/ or configs/ subfolder."""
    for path in sorted({*run_dir.glob('*.json'), *run_dir.glob('config/*.json'), *run_dir.glob('configs/*.json')}):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and {'total_iteration', 'gamma', 'eval_gamma', 'total_timesteps'} & set(data):
            return data
    return {}


def discount(config: dict, run_dir: Path) -> float | None:
    """Discount factor of the evaluated returns: eval_gamma or gamma from the configuration, else the path token."""
    for key in ('eval_gamma', 'gamma'):
        if config.get(key) is not None:
            return float(config[key])
    for part in run_dir.parts:
        if m := re.fullmatch(r'gamma_([0-9pm]+)', part):
            return float(m.group(1).replace('m', '-').replace('p', '.'))
    return None


def unfinished(run_dir: Path, final: dict, config: dict) -> str:
    """Why an MO-PPO/MO-SAC run did not finish, judged by its population store; '' if it finished."""
    stores = sorted((run_dir / 'policies').glob('*.npz'))
    if len(stores) != 1:
        return ''
    population = last_front(stores[0])
    total = config.get('total_iteration')
    iteration = final['iteration'] if final['iteration'] is not None else population['iteration']
    if population['step'] != final['step']:
        return f"archive ends at step {final['step']}, population at step {population['step']}"
    if total is not None and iteration is not None and iteration < int(total):
        return f'last evaluation at iteration {iteration} of {total}'
    return ''


def ipro_records(args: argparse.Namespace, env: str) -> tuple[list[dict], list[str]]:
    """IPRO's final fronts from check_ipro_runs.py and the seeds without one."""
    table = pd.read_csv(args.ipro_fronts)
    table = table[table.environment == env]
    records = []
    missing = []
    for seed in args.seeds:
        rows = table[table.seed == seed]
        if rows.empty:
            missing.append(f'IPRO seed {seed}: no front in {args.ipro_fronts}')
            continue
        gamma = rows.gamma.iloc[0]
        records.append(dict(environment=env, algorithm='IPRO', variant='', seed=seed, label='IPRO',
                            source=str(args.ipro_fronts), step=int(rows.step.iloc[0]),
                            gamma=None if pd.isna(gamma) else float(gamma),
                            front=rows.filter(like='objective_').dropna(axis=1, how='all').to_numpy(float)))
    return records, missing


def collect(args: argparse.Namespace, env: str) -> list[dict]:
    """Every source run of one environment; stops if a run is missing or unfinished."""
    algorithms = {
        'ppo': dict(label='MO-PPO', directory='mo_ppo_nl_baselines', archive=True,
                    variants=('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm')),
        'sac': dict(label='MO-SAC', directory='mo_sac_nl_baselines', archive=True, variants=('ws_raw', 'ws_norm')),
        'gpi_ls': dict(label='GPI-LS', directory='gpi_ls', archive=False, variants=('',)),
        'pcn': dict(label='PCN', directory='pcn', archive=False, variants=('',)),
        'c_morl': dict(label='C-MORL', directory='c_morl', archive=False, variants=('',)),
        'morl_d': dict(label='MORL/D', directory='morl_d', archive=False, variants=('',)),
    }
    if env in ('ant', 'halfcheetah', 'hopper', 'humanoid'):
        sources = ('ppo', 'sac', 'c_morl', 'morl_d', 'gpi_ls')
    else:
        sources = ('ppo', 'sac', 'gpi_ls', 'pcn', 'ipro')
    records = []
    problems = []
    for algo in sources:
        if algo == 'ipro':
            ipro, missing = ipro_records(args, env)
            records += ipro
            problems += missing
            continue
        spec = algorithms[algo]
        for variant in ('ws_raw',) if args.calibration and spec['archive'] else spec['variants']:
            name = f"{spec['label']} {variant}".strip()
            group = []
            for seed in args.seeds:
                dirs = run_dirs(args, spec['directory'], env, variant, seed)
                if len(dirs) != 1:
                    problems.append(f"{name} seed {seed}: {len(dirs)} run folders s_{seed:04d} under "
                                    f"{args.results_root / args.prefix / spec['directory'] / env}, expected one")
                    continue
                stores = front_stores(dirs[0], spec['archive'])
                if len(stores) != 1:
                    problems.append(f'{name} seed {seed}: {len(stores)} front stores in {dirs[0]}, expected one')
                    continue
                final = last_front(stores[0])
                config = run_config(dirs[0])
                if spec['archive'] and (note := unfinished(dirs[0], final, config)):
                    problems.append(f'{name} seed {seed}: {note}')
                    continue
                group.append(dict(environment=env, algorithm=spec['label'], variant=variant, seed=seed, label=name,
                                  source=str(stores[0]), step=final['step'], gamma=discount(config, dirs[0]),
                                  front=final['front']))
            if group and not spec['archive']:
                longest = max(r['step'] for r in group)
                problems += [f"{name} seed {r['seed']}: last front at step {r['step']}, other seeds at {longest}"
                             for r in group if r['step'] < 0.9 * longest]
            records += group
    if problems:
        sys.exit(f'STOP: {env}: {len(problems)} runs missing or unfinished:\n  ' + '\n  '.join(problems[:40]))
    return records


# Non-dominated union

def nondominated(points: np.ndarray) -> np.ndarray:
    """Non-dominated rows (maximization), each distinct vector once."""
    unique = np.unique(points, axis=0)
    kept: list[int] = []
    for i in np.lexsort(unique.T[::-1])[::-1]:
        if not (kept and np.all(unique[kept] >= unique[i], axis=1).any()):
            kept.append(int(i))
    return unique[sorted(kept)]


def merge_near_duplicates(front: np.ndarray) -> np.ndarray:
    """One point per group of points that agree within 1e-5 in every objective."""
    front = front[np.lexsort(front.T[::-1])[::-1]]
    keep = np.ones(len(front), bool)
    for i in range(len(front)):
        if keep[i]:
            keep[i + 1:][np.abs(front[i + 1:] - front[i]).max(1) <= 1e-5] = False
    return front[keep]


def nearest_reference(points: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Per point the index of the nearest reference point within 1e-5 in every objective, else -1."""
    hits = np.full(len(points), -1)
    for start in range(0, len(points), 256):
        distance = np.abs(points[start:start + 256, None, :] - front[None, :, :]).max(-1)
        nearest = distance.argmin(1)
        close = distance[np.arange(len(nearest)), nearest] <= 1e-5
        hits[start:start + len(nearest)][close] = nearest[close]
    return hits


def build(args: argparse.Namespace, env: str, records: list[dict]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Reference front of one environment, its summary and the table of every run's final front."""
    gammas = {}
    for r in records:
        gammas.setdefault('unknown' if r['gamma'] is None else f"{r['gamma']:g}", set()).add(r['label'])
    if len([g for g in gammas if g != 'unknown']) > 1:
        sys.exit(f'STOP: {env}: fronts evaluated with different discount factors: '
                 + '; '.join(f"gamma {g}: {', '.join(sorted(v))}" for g, v in gammas.items()))
    for r in records:
        r['points'] = np.round(r['front'], 6) + 0.0
    front = merge_near_duplicates(nondominated(np.vstack([nondominated(r['points']) for r in records
                                                          if len(r['points'])])))
    variants = [set() for _ in front]
    runs_reaching = np.zeros(len(front), int)
    rows = []
    for r in records:
        hits = nearest_reference(r['points'], front)
        for h in set(hits[hits >= 0].tolist()):
            variants[h].add(r['label'])
            runs_reaching[h] += 1
        rows += [dict(environment=env, algorithm=r['algorithm'], variant=r['variant'], label=r['label'],
                      seed=r['seed'], step=r['step'], point=j, on_reference=bool(h >= 0),
                      reference_point=int(h) if h >= 0 else None)
                 | {f'objective_{i}': float(v) for i, v in enumerate(p)}
                 for j, (p, h) in enumerate(zip(r['points'], hits))]
    ideal = front.max(0)
    nadir = front.min(0)
    span = ideal - nadir
    summary = dict(objectives=front.shape[1], points=len(front), runs=len(records), ideal=ideal.tolist(),
                   nadir=nadir.tolist(), range=span.tolist(), hv_offset=args.offset,
                   hv_reference_point=(nadir - args.offset * span).tolist(),
                   discount_factors={g: sorted(v) for g, v in gammas.items()},
                   reference_points_reached={name: sum(name in v for v in variants)
                                             for name in dict.fromkeys(r['label'] for r in records)})
    table = pd.DataFrame(front, columns=[f'objective_{i}' for i in range(front.shape[1])])
    table.insert(0, 'point', np.arange(len(front)))
    table['runs'] = runs_reaching
    table['variants'] = ['; '.join(sorted(v)) for v in variants]
    return summary, table, pd.DataFrame(rows)


# Main

def parse_args() -> argparse.Namespace:
    """Command line arguments; 'mujoco' in --envs expands to the four MuJoCo tasks."""
    mujoco = ('ant', 'halfcheetah', 'hopper', 'humanoid')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_root', type=Path, default=Path('results'))
    parser.add_argument('--prefix', default='scalarization/baselines',
                        help="Launchers' path prefix below --results_root")
    parser.add_argument('--output', type=Path, default=None,
                        help='Default: results/evaluation/scalarization/reference_fronts (or calibration_references), '
                             'with the suffix _mujoco when only MuJoCo tasks are built')
    parser.add_argument('--calibration', action='store_true',
                        help='Only runs without ideal/nadir references; writes the frozen references')
    parser.add_argument('--ipro_fronts', type=Path,
                        default=Path('results/evaluation/scalarization/ipro_checks/ipro_fronts.csv'))
    parser.add_argument('--envs', nargs='+', default=['minecart', 'reacher', 'mujoco'],
                        help=f"Default: all environments; 'mujoco' stands for {' '.join(mujoco)}")
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--offset', type=float, default=0.1, help='Hypervolume reference point: nadir - offset * range')
    args = parser.parse_args()
    args.envs = [env for name in args.envs for env in (mujoco if name == 'mujoco' else (name,))]
    folder = 'calibration_references' if args.calibration else 'reference_fronts'
    args.output = args.output or Path('results/evaluation/scalarization') / (
        folder + ('_mujoco' if set(args.envs) <= set(mujoco) else ''))
    return args


def main() -> None:
    """Builds the reference front of every environment and writes fronts, summaries and frozen references."""
    # Arguments
    args = parse_args()

    # Reference fronts
    summaries = {}
    tables = {}
    fronts = []
    sources = []
    for env in args.envs:
        records = collect(args, env)
        summaries[env], tables[env], rows = build(args, env, records)
        fronts.append(rows)
        sources += [{k: v for k, v in r.items() if k not in ('front', 'points')} for r in records]

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    for env, table in tables.items():
        table.to_csv(args.output / f'reference_front_{env}.csv', index=False, float_format='%.10g')
    pd.concat(fronts, ignore_index=True).to_csv(args.output / 'final_fronts.csv', index=False, float_format='%.10g')
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / 'reference_fronts.json').write_text(
        json.dumps(dict(settings=settings, environments=summaries, runs=sources), indent=2, default=str) + '\n')
    for env, s in summaries.items():
        print(f"\n{env}: {s['points']} reference points from {s['runs']} runs\n"
              f"  ideal {np.round(s['ideal'], 4).tolist()}  nadir {np.round(s['nadir'], 4).tolist()}  "
              f"HV reference point {np.round(s['hv_reference_point'], 4).tolist()}")
        for name, n in s['reference_points_reached'].items():
            print(f'  {name:16s} reaches {n:6d} points ({100 * n / s["points"]:5.1f}%)')
    if args.calibration:
        references = {env: dict(ideal=s['ideal'], nadir=s['nadir'], hv_reference_point=s['hv_reference_point'])
                      for env, s in summaries.items()}
        (args.output / 'references.json').write_text(json.dumps(references, indent=2) + '\n')
        print("\nFrozen references ('ideal' and 'nadir' for configs/environment_configs.json; "
              f'hv_reference_point = nadir - {args.offset} * range):\n' + json.dumps(references, indent=2))
    print(f'\nOutput: {args.output}')


if __name__ == '__main__':
    main()