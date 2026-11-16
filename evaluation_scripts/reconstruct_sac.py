"""MO-SAC versus MO-PPO with the weighted sum on DST/LDST, both reconstructed with E1's definitions.

Per run (algorithm, environment, scalarizer, seed), at the end of the algorithm's own run with every evaluated
checkpoint observed: archive and active recall, complete recovery, and both within the unsupported returns, with the
schedule and training budget. sac_training.csv holds the seed-mean share of actors at each Pareto return, another
treasure return (off_front) or no treasure (zero) at every checkpoint. PPO runs are E1's standard runs; SAC runs are
found by their folder structure (--sac_select picks one configuration). Outputs (in --output): sac_runs.csv,
sac_training.csv and sac_manifest.json.
"""
import re
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path


# Protocol

SCHEMA = 'sac_v2'
DATA_DIR = Path('results/evaluation/scalarization/dst/sac')
ENVS = ('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2')
ALGORITHMS = ('PPO', 'SAC')
METHODS = ('ws_raw', 'ws_norm')
SEEDS = tuple(range(1, 11))
METRICS = ('archive_recall', 'active_recall', 'complete', 'unsupported_archive_recall', 'unsupported_active_recall')
MATCH_ATOL = 1e-4


# Data

def load_geometry(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Exact front, point ids and linear support per environment (E1's geometry file, gamma = 1)."""
    f = pd.read_csv(path, low_memory=False)
    if 'gamma' in f:
        f = f[np.isclose(f.gamma.astype(float), 1, rtol=0, atol=1e-12)]
    out = {}
    for env in ENVS:
        g = f[f.environment == env].sort_values('point_index')
        out[env] = (g[['objective_0', 'objective_1']].to_numpy(float), g.point_index.to_numpy(int),
                    g.linear_supported.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy())
    return out


def saved_config(path: Path) -> dict:
    """The run's saved configuration next to its policies folder, as in E1; empty if there is none."""
    root = path.parent.parent
    for p in sorted({*root.glob('*.json'), *root.glob('config/*.json'), *root.glob('configs/*.json')}):
        config = json.loads(p.read_text(encoding='utf-8-sig'))
        if isinstance(config, dict) and {'num_steps', 'scalarization_method', 'num_subproblems'} <= config.keys():
            return config
    return {}


def find(args: argparse.Namespace,
         algorithm: str,
         env: str,
         method: str,
         seed: int) -> tuple[list[Path], Path]:
    """Matching histories (E1's standard PPO run, or every SAC configuration containing --sac_select) and the
    searched location."""
    runs = {'PPO': '{env}/gamma_1/das_dennis/P_{k:04d}/K_{k:04d}/R_0001/H_00100/eval_0001/lr_0p0003/ec_0p05/'
                   '{method}/s_{seed:04d}/policies/mo_ppo_nl_dst.npz',
            'SAC': '{env}/gamma_*/das_dennis/P_*/K_*/R_0001/H_*/eval_*/lr_*/alpha_*/{method}/s_{seed:04d}/policies/'
                   'mo_sac_nl.npz'}
    root = args.ppo_root if algorithm == 'PPO' else args.sac_root
    pattern = runs[algorithm].format(env=env, k=20 if env.startswith('ldst') else 10, method=method, seed=seed)
    return sorted(p for p in root.glob(pattern) if algorithm == 'PPO' or args.sac_select in str(p)), root / pattern


def checkpoints(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Greedy returns [checkpoints, actors, 2] and each checkpoint's iteration (global step = it * P * num_steps)."""
    config = saved_config(path)
    steps = int(config.get('num_steps') or 64)
    found = {}
    with np.load(path, allow_pickle=False) as data:
        for key in (x for x in data.files if re.fullmatch(r't_\d+', x)):
            returns = np.asarray(data[key], dtype=float)
            if f'{key}_agent_ids' in data:
                returns = returns[np.argsort(np.asarray(data[f'{key}_agent_ids']))]
            if f'{key}_iteration' in data:
                it = int(np.asarray(data[f'{key}_iteration']).item())
            else:
                it = int(key[2:]) // (len(returns) * steps)
            found[it] = returns
    its = np.array(sorted(found))
    return np.stack([found[t] for t in its]), its, config


# Analysis

def match_matrix(points: np.ndarray, front: np.ndarray) -> np.ndarray:
    """[..., front]: the return equals the Pareto return within MATCH_ATOL in both objectives."""
    return np.all(np.abs(points[..., None, :] - front) <= MATCH_ATOL, axis=-1)


def unsupported_share(found: np.ndarray, supported: np.ndarray) -> float:
    """Share of the unsupported returns that are found; NaN if there are none."""
    return float(found[~supported].mean()) if (~supported).any() else np.nan


def measures(returns: np.ndarray, front: np.ndarray, supported: np.ndarray) -> dict:
    """E1's quantities at the last checkpoint, with the archive over all checkpoints."""
    seen = match_matrix(returns, front).any(1)
    archive = seen.any(0)
    active = seen[-1]
    return dict(archive_recall=float(archive.mean()), active_recall=float(active.mean()), complete=bool(archive.all()),
                unsupported_archive_recall=unsupported_share(archive, supported),
                unsupported_active_recall=unsupported_share(active, supported))


def outcome_shares(returns: np.ndarray, front: np.ndarray) -> np.ndarray:
    """[checkpoints, front + 2]: share of actors at each Pareto return, at another treasure return, at no treasure."""
    hit = match_matrix(returns, front)
    none = np.abs(returns[..., 0]) <= MATCH_ATOL
    return np.concatenate((hit.mean(1), (~hit.any(-1) & ~none).mean(1)[:, None], none.mean(1)[:, None]), axis=1)


# Main

def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sac_root', type=Path, default=Path('results/scalarization/baselines/mo_sac_nl_baselines'))
    parser.add_argument('--ppo_root', type=Path,
                        default=Path('results/scalarization/dst_weight_resolution/mo_ppo_nl_dst'))
    parser.add_argument('--geometry', type=Path,
                        default=Path('results/evaluation/scalarization/dst/geometry/dst_points.csv'))
    parser.add_argument('--output', type=Path, default=DATA_DIR)
    parser.add_argument('--sac_select', default='',
                        help='Substring of the SAC run path, if several configurations exist')
    parser.add_argument('--allow_missing', action='store_true',
                        help='Exclude missing runs (manifest becomes provisional)')
    return parser.parse_args()


def main() -> None:
    """Reconstructs every PPO and SAC run and writes the run table, the training shares and the manifest."""
    # Arguments
    args = parse_args()

    # Runs
    geometry = load_geometry(args.geometry)
    rows = []
    training = {}
    inventory = []
    schedules = {}
    for env, algorithm, method, seed in ((e, a, m, s) for e in ENVS for a in ALGORITHMS for m in METHODS
                                         for s in SEEDS):
        front, ids, supported = geometry[env]
        ident = dict(algorithm=algorithm, environment=env, scalarizer=method, seed=seed)
        paths, location = find(args, algorithm, env, method, seed)
        if not paths and args.allow_missing:
            inventory.append(ident | dict(status='missing', reason=f'no {algorithm} run: {location}'))
            continue
        path = paths[0]
        returns, its, config = checkpoints(path)
        per_iteration = int(config.get('num_steps') or 64) * (
            int(config.get('num_processes') or 1) if algorithm == 'PPO' else 1)
        schedule = dict(horizon=int(its[-1]), checkpoints=len(its),
                        cadence=int(np.median(np.diff(its))) if len(its) > 1 else 0,
                        transitions_per_policy=int(its[-1]) * per_iteration)
        schedules.setdefault((algorithm, env), set()).add(tuple(schedule.values()))
        rows.append(ident | schedule | measures(returns, front, supported))
        total, n, _ = training.setdefault((algorithm, env, method), [0.0, 0, its])
        training[algorithm, env, method] = [total + outcome_shares(returns, front), n + 1, its]
        inventory.append(ident | dict(status='ok', path=str(path.resolve())))

    # Training shares
    train_rows = []
    for (algorithm, env, method), (total, n, its) in sorted(training.items()):
        front, ids, _ = geometry[env]
        labels = ([(str(int(i)), 'pareto', f[0], f[1]) for i, f in zip(ids, front)]
                  + [('off_front', 'off_front', np.nan, np.nan), ('zero', 'zero', np.nan, np.nan)])
        train_rows += [dict(algorithm=algorithm, environment=env, scalarizer=method, iteration=int(it),
                            progress=100.0 * it / its[-1], outcome=outcome, kind=kind, J0=j0, J1=j1,
                            share=float(total[t, j] / n), n_seeds=n)
                       for j, (outcome, kind, j0, j1) in enumerate(labels) for t, it in enumerate(its)]

    # Output
    manifest = dict(
        schema=SCHEMA, status='complete' if all(r['status'] == 'ok' for r in inventory) else 'provisional',
        expected_runs=len(inventory), complete_runs=sum(r['status'] == 'ok' for r in inventory),
        schedules=[dict(algorithm=a, environment=e, horizon=s[0], checkpoints=s[1], cadence=s[2],
                        transitions_per_policy=s[3]) for (a, e), v in sorted(schedules.items()) for s in sorted(v)],
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, source_runs=inventory)
    args.output.mkdir(parents=True, exist_ok=True)
    (pd.DataFrame(rows).sort_values(['algorithm', 'environment', 'scalarizer', 'seed'])
     .to_csv(args.output / 'sac_runs.csv', index=False, float_format='%.17g'))
    pd.DataFrame(train_rows).to_csv(args.output / 'sac_training.csv', index=False, float_format='%.17g')
    (args.output / 'sac_manifest.json').write_text(json.dumps(manifest, indent=2, default=float) + '\n',
                                                   encoding='utf-8')
    print(f"SAC vs PPO: {manifest['complete_runs']}/{manifest['expected_runs']} runs ({manifest['status']}) in "
          f'{args.output}')


if __name__ == '__main__':
    main()