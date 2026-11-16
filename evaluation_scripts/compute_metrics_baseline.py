#!/usr/bin/env python3
"""Hypervolume and IGD+ of every run's final front: MO-PPO and MO-SAC (all scalarizers) and the baselines.

Reference front Z and HV reference point: on DST1-4 and LDST1-2 the exact Pareto front under the runs' discount factor
with nadir - 0.1 * range(Z); on Minecart, Reacher and MuJoCo the constructed reference front of
build_reference_fronts.py (without --calibration), which also holds every run's final front. Metrics: hv, hv_ratio
(hv / hypervolume of Z) and igd_plus (normalized by range(Z)).

Outputs (in --output): metrics_runs.csv (one row per run), metrics_summary.csv (mean and std over seeds) and
metrics_references.json (reference front source and size, HV reference point and hypervolume).

  python compute_metrics_baseline.py                   all environments
  python compute_metrics_baseline.py --envs mujoco     the MuJoCo environments
"""
import sys
import json
import argparse

import numpy as np
import pandas as pd
import mo_gymnasium as mo_gym

from pathlib import Path
from pymoo.indicators.hv import Hypervolume

from build_reference_fronts import collect, nondominated
from envs import dst_envs
from misc.utils import read_algo_config, read_env_config


# Metrics

def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume of the points that dominate the reference point (maximization)."""
    points = points[np.all(points > ref, axis=1)]
    if not len(points):
        return 0.0
    return float(Hypervolume(ref_point=-ref)(-points))


def igd_plus(points: np.ndarray, front: np.ndarray) -> float:
    """Mean over the reference points z of min over the points a of ||max((z - a) / range(front), 0)||_2."""
    if not len(points):
        return float('inf')
    span = np.where(np.ptp(front, 0) > 0, np.ptp(front, 0), 1.0)
    best = np.empty(len(front))
    rows = max(1, 2_000_000 // len(points))
    for i in range(0, len(front), rows):
        gaps = np.maximum((front[i:i + rows, None, :] - points[None, :, :]) / span, 0.0)
        best[i:i + rows] = np.sqrt((gaps ** 2).sum(-1)).min(1)
    return float(best.mean())


# Reference fronts

def exact_reference(args: argparse.Namespace, env: str) -> tuple[dict, list[dict]]:
    """DST/LDST: the exact front under the runs' discount factor and the runs."""
    runs = collect(args, env)
    gammas = {r['gamma'] for r in runs if r['gamma'] is not None}
    if len(gammas) != 1:
        sys.exit(f'STOP: {env}: need one discount factor for the exact front, the runs have {sorted(gammas)}')
    config = read_algo_config(f"configs/mo_ppo_nl_{'ldst' if env.startswith('ldst') else 'dst'}.json")
    dst_envs.register_envs(max_episode_steps=config['max_episode_steps'])
    env_id = read_env_config('configs/environment_configs.json')[env]['env_id']
    front = np.asarray(mo_gym.make(env_id).unwrapped.pareto_front(gamma=gammas.pop()), dtype=float)
    return dict(source='exact front', front=front, hv_reference_point=front.min(0) - 0.1 * np.ptp(front, 0)), runs


def constructed_reference(args: argparse.Namespace, env: str) -> tuple[dict, list[dict]]:
    """Reference front, HV reference point and the runs' final fronts from build_reference_fronts.py."""
    folder = next(f for f in args.reference_fronts if (f / 'reference_fronts.json').is_file()
                  and env in json.loads((f / 'reference_fronts.json').read_text())['environments'])
    summaries = json.loads((folder / 'reference_fronts.json').read_text())['environments']
    front = pd.read_csv(folder / f'reference_front_{env}.csv').filter(like='objective_').to_numpy(float)
    finals = pd.read_csv(folder / 'final_fronts.csv')
    finals = finals[(finals.environment == env) & finals.seed.isin(args.seeds)].fillna({'variant': ''})
    runs = [dict(algorithm=algorithm, variant=variant, label=name, seed=seed, step=group.step.iloc[0],
                 front=group.filter(like='objective_').dropna(axis=1, how='all').to_numpy(float))
            for (algorithm, variant, name, seed), group in finals.groupby(['algorithm', 'variant', 'label', 'seed'],
                                                                          sort=False)]
    hv_reference_point = np.asarray(summaries[env]['hv_reference_point'])
    reference = dict(source=str(folder), front=front, hv_reference_point=hv_reference_point)
    return reference, runs


def evaluate(env: str, reference: dict, runs: list[dict]) -> list[dict]:
    """Hypervolume, HV ratio and IGD+ of every run's final front."""
    front = reference['front']
    ref = reference['hv_reference_point']
    reference['hypervolume'] = hypervolume(front, ref)
    rows = []
    for r in runs:
        points = nondominated(np.round(r['front'], 6) + 0.0) if len(r['front']) else front[:0]
        hv = hypervolume(points, ref)
        rows.append(dict(environment=env, algorithm=r['algorithm'], variant=r['variant'], label=r['label'],
                         seed=r['seed'], step=r['step'], points=len(points), hv=hv,
                         hv_ratio=hv / reference['hypervolume'], igd_plus=igd_plus(points, front)))
    return rows


# Main

def parse_args() -> argparse.Namespace:
    """Command line arguments; 'dst' and 'mujoco' in --envs expand to their variants."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--envs', nargs='+', default=['dst', 'minecart', 'reacher', 'mujoco'],
                        help="Default: all environments; 'dst' and 'mujoco' stand for their variants")
    parser.add_argument('--reference_fronts', type=Path, nargs='+',
                        default=[Path('results/evaluation/scalarization/reference_fronts'),
                                 Path('results/evaluation/scalarization/reference_fronts_mujoco')],
                        help='Output folders of build_reference_fronts.py, searched in this order')
    parser.add_argument('--results_root', type=Path, default=Path('results'))
    parser.add_argument('--prefix', default='scalarization/baselines',
                        help="Launchers' path prefix below --results_root")
    parser.add_argument('--ipro_fronts', type=Path,
                        default=Path('results/evaluation/scalarization/ipro_checks/ipro_fronts.csv'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--output', type=Path, default=Path('results/evaluation/scalarization/metrics'))
    args = parser.parse_args()
    groups = dict(dst=('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2'),
                  mujoco=('ant', 'halfcheetah', 'hopper', 'humanoid'))
    args.envs = [env for name in args.envs for env in groups.get(name, (name,))]
    args.calibration = False
    return args


def main() -> None:
    """Evaluates every run's final front against the reference front of its environment."""
    # Arguments
    args = parse_args()

    # Metrics
    rows = []
    references = {}
    for env in args.envs:
        if env.startswith(('dst', 'ldst')):
            reference, runs = exact_reference(args, env)
        else:
            reference, runs = constructed_reference(args, env)
        rows += evaluate(env, reference, runs)
        references[env] = dict(source=reference['source'], points=len(reference['front']),
                               hv_reference_point=reference['hv_reference_point'].tolist(),
                               hypervolume=reference['hypervolume'])
    runs = pd.DataFrame(rows)
    summary = runs.groupby(['environment', 'algorithm', 'variant', 'label'], sort=False).agg(
        runs=('seed', 'size'), **{f'{m}_{s}': (m, s) for m in ('hv', 'hv_ratio', 'igd_plus') for s in ('mean', 'std')})

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output / 'metrics_runs.csv', index=False, float_format='%.10g')
    summary.reset_index().to_csv(args.output / 'metrics_summary.csv', index=False, float_format='%.10g')
    (args.output / 'metrics_references.json').write_text(json.dumps(references, indent=2) + '\n')
    for env, s in summary.reset_index().groupby('environment', sort=False):
        r = references[env]
        print(f"\n{env}: reference front of {r['points']} points ({r['source']}), HV reference point "
              f"{np.round(r['hv_reference_point'], 4).tolist()}, hypervolume {r['hypervolume']:.6g}")
        for m in s.itertuples():
            print(f'  {m.label:18s} {m.runs:3d} runs   HV ratio {m.hv_ratio_mean:.4f} ± {m.hv_ratio_std:.4f}   '
                  f'IGD+ {m.igd_plus_mean:.4f} ± {m.igd_plus_std:.4f}')
    print(f'\nOutput: {args.output}')


if __name__ == '__main__':
    main()