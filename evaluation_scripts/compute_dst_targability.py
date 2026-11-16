"""Oracle finite-grid targetability of the six scalarizers on the exact DST/LDST fronts; run from the repo root."""
import os

# Thread limits must be set before NumPy/PyTorch are imported
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import math
import torch

import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass

from agents.utils.scalarizer import Scalarizer, calibrate_stch_raw_mu
from misc.utils import read_algo_config
from misc.weights import generate_das_dennis_weights


@dataclass(frozen=True)
class References:
    """Ideal point, nadir point and objective ranges of an exact Pareto front."""
    ideal: np.ndarray
    nadir: np.ndarray
    ranges: np.ndarray


# Geometry

def linearly_supported(front: np.ndarray) -> np.ndarray:
    """Mask of the front points that maximize a 2-D weighted sum for some weight."""
    supported = np.zeros(len(front), dtype=bool)
    scaled_front = front / max(1.0, float(np.max(np.abs(front))))
    for i, point in enumerate(scaled_front):
        lower = 0.0
        upper = 1.0
        feasible = True
        for competitor in scaled_front:
            dx = point[0] - competitor[0]
            dy = point[1] - competitor[1]
            coefficient = dx - dy
            if abs(coefficient) <= 1e-12:
                if dy < -1e-12:
                    feasible = False
                    break
            elif coefficient > 0:
                lower = max(lower, -dy / coefficient)
            else:
                upper = min(upper, -dy / coefficient)
            if lower > upper + 1e-12:
                feasible = False
                break
        supported[i] = feasible and lower <= 1.0 + 1e-12 and upper >= -1e-12
    return supported


def load_geometry(environment: str) -> tuple[pd.DataFrame, References]:
    """Exact front of one environment at gamma=1 with its supported points and references."""
    geometry = pd.read_csv('results/evaluation/scalarization/dst/geometry/dst_points.csv')
    geometry = geometry[geometry['environment'] == environment].copy()
    if 'gamma' in geometry.columns:
        geometry = geometry[np.isclose(pd.to_numeric(geometry['gamma']), 1.0, rtol=0.0, atol=1e-12)]
    geometry = geometry.sort_values('point_index').reset_index(drop=True)
    front = geometry[['objective_0', 'objective_1']].to_numpy(dtype=np.float64)
    ideal = front.max(axis=0)
    nadir = front.min(axis=0)
    geometry['linear_supported'] = linearly_supported(front)
    return geometry, References(ideal=ideal, nadir=nadir, ranges=ideal - nadir)


# Targetability

def project_scores(scalarizer: Scalarizer,
                   values: np.ndarray,
                   weight: np.ndarray,
                   refs: References) -> np.ndarray:
    """Scores of the project scalarizer for every front point under one weight."""
    with torch.no_grad():
        scores = scalarizer(torch.as_tensor(values, dtype=torch.float64), torch.as_tensor(weight, dtype=torch.float64),
                            torch.as_tensor(refs.ideal, dtype=torch.float64),
                            torch.as_tensor(refs.nadir, dtype=torch.float64))
    return scores.numpy().reshape(-1)


def compute_targetability(geometry: pd.DataFrame,
                          weights: np.ndarray,
                          scalarizer: Scalarizer,
                          refs: References) -> tuple[pd.DataFrame, dict]:
    """Front points that maximize the scalarizer for some grid weight, keeping numerical ties."""
    front = geometry[['objective_0', 'objective_1']].to_numpy(dtype=np.float64)
    targeting_weight_ids = [[] for _ in range(len(front))]
    uniquely_targeted = np.zeros(len(front), dtype=bool)
    num_weights_with_ties = 0
    for weight_id, weight in enumerate(weights):
        scores = project_scores(scalarizer, front, weight, refs)
        target_rows = np.flatnonzero(np.isclose(scores, float(scores.max()), rtol=1e-10, atol=1e-12))
        if len(target_rows) > 1:
            num_weights_with_ties += 1
        for row in target_rows:
            targeting_weight_ids[row].append(weight_id)
        if len(target_rows) == 1:
            uniquely_targeted[target_rows[0]] = True
    targeted = np.asarray([len(ids) > 0 for ids in targeting_weight_ids], dtype=bool)
    supported = geometry['linear_supported'].to_numpy(dtype=bool)
    point_table = geometry[['point_index', 'objective_0', 'objective_1', 'linear_supported']].copy()
    point_table['finite_grid_targeted'] = targeted
    point_table['uniquely_targeted'] = uniquely_targeted
    point_table['targeting_weight_ids'] = [','.join(map(str, ids)) for ids in targeting_weight_ids]
    summary = {
        'num_pf_points': int(len(front)),
        'num_finite_grid_targeted': int(targeted.sum()),
        'finite_grid_target_recall': float(targeted.mean()),
        'supported_target_recall': float(np.mean(targeted[supported])) if supported.any() else math.nan,
        'unsupported_target_recall': float(np.mean(targeted[~supported])) if (~supported).any() else math.nan,
        'num_weights_with_ties': num_weights_with_ties,
        'num_uniquely_targeted': int(uniquely_targeted.sum()),
    }
    return point_table, summary


def main() -> None:
    """Computes targetability for every environment, distinct weight count K and scalarizer and writes two CSVs."""
    # Settings
    torch.set_default_dtype(torch.float64)
    mu = 0.1
    summaries = []
    point_tables = []
    print('Computing oracle targetability for the finalized DST/LDST protocol.')
    print(f'K=(2, 5, 10, 15, 20, 25, 30, 50), mu={mu}, gamma=1.0')

    # Targetability
    for environment in ('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2'):
        geometry, refs = load_geometry(environment)
        config = read_algo_config(f"configs/mo_ppo_nl_{'ldst' if environment.startswith('ldst') else 'dst'}.json")
        raw_mu = float(calibrate_stch_raw_mu(mu=mu,
                                             objective_ranges=refs.ranges,
                                             num_weights=int(config['num_subproblems']),
                                             reward_dim=2))
        print(f'\n{environment}: raw STCH effective mu={raw_mu:.12g}')
        for K in (2, 5, 10, 15, 20, 25, 30, 50):
            weights = np.asarray(generate_das_dennis_weights(num_weights=K, reward_dim=2), dtype=np.float64)
            for method in ('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm'):
                effective_mu = raw_mu if method == 'stch_raw' else mu
                point_table, summary = compute_targetability(geometry, weights, Scalarizer(method, effective_mu), refs)
                summary.update({
                    'environment': environment,
                    'scalarizer': method,
                    'num_distinct_weights': K,
                    'mu': mu if method.startswith('stch_') else math.nan,
                    'mu_effective': effective_mu if method.startswith('stch_') else math.nan,
                })
                point_table.insert(0, 'num_distinct_weights', K)
                point_table.insert(0, 'scalarizer', method)
                point_table.insert(0, 'environment', environment)
                summaries.append(summary)
                point_tables.append(point_table)
                print(f"{environment:5s} | {method:9s} | K={K:2d} | "
                      f"targeted={summary['num_finite_grid_targeted']:2d}/{summary['num_pf_points']:2d} | "
                      f"supported={summary['supported_target_recall']:.3f} | "
                      f"unsupported={summary['unsupported_target_recall']:.3f} | "
                      f"ties={summary['num_weights_with_ties']:2d}")

    # Output
    summary_columns = ['environment', 'scalarizer', 'num_distinct_weights', 'mu', 'mu_effective', 'num_pf_points',
                       'num_finite_grid_targeted', 'finite_grid_target_recall', 'supported_target_recall',
                       'unsupported_target_recall', 'num_weights_with_ties', 'num_uniquely_targeted']
    point_columns = ['environment', 'scalarizer', 'num_distinct_weights', 'point_index', 'objective_0', 'objective_1',
                     'linear_supported', 'finite_grid_targeted', 'uniquely_targeted', 'targeting_weight_ids']
    summary_frame = (pd.DataFrame(summaries)[summary_columns]
                     .sort_values(['environment', 'scalarizer', 'num_distinct_weights']).reset_index(drop=True))
    point_frame = (pd.concat(point_tables, ignore_index=True)[point_columns]
                   .sort_values(['environment', 'scalarizer', 'num_distinct_weights', 'point_index'])
                   .reset_index(drop=True))
    output_dir = Path('results/evaluation/scalarization/dst/targetability')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output_dir / 'targetability_summary.csv', index=False)
    point_frame.to_csv(output_dir / 'targetability_points.csv', index=False)
    print(f"\nSaved summary: {output_dir / 'targetability_summary.csv'}")
    print(f"Saved point targets: {output_dir / 'targetability_points.csv'}")


if __name__ == '__main__':
    main()