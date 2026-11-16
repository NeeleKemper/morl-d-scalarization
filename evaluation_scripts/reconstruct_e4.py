"""Fixed E4 protocol: saved greedy histories -> dense run-level measurements. No training or actor evaluation; only
the four source and output paths are configurable. The six-scalarizer cohort is retained; the paper shows four.
"""
import re
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product


# Protocol

ENVS = ('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2')
METHODS = ('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm')
SEEDS = tuple(range(1, 11))
RATES = (1e-4, 3e-4, 6e-4, 1e-3)
DEFAULT_LR = 3e-4
H = 100
ENTROPY = 0.05
MATCH_ATOL = 1e-4
TARGET_ATOL = 1e-12
TARGET_RTOL = 1e-10
SCHEMA = 'e4_lean_v1'
DATA_DIR = Path('results/evaluation/scalarization/dst/e4_lean')
ROOTS = {'sweep': 'dst_optimizer_axis_sweeps/mo_ppo_nl_dst',
         'baseline': 'dst_weight_resolution/mo_ppo_nl_dst'}
RUN_ID = ['environment', 'scalarizer', 'learning_rate', 'seed']
KEY = RUN_ID + ['h']
PRIMARY = ('oracle_recall', 'archive_recall', 'active_recall', 'gap', 'complete',
           'target_coverage', 'off_target_recall', 'target_absence', 'off_target_absence',
           'active_target_coverage', 'active_off_target_recall', 'weight_hit_ever', 'weight_hit_active')
METRICS = PRIMARY + ('active_front_actor_rate', 'redundant_pareto_actor_rate',
                     'no_treasure_rate', 'off_front_treasure_rate')
CONDITIONAL = ('off_target_recall', 'target_absence', 'off_target_absence', 'active_off_target_recall')
FIXED_SETTINGS = ('num_steps', 'num_processes', 'net_arch', 'layernorm', 'clip_param', 'ppo_epoch',
                  'num_mini_batches', 'value_loss_coef', 'max_grad_norm', 'use_clipped_value_loss',
                  'use_linear_lr_decay', 'lr_decay_ratio', 'use_gae', 'gae_lambda', 'obj_rms', 'ob_rms',
                  'use_proper_time_limits', 'max_episode_steps', 'advantage_normalization')


# Helpers

def standard_k(env: str) -> int:
    """Standard number of distinct preferences: 20 on LDST, 10 on DST."""
    return 20 if env.startswith('ldst') else 10


def expected_runs() -> set[tuple]:
    """Run IDs of the complete fixed cohort."""
    return set(product(ENVS, METHODS, RATES, SEEDS))


def lattice(k: int) -> np.ndarray:
    """Two-objective K-point simplex grid."""
    x = np.linspace(0, 1, k)
    return np.column_stack((x, 1 - x))


def divide(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray:
    """Elementwise num / den, NaN where den is not positive."""
    num, den = np.broadcast_arrays(np.asarray(num, float), np.asarray(den, float))
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def scalar_scores(method: str,
                  front: np.ndarray,
                  weights: np.ndarray,
                  mu: float,
                  values: np.ndarray | None = None) -> np.ndarray:
    """The six score formulas of E1/E3: signed gaps, zero weights omitted."""
    values = front if values is None else np.asarray(values, float)
    delta = np.ptp(front, axis=0)
    if method == 'ws_raw':
        return weights @ values.T
    if method == 'ws_norm':
        return weights @ ((values - front.min(0)) / delta).T
    gap = front.max(0) - values
    if method.endswith('_norm'):
        gap = gap / delta
    z = np.where(weights[:, None, :] > 0, weights[:, None, :] * gap[None], -np.inf)
    maximum = z.max(2)
    if method.startswith('tch_'):
        return -maximum
    return -maximum - mu * np.log(np.exp((z - maximum[:, :, None]) / mu).sum(2))


# Data

def load_geometry(path: Path) -> dict:
    """Exact gamma=1 front, point IDs and supportedness of every environment."""
    f = pd.read_csv(path)
    if 'gamma' in f:
        f = f[np.isclose(f.gamma.astype(float), 1, rtol=0, atol=1e-12)]
    out = {}
    for env in ENVS:
        g = f[f.environment == env].sort_values('point_index')
        front = g[['objective_0', 'objective_1']].to_numpy(float)
        ids = np.asarray(g.point_index, float).astype(np.int64)
        supported = g.linear_supported.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()
        out[env] = (front, ids, supported)
    return out


def saved_config(path: Path) -> tuple[dict, Path]:
    """Saved run configuration and its path."""
    root = path.parent.parent
    for p in sorted({*root.glob('*.json'), *root.glob('config/*.json'), *root.glob('configs/*.json')}):
        c = json.loads(p.read_text(encoding='utf-8-sig'))
        if isinstance(c, dict) and {'num_steps', 'scalarization_method'} <= c.keys():
            return c, p


def load_run(path: Path, ident: dict) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Returns aligned by saved actor IDs, the preference index of every actor, the configuration and provenance."""
    c, cp = saved_config(path)
    p = ident['P']
    k = ident['K']
    protocol_fields = ('seed', 'gamma', 'learning_rate', 'entropy_coef', 'total_iteration', 'eval_iteration',
                       'num_subproblems', 'population_size', 'num_distinct_weights', 'replicas_per_weight',
                       'scalarization_method', 'init_w_sampling')
    history = {}
    ids = None
    metadata = {}
    with np.load(path, allow_pickle=False) as data:
        keys = sorted((x for x in data.files if re.fullmatch(r't_\d+', x)), key=lambda x: int(x[2:]))
        has_ids = all(f'{x}_agent_ids' in data for x in keys)
        explicit = all(f'{x}_iteration' in data for x in keys)
        for key in keys:
            if explicit:
                t = int(np.asarray(data[f'{key}_iteration'], float))
            else:
                t = int(key[2:]) // (p * int(c['num_steps']))
            current = np.asarray(data[f'{key}_agent_ids'], float).astype(np.int64) if has_ids else np.arange(p)
            order = np.argsort(current)
            ids = current[order]
            history[t] = np.asarray(data[key], float)[order]
            for suffix in ('weights', 'weight_ids'):
                if f'{key}_{suffix}' in data:
                    metadata[suffix] = np.asarray(data[f'{key}_{suffix}'], float)[order]
    candidates = []
    if 'weights' in metadata:
        candidates.append((metadata['weights'], 'npz_weights'))
    if 'weight_ids' in metadata and c.get('distinct_weights') is not None:
        distinct = np.asarray(c['distinct_weights'], float)
        candidates.append((distinct[metadata['weight_ids'].astype(np.int64)], 'npz_weight_ids+json_distinct_weights'))
    for name in ('initial_weights', 'subproblem_weights', 'weights'):
        if c.get(name) is None or np.shape(c[name]) != (p, 2):
            continue
        a = np.asarray(c[name], float)
        if c.get('agent_ids') is not None:
            a = a[np.argsort(np.asarray(c['agent_ids'], float).astype(np.int64))]
        elif has_ids and not np.array_equal(ids, np.arange(p)):
            continue
        candidates.append((a, f'json_{name}'))
    index = np.rint(candidates[0][0][:, 0] * (k - 1)).astype(int)
    provenance = dict(config_path=str(cp.resolve()), identity_verified=has_ids,
                      weight_source=';'.join(source for _, source in candidates),
                      identity_source='saved_agent_ids' if has_ids else 'row_order_assumed_stable',
                      iteration_source='saved_iteration' if explicit else 'legacy_step_div_P_num_steps',
                      num_steps=int(c['num_steps']), num_processes=int(c['num_processes']),
                      missing_protocol_fields=[name for name in protocol_fields if c.get(name) is None])
    return np.stack([history[t] for t in range(H + 1)]), index, c, provenance


def effective_mu(method: str,
                 config: dict,
                 env: str,
                 table: pd.DataFrame | None) -> tuple[float, str]:
    """STCH temperature from the run configuration, the protocol (normalized) or the targetability summary (raw)."""
    if not method.startswith('stch_'):
        return np.nan, 'not_used'
    if config.get('mu') is not None:
        return float(config['mu']), 'run_config'
    if method == 'stch_norm':
        return 0.1, 'fixed_E4_protocol'
    s = table[(table.environment == env) & (table.scalarizer == method)]
    return float(s.mu_effective.to_numpy(float)[0]), 'targetability_summary'


def find_history(root: Path, ident: dict) -> tuple[Path, str]:
    """History of one run: the sweep first, the E1 baseline as fallback for the default rate only."""
    k = ident['K']
    lr = ident['learning_rate']
    token = f'{lr:.6g}'.replace('.', 'p')
    relative = Path(ident['environment']) / 'gamma_1/das_dennis' / f'P_{k:04d}/K_{k:04d}/R_0001'
    relative /= f"H_{H:05d}/eval_0001/lr_{token}/ec_0p05/{ident['scalarizer']}"
    for source in ('sweep', 'baseline') if lr == DEFAULT_LR else ('sweep',):
        base = root / ROOTS[source] / relative
        bases = [base, base / 'mu_0p1'] if ident['scalarizer'].startswith('stch_') else [base]
        paths = [p / f"s_{ident['seed']:04d}/policies/mo_ppo_nl_dst.npz" for p in bases]
        found = [p.resolve() for p in paths if p.is_file()]
        if found:
            return found[0], source


# Analysis

def analyze(returns: np.ndarray,
            index: np.ndarray,
            front: np.ndarray,
            method: str,
            mu: float,
            k: int) -> pd.DataFrame:
    """E4/E3 measurements at every cutoff 1..H; initialization contributes to discovery."""
    scores = scalar_scores(method, front, lattice(k), mu)
    own = np.isclose(scores, scores.max(1, keepdims=True), rtol=TARGET_RTOL, atol=TARGET_ATOL)
    targeted = own.any(0)
    hit = (np.abs(returns[:, :, None] - front) <= MATCH_ATOL).all(-1)
    active = hit.any(1)
    archive = np.logical_or.accumulate(active, axis=0)
    own_hit = (hit & own[index][None]).any(2)
    ever = np.logical_or.accumulate(own_hit, axis=0)
    a = archive.sum(1)
    p = active.sum(1)
    at = archive[:, targeted].sum(1)
    pt = active[:, targeted].sum(1)
    ao = a - at
    po = p - pt
    n = len(front)
    nt = int(targeted.sum())
    front_actors = hit.any(2)
    none = np.abs(returns[:, :, 0]) <= MATCH_ATOL
    complete_at = np.flatnonzero(a == n)
    first = int(complete_at[0]) if len(complete_at) else np.nan
    data = dict(
        h=np.arange(H + 1), n_front=n, n_targets=nt,
        archive_count=a, active_count=p, archive_targeted_count=at, active_targeted_count=pt,
        archive_off_target_count=ao, active_off_target_count=po,
        discovered_target_absent_count=at - pt, discovered_off_target_absent_count=ao - po,
        own_hit_ever_count=ever.sum(1), own_hit_active_count=own_hit.sum(1),
        active_front_actor_count=front_actors.sum(1), redundant_pareto_actor_count=front_actors.sum(1) - p,
        no_treasure_actor_count=none.sum(1), off_front_treasure_actor_count=(~front_actors & ~none).sum(1),
        oracle_recall=nt / n, archive_recall=a / n, active_recall=p / n, gap=(a - p) / n, complete=a == n,
        target_coverage=divide(at, nt), off_target_recall=divide(ao, n - nt),
        active_target_coverage=divide(pt, nt), active_off_target_recall=divide(po, n - nt),
        target_absence=divide(at - pt, at), off_target_absence=divide(ao - po, ao),
        weight_hit_ever=ever.mean(1), weight_hit_active=own_hit.mean(1),
        active_front_actor_rate=front_actors.mean(1), redundant_pareto_actor_rate=(front_actors.sum(1) - p) / k,
        no_treasure_rate=none.mean(1), off_front_treasure_rate=(~front_actors & ~none).mean(1),
        first_complete_iteration=np.where(np.arange(H + 1) >= first, first, np.nan))
    return pd.DataFrame(data).iloc[1:].reset_index(drop=True)


def main() -> None:
    """Reconstructs every E4 run and writes the dense run table and the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_root', type=Path, default=Path('results/scalarization'))
    parser.add_argument('--geometry', type=Path,
                        default=Path('results/evaluation/scalarization/dst/geometry/dst_points.csv'))
    parser.add_argument('--targetability_dir', type=Path,
                        default=Path('results/evaluation/scalarization/dst/targetability'))
    parser.add_argument('--output', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Runs
    refs = load_geometry(args.geometry)
    temp_path = args.targetability_dir / 'targetability_summary.csv'
    table = pd.read_csv(temp_path) if temp_path.is_file() else None
    blocks = []
    inventory = []
    oracles = {}
    for key in sorted(expected_runs()):
        ident = dict(zip(RUN_ID, key))
        env = ident['environment']
        method = ident['scalarizer']
        k = standard_k(env)
        ident |= dict(P=k, K=k, R=1)
        path, source = find_history(args.result_root, ident)
        values, index, config, provenance = load_run(path, ident)
        front, ids, _ = refs[env]
        mu, mu_source = effective_mu(method, config, env, table)
        result = analyze(values, index, front, method, mu, k)
        transitions = result.h * k * provenance['num_steps'] * provenance['num_processes']
        result = result.assign(**ident, training_transitions=transitions, mu_effective=mu,
                               identity_verified=provenance['identity_verified'])
        blocks.append(result)
        inventory.append(ident | provenance | dict(
            path=str(path), source=source, mu_source=mu_source,
            missing_fixed_settings=[name for name in FIXED_SETTINGS if config.get(name) is None]))
        if f'{env}/{method}' not in oracles:
            scores = scalar_scores(method, front, lattice(k), mu)
            own = np.isclose(scores, scores.max(1, keepdims=True), rtol=TARGET_RTOL, atol=TARGET_ATOL)
            oracles[f'{env}/{method}'] = dict(K=k, weights=lattice(k).tolist(),
                                              target_ids_by_weight=[ids[x].tolist() for x in own],
                                              mu_effective=float(mu) if np.isfinite(mu) else None)
    frame = pd.concat(blocks, ignore_index=True).sort_values(KEY)

    # Manifest
    notes = dict(archive='Original E4 definition: distinct exact-front returns observed through h, including '
                         'initialization.',
                 identity='Past own-target hits use saved actor IDs or a recorded stable-row-order assumption.',
                 budgets='Actual training transitions include num_processes; legacy NPZ step labels may not. '
                         'Evaluation excluded.',
                 default='Sweep histories take precedence. E1 is a fallback only; repeated default runs are not '
                         'extra replicates.',
                 scope='No new training, sampled execution, regret, or causal optimizer diagnostics. H=500 is not '
                       'merged.')
    protocol = dict(environments=ENVS, scalarizers=METHODS, learning_rates=RATES, default_lr=DEFAULT_LR, seeds=SEEDS,
                    H=H, cutoffs=list(range(1, H + 1)), gamma=1, entropy_coef=ENTROPY, cadence=1, mu_base=0.1,
                    roots=ROOTS, match_atol=MATCH_ATOL, target_atol=TARGET_ATOL, target_rtol=TARGET_RTOL)
    manifest = dict(schema=SCHEMA, status='complete', expected_runs=len(expected_runs()),
                    complete_runs=len(inventory), protocol=protocol,
                    inputs={key: str(value) for key, value in vars(args).items()},
                    references={e: dict(front=f.tolist(), point_ids=i.tolist(), supported=s.tolist())
                                for e, (f, i, s) in refs.items()},
                    oracles=oracles, source_runs=inventory, notes=notes)

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'e4_manifest.json').unlink(missing_ok=True)
    frame.to_csv(args.output / 'e4_runs.csv', index=False, float_format='%.17g')
    (args.output / 'e4_summary.csv').unlink(missing_ok=True)
    (args.output / 'e4_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n',
                                                  encoding='utf-8')
    print(f'E4: {len(inventory)} runs; {len(frame):,} dense cutoff rows. Output: {args.output}')


if __name__ == '__main__':
    main()