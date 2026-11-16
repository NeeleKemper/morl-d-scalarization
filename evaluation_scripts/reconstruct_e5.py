"""E5: fixed H=500 extension of the standard E1 populations; offline analysis only. Separate exports prevent counting
the h=100 prefix as additional E1 replications. Only filesystem paths are arguments.
"""
import re
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product
from collections import Counter


# Protocol

ENVS = ('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2')
METHODS = ('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm')
SEEDS = tuple(range(1, 11))
H = 500
REFERENCE_H = 100
LEARNING_RATE = 3e-4
ENTROPY = 0.05
MATCH_ATOL = 1e-4
TARGET_ATOL = 1e-12
TARGET_RTOL = 1e-10
SCHEMA = 'e5_lean_v1'
DATA_DIR = Path('results/evaluation/scalarization/dst/e5_lean')
E1_RUNS = DATA_DIR.parent / 'e1_lean/e1_runs.csv'
RUN_ID = ['environment', 'scalarizer', 'seed']
KEY = RUN_ID + ['h']
PRIMARY = ('archive_recall', 'active_recall', 'gap', 'complete', 'target_coverage',
           'off_target_recall', 'target_absence', 'off_target_absence')
METRICS = PRIMARY + ('oracle_recall', 'weight_hit_ever', 'weight_hit_active',
                     'active_front_actor_rate', 'redundant_pareto_actor_rate',
                     'no_treasure_rate', 'off_front_treasure_rate')
CONDITIONAL = ('off_target_recall', 'target_absence', 'off_target_absence')
FIXED_SETTINGS = ('num_steps', 'num_processes', 'net_arch', 'layernorm', 'clip_param', 'ppo_epoch',
                  'num_mini_batches', 'value_loss_coef', 'max_grad_norm', 'use_clipped_value_loss',
                  'use_linear_lr_decay', 'lr_decay_ratio', 'use_gae', 'gae_lambda', 'obj_rms', 'ob_rms',
                  'use_proper_time_limits', 'max_episode_steps', 'advantage_normalization', 'eval_rep')


# Helpers

def expected_runs() -> set[tuple]:
    """Run IDs of the complete fixed cohort."""
    return set(product(ENVS, METHODS, SEEDS))


def standard_k(env: str) -> int:
    """Standard number of distinct preferences: 20 on LDST, 10 on DST."""
    return 20 if env.startswith('ldst') else 10


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


def load_run(path: Path, ident: dict, horizon: int = H) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Returns aligned by saved actor IDs, the preference index of every actor, the configuration and provenance."""
    c, cp = saved_config(path)
    p = ident['P']
    k = ident['K']
    protocol_fields = ('eval_rep', 'seed', 'gamma', 'learning_rate', 'entropy_coef', 'total_iteration',
                       'eval_iteration', 'num_subproblems', 'population_size', 'num_distinct_weights',
                       'replicas_per_weight', 'scalarization_method', 'init_w_sampling')
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
    return np.stack([history[t] for t in range(horizon + 1)]), index, c, provenance


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
        return 0.1, 'fixed_E5_protocol'
    s = table[(table.environment == env) & (table.scalarizer == method)]
    return float(s.mu_effective.to_numpy(float)[0]), 'targetability_summary'


def find_history(root: Path, ident: dict) -> Path:
    """History of one run; STCH runs may lie below mu_0p1."""
    k = ident['K']
    method = ident['scalarizer']
    base = (root / ident['environment'] / 'gamma_1/das_dennis' / f'P_{k:04d}/K_{k:04d}/R_0001'
            / f'H_{H:05d}/eval_0001/lr_0p0003/ec_0p05/{method}')
    bases = [base, base / 'mu_0p1'] if method.startswith('stch_') else [base]
    paths = [p / f"s_{ident['seed']:04d}/policies/mo_ppo_nl_dst.npz" for p in bases]
    return [p.resolve() for p in paths if p.is_file()][0]


def schedule_record(config: dict) -> dict:
    """Saved learning-rate schedule (decay flag and ratio), not a realized learning-rate sequence."""
    flag = config.get('use_linear_lr_decay')
    if flag is not None:
        flag = str(flag).lower() in ('true', '1', '1.0')
    ratio = config.get('lr_decay_ratio')
    if ratio is not None:
        ratio = float(ratio)
    return dict(use_linear_lr_decay=flag, lr_decay_ratio=ratio)


# Analysis

def analyze(returns: np.ndarray,
            index: np.ndarray,
            front: np.ndarray,
            method: str,
            mu: float,
            k: int) -> pd.DataFrame:
    """E5 recalls and the actor-outcome decomposition at every cutoff 1..H; initialization is included."""
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
        target_absence=divide(at - pt, at), off_target_absence=divide(ao - po, ao),
        weight_hit_ever=ever.mean(1), weight_hit_active=own_hit.mean(1),
        active_front_actor_rate=front_actors.mean(1), redundant_pareto_actor_rate=(front_actors.sum(1) - p) / k,
        no_treasure_rate=none.mean(1), off_front_treasure_rate=(~front_actors & ~none).mean(1),
        first_complete_iteration=np.where(np.arange(H + 1) >= first, first, np.nan))
    return pd.DataFrame(data).iloc[1:].reset_index(drop=True)


class E1Check:
    def __init__(self, path: Path) -> None:
        """E1 runs at h=100 and their source histories, where available, for the prefix comparison."""
        self.path = path
        self.base = None
        self.sources = {}
        if not path.is_file():
            return
        data = pd.read_csv(path, float_precision='round_trip')
        data = data[(data.horizon == REFERENCE_H) & (data.cadence == 1)
                    & data.K.eq(data.environment.map(standard_k))]
        self.base = data.set_index(RUN_ID)
        manifest = path.with_name('e1_manifest.json')
        if manifest.is_file():
            content = json.loads(manifest.read_text())
            for item in content.get('runs', content.get('source_runs', [])):
                if item.get('K') != standard_k(item.get('environment', '')):
                    continue
                self.sources[tuple(item[name] for name in RUN_ID)] = item

    def check(self,
              ident: dict,
              curve: pd.DataFrame,
              values: np.ndarray,
              index: np.ndarray,
              config: dict,
              mu: float,
              temperatures: pd.DataFrame | None) -> dict:
        """Counts at h=100 and, if the E1 history exists, its return prefix and recorded settings."""
        if self.base is None:
            return dict(status='not_checked', reason='E1 run CSV is unavailable')
        key = tuple(ident[name] for name in RUN_ID)
        base = self.base.loc[key]
        current = curve[curve.h == REFERENCE_H].iloc[0]
        names = ('archive_count', 'active_count')
        same = all(int(base[name]) == int(current[name]) for name in names)
        result = dict(status='counts_checked', counts_match=same,
                      e1_counts={name: int(base[name]) for name in names}, return_prefix_match=None,
                      history_status='unavailable', recorded_settings_differ=[], recorded_settings_unverified=[])
        item = self.sources.get(key, {})
        name = item.get('history', item.get('policy_path', item.get('path')))
        if not name:
            return result
        path = Path(name)
        if not path.is_absolute() and not path.is_file():
            path = self.path.parent / path
        result['history_path'] = str(path)
        if not path.is_file():
            return result
        old_values, old_index, old_config, provenance = load_run(path, ident, horizon=REFERENCE_H)
        old_mu, _ = effective_mu(ident['scalarizer'], old_config, ident['environment'], temperatures)
        result.update(history_status='checked', e1_schedule=schedule_record(old_config),
                      return_prefix_match=bool(np.array_equal(values[:REFERENCE_H + 1, np.argsort(index)],
                                                              old_values[:, np.argsort(old_index)])),
                      e1_identity_verified=provenance['identity_verified'])
        for name in FIXED_SETTINGS:
            if config.get(name) is None or old_config.get(name) is None:
                result['recorded_settings_unverified'].append(name)
            elif config[name] != old_config[name]:
                result['recorded_settings_differ'].append(name)
        if np.isfinite(mu) and not np.isclose(mu, old_mu, rtol=TARGET_RTOL, atol=TARGET_ATOL):
            result['recorded_settings_differ'].append('mu_effective')
        return result

    def report(self, inventory: list[dict]) -> dict:
        """Number of runs and histories compared and how many agree."""
        rows = [r['e1_check'] for r in inventory]
        checked = [r for r in rows if r['status'] == 'counts_checked']
        histories = [r for r in checked if r['history_status'] == 'checked']
        return dict(status='checked' if checked else 'not_checked', csv=str(self.path),
                    runs_compared=len(checked), equal_counts=sum(r['counts_match'] for r in checked),
                    histories_compared=len(histories),
                    equal_return_prefixes=sum(r['return_prefix_match'] for r in histories),
                    evidence='Counts compare only h=100. When available, raw greedy returns at 0..100 are compared '
                             'after preference alignment. Neither check establishes identical actor parameters or '
                             'optimizer state.')


def main() -> None:
    """Reconstructs every E5 run, compares the h=100 prefix with E1, and writes the run table and the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_root', type=Path,
                        default=Path('results/scalarization/dst_long_horizon/mo_ppo_nl_dst'))
    parser.add_argument('--geometry', type=Path,
                        default=Path('results/evaluation/scalarization/dst/geometry/dst_points.csv'))
    parser.add_argument('--targetability_dir', type=Path,
                        default=Path('results/evaluation/scalarization/dst/targetability'))
    parser.add_argument('--e1_runs', type=Path, help=f'Optional E1 comparison CSV; default: {E1_RUNS}, when available.')
    parser.add_argument('--output', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Runs
    refs = load_geometry(args.geometry)
    path = args.targetability_dir / 'targetability_summary.csv'
    temperatures = pd.read_csv(path) if path.is_file() else None
    e1 = E1Check(args.e1_runs or E1_RUNS)
    blocks = []
    inventory = []
    oracles = {}
    schedules = Counter()
    for key in sorted(expected_runs()):
        ident = dict(zip(RUN_ID, key))
        env = ident['environment']
        method = ident['scalarizer']
        k = standard_k(env)
        ident |= dict(P=k, K=k, R=1)
        path = find_history(args.result_root, ident)
        values, index, config, provenance = load_run(path, ident)
        front, ids, _ = refs[env]
        mu, mu_source = effective_mu(method, config, env, temperatures)
        schedule = schedule_record(config)
        schedules[json.dumps(schedule, sort_keys=True)] += 1
        curve = analyze(values, index, front, method, mu, k)
        check = e1.check(ident, curve, values, index, config, mu, temperatures)
        transitions = curve.h * k * provenance['num_steps'] * provenance['num_processes']
        curve = curve.assign(**ident, training_transitions=transitions, mu_effective=mu,
                             identity_verified=provenance['identity_verified'])
        blocks.append(curve)
        inventory.append(ident | provenance | dict(
            path=str(path), mu_source=mu_source, schedule=schedule, e1_check=check,
            missing_fixed_settings=[name for name in FIXED_SETTINGS if config.get(name) is None]))
        if f'{env}/{method}' not in oracles:
            scores = scalar_scores(method, front, lattice(k), mu)
            own = np.isclose(scores, scores.max(1, keepdims=True), rtol=TARGET_RTOL, atol=TARGET_ATOL)
            oracles[f'{env}/{method}'] = dict(K=k, weights=lattice(k).tolist(),
                                              target_ids_by_weight=[ids[x].tolist() for x in own],
                                              mu_effective=float(mu) if np.isfinite(mu) else None)
        if len(inventory) % 60 == 0:
            print(f'Read {len(inventory)} E5 runs', flush=True)
    frame = pd.concat(blocks, ignore_index=True).sort_values(KEY)

    # Manifest
    notes = dict(archive='Original E5 definition: distinct exact-front returns observed through h, including '
                         'initialization.',
                 identity='Past own-target hits use saved actor IDs or the explicitly recorded stable-row-order '
                          'assumption.',
                 schedule='A run-horizon-dependent decay can change the first 100 iterations. Equal saved decay '
                          'flags/ratios are not proof of equal realized rates. Missing schedule metadata remains '
                          'unknown.',
                 scope='Long-run cutoffs are repeated observations, not independent replicates. H=100 is not '
                       'appended to E1. Only the default learning-rate/entropy setting is extended; asymptotic '
                       'convergence is not tested.',
                 budgets='Training transitions include num_processes. Evaluation episodes are excluded.')
    protocol = dict(environments=ENVS, scalarizers=METHODS, seeds=SEEDS, H=H, reference_h=REFERENCE_H,
                    cutoffs=list(range(1, H + 1)), gamma=1, learning_rate=LEARNING_RATE, entropy_coef=ENTROPY,
                    cadence=1, mu_base=0.1, match_atol=MATCH_ATOL, target_atol=TARGET_ATOL, target_rtol=TARGET_RTOL)
    manifest = dict(schema=SCHEMA, status='complete', expected_runs=len(expected_runs()),
                    complete_runs=len(inventory), protocol=protocol,
                    inputs={key: str(value) if value is not None else None for key, value in vars(args).items()},
                    references={e: dict(front=f.tolist(), point_ids=i.tolist(), supported=s.tolist())
                                for e, (f, i, s) in refs.items()},
                    oracles=oracles, source_runs=inventory, e1_check=e1.report(inventory),
                    lr_decay=[json.loads(key) | dict(runs=n) for key, n in sorted(schedules.items())], notes=notes)

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'e5_manifest.json').write_text(json.dumps(dict(schema=SCHEMA, status='writing')))
    (args.output / 'e5_summary.csv').unlink(missing_ok=True)
    frame.to_csv(args.output / 'e5_runs.csv', index=False, float_format='%.17g')
    (args.output / 'e5_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n',
                                                  encoding='utf-8')
    print(f'E5: {len(inventory)} runs; {len(frame):,} cutoff rows. Output: {args.output}')
    print('E1 comparison:', json.dumps(manifest['e1_check']))
    print('Saved learning-rate schedules:', manifest['lr_decay'])


if __name__ == '__main__':
    main()