"""Fixed-protocol E3: saved greedy histories -> run and preference diagnostics. No training or actor execution; both
original budgets are retained separately, and only source and output paths are configurable.
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
H = 100
MATCH_ATOL = 1e-4
TARGET_ATOL = 1e-12
TARGET_RTOL = 1e-10
DEFAULT_OUTPUT = Path('results/evaluation/scalarization/dst/e3_lean')
ROOTS = {'standard': 'dst_weights_vs_replicas_default_budget/mo_ppo_nl_dst',
         'large': 'dst_weights_vs_replicas/mo_ppo_nl_dst', 'baseline': 'dst_weight_resolution/mo_ppo_nl_dst'}
KEY = ['budget', 'environment', 'scalarizer', 'P', 'K', 'R', 'seed']
PRIMARY = ('oracle_recall', 'archive_recall', 'active_recall', 'complete', 'target_coverage',
           'off_target_recall', 'active_target_coverage', 'active_off_target_recall',
           'weight_hit_ever', 'weight_hit_active')
METRICS = PRIMARY + ('archive_active_gap', 'archive_target_fraction', 'archive_off_target_fraction',
                     'active_target_fraction', 'active_off_target_fraction', 'actor_hit_ever', 'actor_hit_active',
                     'active_front_actor_rate', 'redundant_pareto_actor_rate', 'no_treasure_rate',
                     'off_front_treasure_rate', 'target_treasure_extra_rate', 'score_tied_miss_rate',
                     'corner_score_tied_miss_rate', 'interior_score_tied_miss_rate')


# Helpers

def allocations(env: str) -> list[tuple[str, int, int, int]]:
    """(budget, P, K, R) of every allocation of an environment at the standard and the large budget."""
    p = 20 if env.startswith('ldst') else 10
    ks = (20, 10, 5, 2) if env.startswith('ldst') else (10, 5, 2)
    return [('standard', p, k, p // k) for k in ks] + [('large', 50, k, 50 // k) for k in (50, 25, 10, 5, 2)]


def allocation(k: int, r: int) -> str:
    """Name of an allocation: all, corners, R2 or K<k>."""
    return 'all' if r == 1 else 'corners' if k == 2 else 'R2' if r == 2 else f'K{k}'


def expected_keys() -> set[tuple]:
    """Run keys of the complete fixed cohort."""
    return {(b, e, m, p, k, r, s) for e, m, s in product(ENVS, METHODS, SEEDS)
            for b, p, k, r in allocations(e)}


def ratio(n: int, d: int) -> float:
    """n / d, NaN for d = 0."""
    return float(n / d) if d else np.nan


def lattice(k: int) -> np.ndarray:
    """Two-objective K-point simplex grid."""
    x = np.linspace(0, 1, k)
    return np.column_stack((x, 1 - x))


def joined(ids: np.ndarray, mask: np.ndarray) -> str:
    """Selected front IDs joined by semicolons."""
    return ';'.join(map(str, ids[mask]))


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


def find_history(root: Path, ident: dict) -> tuple[Path, str]:
    """History of one run and its source; R=1 runs may come from the E1 baseline."""
    method = ident['scalarizer']
    r = ident['R']
    relative = Path(ident['environment']) / 'gamma_1/das_dennis' / f"P_{ident['P']:04d}/K_{ident['K']:04d}/R_{r:04d}"
    relative /= f'H_{H:05d}/eval_0001/lr_0p0003/ec_0p05/{method}'
    for source in [ident['budget']] + (['baseline'] if r == 1 else []):
        base = root / ROOTS[source] / relative
        bases = [base, base / 'mu_0p1'] if method.startswith('stch_') else [base]
        found = [x / f"s_{ident['seed']:04d}/policies/mo_ppo_nl_dst.npz" for x in bases]
        found = [x.resolve() for x in found if x.is_file()]
        if found:
            return found[0], source


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
                      num_steps=int(c['num_steps']), num_processes=int(c['num_processes']))
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
        return 0.1, 'fixed_E3_protocol'
    s = table[(table.environment == env) & (table.scalarizer == method)]
    return float(s.mu_effective.to_numpy(float)[0]), 'targetability_summary'


# Analysis

def analyze(returns: np.ndarray,
            index: np.ndarray,
            front: np.ndarray,
            ids: np.ndarray,
            method: str,
            mu: float,
            k: int) -> tuple[dict, list[dict]]:
    """E3 recalls of one run plus per-preference counts; no inferred trajectory mechanism."""
    p = returns.shape[1]
    scores = scalar_scores(method, front, lattice(k), mu)
    best = scores.max(1)
    own = np.isclose(scores, best[:, None], rtol=TARGET_RTOL, atol=TARGET_ATOL)
    targeted = own.any(0)
    hit = (np.abs(returns[:, :, None] - front) <= MATCH_ATOL).all(-1)
    archive = hit.any((0, 1))
    active = hit[-1].any(0)
    own_hit = (hit & own[index][None]).any(2)
    ever = own_hit.any(0)
    final = own_hit[-1]
    actual = scalar_scores(method, front, lattice(k), mu, returns[-1])[index, np.arange(p)]
    gap = best[index] - actual
    tied = np.abs(gap) <= TARGET_ATOL + TARGET_RTOL * np.abs(best[index])
    corner = (index == 0) | (index == k - 1)
    on_front = hit[-1].any(1)
    none = np.abs(returns[-1, :, 0]) <= MATCH_ATOL
    same_treasure = np.abs(returns[-1, :, None, 0] - front[:, 0]) <= MATCH_ATOL
    extra = (same_treasure & own[index] & (front[None, :, 1] - returns[-1, :, None, 1] > MATCH_ATOL)).any(1)
    preferences = []
    for w in range(k):
        group = index == w
        discovered = hit[:, group].any((0, 1))
        current = hit[-1, group].any(0)
        preferences.append(dict(
            weight_index=w, w0=lattice(k)[w, 0], w1=lattice(k)[w, 1], is_corner=bool(w in (0, k - 1)),
            target_pf_ids=joined(ids, own[w]), n_targets=int(own[w].sum()), n_replicas=int(group.sum()),
            replicas_hit_ever=int(ever[group].sum()), replicas_hit_active=int(final[group].sum()),
            weight_hit_ever=bool(ever[group].any()), weight_hit_active=bool(final[group].any()),
            archive_pf_ids=joined(ids, discovered), active_pf_ids=joined(ids, current),
            n_active_front_actors=int(on_front[group].sum()), n_distinct_active_returns=int(current.sum()),
            no_treasure_actors=int(none[group].sum()), target_treasure_extra_actors=int(extra[group].sum()),
            exact_misses=int((~final[group]).sum()), score_tied_misses=int((tied[group] & ~final[group]).sum()),
            gap_min=float(gap[group].min()), gap_median=float(np.median(gap[group])),
            gap_max=float(gap[group].max())))
    counts = dict(n_front=len(front), n_targets=int(targeted.sum()), archive_count=int(archive.sum()),
                  active_count=int(active.sum()), archive_targeted_count=int((archive & targeted).sum()),
                  active_targeted_count=int((active & targeted).sum()),
                  n_weights_hit_ever=sum(x['weight_hit_ever'] for x in preferences),
                  n_weights_hit_active=sum(x['weight_hit_active'] for x in preferences),
                  n_actors_hit_ever=int(ever.sum()), n_actors_hit_active=int(final.sum()),
                  n_active_front_actors=int(on_front.sum()),
                  n_redundant_front_actors=int(on_front.sum() - active.sum()),
                  n_no_treasure=int(none.sum()), n_off_front_treasure=int((~on_front & ~none).sum()),
                  n_target_treasure_extra=int(extra.sum()))
    n = len(front)
    t = int(targeted.sum())
    row = counts | dict(
        oracle_recall=t / n, archive_recall=float(archive.mean()), active_recall=float(active.mean()),
        complete=bool(archive.all()), target_coverage=ratio((archive & targeted).sum(), t),
        off_target_recall=ratio((archive & ~targeted).sum(), n - t),
        active_target_coverage=ratio((active & targeted).sum(), t),
        active_off_target_recall=ratio((active & ~targeted).sum(), n - t),
        weight_hit_ever=counts['n_weights_hit_ever'] / k, weight_hit_active=counts['n_weights_hit_active'] / k,
        actor_hit_ever=float(ever.mean()), actor_hit_active=float(final.mean()),
        archive_active_gap=float(archive.mean() - active.mean()),
        archive_target_fraction=float((archive & targeted).mean()),
        archive_off_target_fraction=float((archive & ~targeted).mean()),
        active_target_fraction=float((active & targeted).mean()),
        active_off_target_fraction=float((active & ~targeted).mean()), active_front_actor_rate=float(on_front.mean()),
        redundant_pareto_actor_rate=counts['n_redundant_front_actors'] / p, no_treasure_rate=float(none.mean()),
        off_front_treasure_rate=float((~on_front & ~none).mean()), target_treasure_extra_rate=float(extra.mean()))
    for prefix, group in (('', np.ones(p, bool)), ('corner_', corner), ('interior_', ~corner)):
        misses = int((group & ~final).sum())
        ties = int((group & ~final & tied).sum())
        row.update({f'{prefix}exact_misses': misses, f'{prefix}score_tied_misses': ties,
                    f'{prefix}score_tied_miss_rate': ratio(ties, misses)})
    return row, preferences


def main() -> None:
    """Reconstructs every E3 run and writes the run and preference CSVs and the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_root', type=Path, default=Path('results/scalarization'))
    parser.add_argument('--geometry', type=Path,
                        default=Path('results/evaluation/scalarization/dst/geometry/dst_points.csv'))
    parser.add_argument('--targetability_dir', type=Path,
                        default=Path('results/evaluation/scalarization/dst/targetability'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Runs
    refs = load_geometry(args.geometry)
    temp_path = args.targetability_dir / 'targetability_summary.csv'
    temp_table = pd.read_csv(temp_path) if temp_path.is_file() else None
    runs = []
    preferences = []
    inventory = []
    for key in sorted(expected_keys()):
        ident = dict(zip(KEY, key))
        path, source = find_history(args.result_root, ident)
        values, index, config, provenance = load_run(path, ident)
        front, ids, _ = refs[ident['environment']]
        mu, mu_source = effective_mu(ident['scalarizer'], config, ident['environment'], temp_table)
        row, pref = analyze(values, index, front, ids, ident['scalarizer'], mu, ident['K'])
        labels = ident | dict(allocation=allocation(ident['K'], ident['R']), h=H)
        transitions = H * ident['P'] * provenance['num_steps'] * provenance['num_processes']
        runs.append(labels | row | dict(training_transitions=transitions, mu_effective=mu, mu_source=mu_source,
                                        identity_verified=provenance['identity_verified']))
        preferences.extend(labels | x for x in pref)
        inventory.append(ident | provenance | dict(path=str(path), source=source, mu_source=mu_source,
                                                  mu_effective=mu if np.isfinite(mu) else None))
    frame = pd.DataFrame(runs).sort_values(KEY)

    # Manifest
    notes = dict(archive='Exact-front discoveries over all dense checkpoints; original E3 definition.',
                 weight_success='At least one assigned replica hits an own target; denominator K, not P.',
                 score_ties='Exact misses with numerical score equality; score gaps are not pooled across tasks.',
                 identity='Without saved IDs, within-actor history assumes stable row order; recorded per run.',
                 budgets='Training transitions include num_processes; legacy NPZ step labels may not. Budgets never '
                         'pooled together.',
                 scope='No replica-overlap, checkpoint-replacement, or causal optimization analysis.')
    protocol = dict(environments=ENVS, scalarizers=METHODS, seeds=SEEDS, h=H, cadence=1, gamma=1,
                    learning_rate=3e-4, entropy_coef=0.05, mu_base=0.1, roots=ROOTS,
                    allocations={e: allocations(e) for e in ENVS}, match_atol=MATCH_ATOL, target_atol=TARGET_ATOL,
                    target_rtol=TARGET_RTOL)
    manifest = dict(status='complete', schema_version=1, expected_runs=len(expected_keys()),
                    complete_runs=len(frame), protocol=protocol, inputs={n: str(v) for n, v in vars(args).items()},
                    references={e: dict(front=f.tolist(), point_ids=i.tolist(), supported=s.tolist())
                                for e, (f, i, s) in refs.items()},
                    source_runs=inventory, notes=notes)

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'e3_manifest.json').unlink(missing_ok=True)
    frame.to_csv(args.output / 'e3_runs.csv', index=False, float_format='%.17g')
    pd.DataFrame(preferences).sort_values(KEY + ['weight_index']).to_csv(args.output / 'e3_preferences.csv',
                                                                         index=False, float_format='%.17g')
    (args.output / 'e3_summary.csv').unlink(missing_ok=True)
    (args.output / 'e3_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f'E3: {len(frame)} runs; {len(preferences)} preference rows. Output: {args.output}')


if __name__ == '__main__':
    main()