"""Fixed-protocol E1 reconstruction from saved returns; no training or evaluation. Only the four filesystem locations
are command-line options; the paper protocol below is fixed. Run summarize_e1.py and plot_e1_paper.py afterwards, from
the same directory.
"""
import re
import json
import hashlib
import argparse
import itertools

import numpy as np
import pandas as pd

from typing import Sequence
from pathlib import Path
from dataclasses import dataclass


# Protocol

ENVS = ('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2')
METHODS = ('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm')
K_VALUES = (2, 5, 10, 15, 20, 30, 50)
SEEDS = tuple(range(1, 11))
HORIZONS = (10, 25, 50, 75, 100)
CADENCES = (1, 2, 4, 5, 10, 20, 25, 50, 100)
H = 100
LEARNING_RATE = 3e-4
ENTROPY = 0.05
MU = 0.1
MATCH_ATOL = 1e-4
TARGET_ATOL = 1e-12
TARGET_RTOL = 1e-10
WEIGHT_ATOL = 1e-6
HV_MARGIN = 0.1
RUN_KEYS = ['environment', 'scalarizer', 'K', 'seed']
SCHEDULE_KEYS = RUN_KEYS + ['horizon', 'cadence']
QUALITY_COLUMNS = ('archive_hv_ratio', 'active_hv_ratio', 'archive_igd_plus_norm', 'active_igd_plus_norm')
DATA_DIR = Path('results/evaluation/scalarization/dst/e1_lean')
SCHEMA = 'e1-lean-v1'


# Helpers

def standard_k(env: str) -> int:
    """Standard number of distinct preferences: 20 on LDST, 10 on DST."""
    return 20 if env.startswith('ldst') else 10


def read_json(path: Path) -> dict:
    """JSON file, with or without byte-order mark."""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def boolean(value: object) -> bool:
    """Boolean from a CSV value such as True, 1 or 1.0."""
    return str(value).strip().lower() in {'true', '1', '1.0'}


def float_token(x: float) -> str:
    """Number as a path token, e.g. 0.05 -> 0p05."""
    return f'{x:.6g}'.replace('-', 'm').replace('.', 'p')


def nd_max(points: np.ndarray) -> np.ndarray:
    """Exact 2D maximization nondominated set, with exact duplicate removal."""
    x = np.asarray(points, dtype=np.float64)
    if x.size == 0:
        return np.empty((0, 2))
    x = np.unique(x, axis=0)
    x = x[np.lexsort((-x[:, 1], -x[:, 0]))]
    keep = x[:, 1] > np.r_[-np.inf, np.maximum.accumulate(x[:-1, 1])]
    return x[keep]


def hv_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume of 2D maximization vectors above a fixed reference (actual vectors, no clipping)."""
    x = np.asarray(points, dtype=float)
    r = np.asarray(ref, dtype=float)
    if x.size == 0:
        return 0.0
    x = nd_max(x[np.all(x >= r, axis=1)])
    if not len(x):
        return 0.0
    x = x[np.argsort(x[:, 0])]
    return float(np.sum(np.diff(np.r_[r[0], x[:, 0]]) * (x[:, 1] - r[1])))


def igd_plus_norm(points: np.ndarray, front: np.ndarray) -> float:
    """Maximization IGD+: mean_f min_x ||max((f-x)/Delta,0)||_2, Delta = exact-front ranges."""
    f = np.asarray(front, dtype=float)
    x = np.asarray(points, dtype=float)
    delta = np.ptp(f, axis=0)
    if x.size == 0:
        return float('inf')
    best = np.full(len(f), np.inf)
    for start in range(0, len(x), 4096):
        gaps = np.maximum((f[:, None, :] - x[None, start:start + 4096, :]) / delta, 0.0)
        best = np.minimum(best, np.linalg.norm(gaps, axis=2).min(axis=1))
    return float(best.mean())


def scalar_scores(method: str,
                  front: np.ndarray,
                  weights: np.ndarray,
                  mu: float,
                  values: np.ndarray | None = None) -> np.ndarray:
    """Scores [weights, values] with the fixed exact-front references; signed gaps and zero-weight masking."""
    front = np.asarray(front, dtype=float)
    weights = np.asarray(weights, dtype=float)
    values = front if values is None else np.asarray(values, dtype=float)
    upper = front.max(0)
    lower = front.min(0)
    delta = upper - lower
    if method == 'ws_raw':
        return weights @ values.T
    if method == 'ws_norm':
        return weights @ ((values - lower) / delta).T
    gap = upper - values
    if method.endswith('_norm'):
        gap = gap / delta
    z = weights[:, None, :] * gap[None, :, :]
    z = np.where(weights[:, None, :] > 0, z, -np.inf)
    maximum = z.max(axis=2)
    if method.startswith('tch_'):
        return -maximum
    return -maximum - mu * np.log(np.exp((z - maximum[:, :, None]) / mu).sum(axis=2))


def match_matrix(points: np.ndarray, front: np.ndarray, atol: float = MATCH_ATOL) -> np.ndarray:
    """Whether every point equals every front point within atol in both objectives."""
    return np.all(np.abs(points[..., None, :] - front) <= atol, axis=-1)


# Data

@dataclass
class Reference:
    """Exact front of one condition with its oracle targets and hypervolume reference."""
    front: np.ndarray
    supported: np.ndarray
    targeted: np.ndarray
    targets_by_weight: np.ndarray | None
    effective_mu: float
    point_ids: np.ndarray
    hv_ref: np.ndarray | None = None
    true_hv: float | None = None


@dataclass
class History:
    """Dense actor returns of one run with its settings, preferences and actor identities."""
    returns: np.ndarray
    num_steps: int
    num_processes: int
    eval_rep: int
    assigned_targets: np.ndarray | None
    config_path: str | None
    config: dict
    weights: np.ndarray | None = None
    weight_source: str = 'unavailable'
    policy_index_source: str = 'row_index_assumed_stable'
    policy_ids: np.ndarray | None = None


def load_references(geometry_path: Path, targetability_dir: Path) -> dict:
    """Exact front and oracle targets of every environment, scalarizer and K."""
    geometry = pd.read_csv(geometry_path)
    points = pd.read_csv(targetability_dir / 'targetability_points.csv', keep_default_na=False)
    summary = pd.read_csv(targetability_dir / 'targetability_summary.csv')
    result = {}
    for env in ENVS:
        geom = geometry.loc[geometry.environment.eq(env)]
        if 'gamma' in geom:
            geom = geom.loc[np.isclose(geom.gamma.astype(float), 1, atol=1e-12, rtol=0)]
        geom = geom.sort_values('point_index')
        front = geom[['objective_0', 'objective_1']].to_numpy(float)
        ids = geom.point_index.to_numpy(int)
        supported = np.array([boolean(v) for v in geom.linear_supported])
        href = front.min(0) - HV_MARGIN * np.ptp(front, axis=0)
        for method, k in itertools.product(METHODS, K_VALUES):
            table = points.loc[points.environment.eq(env) & points.scalarizer.eq(method)
                               & points.num_distinct_weights.astype(int).eq(k)]
            s = summary.loc[summary.environment.eq(env) & summary.scalarizer.eq(method)
                            & summary.num_distinct_weights.eq(k)]
            table = table.set_index('point_index').loc[ids]
            targeted = np.array([boolean(v) for v in table.finite_grid_targeted])
            targets = None
            if 'targeting_weight_ids' in table:
                targets = np.zeros((k, len(front)), dtype=bool)
                for j, value in enumerate(table.targeting_weight_ids):
                    for token in str(value).split(','):
                        if token.strip():
                            targets[int(float(token)), j] = True
            mu = float(s.iloc[0].mu_effective) if method.startswith('stch_') else np.nan
            result[env, method, k] = Reference(front, supported, targeted, targets, mu, ids, href, hv_2d(front, href))
    return result


def find_history(root: Path,
                 env: str,
                 method: str,
                 k: int,
                 seed: int) -> Path:
    """NPZ history of one run; raw and normalized STCH runs may lie below mu_<MU>."""
    base = (root / env / 'gamma_1' / 'das_dennis' / f'P_{k:04d}' / f'K_{k:04d}' / 'R_0001' / f'H_{H:05d}' /
            'eval_0001' / f'lr_{float_token(LEARNING_RATE)}' / f'ec_{float_token(ENTROPY)}' / method)
    suffix = Path(f's_{seed:04d}') / 'policies' / 'mo_ppo_nl_dst.npz'
    candidates = [base / suffix]
    if method.startswith('stch_'):
        candidates.append(base / f'mu_{float_token(MU)}' / suffix)
    return [p for p in candidates if p.is_file()][0]


def saved_config(path: Path) -> tuple[dict, Path]:
    """Saved learner configuration of a run and its path."""
    root = path.parent.parent
    for p in sorted({*root.glob('*.json'), *root.glob('config/*.json'), *root.glob('configs/*.json')}):
        obj = read_json(p)
        if isinstance(obj, dict) and {'num_steps', 'scalarization_method', 'num_subproblems'} <= obj.keys():
            return obj, p


def load_history(path: Path,
                 method: str,
                 k: int,
                 ref: Reference) -> History:
    """Raw actor returns aligned by actor IDs, with the preference grid and the assigned oracle targets."""
    config, cpath = saved_config(path)
    steps = int(config['num_steps'])
    processes = int(config['num_processes'])
    history = {}
    weights = None
    weight_ids = None
    policy_ids = None
    weight_source = 'unavailable'
    index_source = 'row_index_assumed_stable'
    metadata_presence = []
    with np.load(path, allow_pickle=False) as data:
        keys = sorted((x for x in data.files if re.fullmatch(r't_\d+', x)), key=lambda x: int(x[2:]))
        for key in keys:
            step = int(key[2:])
            if f'{key}_iteration' in data:
                iteration = int(np.asarray(data[f'{key}_iteration']).item())
            else:
                iteration = step // (k * steps)
            order = np.arange(k)
            has_ids = f'{key}_agent_ids' in data
            metadata_presence.append(has_ids)
            if has_ids:
                ids = np.asarray(data[f'{key}_agent_ids'])
                order = np.argsort(ids)
                policy_ids = ids[order]
            if f'{key}_weight_ids' in data:
                weight_ids = np.asarray(data[f'{key}_weight_ids'])[order].astype(int)
            if f'{key}_weights' in data:
                weights = np.asarray(data[f'{key}_weights'], float)[order]
                weight_source = 'npz_weights'
            history[iteration] = np.asarray(data[key], dtype=float)[order]
    if metadata_presence and all(metadata_presence):
        index_source = 'agent_ids'
    for name in ('initial_weights', 'distinct_weights'):
        if config.get(name) is None:
            continue
        candidate = np.asarray(config[name], float)
        if name == 'initial_weights' and policy_ids is not None:
            canonical = (np.issubdtype(policy_ids.dtype, np.number)
                         and np.array_equal(policy_ids, policy_ids.astype(int))
                         and np.array_equal(np.sort(policy_ids.astype(int)), np.arange(k)))
            if canonical:
                candidate = candidate[policy_ids.astype(int)]
            elif weights is None:
                continue
        elif name == 'distinct_weights' and weight_ids is not None:
            candidate = candidate[weight_ids]
        elif name == 'distinct_weights' and weights is None and policy_ids is not None:
            continue
        weights = candidate
        weight_source = f'json_{name}'
        break
    assigned = None
    exact_weights = None
    if weights is not None:
        lattice = np.linspace(0, 1, k)
        indices = np.rint(weights[:, 0] * (k - 1)).astype(int)
        exact_weights = np.column_stack((lattice[indices], 1 - lattice[indices]))
        scores = scalar_scores(method, ref.front, exact_weights, ref.effective_mu)
        assigned = np.isclose(scores, scores.max(1, keepdims=True), rtol=TARGET_RTOL, atol=TARGET_ATOL)
    elif weight_ids is not None and ref.targets_by_weight is not None:
        assigned = ref.targets_by_weight[weight_ids]
    return History(np.stack([history[t] for t in range(H + 1)]), steps, processes, 1, assigned,
                   str(cpath) if cpath else None, config, exact_weights, weight_source, index_source, policy_ids)


# Analysis

def schedule(h: int, c: int) -> np.ndarray:
    """Observed iterations up to h at cadence c, always including h."""
    return np.unique(np.r_[np.arange(0, h + 1, c), h]).astype(int)


def first_complete(times: Sequence[int], masks: Sequence[np.ndarray]) -> float:
    """First observed iteration whose archive contains the whole front; NaN if none."""
    return next((float(t) for t, mask in zip(times, masks) if mask.all()), np.nan)


def point_evidence(hist: History,
                   ref: Reference,
                   active: np.ndarray,
                   dense: dict) -> list[dict]:
    """Per Pareto return: support, targets, first discovery, and active and archive membership at the cutoffs."""
    targets = ref.targets_by_weight if ref.targets_by_weight is not None else hist.assigned_targets
    multiplicity = targets.sum(0) if targets is not None else np.full(len(ref.front), np.nan)
    rows = []
    for j, value in enumerate(ref.front):
        seen = np.flatnonzero(active[:, j])
        row = dict(pf_id=int(ref.point_ids[j]), J0=value[0], J1=value[1], supported=ref.supported[j],
                   targeted=ref.targeted[j], n_targeting_weights=multiplicity[j],
                   first_discovered_iteration_dense=int(seen[0]) if len(seen) else None)
        for h in HORIZONS:
            row.update({f'active_h{h}': bool(active[h, j]), f'archive_h{h}': bool(dense[h][j])})
        rows.append(row)
    return rows


def policy_evidence(hist: History,
                    ref: Reference,
                    method: str,
                    matches: np.ndarray) -> list[dict]:
    """All actors at the fixed cutoffs; exact vector hits and score ties are separate."""
    targets = hist.assigned_targets
    weights = hist.weights
    hits = (matches & targets[None]).any(2) if targets is not None else None
    best = scalar_scores(method, ref.front, weights, ref.effective_mu).max(1) if weights is not None else None
    rows = []
    for h in HORIZONS:
        scores = (np.diag(scalar_scores(method, ref.front, weights, ref.effective_mu, hist.returns[h]))
                  if weights is not None else None)
        for i, value in enumerate(hist.returns[h]):
            pf = np.flatnonzero(matches[h, i])
            target = targets[i] if targets is not None else None
            own = bool(hits[h, i]) if hits is not None else None
            grid_hit = bool(ref.targeted[pf].any())
            relation = ('off_front' if not len(pf) else 'own_target' if own else
                        'grid_target_assignment_unknown' if grid_hit and target is None else
                        'other_grid_target' if grid_hit else 'off_target_pareto')
            treasure = np.isclose(ref.front[:, 0], value[0], atol=MATCH_ATOL, rtol=0)
            excess = float(ref.front[treasure, 1].max() - value[1]) if treasure.any() else None
            seen = np.flatnonzero(hits[:h + 1, i]) if hits is not None else None
            gap = float(best[i] - scores[i]) if scores is not None else None
            row = dict(horizon=h, policy_index=i,
                       policy_id=str(hist.policy_ids[i]) if hist.policy_ids is not None else None,
                       identity_verified=hist.policy_index_source == 'agent_ids',
                       w0=weights[i, 0] if weights is not None else None,
                       w1=weights[i, 1] if weights is not None else None,
                       J0=value[0], J1=value[1], matched_pf_id=int(ref.point_ids[pf[0]]) if len(pf) else None,
                       assigned_target_pf_ids=';'.join(map(str, ref.point_ids[target])) if target is not None else None,
                       matches_assigned_target=own,
                       ever_matched_assigned_target=bool(len(seen)) if seen is not None else None,
                       first_assigned_hit_iteration=int(seen[0]) if seen is not None and len(seen) else None,
                       front_relation=relation, zero_treasure=bool(abs(value[0]) <= MATCH_ATOL),
                       own_target_treasure_reached=bool((treasure & target).any()) if target is not None else None,
                       excess_steps=excess, scalarized_score=float(scores[i]) if scores is not None else None,
                       best_front_score=float(best[i]) if best is not None else None, scalarized_gap=gap,
                       score_tied=(bool(abs(gap) <= TARGET_ATOL + TARGET_RTOL * abs(best[i]))
                                   if gap is not None else None))
            rows.append(row)
    return rows


def analyze_run(hist: History,
                ref: Reference,
                env: str,
                method: str,
                k: int) -> tuple[list[dict], list[dict], list[dict], np.ndarray]:
    """Run rows per cutoff and cadence, point evidence, policy evidence and the exact-front matches."""
    matches = match_matrix(hist.returns, ref.front)
    active = matches.any(1)
    plans = {(h, 1) for h in HORIZONS}
    if k == standard_k(env):
        plans |= {(H, c) for c in CADENCES}
    caches = {}
    for c in sorted({c for _, c in plans}):
        archive = np.empty((0, 2))
        prefixes = {}
        for t in range(0, H + 1, c):
            archive = nd_max(np.vstack((archive, hist.returns[t])))
            prefixes[t] = (archive, match_matrix(archive, ref.front).any(0))
        caches[c] = prefixes
    dense = {t: pair[1] for t, pair in caches[1].items()}
    assigned_hits = (matches & hist.assigned_targets[None]).any(2) if hist.assigned_targets is not None else None
    rows = []
    for h, c in sorted(plans):
        times = schedule(h, c)
        pairs = [caches[c][int(t)] for t in times if t % c == 0]
        if h % c:
            vectors = nd_max(np.vstack((pairs[-1][0], hist.returns[h])))
            pairs.append((vectors, match_matrix(vectors, ref.front).any(0)))
        vectors, archive = pairs[-1]
        current = active[h]
        tm = ref.targeted
        sm = ref.supported
        first = first_complete(times, [p[1] for p in pairs])
        dense_first = first_complete(range(h + 1), [dense[t] for t in range(h + 1)])
        current_vectors = nd_max(hist.returns[h])
        row = dict(horizon=h, cadence=c, training_transitions=h * k * hist.num_steps * hist.num_processes,
                   observed_policy_evaluations=k * len(times),
                   scheduled_evaluation_episodes=k * len(times) * hist.eval_rep,
                   num_pf_points=len(ref.front), num_oracle_targets=int(tm.sum()), num_supported_points=int(sm.sum()),
                   archive_count=int(archive.sum()), archive_targeted_count=int((archive & tm).sum()),
                   archive_off_target_count=int((archive & ~tm).sum()),
                   archive_supported_count=int((archive & sm).sum()),
                   archive_unsupported_count=int((archive & ~sm).sum()), active_count=int(current.sum()),
                   active_targeted_count=int((current & tm).sum()),
                   active_off_target_count=int((current & ~tm).sum()),
                   active_supported_count=int((current & sm).sum()),
                   active_unsupported_count=int((current & ~sm).sum()),
                   discovered_target_absent_count=int((archive & tm & ~current).sum()),
                   discovered_off_target_absent_count=int((archive & ~tm & ~current).sum()),
                   dense_archive_count=int(dense[h].sum()), dense_targeted_count=int((dense[h] & tm).sum()),
                   dense_off_target_count=int((dense[h] & ~tm).sum()),
                   assigned_policy_count=k if assigned_hits is not None else 0,
                   assigned_hit_ever_count=int(assigned_hits[:h + 1].any(0).sum()) if assigned_hits is not None else 0,
                   assigned_hit_active_count=int(assigned_hits[h].sum()) if assigned_hits is not None else 0,
                   oracle_recall=float(tm.mean()), archive_recall=float(archive.mean()),
                   active_recall=float(current.mean()),
                   complete_archive=bool(archive.all()), first_complete_iteration=first,
                   dense_complete_archive=bool(dense[h].all()), dense_first_complete_iteration=dense_first,
                   completion_delay=first - dense_first if np.isfinite(first) else np.nan,
                   archive_hv_ratio=hv_2d(vectors, ref.hv_ref) / ref.true_hv,
                   active_hv_ratio=hv_2d(current_vectors, ref.hv_ref) / ref.true_hv,
                   archive_igd_plus_norm=igd_plus_norm(vectors, ref.front),
                   active_igd_plus_norm=igd_plus_norm(current_vectors, ref.front))
        rows.append(row)
    return rows, point_evidence(hist, ref, active, dense), policy_evidence(hist, ref, method, matches), matches


def history_signature(hist: History, ref: Reference) -> str:
    """Hash of the full recorded interpretation, to consolidate only identical K=2 aliases."""
    digest = hashlib.sha256()
    for value in (hist.returns, hist.weights, hist.assigned_targets, ref.front, ref.supported,
                  ref.targeted, np.array([hist.num_steps, hist.num_processes, hist.eval_rep])):
        if value is None:
            digest.update(b'None')
        else:
            value = np.asarray(value)
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    """Reads every E1 run and writes the four evidence CSVs and the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_root', type=Path,
                        default=Path('results/scalarization/dst_weight_resolution/mo_ppo_nl_dst'))
    parser.add_argument('--geometry', type=Path,
                        default=Path('results/evaluation/scalarization/dst/geometry/dst_points.csv'))
    parser.add_argument('--targetability_dir', type=Path,
                        default=Path('results/evaluation/scalarization/dst/targetability'))
    parser.add_argument('--output', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Runs
    refs = load_references(args.geometry, args.targetability_dir)
    frames = {'e1_runs.csv': [], 'e1_points.csv': [], 'e1_policy_diagnostics.csv': []}
    audit = []
    aliases = {}
    training = {}
    for env, method, k, seed in itertools.product(ENVS, METHODS, K_VALUES, SEEDS):
        path = find_history(args.result_root, env, method, k, seed)
        ref = refs[env, method, k]
        hist = load_history(path, method, k, ref)
        rows, points, policies, matches = analyze_run(hist, ref, env, method, k)
        representative = method
        if k == 2:
            key = (env, seed, method.rsplit('_', 1)[1], history_signature(hist, ref))
            representative = aliases.setdefault(key, method)
        base = dict(zip(RUN_KEYS, (env, method, k, seed)))
        base.update(include_in_pooled=representative == method, pool_representative=representative)
        for name, records in zip(frames, (rows, points, policies)):
            frames[name].extend(base | row for row in records)
        audit.append(base | dict(history=str(path.resolve()), config=hist.config_path,
                                identity=hist.policy_index_source, weights=hist.weight_source,
                                num_steps=hist.num_steps, num_processes=hist.num_processes, eval_rep=hist.eval_rep))
        if k == standard_k(env):
            none = abs(hist.returns[..., 0]) <= MATCH_ATOL
            other = ~matches.any(2) & ~none
            shares = np.column_stack((matches.mean(1), other.mean(1), none.mean(1)))
            training.setdefault((env, method, k), []).append(shares)
        if len(audit) % 60 == 0:
            print(f'Read {len(audit)} E1 runs', flush=True)

    # Evidence
    result = {name: pd.DataFrame(rows) for name, rows in frames.items()}
    result['e1_training.csv'] = pd.DataFrame([
        dict(environment=env, scalarizer=method, K=k, iteration=t, outcome=outcome, kind=kind,
             J0=j0, J1=j1, share=float(means[t, j]), n_seeds=len(values))
        for (env, method, k), values in training.items()
        for means in [np.mean(values, axis=0)]
        for j, (outcome, kind, j0, j1) in enumerate(
            [(str(int(pid)), 'pareto', *v) for pid, v in zip(refs[env, method, k].point_ids,
                                                               refs[env, method, k].front)]
            + [('off_front', 'off_front', np.nan, np.nan), ('zero', 'zero', np.nan, np.nan)])
        for t in range(H + 1)])
    manifest = dict(schema=SCHEMA, status='complete', n_runs=len(audit),
                    paths={key: str(value.resolve()) for key, value in vars(args).items()},
                    protocol=dict(envs=ENVS, scalarizers=METHODS, k_values=K_VALUES, seeds=SEEDS, horizon=H,
                                  cutoffs=HORIZONS, cadences=CADENCES, learning_rate=LEARNING_RATE,
                                  entropy_coef=ENTROPY, mu=MU, match_atol=MATCH_ATOL,
                                  target_atol=TARGET_ATOL, target_rtol=TARGET_RTOL, hv_margin=HV_MARGIN),
                    references={env: dict(front=refs[env, METHODS[0], K_VALUES[0]].front.tolist(),
                                          point_ids=refs[env, METHODS[0], K_VALUES[0]].point_ids.tolist())
                                for env in ENVS},
                    runs=audit)

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'e1_manifest.json').write_text(json.dumps({'schema': SCHEMA, 'status': 'writing'}))
    (args.output / 'e1_summary.csv').unlink(missing_ok=True)
    for name, frame in result.items():
        for col in ('pf_id', 'matched_pf_id', 'n_targeting_weights', 'first_discovered_iteration_dense',
                    'first_assigned_hit_iteration'):
            if col in frame:
                frame[col] = pd.array(frame[col], dtype='Int64')
        frame.to_csv(args.output / name, index=False, float_format='%.17g')
    (args.output / 'e1_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved four evidence CSVs and e1_manifest.json to {args.output}')


if __name__ == '__main__':
    main()