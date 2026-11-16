"""Fixed E2 protocol, reconstructed from paired return sidecars; no training or actor execution. Run this,
summarize_e2.py, then plot_e2_paper.py. Only source and output paths are configurable; the first-discovery archive and
the exact-coverage, held-out and HV definitions are retained.
"""
import re
import json
import argparse
import itertools

import numpy as np
import pandas as pd

from typing import Iterator
from pathlib import Path


# Protocol

ENVS = ('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2')
METHODS = ('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm')
ENTROPIES = (0.0, 0.025, 0.05, 0.1)
SEEDS = tuple(range(1, 11))
H = 100
EPISODES = 100
HALF = 50
QS = (0.5, 0.9)
MATCH_ATOL = 1e-4
HV_OFFSET = 0.1
LEARNING_RATE = 3e-4
STCH_MU = 0.1
DEFAULT_ENTROPY = 0.05
DATA_DIR = Path('results/evaluation/scalarization/dst/e2_lean')
SCHEMA = 'e2-lean-v1'
RUN_KEY = ['entropy_coef', 'environment', 'scalarizer', 'K', 'seed']
KEY = RUN_KEY + ['h', 'collection', 'execution']
PRIMARY = ('coverage_q050', 'coverage_q090', 'coverage_heldout_q050', 'coverage_heldout_q090', 'hv')
DIAGNOSTICS = ('treasure_coverage_q050', 'treasure_coverage_q090', 'pareto_episode_rate',
               'detour_episode_rate', 'no_treasure_episode_rate', 'other_episode_rate',
               'greedy_exact_rate', 'greedy_treasure_rate', 'checkpoint_age_mean')
METRICS = PRIMARY + DIAGNOSTICS + ('n_policies', 'n_greedy_front_policies')


# Helpers

def standard_k(env: str) -> int:
    """Standard number of distinct preferences: 20 on LDST, 10 on DST."""
    return 20 if env.startswith('ldst') else 10


def token(value: float) -> str:
    """Number as a path token, e.g. 0.05 -> 0p05."""
    return f'{value:.6g}'.replace('-', 'm').replace('.', 'p')


def read_json(path: Path) -> dict:
    """JSON file, with or without byte-order mark."""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def matches(values: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Whether every value equals every front point within the matching tolerance."""
    return np.all(np.abs(np.asarray(values)[..., None, :] - front) <= MATCH_ATOL, axis=-1)


def dominated_mask(values: np.ndarray) -> np.ndarray:
    """Points dominated by another point under the E2 tolerance convention."""
    difference = values[None, :, :] - values[:, None, :]
    return ((difference >= -MATCH_ATOL).all(-1) & (difference > MATCH_ATOL).any(-1)).any(1)


def hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    """Exact 2D maximization HV of the actual vectors; no snapping to the front."""
    points = np.asarray(points, float).reshape(-1, 2)
    points = points[(points > reference).all(1)]
    area = 0.0
    best_y = float(reference[1])
    for x, y in points[np.lexsort((-points[:, 1], -points[:, 0]))]:
        if y > best_y:
            area += (x - reference[0]) * (y - best_y)
            best_y = y
    return float(area)


def conditional_mean(values: np.ndarray, greedy_front: np.ndarray) -> float:
    """Mean over the policies whose greedy return lies on the front; NaN if there are none."""
    return float(values[greedy_front].mean()) if greedy_front.any() else np.nan


# Data

def load_references(path: Path) -> dict:
    """Exact gamma=1 front of every environment with its point IDs and hypervolume reference."""
    frame = pd.read_csv(path)
    if 'gamma' in frame:
        frame = frame[np.isclose(pd.to_numeric(frame.gamma), 1.0, rtol=0, atol=1e-12)]
    references = {}
    for env in ENVS:
        group = frame[frame.environment.eq(env)].sort_values('point_index')
        front = group[['objective_0', 'objective_1']].to_numpy(float)
        ids = group.point_index.to_numpy(float)
        href = front.min(0) - HV_OFFSET * np.ptp(front, axis=0)
        references[env] = dict(front=front, ids=ids.astype(int), hv_reference=href,
                               hv_front=hypervolume(front, href))
    return references


def sidecar_path(root: Path, ident: dict) -> Path:
    """Paired return sidecar of one run; raw STCH runs lie below mu_<STCH_MU>."""
    env = ident['environment']
    method = ident['scalarizer']
    k = ident['K']
    base = (root / env / 'gamma_1' / 'das_dennis' / f'P_{k:04d}' / f'K_{k:04d}' / 'R_0001'
            / f'H_{H:05d}' / 'eval_0001' / f'lr_{token(LEARNING_RATE)}' / f"ec_{token(ident['entropy_coef'])}" / method)
    if method == 'stch_raw':
        base = base / f'mu_{token(STCH_MU)}'
    return base / f"s_{ident['seed']:04d}" / 'policies' / 'stoch_vs_deter.npz'


def checkpoints(path: Path) -> Iterator[dict]:
    """Greedy and sampled returns of every checkpoint, paired within the checkpoint by actor row."""
    with np.load(path, allow_pickle=False) as side:
        names = sorted((n[:-10] for n in side.files if re.fullmatch(r't_\d+_iteration', n)),
                       key=lambda name: int(name[2:]))
        for index, prefix in enumerate(names):
            det = np.asarray(side[f'{prefix}_det_returns'], float)
            episodes = np.asarray(side[f'{prefix}_episode_returns'], float)
            yield dict(checkpoint=np.full(len(det), index), actor_row=np.arange(len(det)), det=det, episodes=episodes)


# Analysis

def merge_archive(previous: dict | None, current: dict) -> dict:
    """First-discovery archive: first representative after six-decimal duplicate removal, greedy dominance only."""
    if previous is None:
        records = current
    else:
        records = {name: np.concatenate((previous[name], current[name])) for name in current}
    _, first = np.unique(np.round(records['det'], 6), axis=0, return_index=True)
    first = np.sort(first)
    keep = first[~dominated_mask(records['det'][first])]
    return {name: values[keep] for name, values in records.items()}


def collection_evidence(records: dict,
                        ident: dict,
                        collection: str,
                        ref: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Primary metrics and descriptive diagnostics of one fixed checkpoint collection."""
    det = records['det']
    episodes = records['episodes']
    front = ref['front']
    n = len(det)
    base = ident | dict(h=H, collection=collection)
    exact = matches(episodes, front)
    treasure = np.abs(episodes[:, :, None, 0] - front[None, None, :, 0]) <= MATCH_ATOL
    det_match = matches(det, front)
    counts = exact.sum(1)
    chosen = counts.argmax(0)
    j = np.arange(len(front))
    best = counts[chosen, j] / EPISODES
    first = exact[:, :HALF].sum(1)
    second = exact[:, HALF:].sum(1)
    held_choice = first.argmax(0)
    observed = first.max(0) > 0
    held = np.where(observed, second[held_choice, j] / HALF, 0.0)
    best_treasure = treasure.mean(1).max(0)
    detour = (treasure & (episodes[:, :, None, 1] < front[None, None, :, 1] - MATCH_ATOL)).any(-1)
    on_front = exact.any(-1)
    none = np.abs(episodes[:, :, 0]) <= MATCH_ATOL
    other = ~(on_front | detour | none)
    own_greedy_exact = np.all(np.abs(episodes - det[:, None, :]) <= MATCH_ATOL, axis=-1)
    own_greedy_treasure = np.abs(episodes[:, :, 0] - det[:, None, 0]) <= MATCH_ATOL
    greedy_front = det_match.any(1)
    rows = []
    for execution, vectors, frequencies, held_frequencies, treasure_freq in (
            ('deterministic', det, det_match.any(0).astype(float), det_match.any(0).astype(float),
             (np.abs(det[:, None, 0] - front[None, :, 0]) <= MATCH_ATOL).any(0).astype(float)),
            ('sampled', episodes.mean(1), best, held, best_treasure)):
        if execution == 'sampled':
            rates = dict(pareto_episode_rate=float(on_front.mean()), detour_episode_rate=float(detour.mean()),
                         no_treasure_episode_rate=float(none.mean()), other_episode_rate=float(other.mean()),
                         greedy_exact_rate=conditional_mean(own_greedy_exact, greedy_front),
                         greedy_treasure_rate=conditional_mean(own_greedy_treasure, greedy_front))
        else:
            dtreasure = np.abs(det[:, None, 0] - front[None, :, 0]) <= MATCH_ATOL
            ddetour = (dtreasure & (det[:, None, 1] < front[None, :, 1] - MATCH_ATOL)).any(1)
            dnone = np.abs(det[:, 0]) <= MATCH_ATOL
            rates = dict(pareto_episode_rate=float(greedy_front.mean()), detour_episode_rate=float(ddetour.mean()),
                         no_treasure_episode_rate=float(dnone.mean()),
                         other_episode_rate=float((~(greedy_front | ddetour | dnone)).mean()),
                         greedy_exact_rate=1.0 if greedy_front.any() else np.nan,
                         greedy_treasure_rate=1.0 if greedy_front.any() else np.nan)
        rows.append(base | dict(execution=execution, n_policies=n, n_front=len(front),
                               n_episodes=EPISODES if execution == 'sampled' else 1,
                               n_greedy_front_policies=int(greedy_front.sum()),
                               checkpoint_age_mean=float((H - records['checkpoint']).mean()),
                               hv=hypervolume(vectors, ref['hv_reference']) / ref['hv_front'])
                    | {f'coverage_q{int(q * 100):03d}': float((frequencies >= q - 1e-12).mean()) for q in QS}
                    | {f'coverage_heldout_q{int(q * 100):03d}': float((held_frequencies >= q - 1e-12).mean())
                       for q in QS}
                    | {f'treasure_coverage_q{int(q * 100):03d}': float((treasure_freq >= q - 1e-12).mean())
                       for q in QS}
                    | rates)
    point_rows = [base | dict(pf_id=int(pid), J0=float(v[0]), J1=float(v[1]),
                             greedy_present=bool(det_match[:, j].any()),
                             best_exact_count=int(counts[chosen[j], j]),
                             best_checkpoint=int(records['checkpoint'][chosen[j]]),
                             best_actor_row=int(records['actor_row'][chosen[j]]),
                             best_treasure_count=int(treasure[:, :, j].sum(1).max()),
                             selection_exact_count=int(first[held_choice[j], j]),
                             heldout_exact_count=int(second[held_choice[j], j]) if observed[j] else 0,
                             heldout_checkpoint=int(records['checkpoint'][held_choice[j]]) if observed[j] else None,
                             heldout_actor_row=int(records['actor_row'][held_choice[j]]) if observed[j] else None)
                  for j, (pid, v) in enumerate(zip(ref['ids'], front))]
    policy_rows = [base | dict(checkpoint=int(records['checkpoint'][i]), actor_row=int(records['actor_row'][i]),
                              checkpoint_age=int(H - records['checkpoint'][i]),
                              greedy_J0=float(det[i, 0]), greedy_J1=float(det[i, 1]),
                              mean_J0=float(episodes[i, :, 0].mean()), mean_J1=float(episodes[i, :, 1].mean()),
                              greedy_on_front=bool(greedy_front[i]),
                              sampled_greedy_return_count=int(own_greedy_exact[i].sum()),
                              sampled_greedy_treasure_count=int(own_greedy_treasure[i].sum()),
                              pareto_count=int(on_front[i].sum()), detour_count=int(detour[i].sum()),
                              no_treasure_count=int(none[i].sum()), other_count=int(other[i].sum()))
                   for i in range(n)]
    return rows, point_rows, policy_rows


def analyze_run(path: Path, ident: dict, ref: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Evidence of the final active population and of the first-discovery archive of one run."""
    archive = None
    for current in checkpoints(path):
        archive = merge_archive(archive, current)
    runs = []
    points = []
    policies = []
    for collection, records in (('population', current), ('archive', archive)):
        r, p, a = collection_evidence(records, ident, collection, ref)
        runs.extend(r)
        points.extend(p)
        policies.extend(a)
    return runs, points, policies


def main() -> None:
    """Reconstructs every E2 run and writes the three evidence CSVs and the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired_root', type=Path,
                        default=Path('results/scalarization/dst_stoch_vs_deter/paired/mo_ppo_nl_dst'))
    parser.add_argument('--geometry', type=Path,
                        default=Path('results/evaluation/scalarization/dst/geometry/dst_points.csv'))
    parser.add_argument('--output', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Runs
    refs = load_references(args.geometry)
    runs = []
    points = []
    policies = []
    inventory = []
    for ec, env, method, seed in itertools.product(ENTROPIES, ENVS, METHODS, SEEDS):
        ident = dict(entropy_coef=ec, environment=env, scalarizer=method, K=standard_k(env), seed=seed)
        path = sidecar_path(args.paired_root, ident)
        r, p, a = analyze_run(path, ident, refs[env])
        runs.extend(r)
        points.extend(p)
        policies.extend(a)
        inventory.append(ident | dict(path=str(path.resolve()), size_bytes=path.stat().st_size))
        if len(inventory) % 120 == 0:
            print(f'E2: reconstructed {len(inventory)}/1440 runs', flush=True)

    # Manifest
    conventions = dict(
        archive='First occurrence after six-decimal duplicate removal; original E2 tolerance dominance. '
                'Membership uses greedy returns only; no truncation or sampled reselection.',
        identity='checkpoint + actor_row identify a saved paired record, not a verified cross-time actor ID.',
        heldout='Select by exact counts on episodes 1-50, ties use first collection record. '
                'No first-half hit gives zero. Evaluate on episodes 51-100.',
        treasure='Auxiliary treasure-only coverage ignores route length; its best actor may differ from exact '
                 'coverage.',
        diagnostics='Per-policy rates use all 100 episodes. Greedy-reproduction means condition on greedy-front '
                    'actors; empty groups are undefined. Checkpoint age is descriptive, not a cause or replacement '
                    'test.',
        hv='One greedy vector or sampled mean per policy; common exact-front reference; ratios are not clipped.')
    manifest = dict(schema=SCHEMA, status='complete', expected_runs=len(inventory), complete_runs=len(inventory),
                    protocol=dict(environments=ENVS, scalarizers=METHODS, entropy_coefs=ENTROPIES, seeds=SEEDS,
                                  h=H, sampled_episodes=EPISODES, heldout_split=HALF, coverage_q=QS,
                                  archive_cadence=1, archive_keep='first', learning_rate=LEARNING_RATE,
                                  stch_mu=STCH_MU, match_atol=MATCH_ATOL, hv_offset=HV_OFFSET),
                    paths={k: str(v.resolve()) for k, v in vars(args).items()}, source_runs=inventory,
                    references={env: {key: v.tolist() if isinstance(v, np.ndarray) else v for key, v in ref.items()}
                                for env, ref in refs.items()},
                    conventions=conventions)

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'e2_manifest.json').unlink(missing_ok=True)
    (args.output / 'e2_summary.csv').unlink(missing_ok=True)
    for name, data, keys in (('e2_runs.csv', runs, KEY),
                             ('e2_points.csv', points, RUN_KEY + ['h', 'collection', 'pf_id']),
                             ('e2_policy_diagnostics.csv', policies,
                              RUN_KEY + ['h', 'collection', 'checkpoint', 'actor_row'])):
        frame = pd.DataFrame(data).sort_values(keys, kind='stable')
        frame.to_csv(args.output / name, index=False, float_format='%.17g')
    (args.output / 'e2_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'E2: saved {len(runs)} collection/execution rows and diagnostics to {args.output}')


if __name__ == '__main__':
    main()