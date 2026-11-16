"""Supported and unsupported parts of each reference front at stated resolutions, which methods reach them, and the
supported and unsupported returns of every run's own front.

The weighted-sum gap of a point z of a front Z (objectives maximized, in units of their ranges on the reference front)
is min over weights w of max over q in Z of w.(q - z); it is 0 exactly for supported points. The reference analysis
uses the reference front as Z (exact on DST/LDST, constructed otherwise), the own-front analysis the non-dominated set
of one run. At every resolution eps, a front is thinned to an eps-net in the max-norm (greedily by increasing gap); a
representative is unsupported if its gap exceeds eps, reached if some return is at least it minus eps in every
objective, and on the reference front if no reference point exceeds it by more than eps in every objective. eps = 0 is
the exact analysis.

Outputs (in --output): supported_points.csv, supported_representatives.csv, supported_runs.csv,
supported_summary.csv, supported_own_points.csv, supported_own_runs.csv and supported_own_summary.csv.

  python supported_share.py
  python supported_share.py --envs reacher --eps 0 0.01 0.02 0.05
"""
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from build_reference_fronts import nondominated
from compute_metrics_baseline import constructed_reference, exact_reference


# Weighted-sum gap

def gap_lp(p: np.ndarray, rivals: np.ndarray) -> float:
    """min over w >= 0, sum(w) = 1, of max over the rivals q of w.(q - p), clipped at 0."""
    d = len(p)
    res = linprog(np.r_[np.zeros(d), 1.0], A_ub=np.hstack([rivals - p, -np.ones((len(rivals), 1))]),
                  b_ub=np.zeros(len(rivals)), A_eq=np.r_[np.ones(d), 0.0][None], b_eq=[1.0],
                  bounds=[(0, None)] * d + [(None, None)], method='highs')
    return max(res.fun, 0.0)


def gaps(x: np.ndarray) -> np.ndarray:
    """Weighted-sum gap of every point of a non-dominated set: exit of the ray z + t(1, ..., 1) from the hull of x and
    its lowered copies where that facet has a nonnegative normal, otherwise a linear program."""
    n, d = x.shape
    if n == 0:
        return np.zeros(0)
    lowered = np.repeat(x[None], d, 0)
    lowered[np.arange(d), :, np.arange(d)] = x.min(0)[:, None] - 1.0
    try:
        hull = ConvexHull(np.vstack([x, lowered.reshape(-1, d)]), qhull_options='Qc')
    except (RuntimeError, ValueError):
        return np.array([gap_lp(p, x) for p in x])
    on_hull = np.isin(np.arange(n), np.union1d(hull.vertices, hull.coplanar[:, 0]))
    normal = hull.equations[:, :d]
    offset = hull.equations[:, d]
    rising = normal.sum(1) > 1e-12
    upper = (normal[rising] >= -1e-12).all(1)
    normal = normal[rising]
    offset = offset[rising]
    gap = np.where(on_hull, 0.0, np.nan)
    inner = np.flatnonzero(~on_hull)
    rows = max(1, 4_000_000 // max(1, len(normal)))
    for i in range(0, len(inner), rows):
        idx = inner[i:i + rows]
        t = -(offset + x[idx] @ normal.T) / normal.sum(1)
        exit_t = t.min(1)
        exact = (upper & (t <= exit_t[:, None] + 1e-12)).any(1)
        gap[idx[exact]] = exit_t[exact]
    for i in np.flatnonzero(np.isnan(gap)):
        gap[i] = gap_lp(x[i], x[on_hull])
    return np.maximum(gap, 0.0)


# Resolution

def thin(x: np.ndarray, eps: float, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy eps-net of x in the max-norm in the given order: sorted representatives and the owner of every point."""
    if eps <= 0 or len(x) == 0:
        return np.arange(len(x)), np.arange(len(x))
    kept = np.empty_like(x)
    reps = []
    owner = np.empty(len(x), int)
    for i in order:
        distance = np.abs(kept[:len(reps)] - x[i]).max(1)
        if len(reps) and distance.min() <= eps:
            owner[i] = reps[int(distance.argmin())]
        else:
            kept[len(reps)] = x[i]
            owner[i] = i
            reps.append(int(i))
    return np.sort(reps), owner


def reached(reps: np.ndarray, points: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Per representative: some point is at least the representative minus tau in every objective."""
    out = np.zeros(len(reps), bool)
    if len(points):
        rows = max(1, 4_000_000 // (len(points) * reps.shape[1]))
        for i in range(0, len(reps), rows):
            out[i:i + rows] = (points[None] >= reps[i:i + rows, None] - tau).all(2).any(1)
    return out


def below_reference(points: np.ndarray, x: np.ndarray, slack: np.ndarray) -> np.ndarray:
    """Per point: the largest margin by which a reference point exceeds it in every objective beyond the slack."""
    out = np.full(len(points), -np.inf)
    if len(points) and len(x):
        rows = max(1, 4_000_000 // (len(x) * x.shape[1]))
        for i in range(0, len(points), rows):
            out[i:i + rows] = (x[None] - points[i:i + rows, None] - slack).min(2).max(1)
    return out


# Statistics

def recalls(hit: np.ndarray, unsupported: np.ndarray) -> dict:
    """Reached representatives and recall of the supported and the unsupported ones."""
    out = {}
    for group, members in zip(('supported', 'unsupported'), (~unsupported, unsupported)):
        out[f'{group}_reached'] = int(hit[members].sum())
        out[f'{group}_recall'] = hit[members].mean() if members.any() else np.nan
    return out


def interval(values: pd.Series) -> tuple[float, float]:
    """95% percentile seed-bootstrap interval of the mean."""
    v = values.dropna().to_numpy(float)
    if len(v) < 2:
        return np.nan, np.nan
    means = v[np.random.default_rng(0).integers(len(v), size=(2000, len(v)))].mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def own_fronts(env: str,
               grid: list[float],
               finals: list[tuple],
               labels: list[str],
               x: np.ndarray,
               lo: np.ndarray,
               span: np.ndarray) -> list[pd.DataFrame]:
    """Supported and unsupported returns of every run's own front: per point, per eps and run, per eps and method."""
    point_tables = []
    run_rows = []
    for label, seed, xr in finals:
        gap = gaps(xr)
        below = below_reference(xr, x, 1e-5 / span)
        table = pd.DataFrame(xr * span + lo, columns=[f'objective_{i}' for i in range(x.shape[1])])
        for position, (name, value) in enumerate((('environment', env), ('label', label), ('seed', seed),
                                                  ('point', np.arange(len(xr))))):
            table.insert(position, name, value)
        table['own_gap'] = gap
        table['below_reference'] = below
        point_tables.append(table)
        order = np.argsort(gap, kind='stable')
        for eps in grid:
            reps, _ = thin(xr, eps, order)
            unsupported = gap[reps] > max(eps, 1e-9)
            run_rows.append(dict(
                environment=env, eps=eps, label=label, seed=seed, own_points=len(xr), own_representatives=len(reps),
                own_unsupported=int(unsupported.sum()),
                own_unsupported_share=unsupported.mean() if len(reps) else np.nan,
                own_unsupported_on_reference=int((unsupported & (below[reps] <= eps)).sum()),
                own_gap_max=gap.max() if len(gap) else np.nan))
    runs = pd.DataFrame(run_rows)
    summary_rows = []
    for eps in grid:
        for label in labels:
            r = runs[runs.eps.eq(eps) & runs.label.eq(label)]
            low, high = interval(r.own_unsupported_share)
            total = int(r.own_unsupported.sum())
            summary_rows.append(dict(
                environment=env, eps=eps, label=label, runs=len(r),
                runs_with_unsupported=int((r.own_unsupported > 0).sum()),
                own_unsupported=r.own_unsupported.mean(), own_unsupported_share=r.own_unsupported_share.mean(),
                own_unsupported_share_low=low, own_unsupported_share_high=high,
                runs_with_unsupported_on_reference=int((r.own_unsupported_on_reference > 0).sum()),
                on_reference_share=r.own_unsupported_on_reference.sum() / total if total else np.nan,
                own_gap_max=r.own_gap_max.max()))
    return [pd.concat(point_tables, ignore_index=True), runs, pd.DataFrame(summary_rows)]


def analyse(args: argparse.Namespace, env: str) -> list[pd.DataFrame]:
    """All seven tables of one environment."""
    if env.startswith(('dst', 'ldst')):
        reference, runs = exact_reference(args, env)
    else:
        reference, runs = constructed_reference(args, env)
    front = reference['front']
    lo = front.min(0)
    span = np.ptp(front, 0)
    span = np.where(span > 0, span, 1.0)
    x = (front - lo) / span
    gap = gaps(x)
    finals = [(r['label'], r['seed'], (nondominated(np.round(r['front'], 6) + 0.0) - lo) / span
               if len(r['front']) else x[:0]) for r in runs]
    labels = list(dict.fromkeys(label for label, _, _ in finals))
    points = pd.DataFrame(front, columns=[f'objective_{i}' for i in range(front.shape[1])])
    points.insert(0, 'point', np.arange(len(front)))
    points.insert(0, 'environment', env)
    points['gap'] = gap
    points['supported'] = gap <= 1e-9
    reps_tables = []
    runs_tables = []
    summary_rows = []
    for eps in args.eps:
        reps, owner = thin(x, eps, np.argsort(gap, kind='stable'))
        unsupported = gap[reps] > max(eps, 1e-9)
        hits = [reached(x[reps], xr, eps + 1e-5 / span) for _, _, xr in finals]
        pooled = {label: np.any([h for (l, _, _), h in zip(finals, hits) if l == label], axis=0) for label in labels}
        methods = np.sum(list(pooled.values()), axis=0)
        reps_tables.append(pd.DataFrame(dict(
            environment=env, eps=eps, point=reps, gap=gap[reps], unsupported=unsupported,
            members=np.bincount(owner, minlength=len(x))[reps], runs=np.sum(hits, axis=0),
            methods=['; '.join(l for l in labels if pooled[l][j]) for j in range(len(reps))])))
        table = pd.DataFrame([dict(environment=env, eps=eps, label=label, seed=seed) | recalls(h, unsupported)
                              for (label, seed, _), h in zip(finals, hits)])
        runs_tables.append(table)
        for label, r in table.groupby('label', sort=False):
            row = dict(environment=env, eps=eps, label=label, runs=len(r),
                       n_supported=int((~unsupported).sum()), n_unsupported=int(unsupported.sum()))
            for group, members in zip(('supported', 'unsupported'), (~unsupported, unsupported)):
                low, high = interval(r[f'{group}_recall'])
                row |= {f'{group}_recall': r[f'{group}_recall'].mean(), f'{group}_low': low, f'{group}_high': high,
                        f'{group}_pooled': pooled[label][members].mean() if members.any() else np.nan}
            only = pooled[label] & (methods == 1)
            row['unsupported_only'] = only[unsupported].mean() if unsupported.any() else np.nan
            summary_rows.append(row)
    return [points, pd.concat(reps_tables), pd.concat(runs_tables), pd.DataFrame(summary_rows),
            *own_fronts(env, args.eps, finals, labels, x, lo, span)]


# Report

def show(value: float) -> str:
    """Number with three decimals, or a dash."""
    return f'{value:.3f}' if np.isfinite(value) else '--'


def cell(row: tuple, group: str) -> str:
    """Mean recall of a group with its interval, or a dash."""
    mean = getattr(row, f'{group}_recall')
    low = getattr(row, f'{group}_low')
    high = getattr(row, f'{group}_high')
    return f'{mean:.3f} [{low:.3f}, {high:.3f}]' if np.isfinite(mean) else '--'


def report(points: pd.DataFrame,
           reps: pd.DataFrame,
           summary: pd.DataFrame,
           own: pd.DataFrame,
           grid: list[float]) -> None:
    """Gap shares, recall per method and own-front counts per environment and resolution."""
    for env, s in summary.groupby('environment', sort=False):
        gap = points.loc[points.environment.eq(env), 'gap'].to_numpy()
        shares = ', '.join(f'> {t:g}: {(gap > max(t, 1e-9)).mean():.1%}' for t in sorted({0.0, *grid}))
        print(f'\n{env}: {len(gap)} points on the reference front; weighted-sum gap {shares}')
        for eps, e in s.groupby('eps', sort=False):
            r = reps[reps.environment.eq(env) & reps.eps.eq(eps)]
            print(f'  eps {eps:g}: {len(r)} representatives, {r.unsupported.sum()} unsupported; '
                  f'the largest stands for {r.members.max()} points')
            print(f"    {'':18s} {'runs':>4s}  {'unsupported: recall [95% CI]':28s} {'pooled':>6s} {'only':>6s}"
                  f"  {'supported: recall [95% CI]':28s} {'pooled':>6s}")
            for m in e.itertuples():
                print(f"    {m.label:18s} {m.runs:4d}  {cell(m, 'unsupported'):28s} {show(m.unsupported_pooled):>6s} "
                      f"{show(m.unsupported_only):>6s}  {cell(m, 'supported'):28s} {show(m.supported_pooled):>6s}")
            print(f"    {'own fronts':18s} {'runs':>4s}  {'with unsupported':>16s}  {'per run':>7s}  "
                  f"{'share per run [95% CI]':24s} {'runs on ref.':>12s} {'on ref.':>7s} {'max gap':>7s}")
            for m in own[own.environment.eq(env) & own.eps.eq(eps)].itertuples():
                share = (f'{m.own_unsupported_share:.3f} [{m.own_unsupported_share_low:.3f}, '
                         f'{m.own_unsupported_share_high:.3f}]' if np.isfinite(m.own_unsupported_share) else '--')
                print(f'    {m.label:18s} {m.runs:4d}  {m.runs_with_unsupported:16d}  {m.own_unsupported:7.1f}  '
                      f'{share:24s} {m.runs_with_unsupported_on_reference:12d} {show(m.on_reference_share):>7s} '
                      f'{show(m.own_gap_max):>7s}')


# Main

def parse_args() -> argparse.Namespace:
    """Command line arguments; 'dst' and 'mujoco' in --envs expand to their variants."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--envs', nargs='+', default=['dst', 'minecart', 'reacher', 'mujoco'],
                        help="Default: all environments; 'dst' and 'mujoco' stand for their variants")
    parser.add_argument('--eps', type=float, nargs='+', default=[0.0, 0.005, 0.01, 0.02],
                        help="Resolutions as fractions of each objective's range on the reference front")
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
    parser.add_argument('--output', type=Path, default=Path('results/evaluation/scalarization/supported'))
    args = parser.parse_args()
    groups = dict(dst=('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2'),
                  mujoco=('ant', 'halfcheetah', 'hopper', 'humanoid'))
    args.envs = [env for name in args.envs for env in groups.get(name, (name,))]
    args.calibration = False
    return args


def main() -> None:
    """Writes the seven tables of every environment and prints the report."""
    # Arguments
    args = parse_args()

    # Analysis
    tables = [pd.concat(t, ignore_index=True) for t in zip(*(analyse(args, env) for env in args.envs))]

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    names = ('points', 'representatives', 'runs', 'summary', 'own_points', 'own_runs', 'own_summary')
    for table, name in zip(tables, names):
        table.to_csv(args.output / f'supported_{name}.csv', index=False, float_format='%.10g')
    report(tables[0], tables[1], tables[3], tables[6], args.eps)
    print(f'\nOutput: {args.output}')


if __name__ == '__main__':
    main()