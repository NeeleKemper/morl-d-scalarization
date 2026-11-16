"""E5: dense recall curves, paired h-minus-100 changes and descriptive mean-curve peaks. Only an input path is
configurable; environments are averaged within seed first.
"""
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from reconstruct_e5 import DATA_DIR, ENVS, H, REFERENCE_H, METRICS, SEEDS


def stats_many(values: np.ndarray) -> dict[str, np.ndarray]:
    """Mean, standard deviation and 95% seed-bootstrap interval per row (metric) over the finite seed columns."""
    finite = np.isfinite(values)
    n = finite.sum(1)
    mean = np.divide(np.where(finite, values, 0).sum(1), n, out=np.full(len(n), np.nan), where=n > 0)
    ss = np.where(finite, values - mean[:, None], 0) ** 2
    std = np.sqrt(np.divide(ss.sum(1), n - 1, out=np.full(len(n), np.nan), where=n > 1))
    low = np.full(len(n), np.nan)
    high = np.full(len(n), np.nan)
    for size in np.unique(n[n > 1]):
        selected = n == size
        compact = values[selected][finite[selected]].reshape(-1, size)
        indices = np.random.default_rng(2027).integers(0, size, (2000, size))
        counts = np.stack([(indices == i).sum(1) for i in range(size)], axis=1) / size
        boot = compact @ counts.T
        low[selected], high[selected] = np.quantile(boot, [(1 - 0.95) / 2, (1 + 0.95) / 2], axis=1)
    return dict(mean=mean, std=std, ci_low=low, ci_high=high, n_seeds=n)


def aggregate(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Statistics per scalarizer, environment (and all environments), iteration and metric."""
    horizons = sorted(frame.h.unique())
    columns = pd.MultiIndex.from_product([METRICS, SEEDS])
    names = pd.DataFrame([(h, m) for h in horizons for m in METRICS], columns=['h', 'metric'])
    keep = names.metric.isin(('archive_recall', 'active_recall')) | names.h.isin((100, 200, 300, 400, 500))
    names = names[keep].reset_index(drop=True)
    blocks = []
    for method, group in frame.groupby('scalarizer', sort=True):
        for env in (*ENVS, 'ALL'):
            block = group if env == 'ALL' else group[group.environment == env]
            per_seed = block.groupby(['h', 'seed'])[list(METRICS)]
            values = per_seed.mean().unstack('seed').reindex(index=horizons, columns=columns)
            defined = per_seed.count().unstack('seed').reindex(index=horizons, columns=columns)
            values = values.to_numpy(float).reshape(-1, len(SEEDS))[keep]
            defined = defined.to_numpy(float).reshape(-1, len(SEEDS))[keep]
            finite = np.isfinite(values)
            result = names.copy().assign(**stats_many(values), kind=kind, scalarizer=method,
                                         level='pooled' if env == 'ALL' else 'condition', environment=env,
                                         comparison='minus_h100' if kind == 'comparison' else '',
                                         reference_h=REFERENCE_H if kind == 'comparison' else np.nan,
                                         n_runs=block.groupby('h').size().iloc[0],
                                         n_defined_runs=defined.sum(1).astype(int))
            result['min_defined_environments'] = np.where(finite, defined, np.inf).min(1)
            result['max_defined_environments'] = np.where(finite, defined, 0).max(1)
            result.loc[result.n_seeds == 0, 'min_defined_environments'] = 0
            blocks.append(result)
    return pd.concat(blocks, ignore_index=True)


def horizon_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    """Every reported iteration minus h=100 of the same H=500 run, never the separate E1 CSV."""
    key = ['environment', 'scalarizer', 'seed']
    base = runs[runs.h == REFERENCE_H].set_index(key)[list(METRICS)].astype(float).sort_index()
    blocks = []
    for h in (100, 200, 300, 400, 500):
        if h == REFERENCE_H:
            continue
        current = runs[runs.h == h].set_index(key)[list(METRICS)].astype(float).sort_index()
        blocks.append((current - base).reset_index().assign(h=h))
    return pd.concat(blocks, ignore_index=True)


def timing_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Peaks of the observed mean curves within h <= 100 and overall, and the endpoint minus the early peak."""
    curves = summary[(summary.kind == 'measures') & (summary.metric == 'active_recall')]
    rows = []
    for (env, method), curve in curves.groupby(['environment', 'scalarizer'], sort=True):
        curve = curve.sort_values('h')
        early = curve[curve.h <= REFERENCE_H]
        early_peak = early.loc[early['mean'].idxmax()]
        peak = curve.loc[curve['mean'].idxmax()]
        endpoint = curve.iloc[-1]
        common = dict(kind='timing', comparison='descriptive', level='pooled' if env == 'ALL' else 'condition',
                      environment=env, scalarizer=method, n_seeds=int(endpoint.n_seeds), n_runs=int(endpoint.n_runs),
                      std=np.nan, ci_low=np.nan, ci_high=np.nan)
        for name, row in (('active_peak_first100', early_peak), ('active_peak', peak)):
            rows.append(common | dict(metric=name, mean=float(row['mean']), h=int(row.h), reference_h=np.nan))
        rows.append(common | dict(metric='endpoint_minus_early_peak',
                                  mean=float(endpoint['mean'] - early_peak['mean']), h=H,
                                  reference_h=int(early_peak.h)))
    return pd.DataFrame(rows)


def make_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Measures, paired horizon contrasts and timing rows in long format."""
    measures = aggregate(runs, 'measures')
    return pd.concat([measures, aggregate(horizon_contrasts(runs), 'comparison'), timing_rows(measures)],
                     ignore_index=True)


def main() -> None:
    """Writes e5_summary.csv and records the summary conventions in the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Summary
    manifest = json.loads((args.input / 'e5_manifest.json').read_text())
    runs = pd.read_csv(args.input / 'e5_runs.csv', float_precision='round_trip', low_memory=False)
    summary = make_summary(runs)

    # Output
    summary.to_csv(args.input / 'e5_summary.csv', index=False, float_format='%.17g')
    manifest['summary'] = dict(
        bootstrap=2000, confidence=0.95, bootstrap_seed=2027,
        curves=('archive_recall', 'active_recall'), diagnostic_cutoffs=(100, 200, 300, 400, 500),
        convention='Horizon differences are paired within the same long run before equal-environment seed '
                   'averaging. Undefined conditional rates are omitted only for the corresponding metric; pairwise '
                   'differences require both values. Intervals are pointwise, conditional on the fixed references '
                   'and tasks.',
        timing='Mean active curves are maximized over h=1..100 and h=1..500; earliest maximizer wins. '
               'No per-run maximum is averaged. Timing rows are descriptive, without confidence intervals. '
               'They do not validate early stopping or establish asymptotic convergence.')
    (args.input / 'e5_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f"E5: wrote {len(summary):,} long-format rows to {args.input / 'e5_summary.csv'}")


if __name__ == '__main__':
    main()