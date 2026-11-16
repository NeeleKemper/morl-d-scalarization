"""E4: one long-format summary of curves, paired rate effects and descriptive timing. Archive and active curves use
every iteration, other metrics h = 25, 50, 75, 100. Averaging precedes maximization: the timing rows never average
per-run maxima.
"""
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from reconstruct_e4 import DATA_DIR, DEFAULT_LR, ENVS, H, METRICS, RATES, SEEDS


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
    """Statistics per scalarizer, learning rate, environment (and all environments), iteration and metric."""
    columns = pd.MultiIndex.from_product([METRICS, SEEDS])
    names = pd.DataFrame([(h, m) for h in range(1, H + 1) for m in METRICS], columns=['h', 'metric'])
    keep = names.metric.isin(('archive_recall', 'active_recall')) | names.h.isin((25, 50, 75, 100))
    names = names[keep].reset_index(drop=True)
    blocks = []
    for (method, lr), group in frame.groupby(['scalarizer', 'learning_rate'], sort=True):
        for env in (*ENVS, 'ALL'):
            block = group if env == 'ALL' else group[group.environment == env]
            per_seed = block.groupby(['h', 'seed'])[list(METRICS)]
            values = per_seed.mean().unstack('seed').reindex(index=range(1, H + 1), columns=columns)
            defined = per_seed.count().unstack('seed').reindex(index=range(1, H + 1), columns=columns)
            values = values.to_numpy(float).reshape(-1, len(SEEDS))[keep]
            defined = defined.to_numpy(float).reshape(-1, len(SEEDS))[keep]
            finite = np.isfinite(values)
            result = names.copy().assign(**stats_many(values), kind=kind, scalarizer=method, learning_rate=lr,
                                         level='pooled' if env == 'ALL' else 'condition', environment=env,
                                         comparison='minus_default' if kind == 'comparison' else '',
                                         baseline_learning_rate=DEFAULT_LR if kind == 'comparison' else np.nan,
                                         reference_h=np.nan, n_runs=block.groupby('h').size().iloc[0],
                                         n_defined_runs=defined.sum(1).astype(int))
            result['min_defined_environments'] = np.where(finite, defined, np.inf).min(1)
            result['max_defined_environments'] = np.where(finite, defined, 0).max(1)
            result.loc[result.n_seeds == 0, 'min_defined_environments'] = 0
            blocks.append(result)
    return pd.concat(blocks, ignore_index=True)


def rate_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    """Every learning rate minus the default rate, paired by environment, scalarizer, iteration and seed."""
    key = ['environment', 'scalarizer', 'h', 'seed']
    default = runs[runs.learning_rate == DEFAULT_LR].set_index(key)[list(METRICS)].astype(float).sort_index()
    blocks = []
    for lr in RATES:
        if lr == DEFAULT_LR:
            continue
        current = runs[runs.learning_rate == lr].set_index(key)[list(METRICS)].astype(float).sort_index()
        blocks.append((current - default).reset_index().assign(learning_rate=lr))
    return pd.concat(blocks, ignore_index=True)


def timing_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Observed mean-curve maxima and endpoint-minus-default-peak; descriptive, not stopping-rule guarantees."""
    curves = summary[(summary.kind == 'measures') & (summary.metric == 'active_recall')]
    rows = []
    for (env, method), group in curves.groupby(['environment', 'scalarizer'], sort=True):
        base = group[group.learning_rate == DEFAULT_LR].sort_values('h')
        baseline_peak = base.loc[base['mean'].idxmax()]
        for lr, curve in group.groupby('learning_rate', sort=True):
            curve = curve.sort_values('h')
            peak = curve.loc[curve['mean'].idxmax()]
            final = curve[curve.h == H].iloc[0]
            common = dict(kind='timing', comparison='descriptive', environment=env,
                          level='pooled' if env == 'ALL' else 'condition', scalarizer=method, learning_rate=lr,
                          n_seeds=int(final.n_seeds), n_runs=int(final.n_runs), std=np.nan, ci_low=np.nan,
                          ci_high=np.nan)
            rows.append(common | dict(metric='active_peak', mean=float(peak['mean']), h=int(peak.h),
                                      reference_h=np.nan, baseline_learning_rate=np.nan))
            rows.append(common | dict(metric='endpoint_minus_default_peak',
                                      mean=float(final['mean'] - baseline_peak['mean']), h=H,
                                      reference_h=int(baseline_peak.h), baseline_learning_rate=DEFAULT_LR))
    return pd.DataFrame(rows)


def make_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Measures, paired rate contrasts and timing rows in long format."""
    measures = aggregate(runs, 'measures')
    return pd.concat([measures, aggregate(rate_contrasts(runs), 'comparison'), timing_rows(measures)],
                     ignore_index=True)


def main() -> None:
    """Writes e4_summary.csv and records the summary conventions in the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Summary
    manifest = json.loads((args.input / 'e4_manifest.json').read_text())
    runs = pd.read_csv(args.input / 'e4_runs.csv', float_precision='round_trip', low_memory=False)
    summary = make_summary(runs)

    # Output
    summary.to_csv(args.input / 'e4_summary.csv', index=False, float_format='%.17g')
    manifest['summary'] = dict(
        bootstrap=2000, confidence=0.95, bootstrap_seed=2027,
        curves=('archive_recall', 'active_recall'), diagnostic_cutoffs=(25, 50, 75, 100),
        convention='Differences are paired within environment and seed before equal-environment seed averaging. '
                   'Undefined conditional rates are omitted; denominators are explicit. Intervals are pointwise, '
                   'conditional on fixed environments/configurations/references, not unseen-task uncertainty.',
        timing='Maximize the observed environment/seed-mean active curve over h=1..100; earliest maximizer wins. '
               'Endpoint-minus-default-peak uses that same pooled-curve definition, not per-run maxima. '
               'Timing rows are descriptive, without bootstrap intervals or a validated stopping rule.')
    (args.input / 'e4_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f"E4: wrote {len(summary):,} rows to {args.input / 'e4_summary.csv'}")


if __name__ == '__main__':
    main()