"""One E3 summary: seed statistics and allocation-minus-R=1 contrasts, in long format."""
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from reconstruct_e3 import DEFAULT_OUTPUT, ENVS, H, METRICS, SEEDS


def stats(values: np.ndarray) -> dict:
    """Mean, standard deviation and 95% seed-bootstrap interval of the finite seed values."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    result = dict(mean=float(x.mean()) if len(x) else np.nan,
                  std=float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                  ci_low=np.nan, ci_high=np.nan, n_seeds=len(x))
    if len(x) > 1:
        indices = np.random.default_rng(2027).integers(0, len(x), size=(2000, len(x)))
        lo, hi = np.quantile(x[indices].mean(1), [0.025, 0.975])
        result.update(ci_low=float(lo), ci_high=float(hi))
    return result


def emit(g: pd.DataFrame,
         identity: dict,
         level: str,
         kind: str) -> list[dict]:
    """One row per metric: seed means of the group with their counts and statistics."""
    rows = []
    for metric in METRICS:
        values = g[metric].astype(float)
        by_seed = pd.DataFrame(dict(seed=g.seed.to_numpy(), value=values.to_numpy()))
        grouped = by_seed.groupby('seed').value
        means = grouped.mean().reindex(SEEDS)
        defined = grouped.count().reindex(SEEDS, fill_value=0)
        rows.append(identity | dict(kind=kind, comparison='allocation - R1' if kind == 'comparison' else '',
                                    level=level, h=H, metric=metric, n_runs=len(g),
                                    n_values=int(np.isfinite(values).sum()), n_environments=g.environment.nunique(),
                                    n_environments_min=int(defined.min()), n_environments_max=int(defined.max()))
                    | stats(means.to_numpy()))
    return rows


def aggregate(frame: pd.DataFrame, kind: str) -> list[dict]:
    """Rows per condition and pooled over all environments; undefined conditional rates stay undefined."""
    group = ['budget', 'allocation', 'scalarizer']
    rows = []
    for key, g in frame.groupby(group + ['environment'], sort=True):
        identity = dict(zip(group + ['environment'], key))
        identity |= {n: int(g[n].iloc[0]) for n in ('P', 'K', 'R')}
        rows += emit(g, identity, 'condition', kind)
    for key, g in frame.groupby(group, sort=True):
        if set(g.environment) != set(ENVS):
            continue
        identity = dict(zip(group, key)) | dict(environment='ALL')
        identity |= {n: int(g[n].iloc[0]) if g[n].nunique() == 1 else np.nan for n in ('P', 'K', 'R')}
        rows += emit(g, identity, 'pooled', kind)
    return rows


def make_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Measures and allocation-minus-R=1 contrasts, paired by budget, environment, scalarizer and seed."""
    rows = aggregate(runs, 'measures')
    pair = ['budget', 'environment', 'scalarizer', 'seed']
    baseline = runs[runs.R == 1].set_index(pair)[list(METRICS)]
    different = runs[runs.R != 1].copy()
    paired_index = pd.MultiIndex.from_frame(different[pair])
    different[list(METRICS)] = (different[list(METRICS)].to_numpy(float)
                                - baseline.reindex(paired_index).to_numpy(float))
    rows += aggregate(different, 'comparison')
    return pd.DataFrame(rows).sort_values(['kind', 'budget', 'level', 'environment', 'scalarizer', 'allocation',
                                           'metric'])


def main() -> None:
    """Writes e3_summary.csv and records the summary conventions in the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Summary
    manifest = json.loads((args.input / 'e3_manifest.json').read_text())
    runs = pd.read_csv(args.input / 'e3_runs.csv')
    summary = make_summary(runs)

    # Output
    summary.to_csv(args.input / 'e3_summary.csv', index=False, float_format='%.17g')
    manifest['summary'] = dict(
        bootstrap=2000, confidence=0.95, bootstrap_seed=2027,
        pairing='Allocation minus R=1 within budget/environment/scalarizer/training seed.',
        pooling='Equal environment weights within seed, then mean/SD/bootstrap over training seeds. Budgets stay '
                'separate.',
        undefined='Conditional means omit undefined run values; n_values and environment-count ranges record their '
                  'scope.',
        units='Rates on [0,1], contrasts on [-1,1]. Raw scalarized-gap magnitudes are not aggregated.')
    (args.input / 'e3_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f"E3: {len(summary)} summary rows. Output: {args.input / 'e3_summary.csv'}")


if __name__ == '__main__':
    main()