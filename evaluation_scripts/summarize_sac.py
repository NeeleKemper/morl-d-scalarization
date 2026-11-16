"""Summarize sac_runs.csv into sac_summary.csv, one row per (kind, environment, algorithm, scalarizer, metric).

measures: per algorithm and scalarizer, for each environment and pooled as ALL (environments averaged with equal
weight within seed, then the seeds). comparison: SAC minus PPO for the same environment, scalarizer and seed index,
and ws_norm minus ws_raw for the same environment, algorithm and seed index. Statistics: mean, standard deviation and
95% percentile seed-bootstrap interval (2,000 replicates, seed 2027, a fresh generator per statistic).
"""
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from reconstruct_sac import DATA_DIR, METRICS


def stats(x: pd.Series | np.ndarray) -> dict:
    """Mean, standard deviation and 95% seed-bootstrap interval of the finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = dict(mean=x.mean() if len(x) else np.nan, std=x.std(ddof=1) if len(x) > 1 else np.nan, n_seeds=len(x),
               ci_low=np.nan, ci_high=np.nan)
    if len(x) > 1:
        means = x[np.random.default_rng(2027).integers(0, len(x), size=(2000, len(x)))].mean(1)
        out['ci_low'], out['ci_high'] = np.quantile(means, [(1 - 0.95) / 2, (1 + 0.95) / 2])
    return out


def aggregate(frame: pd.DataFrame, kind: str) -> list[dict]:
    """Rows per environment and pooled over the seeds that cover every environment."""
    rows = []
    n_envs = frame.environment.nunique()
    for (algorithm, method), g in frame.groupby(['algorithm', 'scalarizer'], sort=True):
        envs_per_seed = g.groupby('seed').environment.nunique()
        complete = g[g.seed.isin(envs_per_seed.index[envs_per_seed == n_envs])]
        values = [(env, e.sort_values('seed'), 1) for env, e in g.groupby('environment', sort=True)]
        values.append(('ALL', complete.groupby('seed')[list(METRICS)].mean(), n_envs))
        rows += [dict(kind=kind, environment=env, algorithm=algorithm, scalarizer=method, metric=m, **stats(v[m]),
                      n_environments=n) for env, v, n in values for m in METRICS]
    return rows


def difference(runs: pd.DataFrame,
               factor: str,
               minuend: str,
               subtrahend: str) -> pd.DataFrame:
    """minuend minus subtrahend of factor, paired by environment, the other factor and seed index."""
    key = ['environment', 'scalarizer' if factor == 'algorithm' else 'algorithm', 'seed']
    first = runs[runs[factor] == minuend].set_index(key)[list(METRICS)]
    second = runs[runs[factor] == subtrahend].set_index(key)[list(METRICS)]
    return (first - second).reset_index().assign(**{factor: f'{minuend} - {subtrahend}'})


def main() -> None:
    """Writes sac_summary.csv and records the summary settings in the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Summary
    manifest = json.loads((args.input / 'sac_manifest.json').read_text())
    runs = pd.read_csv(args.input / 'sac_runs.csv', low_memory=False)
    runs['complete'] = runs.complete.astype(float)
    rows = aggregate(runs, 'measures')
    rows += aggregate(difference(runs, 'algorithm', 'SAC', 'PPO'), 'comparison')
    rows += aggregate(difference(runs, 'scalarizer', 'ws_norm', 'ws_raw'), 'comparison')
    summary = pd.DataFrame(rows)

    # Output
    summary.to_csv(args.input / 'sac_summary.csv', index=False, float_format='%.17g')
    manifest['summary'] = dict(rows=len(summary), bootstrap=2000, confidence=0.95, seed=2027)
    (args.input / 'sac_manifest.json').write_text(json.dumps(manifest, indent=2, default=float) + '\n',
                                                  encoding='utf-8')
    print(f"SAC vs PPO summary: {len(summary):,} rows in {args.input / 'sac_summary.csv'}")


if __name__ == '__main__':
    main()