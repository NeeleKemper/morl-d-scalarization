"""One tidy E2 summary: within-run metrics and paired contrasts, then equal-environment seed means."""
import json
import argparse

import numpy as np
import pandas as pd

from typing import Any
from pathlib import Path

from reconstruct_e2 import DATA_DIR, DEFAULT_ENTROPY, ENVS, ENTROPIES, METHODS, METRICS, PRIMARY, SEEDS, read_json


# Contrasts

def difference(a: pd.DataFrame, b: pd.DataFrame, **labels: Any) -> pd.DataFrame:
    """Paired difference a - b with its labels as columns."""
    return (a - b).reset_index().assign(**labels)


def contrasts(runs: pd.DataFrame, case: list[str]) -> pd.DataFrame:
    """Paired retention, execution, method, interaction and entropy contrasts per environment and seed."""
    r = runs.set_index(case + ['environment', 'seed'])[list(PRIMARY)].sort_index()
    effect = r.xs('sampled', level='execution') - r.xs('deterministic', level='execution')
    parts = [difference(r.xs('archive', level='collection'), r.xs('population', level='collection'),
                        comparison='retention', collection='archive - population', baseline=''),
             effect.reset_index().assign(comparison='execution', execution='sampled - deterministic', baseline='')]
    for baseline in ('ws_norm', 'ws_raw'):
        for method in METHODS:
            if method == baseline:
                continue
            parts.append(difference(r.xs(method, level='scalarizer'), r.xs(baseline, level='scalarizer'),
                                    comparison='method', scalarizer=method, baseline=baseline))
            parts.append(difference(effect.xs(method, level='scalarizer'), effect.xs(baseline, level='scalarizer'),
                                    comparison='interaction', scalarizer=method, baseline=baseline,
                                    execution='sampled - deterministic'))
    default = r.xs(DEFAULT_ENTROPY, level='entropy_coef')
    for entropy in ENTROPIES:
        if entropy != DEFAULT_ENTROPY:
            parts.append(difference(r.xs(entropy, level='entropy_coef'), default, comparison='entropy',
                                    entropy_coef=entropy, baseline=f'entropy={DEFAULT_ENTROPY:g}'))
    return pd.concat(parts, ignore_index=True)


# Aggregation

def aggregate(frame: pd.DataFrame,
              grouping: list[str],
              metrics: tuple[str, ...],
              kind: str) -> list[dict]:
    """Equal-environment seed means with seed-bootstrap intervals; undefined conditional rates are omitted."""
    rows = []
    draws = {n: np.random.default_rng(2027).integers(0, n, (2000, n)) for n in range(2, 11)}
    for key, group in frame.groupby(grouping, sort=True, dropna=False):
        identity = dict(zip(grouping, key if isinstance(key, tuple) else (key,)))
        for environment, block in [(env, group[group.environment.eq(env)]) for env in ENVS] + [('ALL', group)]:
            for metric in metrics:
                by_seed = block.groupby('seed')[metric].agg(['mean', 'count']).reindex(SEEDS)
                valid = by_seed[by_seed['mean'].notna()]
                values = valid['mean'].to_numpy(float)
                n = len(values)
                mean = float(values.mean()) if n else np.nan
                std = float(values.std(ddof=1)) if n > 1 else np.nan
                if n > 1:
                    lo, hi = np.quantile(values[draws[n]].mean(1), [(1 - 0.95) / 2, (1 + 0.95) / 2])
                else:
                    lo = np.nan
                    hi = np.nan
                rows.append(dict(kind=kind, comparison='', baseline='') | identity |
                            dict(level='pooled' if environment == 'ALL' else 'condition', environment=environment,
                                 metric=metric, mean=mean, std=std, ci_low=lo, ci_high=hi, n_seeds=n,
                                 n_runs=len(block), n_defined_runs=int(block[metric].notna().sum()),
                                 min_defined_environments=int(valid['count'].min()) if n else 0,
                                 max_defined_environments=int(valid['count'].max()) if n else 0))
    return rows


def make_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Measures and paired contrasts of every case in long format."""
    case = ['entropy_coef', 'scalarizer', 'collection', 'execution', 'h']
    rows = aggregate(runs, case, METRICS, 'measures')
    rows += aggregate(contrasts(runs, case), ['comparison', 'baseline'] + case, PRIMARY, 'comparison')
    return pd.DataFrame(rows)


def main() -> None:
    """Writes e2_summary.csv and records the summary conventions in the manifest."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Summary
    manifest = read_json(args.input / 'e2_manifest.json')
    runs = pd.read_csv(args.input / 'e2_runs.csv', float_precision='round_trip')
    summary = make_summary(runs)

    # Output
    summary.to_csv(args.input / 'e2_summary.csv', index=False, float_format='%.17g')
    manifest['summary'] = dict(
        bootstrap=2000, confidence=0.95, bootstrap_seed=2027,
        convention='Contrasts are paired within environment and training seed. Environments are averaged equally '
                   'within seed before bootstrap resampling. Conditional rates omit undefined environment values and '
                   'report their effective denominators. Intervals condition on fixed tasks, references, and saved '
                   'episode evaluations; they are not outcome-probability guarantees.',
        layout='Long format; all reported metrics and paired contrasts are in e2_summary.csv.')
    (args.input / 'e2_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f"E2: wrote {len(summary):,} summary rows to {args.input / 'e2_summary.csv'}")


if __name__ == '__main__':
    main()