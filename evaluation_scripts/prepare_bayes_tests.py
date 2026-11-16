"""Inputs for the Bayesian t-tests of raw WS against the other scalarizers, and a check whether seeds pair the runs.

Compares raw WS with each other scalarizer of the same learner (MO-PPO: WS norm, TCH and STCH raw/norm; MO-SAC: WS
norm) on one group of environments (--group additional or exact). Per comparison the check reports Pearson and
Spearman correlations across seeds and the variance ratio var(raw WS - alternative) / (var(raw WS) + var(alternative));
per learner, the mean Pearson correlation is tested against shuffled raw-WS seed labels. The paired test is recommended
if the mean correlation is positive with p < 0.05, otherwise the independent test.

Outputs (in --output): bayes_pairs.csv (y1 raw WS, y2 alternative, y1 - y2 > 0 means raw WS is better),
bayes_correlation.csv and bayes_design.json.

  python prepare_bayes_tests.py
  python prepare_bayes_tests.py --group exact
  python prepare_bayes_tests.py --metric igd_plus
"""
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path


def make_pairs(runs: pd.DataFrame,
               metric: str,
               envs: tuple[str, ...],
               alternatives: dict) -> pd.DataFrame:
    """Raw WS and one alternative per row, matched by seed; seeds missing for either side are dropped."""
    sign = dict(hv_ratio=1.0, igd_plus=-1.0)[metric]
    rows = []
    for algorithm, alts in alternatives.items():
        for env in envs:
            group = runs[(runs.algorithm == algorithm) & (runs.environment == env)]
            values = group.pivot(index='seed', columns='variant', values=metric)
            for alt in alts:
                if 'ws_raw' not in values or alt not in values:
                    print(f'note: {algorithm} on {env}: ws_raw or {alt} missing')
                    continue
                both = values[['ws_raw', alt]].dropna()
                rows += [dict(algorithm=algorithm, environment=env, alternative=alt, seed=int(seed),
                              ws_raw_value=a, alternative_value=b, y1=sign * a, y2=sign * b)
                         for seed, (a, b) in both.iterrows()]
    return pd.DataFrame(rows)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN if one side is constant."""
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan


def summary(g: pd.DataFrame) -> pd.Series:
    """Means, standard deviations, correlations and variance ratio of one comparison."""
    a = g.ws_raw_value.to_numpy()
    b = g.alternative_value.to_numpy()
    d = g.y1.to_numpy() - g.y2.to_numpy()
    spread = a.var(ddof=1) + b.var(ddof=1)
    return pd.Series(dict(
        n=len(g), mean_ws_raw=a.mean(), sd_ws_raw=a.std(ddof=1), mean_alternative=b.mean(),
        sd_alternative=b.std(ddof=1), mean_advantage=d.mean(), sd_advantage=d.std(ddof=1),
        pearson=correlation(a, b), spearman=correlation(pd.Series(a).rank().to_numpy(),
                                                        pd.Series(b).rank().to_numpy()),
        variance_ratio=d.var(ddof=1) / spread if spread > 0 else np.nan))


def pooled_check(pairs: pd.DataFrame,
                 algorithm: str,
                 rng: np.random.Generator,
                 permutations: int,
                 alpha: float) -> dict:
    """Mean Pearson correlation of a learner's comparisons against one raw-WS seed shuffle per environment."""
    tests = dict(paired='BayesianCorrelatedTTest(rho=0)', independent='BayesianIndependentTTest')
    observed = []
    null = []
    for env, g in pairs[pairs.algorithm == algorithm].groupby('environment'):
        wide = g.pivot(index='seed', columns='alternative', values='alternative_value').dropna()
        y1 = g.groupby('seed').ws_raw_value.first().reindex(wide.index).to_numpy()
        wide = wide.loc[:, wide.std() > 0]
        if wide.empty or y1.std() == 0 or len(y1) < 3:
            continue
        n = len(y1)
        z1 = (y1 - y1.mean()) / y1.std(ddof=1)
        z2 = ((wide - wide.mean()) / wide.std(ddof=1)).to_numpy()
        shuffles = rng.permuted(np.tile(np.arange(n), (permutations, 1)), axis=1)
        observed.append(z1 @ z2 / (n - 1))
        null.append(z1[shuffles] @ z2 / (n - 1))
    if not observed:
        return dict(comparisons=0, test=None)
    observed = np.concatenate(observed)
    null = np.concatenate(null, axis=1)
    mean = float(observed.mean())
    null_mean = null.mean(axis=1)
    p = float((1 + np.sum(null_mean >= mean)) / (1 + permutations))
    kind = 'paired' if mean > 0 and p < alpha else 'independent'
    return dict(comparisons=len(observed), mean_pearson=mean, p_permutation=p, design=kind, test=tests[kind])


def main() -> None:
    """Writes the test inputs, the per-comparison correlations and the recommended design per learner."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics', type=Path,
                        default=Path('results/evaluation/scalarization/metrics/metrics_runs.csv'))
    parser.add_argument('--group', choices=('additional', 'exact'), default='exact', help='Environment group')
    parser.add_argument('--output', type=Path, default=None,
                        help='Default: bayes/ or bayes_exact/ under results/evaluation/scalarization')
    parser.add_argument('--metric', choices=('hv_ratio', 'igd_plus'), default='hv_ratio')
    args = parser.parse_args()
    groups = dict(additional=('minecart', 'reacher', 'ant', 'halfcheetah', 'hopper', 'humanoid'),
                  exact=('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2'))
    folders = dict(additional='bayes', exact='bayes_exact')
    envs = groups[args.group]
    args.output = args.output or Path('results/evaluation/scalarization') / folders[args.group]

    # Pairs and checks
    alternatives = {'MO-PPO': ('ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm'), 'MO-SAC': ('ws_norm',)}
    permutations = 10_000
    seed = 2027
    alpha = 0.05
    runs = pd.read_csv(args.metrics).fillna({'variant': ''})
    pairs = make_pairs(runs, args.metric, envs, alternatives)
    keys = ['algorithm', 'environment', 'alternative']
    values = ['ws_raw_value', 'alternative_value', 'y1', 'y2']
    table = pairs.groupby(keys, sort=False)[values].apply(summary).reset_index()
    rng = np.random.default_rng(seed)
    learners = {a: pooled_check(pairs, a, rng, permutations, alpha) for a in alternatives
                if (pairs.algorithm == a).any()}
    for a, check in learners.items():
        check['mean_variance_ratio'] = float(table[table.algorithm == a].variance_ratio.mean())

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output / 'bayes_pairs.csv', index=False, float_format='%.17g')
    table.to_csv(args.output / 'bayes_correlation.csv', index=False, float_format='%.6g')
    design = dict(metric=args.metric, group=args.group, orientation='y1 - y2 > 0 means raw WS is better',
                  baseline='ws_raw', environments=envs, alternatives=alternatives, permutations=permutations,
                  seed=seed, alpha=alpha, learners=learners)
    (args.output / 'bayes_design.json').write_text(json.dumps(design, indent=2) + '\n', encoding='utf-8')
    with pd.option_context('display.width', 140, 'display.float_format', '{:.2f}'.format):
        for a, check in learners.items():
            print(f'\n{a}: Pearson correlation of raw WS with each alternative across seeds')
            print(table[table.algorithm == a].pivot(index='environment', columns='alternative', values='pearson')
                  .reindex(index=[e for e in envs if e in set(table.environment)],
                           columns=[v for v in alternatives[a] if v in set(table.alternative)]))
            if check['test'] is None:
                print('no comparison with variation on both sides')
                continue
            print(f"mean r = {check['mean_pearson']:.3f} over {check['comparisons']} comparisons, permutation "
                  f"p = {check['p_permutation']:.4f}, mean variance ratio = {check['mean_variance_ratio']:.2f}"
                  f" -> {check['design']}: {check['test']}")
    print(f'\nwrote {args.output}/bayes_pairs.csv, bayes_correlation.csv, bayes_design.json')


if __name__ == '__main__':
    main()