"""Bayesian independent t-tests (BEST; Kruschke, 2013) of raw WS against the other MO-PPO scalarizers.

Reads bayes_pairs.csv and bayes_design.json of prepare_bayes_tests.py (same --group); y1 - y2 > 0 means raw WS is
better. Each comparison fits BayesianIndependentTTest of the bayesian-model-comparison repository (--repo or
BAYES_REPO). The probabilities that raw WS is better, practically equivalent (|mu1 - mu2| <= --rope) or worse are the
shares of posterior draws in each region; a comparison is decided when one share reaches --threshold. If both
scalarizers are constant across seeds, the difference is exact and no test is fitted.

Outputs (in --output, default --input): bayes_mo_ppo_<metric>.csv and bayes_mo_ppo_<metric>.tex; with --plots the
posterior predictive plots in ppc/.

  python run_bayes_tests.py --repo /path/to/bayesian-model-comparison
  python run_bayes_tests.py --group exact
"""
import os
import sys
import json
import logging
import argparse
import matplotlib

import numpy as np
import pandas as pd

from typing import Any
from pathlib import Path

matplotlib.use('Agg')

import arviz as az
import cmdstanpy
import matplotlib.pyplot as plt

from paper_style import head, label, table_tex

cmdstanpy.install_cmdstan(version='2.36.0', overwrite=True, verbose=True)


# Tests

def load_test(repo: Path) -> type:
    """BayesianIndependentTTest from the repository, with its Stan files addressed by absolute path."""
    repo = repo.expanduser().resolve()
    sys.path.insert(0, str(repo))
    import bayesian_test.AbstractBayesian as base
    from bayesian_test.BayesianIndependentTTest import BayesianIndependentTTest
    base.STAN_FILES_FOLDER = str(repo / 'stan_files')
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    return BayesianIndependentTTest


def hdi(x: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """Shortest interval containing prob of the draws."""
    x = np.sort(x)
    k = int(np.floor(prob * len(x)))
    i = int(np.argmin(x[k:] - x[:len(x) - k]))
    return float(x[i]), float(x[i + k])


def exact(y1: np.ndarray, y2: np.ndarray, rope: float) -> dict | None:
    """Both scalarizers constant across seeds: the difference is exact and no test is fitted (None otherwise)."""
    if np.ptp(y1) > 1e-12 or np.ptp(y2) > 1e-12:
        return None
    d = float(y1[0] - y2[0])
    decision = ('identical' if abs(d) <= 1e-12 else 'better' if d > rope else 'worse' if d < -rope else 'equivalent')
    return dict(fitted=False, difference=d, hdi_low=d, hdi_high=d, decision=decision)


def compare(test: type,
            y1: np.ndarray,
            y2: np.ndarray,
            name: str,
            args: argparse.Namespace) -> dict:
    """Exact difference or fitted t-test of one comparison with its decision, diagnostics and predictive checks."""
    if (result := exact(y1, y2, args.rope)) is not None:
        print(f"both constant across seeds: difference {result['difference']:+.4f}, no test fitted")
        return result
    model = test(y1=y1, y2=y2, rope=(-args.rope, args.rope), seed=args.seed)
    model.fit(iter_sampling=args.draws, iter_warmup=args.warmup, chains=args.chains)
    d = model._fit.stan_variable('difference_mean')
    shares = dict(better=np.mean(d > args.rope), equivalent=np.mean(np.abs(d) <= args.rope),
                  worse=np.mean(d < -args.rope))
    region = max(shares, key=shares.get)
    diagnostics = az.summary(model.inf_data, var_names=['difference_mean'], kind='diagnostics').loc['difference_mean']
    hdi_low, hdi_high = hdi(d)
    result = dict(fitted=True, difference=float(d.mean()), hdi_low=hdi_low, hdi_high=hdi_high,
                  effect_size=float(model._fit.stan_variable('effect_size').mean()),
                  **{f'p_{k}': float(v) for k, v in shares.items()},
                  decision=region if shares[region] >= args.threshold else 'undecided',
                  r_hat=float(diagnostics['r_hat']), ess_bulk=float(diagnostics['ess_bulk']))
    for side, y, rep in (('ws_raw', y1, 'y1_rep'), ('alternative', y2, 'y2_rep')):
        draws = model._fit.stan_variable(rep)
        result[f'ppc_mean_{side}'] = float(np.mean(draws.mean(axis=1) >= y.mean()))
        result[f'ppc_sd_{side}'] = float(np.mean(draws.std(axis=1, ddof=1) >= y.std(ddof=1)))
    if args.plots:
        model._posterior_predictive_check(directory_path=str(args.output), file_path='ppc',
                                          file_name=f'{name}_ppc', font_size=10, save=True)
        plt.close('all')
    return result


# Table

def signed(x: float) -> str:
    """Signed number with a typographic minus; values that round to zero carry no sign."""
    return '0.000' if round(x, 3) == 0 else f'{x:+.3f}'


def prob(p: float, decided: bool) -> str:
    """Posterior probability, bold if it decides the comparison."""
    return rf'\textbf{{{p:.3f}}}' if decided else f'{p:.3f}'


def cells(r: Any, show: tuple[str, ...]) -> list[str]:
    """Difference, HDI and probability cells of one comparison, or a note for constant scalarizers."""
    out = [f'${signed(r.difference)}$'] if 'delta' in show else []
    if r.fitted:
        out += [f'$[{signed(r.hdi_low)}, {signed(r.hdi_high)}]$'] if 'hdi' in show else []
        return out + [prob(getattr(r, f'p_{k}'), r.decision == k) for k in ('better', 'equivalent', 'worse')]
    out += ['--'] if 'hdi' in show else []
    if r.decision == 'identical':
        text = 'identical in every seed'
    else:
        text = 'constant in every seed' + ('' if 'delta' in show else f' ($\\Delta={signed(r.difference)}$)')
    return out + [rf'\multicolumn{{3}}{{c}}{{\emph{{{text}}}}}']


def latex(table: pd.DataFrame,
          design: dict,
          envs: list[str],
          args: argparse.Namespace) -> str:
    """Appendix table of all comparisons."""
    names = dict(minecart='Minecart', reacher='Reacher', ant='Ant', halfcheetah='HalfCheetah', hopper='Hopper',
                 humanoid='Humanoid', dst1='DST1', dst2='DST2', dst3='DST3', dst4='DST4', ldst1='LDST1', ldst2='LDST2')
    tasks = dict(additional='the tasks without exact fronts', exact='the exact-front tasks')
    metrics = dict(hv_ratio='hypervolume ratios', igd_plus='negated IGD$^+$ values')
    show = dict(full=('delta', 'hdi'), delta=('delta',), probabilities=())[args.columns]
    rows = []
    for env in envs:
        sub = table[table.environment == env]
        if sub.empty:
            continue
        if rows:
            rows.append(r'\midrule')
        rows += [[names.get(env, env) if i == 0 else '', label(r.alternative), *cells(r, show)]
                 for i, r in enumerate(sub.itertuples())]
    seed_counts = sorted(table.n.unique())
    seeds = 'ten' if seed_counts == [10] else ' to '.join(map(str, (seed_counts[0], seed_counts[-1])))
    check = design['learners'].get('MO-PPO', {})
    independent = (f', treated as independent because runs with equal seeds were uncorrelated (mean correlation '
                   f"${check['mean_pearson']:.2f}$, permutation $p={check['p_permutation']:.2f}$)"
                   if check.get('design') == 'independent' else '')
    group = design.get('group', 'additional')
    constant = (' Rows without probabilities: each scalarizer reaches the same value in every seed, so the '
                'difference is exact and no test is fitted.' if not table.fitted.all() else '')
    legend = '; '.join([*(['$\\Delta$: posterior mean of the difference'] if 'delta' in show else []),
                        *(['95\\% HDI: highest-density interval'] if 'hdi' in show else [])])
    tables = dict(additional=r'Tables~\ref{tab:baselines-discrete} and~\ref{tab:baselines-mujoco}',
                  exact=r'Table~\ref{tab:baselines-discrete}')
    means = f' Mean values per scalarizer are given in {tables[group]}.' if 'delta' not in show else ''
    caption = (f'Bayesian comparison of raw WS with each other MO-PPO scalarizer on {tasks[group]}. '
               f'For each environment and alternative, the Bayesian independent t-test estimates the difference of '
               f"mean {metrics[design['metric']]}, raw WS minus the alternative, from {seeds} training seeds per "
               f"scalarizer{independent}. {legend + '. ' if legend else ''}The probability columns give the posterior "
               f'probabilities that raw WS is better by more than {args.rope:g}, that the difference lies within '
               f'$\\pm{args.rope:g}$, and that the alternative is better by more than {args.rope:g}; bold: at least '
               f'{args.threshold:g}.{means}{constant}')
    name = f"tab:bayes-mo-ppo{'-exact' if group == 'exact' else ''}{'' if design['metric'] == 'hv_ratio' else '-igd'}"
    header = ['Env.', 'Alternative', *([r'$\Delta$'] if 'delta' in show else []),
              *([r'95\% HDI'] if 'hdi' in show else []), r'\shortstack[r]{Raw WS\\better}',
              r'\shortstack[r]{Practically\\equivalent}', r'\shortstack[r]{Alternative\\better}']
    return table_tex(name, caption, 'll' + 'r' * (len(header) - 2), [head(*header)], rows,
                     wide=args.columns != 'probabilities')


# Main

def main() -> None:
    """Fits every comparison of raw WS with the other MO-PPO scalarizers and writes the results and the table."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--group', choices=('additional', 'exact'), default='exact',
                        help='Environment group; selects the default --input folder')
    parser.add_argument('--input', type=Path, default=None,
                        help='Folder with bayes_pairs.csv and bayes_design.json (default: bayes/ or bayes_exact/)')
    parser.add_argument('--output', type=Path, default=None, help='Default: --input')
    parser.add_argument('--repo', type=Path,
                        default=Path(os.environ.get('BAYES_REPO', '../bayesian-model-comparison')))
    parser.add_argument('--rope', type=float, default=0.01, help='Half-width of the region of practical equivalence')
    parser.add_argument('--threshold', type=float, default=0.95, help='Posterior probability that decides a comparison')
    parser.add_argument('--draws', type=int, default=10_000, help='Posterior draws per chain')
    parser.add_argument('--warmup', type=int, default=1_000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--plots', action='store_true', help="Save the repository's posterior predictive plots")
    parser.add_argument('--table_only', action='store_true',
                        help='Rewrite the LaTeX table from the saved results CSV without fitting again')
    parser.add_argument('--columns', choices=('full', 'delta', 'probabilities'), default='full',
                        help='Table columns besides the three probabilities: full (difference and HDI), delta '
                             '(difference only), probabilities (none)')
    args = parser.parse_args()
    folders = dict(additional='bayes', exact='bayes_exact')
    args.input = args.input or Path('results/evaluation/scalarization') / folders[args.group]
    args.output = args.output or args.input

    # Design
    design = json.loads((args.input / 'bayes_design.json').read_text(encoding='utf-8'))
    envs = design.get('environments', ('minecart', 'reacher', 'ant', 'halfcheetah', 'hopper', 'humanoid'))
    stem = f"bayes_mo_ppo_{design['metric']}"
    if args.table_only:
        table = pd.read_csv(args.output / f'{stem}.csv')
        table['fitted'] = table.fitted.astype(bool)
        (args.output / f'{stem}.tex').write_text(latex(table, design, envs, args), encoding='utf-8')
        print(f'wrote {args.output / stem}.tex')
        return

    # Tests
    alternatives = ('ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm')
    columns = ['environment', 'alternative', 'n', 'mean_ws_raw', 'mean_alternative', 'fitted', 'difference', 'hdi_low',
               'hdi_high', 'effect_size', 'p_better', 'p_equivalent', 'p_worse', 'decision', 'r_hat', 'ess_bulk',
               'ppc_mean_ws_raw', 'ppc_sd_ws_raw', 'ppc_mean_alternative', 'ppc_sd_alternative']
    pairs = pd.read_csv(args.input / 'bayes_pairs.csv')
    pairs = pairs[pairs.algorithm == 'MO-PPO']
    test = load_test(args.repo)
    rows = []
    for env in envs:
        for alt in alternatives:
            g = pairs[(pairs.environment == env) & (pairs.alternative == alt)].sort_values('seed')
            if len(g) < 3:
                print(f'note: {env}, {alt}: {len(g)} seeds, skipped')
                continue
            print(f'\n=== MO-PPO on {env}: raw WS versus {alt} ({len(g)} seeds)')
            rows.append(dict(environment=env, alternative=alt, n=len(g), mean_ws_raw=g.ws_raw_value.mean(),
                             mean_alternative=g.alternative_value.mean(),
                             **compare(test, g.y1.to_numpy(float), g.y2.to_numpy(float), f'{env}_{alt}', args)))
    table = pd.DataFrame(rows).reindex(columns=columns)
    table['fitted'] = table.fitted.astype(bool)

    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / f'{stem}.csv', index=False, float_format='%.6g')
    (args.output / f'{stem}.tex').write_text(latex(table, design, envs, args), encoding='utf-8')
    symbols = dict(better='+', equivalent='=', worse='-', undecided='?', identical='≡')
    print(f"\nRaw WS versus each alternative ({symbols['better']} better, {symbols['equivalent']} practically "
          f"equivalent, {symbols['worse']} worse, {symbols['undecided']} undecided, {symbols['identical']} identical "
          f'in every seed, * both constant across seeds: exact difference, no test; threshold {args.threshold:g}, '
          f'ROPE +-{args.rope:g}):')
    table['symbol'] = table.decision.map(symbols) + np.where(table.fitted | table.decision.eq('identical'), '', '*')
    print(table.pivot(index='environment', columns='alternative', values='symbol')
          .reindex(index=[e for e in envs if e in set(table.environment)],
                   columns=[a for a in alternatives if a in set(table.alternative)]).to_string())
    print(table.decision.value_counts().to_string())
    ppc = [c for c in columns if c.startswith('ppc_')]
    flagged = table[(table[ppc] < 0.01).any(axis=1) | (table[ppc] > 0.99).any(axis=1)
                    | (table.r_hat > 1.01) | (table.ess_bulk < 1_000)]
    if not flagged.empty:
        print('\ncheck the fit of (R-hat > 1.01, bulk ESS < 1000, or a posterior predictive p-value outside '
              '(0.01, 0.99); a constant side always fails the spread check):')
        print(flagged[['environment', 'alternative', 'r_hat', 'ess_bulk', *ppc]].to_string(index=False))
    print(f'\nwrote {args.output / stem}.csv and .tex')


if __name__ == '__main__':
    main()