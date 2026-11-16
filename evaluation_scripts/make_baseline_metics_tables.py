"""LaTeX tables of the baseline comparison from metrics_runs.csv of compute_metrics_baseline.py (paper_style.py).

baselines_discrete.tex covers DST1-4, LDST1-2, Minecart and Reacher, baselines_mujoco.tex Ant, HalfCheetah, Hopper and
Humanoid (written once their runs are in the metrics). Methods are rows, environments columns; panel (a) is the HV
ratio, panel (b) IGD+. Cells are mean +- standard deviation over seeds, or with --stat median the median (IQR); the
best value per environment is bold.

  python make_baseline_tables.py
  python make_baseline_tables.py --stat median
"""
import argparse

import pandas as pd

from pathlib import Path

from paper_style import head, label, table_tex


def statistics(values: pd.Series, stat: str) -> tuple[float, float]:
    """Mean and standard deviation, or median and interquartile range."""
    if stat == 'mean':
        return values.mean(), values.std()
    return values.median(), values.quantile(0.75) - values.quantile(0.25)


def entry(center: float,
          spread: float,
          stat: str,
          best: bool) -> str:
    """Cell mean{±SD} or median{(IQR)}; the best value of the environment in bold."""
    value = rf'\textbf{{{center:.3f}}}' if best else f'{center:.3f}'
    return value + (rf'{{\scriptsize$\pm${spread:.3f}}}' if stat == 'mean' else rf'{{\scriptsize\,({spread:.3f})}}')


def table(runs: pd.DataFrame,
          group: str,
          group_envs: tuple[str, ...],
          stat: str) -> str | None:
    """LaTeX table of one environment group, or None if the metrics hold none of its environments."""
    names = dict(dst1='DST1', dst2='DST2', dst3='DST3', dst4='DST4', ldst1='LDST1', ldst2='LDST2',
                 minecart='Minecart', reacher='Reacher', ant='Ant', halfcheetah='HalfCheetah', hopper='Hopper',
                 humanoid='Humanoid')
    scalarizers = ('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm')
    methods = ([('MO-PPO', v) for v in scalarizers] + [('MO-SAC', v) for v in scalarizers[:2]]
               + [(b, '') for b in ('GPI-LS', 'PCN', 'IPRO', 'C-MORL', 'MORL/D')])
    metrics = (('hv_ratio', '(a) Hypervolume ratio (higher is better)', max),
               ('igd_plus', '(b) IGD$^+$ (lower is better)', min))
    envs = [e for e in group_envs if e in set(runs.environment)]
    if not envs:
        return None
    runs = runs[runs.environment.isin(envs)]
    rows = [r for r in methods if ((runs.algorithm == r[0]) & (runs.variant == r[1])).any()]
    body = []
    for column, title, best in metrics:
        stats = {key: statistics(g[column], stat) for key, g in runs.groupby(['environment', 'algorithm', 'variant'])}
        top = {e: best(round(c, 3) for (env, *_), (c, _) in stats.items() if env == e) for e in envs}
        if body:
            body.append(r'\midrule')
        body.append(rf'\multicolumn{{{len(envs) + 2}}}{{l}}{{\emph{{{title}}}}} \\')
        for i, (algo, variant) in enumerate(rows):
            first = i == 0 or rows[i - 1][0] != algo
            if i and first and (variant or rows[i - 1][1]):
                body.append(r'\addlinespace')
            name = [algo if first else '', label(variant)] if variant else [rf'\multicolumn{{2}}{{l}}{{{algo}}}']
            cells = [entry(*stats[env, algo, variant], stat, round(stats[env, algo, variant][0], 3) == top[env])
                     if (env, algo, variant) in stats else '--' for env in envs]
            body.append([*name, *cells])
    seed_counts = sorted(runs.groupby(['environment', 'label']).seed.nunique().unique())
    seeds = 'ten' if seed_counts == [10] else ' or '.join(map(str, seed_counts))
    constructed = 'constructed as the non-dominated union of all final fronts'
    statistic = (f'Means $\\pm$ standard deviations over {seeds} training seeds; bold: best mean per environment.'
                 if stat == 'mean' else
                 f'Medians (interquartile ranges) over {seeds} training seeds; bold: best median per environment.')
    caption = (f"Baseline comparison on the {'MuJoCo tasks' if group == 'mujoco' else 'discrete-action environments'}: "
               '(a) hypervolume ratio of the final front, its hypervolume divided by that of the reference front, and '
               '(b) IGD$^+$ of the final front with respect to the reference front. Reference fronts are '
               + (f'exact for DST and LDST and, for Minecart and Reacher, {constructed}. ' if group == 'discrete'
                  else f'{constructed}. ')
               + f'{statistic} Training budgets are listed in the appendix.')
    return table_tex(f"tab:baselines-{group}{'' if stat == 'mean' else '-median'}", caption, 'll' + 'r' * len(envs),
                     [head('Algorithm', 'Scalarizer', *(names[e] for e in envs))], body, wide=True)


def main() -> None:
    """Writes the baseline table of every environment group that has metrics."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics', type=Path,
                        default=Path('results/evaluation/scalarization/metrics/metrics_runs.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/evaluation/scalarization/tables'))
    parser.add_argument('--stat', choices=('mean', 'median'), default='mean')
    args = parser.parse_args()

    # Tables
    groups = dict(discrete=('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2', 'minecart', 'reacher'),
                  mujoco=('ant', 'halfcheetah', 'hopper', 'humanoid'))
    runs = pd.read_csv(args.metrics).fillna({'variant': ''})
    args.output.mkdir(parents=True, exist_ok=True)
    for group, group_envs in groups.items():
        if (tex := table(runs, group, group_envs, args.stat)) is not None:
            path = args.output / f"baselines_{group}{'' if args.stat == 'mean' else '_median'}.tex"
            path.write_text(tex, encoding='utf-8')
            print(f'wrote {path}')


if __name__ == '__main__':
    main()