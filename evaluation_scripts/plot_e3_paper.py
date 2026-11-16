"""Fixed E3 paper outputs: one main table, four appendix tables and one figure in a paper version (DST1, DST4, LDST2)
and an appendix version (DST2, DST3, LDST1). The paper says "deterministic execution" for greedy action selection.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from paper_style import (FIGURE_STATS, POOLED_STATS, SEED_STATS, cell, draw, envs_text, figure_tex, finish, head, label,
                         mark, panels, save, sd_of, table_tex, versions)
from reconstruct_e3 import DEFAULT_OUTPUT, ENVS, METHODS, H


class Summary:
    def __init__(self, frame: pd.DataFrame) -> None:
        """E3 summary table with one row per measurement."""
        self.frame = frame

    def get(self,
            method: str,
            alloc: str,
            metric: str,
            env: str = 'ALL',
            budget: str = 'standard') -> pd.Series:
        """Summary row of one measurement."""
        f = self.frame
        selected = f[(f.kind == 'measures') & (f.scalarizer == method) & (f.allocation == alloc) &
                     (f.metric == metric) & (f.environment == env) & (f.budget == budget)]
        return selected.iloc[0]

    def cell(self,
             method: str,
             alloc: str,
             metric: str,
             env: str = 'ALL',
             budget: str = 'standard') -> str:
        """Percent; mean +- standard deviation over the ten seeds unless the metric has no deviation."""
        row = self.get(method, alloc, metric, env, budget)
        means_only = ('oracle_recall', 'complete', 'score_tied_miss_rate')
        return cell(row['mean'], None if metric in means_only else sd_of(row))


def tables(summary: Summary) -> tuple[str, str]:
    """Main allocation table and the four appendix tables."""
    common = ('all', 'R2', 'corners')
    main_metrics = ('archive_recall', 'active_recall', 'weight_hit_active')
    budgets = '($P=10$ for DST, $P=20$ for LDST)'
    archive_active = ('Archive shows Pareto returns recovered from all evaluated checkpoints ($C=1$), while Active '
                      'shows returns represented by the final actors.')
    groups = [r'\multicolumn{3}{c}{Archive recall} & \multicolumn{3}{c}{Active recall} & '
              r'\multicolumn{3}{c}{Preferences with an own-target actor} \\']
    allocations = [r'$R=1$', r'$R=2$', r'$K=2$'] * 3
    main = table_tex(
        'tab:e3-allocations',
        f'Recall and preference success (\\%) at $h={H}$ under fixed actor budgets {budgets}. We compare one actor '
        'per preference ($R=1$, $K=P$), two actors per preference ($R=2$, $K=P/2$), and populations using only the '
        f'two corner preferences ($K=2$). {archive_active} The last three columns report the proportion of '
        f'preferences with at least one actor matching its own target. {POOLED_STATS}',
        'l' + 'r' * 9,
        ['& ' + groups[0], r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
         head('Scalarizer', *allocations)],
        [[label(m)] + [summary.cell(m, a, metric) for metric in main_metrics for a in common] for m in METHODS],
        wide=True)
    headings = ('Oracle', 'Complete', r'\shortstack{Targets\\found}', r'\shortstack{Off-target\\found}',
                r'\shortstack{Preferences\\hit}', r'\shortstack{Actors\\hit}', 'Redundant',
                r'\shortstack{Tied\\misses}')
    diagnostic = ('oracle_recall', 'complete', 'target_coverage', 'off_target_recall', 'weight_hit_active',
                  'actor_hit_active', 'redundant_pareto_actor_rate', 'score_tied_miss_rate')
    rows = []
    for a, title in zip(common, (r'All preferences ($R=1$)', r'Two replicas ($R=2$)', r'Corner preferences ($K=2$)')):
        if rows:
            rows.append(r'\midrule')
        rows.append(r'\multicolumn{9}{l}{\emph{' + title + r'}} \\')
        rows += [[label(m)] + [summary.cell(m, a, x) for x in diagnostic] for m in METHODS]
    appendix = table_tex(
        'tab:e3-decomposition',
        f'Diagnostics (\\%) at $h={H}$ for the fixed actor budgets of Table~\\ref{{tab:e3-allocations}}. '
        f'{POOLED_STATS} Oracle, Complete, and Tied misses are reported as means, and undefined rates are omitted.',
        'l' + 'r' * 8, [head('Scalarizer', *headings)], rows, wide=True,
        note="\\textit{Definitions.} Oracle: returns targeted by the allocation's preferences. Complete: share of "
             'runs with a complete archive. Targets found and Off-target found: archive recall within target and '
             'non-target returns. Preferences hit: share of preferences with at least one actor at an own target. '
             'Actors hit: share of actors at an own target. Redundant: actors repeating an already represented '
             "Pareto return, as a share of all actors. Tied misses: own-target misses that tie the target's "
             'scalarized score.')
    rows = []
    for e in ENVS:
        if rows:
            rows.append(r'\midrule')
        rows += [[e.upper() if n == 0 else '', label(m)]
                 + [summary.cell(m, a, x, e) for x in main_metrics for a in common]
                 for n, m in enumerate(METHODS)]
    appendix += '\n' + table_tex(
        'tab:e3-environments',
        f'Recall and preference success (\\%) by environment at $h={H}$ under fixed actor budgets {budgets}. '
        f'Allocations and columns are defined as in Table~\\ref{{tab:e3-allocations}}. {SEED_STATS}',
        'll' + 'r' * 9,
        ['& & ' + groups[0], r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',
         head('Env.', 'Scalarizer', *allocations)], rows, wide=True)
    rows = []
    for e in (e for e in ENVS if e.startswith('ldst')):
        if rows:
            rows.append(r'\midrule')
        rows += [[e.upper() if n == 0 else '', label(m)] + [summary.cell(m, 'K5', x, e) for x in main_metrics]
                 for n, m in enumerate(METHODS)]
    appendix += '\n' + table_tex(
        'tab:e3-ldst-k5',
        f'Recall and preference success (\\%) on LDST at $h={H}$ for the additional allocation of five preferences '
        'with four actors each ($K=5$, $R=4$, $P=20$). Columns are defined as in '
        f'Table~\\ref{{tab:e3-allocations}}. {SEED_STATS}',
        'llrrr', [head('Env.', 'Scalarizer', 'Archive recall', 'Active recall',
                       r'\shortstack[r]{Preferences with an\\own-target actor}')], rows)
    large = ('all', 'R2', 'K10', 'K5', 'corners')
    appendix += '\n' + table_tex(
        'tab:e3-large',
        f'Recall (\\%) at the larger fixed actor budget $P=50$ and $h={H}$. Column headers give $K/R$, with $K$ '
        'preferences of $R$ actors each. Archive ($C=1$) and Active are defined as in '
        f'Table~\\ref{{tab:e1-summary}}. {POOLED_STATS}',
        'l' + 'r' * 10,
        [r'& \multicolumn{5}{c}{Archive recall} & \multicolumn{5}{c}{Active recall} \\',
         r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
         head('Scalarizer', *[f'${k}/{50 // k}$' for k in (50, 25, 10, 5, 2)] * 2)],
        [[label(m)] + [summary.cell(m, a, x, budget='large') for x in ('archive_recall', 'active_recall')
                       for a in large]
         for m in METHODS], wide=True)
    return main, appendix


def allocation_figure(summary: Summary, envs: tuple[str, ...]) -> plt.Figure:
    """Rows: oracle, archive, and active recall; columns: environments; x: allocations by decreasing K."""
    rows = (('oracle_recall', 'Oracle (%)'), ('archive_recall', 'Archive (%)'), ('active_recall', 'Active (%)'))
    ylabels = [ylabel for _, ylabel in rows]
    fig, axes = panels(len(rows), envs, ylabels)
    for col, env in enumerate(envs):
        allocations = ('all', 'R2', 'K5', 'corners') if env.startswith('ldst') else ('all', 'R2', 'corners')
        x = np.arange(len(allocations), dtype=float)
        for row, (metric, _) in enumerate(rows):
            ax = axes[row, col]
            mark(ax, 0)
            for method in METHODS:
                v = 100 * np.array([summary.get(method, a, metric, env)[['mean', 'ci_low', 'ci_high']].to_numpy(float)
                                    for a in allocations])
                draw(ax, x, v[:, 0], v[:, 1], v[:, 2], method)
        firsts = [summary.get(METHODS[0], a, 'active_recall', env) for a in allocations]
        axes[-1, col].set(xticks=x, xticklabels=[f'{int(r.K)}/{int(r.R)}' for r in firsts], xlim=(-0.25, x[-1] + 0.25))
    finish(fig, axes, envs, ylabels, 'Preferences / actors per preference ($K/R$)', METHODS)
    return fig


def figure_caption(envs: tuple[str, ...], other: str) -> str:
    """Caption of the allocation figure for its environments."""
    return (f'Oracle, archive, and active recall across allocations of preferences and actors on {envs_text(envs)} '
            f'at $h={H}$, under fixed actor budgets ($P=10$ for DST, $P=20$ for LDST). The horizontal axis gives '
            '$K/R$, with $K$ preferences of $R=P/K$ actors each. The dotted line marks $R=1$, the standard '
            "population. Oracle shows returns targeted by the allocation's preferences. Archive shows Pareto returns "
            'recovered from all evaluated checkpoints ($C=1$), while Active shows returns represented by the final '
            f'actors. {FIGURE_STATS} {other}')


def main() -> None:
    """Writes the E3 tables and the allocation figure to <input>/paper."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Tables
    summary = Summary(pd.read_csv(args.input / 'e3_summary.csv'))
    out = args.input / 'paper'
    (out / 'appendix').mkdir(parents=True, exist_ok=True)
    main_table, appendix = tables(summary)
    (out / 'e3_allocations.tex').write_text(main_table, encoding='utf-8')
    (out / 'appendix/e3_tables.tex').write_text(appendix, encoding='utf-8')

    # Figures
    description = ('Line charts with one column per environment and three rows for oracle, archive and active recall '
                   'in percent. The horizontal axis lists the allocations, from many preferences with one actor each '
                   'to two preferences with many actors each. Each panel has one line per scalarizer.')
    for envs, path, tag, other in versions('e3_allocations', 'fig:e3-allocations'):
        save(allocation_figure(summary, envs), out / path)
        (out / f'{path[:-4]}_figure.tex').write_text(
            figure_tex(path, figure_caption(envs, other), tag, description=description), encoding='utf-8')
    print(f'E3 paper outputs: {out}')


if __name__ == '__main__':
    main()