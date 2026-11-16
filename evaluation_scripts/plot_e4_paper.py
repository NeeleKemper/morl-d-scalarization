"""Fixed E4 paper outputs: one main table, three appendix tables, one learning-rate figure in a paper version (DST1,
DST4, LDST2) and an appendix version (DST2, DST3, LDST1), and one pooled timing figure for the appendix. Tables write
scalarizers with the paper's macros (\\wsraw{} etc.) and signed differences in math mode; figures use the plain names.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Iterable
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

from paper_style import (FIGURE_STATS, POOLED_STATS, SEED_STATS, WIDTH, cell, draw, envs_text, figure_tex, finish, head,
                         label, mark, panels, save, sd_of, style, table_tex, versions)
from reconstruct_e4 import DATA_DIR, DEFAULT_LR, ENVS, H, METHODS, RATES


# Data

class Summary:
    def __init__(self, frame: pd.DataFrame) -> None:
        """E4 summary indexed by kind, environment, scalarizer, learning rate, iteration and metric."""
        self.frame = frame
        self.lookup = frame.set_index(['kind', 'environment', 'scalarizer', 'learning_rate', 'h', 'metric'])

    def get(self,
            method: str,
            lr: float,
            metric: str,
            h: int = H,
            env: str = 'ALL') -> pd.Series:
        """Summary row of one measurement."""
        return self.lookup.loc[('measures', env, method, lr, h, metric)]

    def cell(self,
             method: str,
             lr: float,
             metric: str,
             h: int = H,
             env: str = 'ALL',
             deviation: bool = True) -> str:
        """Percent; mean +- standard deviation over the ten seeds (or the mean only)."""
        row = self.get(method, lr, metric, h, env)
        return cell(row['mean'], sd_of(row) if deviation else None)

    def peak(self, method: str, lr: float, env: str = 'ALL') -> pd.Series:
        """Descriptive peak of the pooled mean active-recall curve and its earliest iteration."""
        f = self.frame
        x = f[(f.kind == 'timing') & (f.metric == 'active_peak') & (f.environment == env)
              & (f.scalarizer == method) & (f.learning_rate == lr)]
        return x.iloc[0]


# Text

def rate_count() -> str:
    """Number of learning rates as a word."""
    return {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}.get(len(RATES), str(len(RATES)))


def rate_text(lr: float) -> str:
    """Learning rate in units of 10^-4, e.g. 3."""
    return f'{lr / 1e-4:g}'


def rate_tex(lr: float) -> str:
    """Learning rate in scientific notation, e.g. $3\\times10^{-4}$."""
    mantissa, exponent = f'{lr:.6e}'.split('e')
    return f'${float(mantissa):g}\\times10^{{{int(exponent)}}}$'


def macro(method: str) -> str:
    """The paper's macro of a scalarizer for tables, e.g. ws_raw -> \\wsraw{}."""
    return '\\' + method.replace('_', '') + '{}'


def series(items: Iterable[str]) -> str:
    """'a, b, and c' (or 'a and b')."""
    items = list(items)
    return ' and '.join(items) if len(items) < 3 else ', '.join(items[:-1]) + f', and {items[-1]}'


def signed(value: float) -> str:
    """A difference of two shares in percentage points, with its sign and in math mode."""
    if not np.isfinite(value):
        return '--'
    points = round(100 * value, 1)
    return '$0.0$' if points == 0 else f'${points:+.1f}$'


def recall_text(verb: str) -> str:
    """Definition of the two recall measures, in the words of the paper."""
    return (f'Archive {verb} Pareto returns recovered from checkpoints evaluated every iteration ($C=1$), while '
            f'Active {verb} returns represented by the final actors.')


# Tables

def tables(summary: Summary) -> tuple[str, str]:
    """Main learning-rate table and the three appendix tables."""
    count = rate_count()
    rates = [f'${rate_text(lr)}$' for lr in RATES]
    default = f'${rate_text(DEFAULT_LR)}$'
    headers = f'The column headings give the learning rate in units of $10^{{-4}}$, and {default} is the default.'
    rate_column = f'The Rate column gives the learning rate in units of $10^{{-4}}$, and {default} is the default.'
    rows = []
    for metric, title in (('archive_recall', 'Archive recall'), ('active_recall', 'Active recall')):
        if rows:
            rows.append(r'\midrule')
        rows.append(r'\multicolumn{' + str(len(RATES) + 1) + r'}{l}{\emph{' + title + r'}} \\')
        rows += [[macro(m)] + [summary.cell(m, lr, metric) for lr in RATES] for m in METHODS]
    main = table_tex('tab:e4-learning-rate',
                     f'Archive and active recall (\\%) at $h={H}$ across {count} learning rates, using the standard '
                     f'populations with all other settings unchanged. The default learning rate is '
                     f"{rate_tex(DEFAULT_LR)}. {recall_text('measures')} {POOLED_STATS}",
                     'l' + 'r' * len(RATES), [head('Scalarizer', *[rate_tex(lr) for lr in RATES])], rows)
    env_rows = []
    for env in ENVS:
        if env_rows:
            env_rows.append(r'\midrule')
        env_rows += [[env.upper() if j == 0 else '', macro(m)] +
                     [summary.cell(m, lr, metric, env=env) for metric in ('archive_recall', 'active_recall')
                      for lr in RATES] for j, m in enumerate(METHODS)]
    environments = table_tex(
        'tab:e4-environments',
        f'Archive and active recall (\\%) by environment for the {count} learning rates, with the standard '
        f'populations at $h={H}$. {headers} {SEED_STATS}',
        'll' + 'r' * 2 * len(RATES),
        [r'& & \multicolumn{' + str(len(RATES)) + r'}{c}{Archive recall} & \multicolumn{' + str(len(RATES))
         + r'}{c}{Active recall} \\',
         rf'\cmidrule(lr){{3-{2 + len(RATES)}}}\cmidrule(lr){{{3 + len(RATES)}-{2 + 2 * len(RATES)}}}',
         head('Env.', 'Scalarizer', *rates * 2)], env_rows, wide=True)
    diagnostics = ('complete', 'off_target_recall', 'target_absence', 'off_target_absence',
                   'weight_hit_active', 'redundant_pareto_actor_rate', 'no_treasure_rate')
    rows = []
    for method in METHODS:
        if rows:
            rows.append(r'\midrule')
        rows += [[macro(method) if j == 0 else '', rates[j]] +
                 [summary.cell(method, lr, metric, deviation=metric != 'complete') for metric in diagnostics]
                 for j, lr in enumerate(RATES)]
    loss = table_tex(
        'tab:e4-loss',
        f'Diagnostics (\\%) at $h={H}$ for the {count} learning rates, with the standard populations. {rate_column} '
        f'{POOLED_STATS} Complete is a share of runs and has no deviation. Environments in which a measure is '
        'undefined are left out of its average.',
        'll' + 'r' * len(diagnostics),
        [head('Scalarizer', 'Rate', 'Complete', r'\shortstack{Off-target\\found}', 'T absent', 'O absent',
              'Own target', 'Redundant', 'None')], rows, wide=True,
        note='\\textit{Definitions.} Complete is the share of runs whose archive contains the complete front. '
             'Off-target found is the archive recall on the Pareto returns outside the target set. T absent and O '
             'absent are the shares of the discovered target and off-target returns that the final actors do not '
             'represent. Own target is the share of actors whose final return is a target of their own preference. '
             'Redundant is the share of all actors that repeat an already represented Pareto return. None is the '
             'share of actors that collect no treasure.')
    rows = []
    pairs = (('archive_recall', 25), ('archive_recall', 50), ('active_recall', 25),
             ('active_recall', 50), ('active_recall', 75), ('active_recall', 100))
    n_archive = sum(metric == 'archive_recall' for metric, _ in pairs)
    for method in METHODS:
        if rows:
            rows.append(r'\midrule')
        baseline = summary.peak(method, DEFAULT_LR)
        for j, lr in enumerate((RATES[0], DEFAULT_LR)):
            peak = summary.peak(method, lr)
            final = summary.get(method, lr, 'active_recall')['mean']
            rows.append([macro(method) if j == 0 else '', f'${rate_text(lr)}$'] +
                        [summary.cell(method, lr, metric, h=h) for metric, h in pairs] +
                        [f"{cell(peak['mean'])} ({int(peak.h)})", signed(final - baseline['mean'])])
    cutoffs = table_tex(
        'tab:e4-cutoffs',
        'Archive and active recall (\\%) at earlier iterations for the lowest and the default learning rate, with '
        f'the standard populations. {rate_column} {POOLED_STATS} Highest and $\\Delta$ highest are descriptive, have '
        'no deviation, and do not yield a rule for when to stop training.',
        'll' + 'r' * 8,
        [rf'& & \multicolumn{{{n_archive}}}{{c}}{{Archive recall}} & \multicolumn{{{len(pairs) - n_archive}}}{{c}}'
         r'{Active recall} & & \\',
         rf'\cmidrule(lr){{3-{2 + n_archive}}}\cmidrule(lr){{{3 + n_archive}-{2 + len(pairs)}}}',
         head('Scalarizer', 'Rate', *[f'$h={h}$' for _, h in pairs], 'Highest ($h$)', r'$\Delta$ highest')],
        rows, wide=True,
        note='\\textit{Definitions.} Highest gives the maximum of the pooled mean active-recall curve over '
             f'iterations 1--{H}, with the earliest iteration that attains it in parentheses. $\\Delta$ highest is '
             'the final active recall minus the highest value of the default learning rate, in percentage points.')
    return main, environments + '\n' + loss + '\n' + cutoffs


# Figures

def learning_rate_figure(summary: Summary, envs: tuple[str, ...]) -> plt.Figure:
    """Rows: archive and active recall at h=H; columns: environments; x: learning rate (linear, units of 1e-4)."""
    rows = (('archive_recall', 'Archive (%)'), ('active_recall', 'Active (%)'))
    ylabels = [ylabel for _, ylabel in rows]
    fig, axes = panels(len(rows), envs, ylabels)
    x = np.array(RATES) / 1e-4
    for col, env in enumerate(envs):
        for row, (metric, _) in enumerate(rows):
            ax = axes[row, col]
            mark(ax, DEFAULT_LR / 1e-4)
            for method in METHODS:
                v = 100 * np.array([summary.get(method, lr, metric, env=env)[['mean', 'ci_low', 'ci_high']]
                                    .to_numpy(float) for lr in RATES])
                draw(ax, x, v[:, 0], v[:, 1], v[:, 2], method)
            ax.set(xticks=x, xlim=(0, x[-1] + 0.5))
            ax.set_xticklabels([f'{v:g}' for v in x])
    finish(fig, axes, envs, ylabels, r'Learning rate ($\times10^{-4}$)', METHODS)
    return fig


def learning_rate_caption(envs: tuple[str, ...], other: str) -> str:
    """Caption of the learning-rate figure for its environments."""
    return (f'Archive and active recall at $h={H}$ across learning rates on {envs_text(envs)}, using the standard '
            f'populations (\\% of the Pareto front). Each learning rate is a separately trained set of runs. '
            f"{recall_text('shows')} The dotted line marks the default learning rate {rate_tex(DEFAULT_LR)}. "
            f'{FIGURE_STATS} {other}')


def shade(color: str | tuple, step: float) -> tuple:
    """Lighter (step > 0, toward white) or darker (step < 0, toward black) version of a color."""
    rgb = np.array(to_rgb(color))
    return tuple(rgb + step * (1 - rgb)) if step >= 0 else tuple(rgb * (1 + step))


def rate_style(lr: float) -> dict:
    """Lightness step, line style with its name, and marker of a learning rate; the default has no marker."""
    styles = {0.0001: dict(step=0.5, linestyle=(0, (1, 1.6)), name='dotted', marker='o'),
              0.0003: dict(step=0.0, linestyle='-', name='solid', marker=None),
              0.0006: dict(step=-0.3, linestyle=(0, (3, 1.5)), name='dashed', marker='^'),
              0.001: dict(step=-0.55, linestyle=(0, (7, 1.5, 1, 1.5)), name='dash-dotted', marker='D')}
    for value, rate in styles.items():
        if np.isclose(lr, value, rtol=1e-6, atol=0):
            return rate


def rate_line(lr: float, color: str | tuple) -> dict:
    """Line properties of one learning rate in the given color, without markers."""
    s = rate_style(lr)
    return dict(color=shade(color, s['step']), linestyle=s['linestyle'], linewidth=1.5 if lr == DEFAULT_LR else 1.0,
                marker=None)


def rate_markers(lr: float, color: str | tuple) -> dict:
    """Marker properties of one learning rate in the given color: open markers with an edge in the line's color."""
    s = rate_style(lr)
    return dict(marker=s['marker'], markersize=3.8, markerfacecolor='white', markeredgewidth=0.8,
                markeredgecolor=shade(color, s['step']))


def timing_figure(summary: Summary) -> plt.Figure:
    """Pooled active recall over training at every learning rate, one panel per scalarizer."""
    height = 3.9
    every = 16
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.57 / WIDTH, right=0.99, top=1 - 0.27 / height, bottom=0.92 / height, wspace=0.12,
                        hspace=0.32)
    h = np.arange(1, H + 1)
    for (r, c), ax in np.ndenumerate(axes):
        method = f"{('ws', 'tch', 'stch')[c]}_{('raw', 'norm')[r]}"
        color = style(method)['color']
        marked = [lr for lr in RATES if rate_style(lr)['marker']]
        for lr in RATES:
            v = 100 * np.array([summary.get(method, lr, 'active_recall', h=int(t))[['mean', 'ci_low', 'ci_high']]
                                .to_numpy(float) for t in h])
            draw(ax, h, v[:, 0], v[:, 1], v[:, 2], method, **rate_line(lr, color), zorder=3 if lr == DEFAULT_LR else 2)
            if lr in marked:
                offset = (marked.index(lr) + 1) * every // (len(marked) + 1)
                ax.plot(h, v[:, 0], linestyle='', markevery=(offset, every), zorder=4, **rate_markers(lr, color))
        ax.set(ylim=(-4, 104), yticks=[0, 50, 100], xlim=(1, H), xticks=[1, 25, 50, 75, 100])
        ax.set_title(label(method), fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.set_ylabel('Active (%)', fontsize=9)
    center = (fig.subplotpars.left + fig.subplotpars.right) / 2
    fig.text(center, 0.62 / height, 'Training iteration $h$', ha='center', va='center', fontsize=9)
    ink = plt.rcParams['text.color']
    handles = []
    for lr in RATES:
        tone = shade(ink, max(rate_style(lr)['step'], 0.0))
        line = rate_line(lr, ink) | rate_markers(lr, ink) | dict(color=tone, markeredgecolor=tone)
        handles.append(Line2D([], [], label=rate_text(lr) + (' (default)' if lr == DEFAULT_LR else ''), **line))
    fig.legend(handles=handles, title=r'Learning rate ($\times10^{-4}$)', title_fontsize=8, loc='lower center',
               bbox_to_anchor=(center, 0), ncol=len(RATES), frameon=False, fontsize=8, handlelength=3.6)
    return fig


def timing_caption() -> str:
    """Caption of the timing figure with the line styles and markers of the learning rates."""
    names = {'o': 'circles', 's': 'squares', '^': 'triangles', 'D': 'diamonds', 'v': 'inverted triangles'}
    rates = series(f'${rate_text(lr)}$' for lr in RATES)
    styles = series(rate_style(lr)['name'] for lr in RATES)
    markers = series(names[rate_style(lr)['marker']] for lr in RATES if rate_style(lr)['marker'])
    return (f'Pooled active recall during training with the standard populations for the {rate_count()} learning '
            'rates (\\% of the Pareto front). Each panel shows one scalarizer. '
            f'The learning rates {rates}, in units of $10^{{-4}}$, are drawn from light to dark with {styles} lines. '
            f'The default ${rate_text(DEFAULT_LR)}$ has no markers, and the other learning rates are marked with '
            f'{markers}. Lines show means over ten seeds, with equal weight per '
            'environment, and shading indicates pointwise 95\\% seed-bootstrap intervals. The comparison of the '
            'curves is descriptive and does not yield a rule for when to stop training.')


def main() -> None:
    """Writes the E4 tables, the learning-rate figure and the timing figure to <input>/paper."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Tables
    summary = Summary(pd.read_csv(args.input / 'e4_summary.csv', float_precision='round_trip', low_memory=False))
    out = args.input / 'paper'
    (out / 'appendix').mkdir(parents=True, exist_ok=True)
    main_table, appendix = tables(summary)
    (out / 'e4_learning_rate.tex').write_text(main_table, encoding='utf-8')
    (out / 'appendix/e4_tables.tex').write_text(appendix, encoding='utf-8')

    # Figures
    rate_description = ('Line charts with one column per environment and two rows for archive and active recall in '
                        'percent, plotted against the learning rate. Each panel has one line per scalarizer.')
    timing_description = ('Six line charts in two rows, with the raw scalarizers on top and the normalized ones '
                          'below. Each chart shows active recall in percent over the training iterations, with one '
                          'line per learning rate.')
    for envs, path, tag, other in versions('e4_learning_rate', 'fig:e4-learning-rate'):
        save(learning_rate_figure(summary, envs), out / path)
        (out / f'{path[:-4]}_figure.tex').write_text(
            figure_tex(path, learning_rate_caption(envs, other), tag, description=rate_description), encoding='utf-8')
    save(timing_figure(summary), out / 'appendix/e4_timing.pdf')
    (out / 'appendix/e4_timing_figure.tex').write_text(
        figure_tex('appendix/e4_timing.pdf', timing_caption(), 'fig:e4-timing', description=timing_description),
        encoding='utf-8')
    print(f'E4 paper outputs written to {out}')


if __name__ == '__main__':
    main()