"""One E5 appendix table and one appendix horizon figure from the lean summary. The paper says "deterministic
execution" for greedy action selection, and captions follow the wording of the paper.
"""
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from paper_style import (APPENDIX_ENVS, FIGURE_STATS, MAIN_ENVS, POOLED_STATS, cell, draw, envs_text, figure_tex,
                         finish, head, label, mark, panels, save, sd_of, table_tex)
from reconstruct_e5 import DATA_DIR, METHODS, H, REFERENCE_H


class Summary:
    def __init__(self, frame: pd.DataFrame) -> None:
        """E5 summary indexed by kind, environment, scalarizer, metric and iteration."""
        self.frame = frame
        self.index = frame.set_index(['kind', 'environment', 'scalarizer', 'metric', 'h']).sort_index()

    def get(self,
            metric: str,
            method: str,
            h: int,
            kind: str = 'measures',
            env: str = 'ALL') -> pd.Series:
        """Summary row of one measurement."""
        return self.index.loc[(kind, env, method, metric, h)]

    def curve(self, metric: str, method: str, env: str) -> pd.DataFrame:
        """Measurements of one metric over all iterations."""
        return self.index.loc[('measures', env, method, metric)].sort_index()

    def peak(self, method: str) -> pd.Series:
        """Pooled descriptive peak of the active-recall curve and its earliest iteration."""
        rows = self.frame[(self.frame.kind == 'timing') & (self.frame.environment == 'ALL')
                          & (self.frame.scalarizer == method) & (self.frame.metric == 'active_peak')]
        return rows.iloc[0]


def horizon_figure(summary: Summary) -> plt.Figure:
    """Two blocks of three environments (paper environments on top), each with an archive and an active row."""
    rows = (('archive_recall', 'Archive (%)'), ('active_recall', 'Active (%)'))
    blocks = (MAIN_ENVS, APPENDIX_ENVS)
    ylabels = [ylabel for _ in blocks for _, ylabel in rows]
    fig, axes = panels(len(ylabels), MAIN_ENVS, ylabels)
    fig.subplots_adjust(hspace=0.45)
    for b, envs in enumerate(blocks):
        for col, env in enumerate(envs):
            for row, (metric, _) in enumerate(rows):
                ax = axes[2 * b + row, col]
                mark(ax, REFERENCE_H)
                for method in METHODS:
                    curve = 100 * summary.curve(metric, method, env)[['mean', 'ci_low', 'ci_high']]
                    draw(ax, curve.index, curve['mean'], curve['ci_low'], curve['ci_high'], method,
                         markevery=(8 * METHODS.index(method), 50))
                ax.set_xticks([0, 100, 250, 500])
    finish(fig, axes, MAIN_ENVS, ylabels, 'Training iteration $h$', METHODS)
    for col, env in enumerate(APPENDIX_ENVS):
        axes[2, col].set_title(env.upper(), fontsize=10)
    return fig


def horizon_table(summary: Summary) -> str:
    """Recall at the reference and the final iteration, the paired change and the highest active recall."""
    rows = []
    for method in METHODS:
        levels = [cell(row['mean'], sd_of(row)) for metric in ('archive_recall', 'active_recall')
                  for row in (summary.get(metric, method, h) for h in (REFERENCE_H, H))]
        diff = summary.get('active_recall', method, H, kind='comparison')
        delta = f"{100 * diff['mean']:+.1f} [{100 * diff.ci_low:+.1f}, {100 * diff.ci_high:+.1f}]"
        peak = summary.peak(method)
        rows.append([label(method), *levels, delta, f"{cell(peak['mean'])} ({int(peak.h)})"])
    return table_tex(
        'tab:e5-horizon',
        f'Archive and active recall (\\%) of the standard populations trained for $H={H}$ iterations, at '
        f'$h={REFERENCE_H}$ and $h={H}$ of the same runs. $\\Delta$ active shows the paired change from '
        f'$h={REFERENCE_H}$ to $h={H}$ in percentage points, with a 95\\% seed-bootstrap interval. Highest shows the '
        f'maximum of the pooled mean active-recall curve over iterations 1--{H}, with the earliest maximizing '
        f'iteration in parentheses. {POOLED_STATS}',
        'lrrrrrr',
        [r'& \multicolumn{2}{c}{Archive recall} & \multicolumn{2}{c}{Active recall} & & \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         head('Scalarizer', f'$h={REFERENCE_H}$', f'$h={H}$', f'$h={REFERENCE_H}$', f'$h={H}$',
              r'$\Delta$ active [95\% CI]', 'Highest ($h$)')], rows, wide=True)


def main() -> None:
    """Writes the E5 horizon table and figure to <input>/paper and prints the manifest checks."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Data
    manifest = json.loads((args.input / 'e5_manifest.json').read_text())
    summary = Summary(pd.read_csv(args.input / 'e5_summary.csv', low_memory=False, float_precision='round_trip'))
    out = args.input / 'paper'
    out.mkdir(parents=True, exist_ok=True)

    # Output
    save(horizon_figure(summary), out / 'e5_horizon.pdf')
    (out / 'e5_horizon.tex').write_text(horizon_table(summary), encoding='utf-8')
    caption = (f'Archive and active recall of the standard populations trained for $H={H}$ iterations on all six '
               f'environments. {envs_text(MAIN_ENVS)} are shown on top, {envs_text(APPENDIX_ENVS)} below. The '
               f'dotted line marks $h={REFERENCE_H}$. {FIGURE_STATS}')
    description = ('Line charts in two blocks of three environments. In each block, the upper row shows archive '
                   'recall and the lower row active recall in percent over the training iterations, with one line '
                   'per scalarizer.')
    (out / 'e5_horizon_figure.tex').write_text(
        figure_tex('e5_horizon.pdf', caption, 'fig:e5-horizon', description=description), encoding='utf-8')
    (out / 'e5_numbers.json').unlink(missing_ok=True)
    print(f'E5: one table and one figure with its LaTeX wrapper in {out}')
    print('E1 prefix check:', manifest['e1_check'])
    print('Saved schedules:', manifest['lr_decay'])


if __name__ == '__main__':
    main()