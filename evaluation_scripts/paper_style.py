"""Shared publication style of the paper's figures and tables (JAIR, acmart acmlarge: one column, text width 6.26in,
Linux Libertine). Figures stay readable in monochrome: families differ in marker and lightness, raw is dashed and
hollow, normalized solid and filled. Fonts come from fonts/ (make_ttf_fonts.py) or the installed fonts."""
import sys
import matplotlib

import numpy as np
import pandas as pd

from typing import Any, Sequence
from pathlib import Path

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgba
from matplotlib.lines import Line2D


# Shared constants

MAIN_ENVS = ('dst1', 'dst4', 'ldst2')
APPENDIX_ENVS = ('dst2', 'dst3', 'ldst1')
WIDTH = 6.26
COLUMN = 3.4
FAMILY_COLORS = {'ws': '#184f95', 'tch': '#e34948', 'stch': '#eda100'}
FAMILY_MARKERS = {'ws': 'o', 'tch': 's', 'stch': '^'}
SHARE_CMAP = LinearSegmentedColormap.from_list('share', ['#ffffff', '#cde2fb', '#6da7ec', '#256abf', '#0d366b'])
OTHER_HATCH = '//////'
FIGURE_STATS = 'Lines show means over ten seeds, with shading indicating pointwise 95\\% seed-bootstrap intervals.'
POOLED_STATS = ('Results are averaged over ten seeds, with equal weight per environment. $\\pm$ indicates the '
                'standard deviation of seed means.')
SEED_STATS = 'Results are reported as means $\\pm$ standard deviations over ten seeds.'


# Fonts

def setup_fonts() -> None:
    """Linux Libertine for text and math, embedded as TrueType; a Times-style fallback with a notice otherwise."""
    for path in sorted(Path(__file__).with_name('fonts').glob('*.ttf')):
        font_manager.fontManager.addfont(path)
    serif = ['Linux Libertine', 'Linux Libertine O', 'Libertinus Serif', 'Times New Roman', 'Liberation Serif',
             'STIXGeneral']
    installed = {font.name for font in font_manager.fontManager.ttflist}
    text_font = next((name for name in serif if name in installed), 'STIXGeneral')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': serif, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'hatch.linewidth': 0.5})
    if text_font.startswith(('Linux Libertine', 'Libertinus')):
        plt.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.rm': text_font,
                             'mathtext.it': f'{text_font}:italic', 'mathtext.bf': f'{text_font}:bold',
                             'mathtext.cal': 'cmsy10', 'mathtext.fallback': 'stix'})
    else:
        plt.rcParams.update({'mathtext.fontset': 'stix'})
        print(f'paper_style: Linux Libertine is not installed; figures use {text_font}.', file=sys.stderr)
    if text_font == 'Linux Libertine O':
        print('paper_style: using the OpenType files of Linux Libertine; run make_ttf_fonts.py for a clean TrueType '
              'embedding.', file=sys.stderr)


setup_fonts()


# Figures

def label(method: str) -> str:
    """Name of a scalarizer as in the text of the paper, e.g. WS-raw."""
    family, scale = method.split('_')
    return f'{family.upper()}-{scale}'


def style(method: str, **extra: Any) -> dict:
    """Color, lightness and marker by family; dashed and hollow for raw, solid and filled for normalized."""
    family, scale = method.split('_')
    color = FAMILY_COLORS[family]
    return dict(color=color, linestyle='--' if scale == 'raw' else '-', marker=FAMILY_MARKERS[family], markersize=3.5,
                linewidth=1.1, markeredgewidth=0.9, markerfacecolor='white' if scale == 'raw' else color) | extra


def ordered_colors(n: int) -> np.ndarray:
    """n colors for ordered categories from light to dark; more than ten are interpolated."""
    steps = np.array([to_rgba(color) for color in ('#84bbf4', '#ca9a1a', '#cd770e', '#c8522b', '#b6304f', '#971d65',
                                                   '#6b1c7b', '#401a7a', '#1a1765', '#081241')])
    if n <= len(steps):
        return steps[np.linspace(0, len(steps) - 1, n).round().astype(int)]
    return LinearSegmentedColormap.from_list('ordered', steps)(np.linspace(0, 1, n))


def envs_text(envs: Sequence[str]) -> str:
    """Environments as an enumeration in the text, e.g. DST1, DST4, and LDST2."""
    return ', '.join(e.upper() for e in envs[:-1]) + f', and {envs[-1].upper()}'


def versions(stem: str, tag: str) -> tuple[tuple, tuple]:
    """Paper and appendix version of a figure: environments, PDF path in the paper folder, label, other version."""
    return ((MAIN_ENVS, f'{stem}.pdf', tag,
             f'Results for {envs_text(APPENDIX_ENVS)} are shown in Appendix Fig.~\\ref{{{tag}-other}}.'),
            (APPENDIX_ENVS, f'appendix/{stem}_other.pdf', f'{tag}-other',
             f'Results for {envs_text(MAIN_ENVS)} are shown in Fig.~\\ref{{{tag}}}.'))


def panels(n_rows: int, envs: Sequence[str], ylabels: Sequence[str]) -> tuple[plt.Figure, np.ndarray]:
    """Metric rows by environment columns at text width, with room for one x label and one legend row."""
    height = 0.95 * n_rows + 0.97
    fig, axes = plt.subplots(n_rows, len(envs), figsize=(WIDTH, height), squeeze=False, sharex='col', sharey=True)
    left = (0.68 if any('\n' in y for y in ylabels) else 0.57) / WIDTH
    fig.subplots_adjust(left=left, right=0.99, top=1 - 0.27 / height, bottom=0.70 / height, wspace=0.12, hspace=0.22)
    return fig, axes


def draw(ax: plt.Axes,
         x: np.ndarray,
         mean: np.ndarray,
         low: np.ndarray,
         high: np.ndarray,
         method: str,
         **extra: Any) -> None:
    """One scalarizer in percent: its mean line and the pointwise interval as shading."""
    kw = style(method, **extra)
    ax.fill_between(x, low, high, color=kw['color'], alpha=0.15, linewidth=0)
    ax.plot(x, mean, **kw)


def mark(ax: plt.Axes, x: float) -> None:
    """Dotted vertical line at the default setting."""
    ax.axvline(x, linestyle=':', linewidth=0.8, color=plt.rcParams['text.color'], alpha=0.5)


def finish(fig: plt.Figure,
           axes: np.ndarray,
           envs: Sequence[str],
           ylabels: Sequence[str],
           xlabel: str,
           methods: Sequence[str]) -> None:
    """Shared axes, environment titles, row labels, one x label, and one legend row centered under the panels."""
    for (r, c), ax in np.ndenumerate(axes):
        ax.set(ylim=(-4, 104), yticks=[0, 50, 100])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(labelsize=8, labelbottom=r == axes.shape[0] - 1, labelleft=c == 0)
        if r == 0:
            ax.set_title(envs[c].upper(), fontsize=10)
        if c == 0:
            ax.set_ylabel(ylabels[r], fontsize=9)
    center = (fig.subplotpars.left + fig.subplotpars.right) / 2
    height = fig.get_figheight()
    fig.text(center, 0.32 / height, xlabel, ha='center', va='center', fontsize=9)
    fig.legend(handles=[Line2D([], [], label=label(m), **style(m)) for m in methods], ncol=len(methods),
               loc='lower center', bbox_to_anchor=(center, 0), frameon=False, fontsize=8, handlelength=2.2,
               columnspacing=1.2, handletextpad=0.5)


def outcome_map(train: pd.DataFrame,
                envs: Sequence,
                rows: Sequence[tuple[str, dict]],
                x: str,
                xlabel: str,
                x_max: int = 100,
                row_height: float = 0.8,
                axis_name: str = 'Treasure') -> plt.Figure:
    """Heatmaps of the mean share of actors per greedy outcome over training: rows by environment columns or blocks."""
    blocks = [envs] if isinstance(envs[0], str) else list(envs)
    gap = 0.35 * 0.8 / row_height
    ratios = [h for b in range(len(blocks)) for h in [gap] * (b > 0) + [1] * len(rows)]
    height = row_height * sum(ratios) + 1.4
    left = 0.86 / WIDTH
    fig = plt.figure(figsize=(WIDTH, height))
    grid = fig.add_gridspec(len(ratios), max(map(len, blocks)), height_ratios=ratios, left=left, right=0.99,
                            top=1 - 0.28 / height, bottom=0.95 / height, wspace=0.27, hspace=0.20)
    for b, block in enumerate(blocks):
        for c, env in enumerate(block):
            data = train.loc[train.environment.eq(env)]
            front = data.loc[data.kind.eq('pareto')].drop_duplicates('outcome').sort_values('J0')
            outcomes = ['zero', 'off_front', *front.outcome]
            first_label = min(len(front) - 1, max(0, int(np.ceil(len(outcomes) * 0.8 / (6 * row_height))) - 1))
            ticks = np.unique(np.linspace(first_label, len(front) - 1, 4 if row_height >= 0.75 else 3).round()
                              .astype(int))
            for r, (row_label, selection) in enumerate(rows):
                ax = fig.add_subplot(grid[b * (len(rows) + 1) + r, c])
                g = data
                for column, value in selection.items():
                    g = g.loc[g[column].eq(value)]
                matrix = (g.pivot(index='outcome', columns=x, values='share')
                          .reindex(index=outcomes, columns=range(x_max + 1)).to_numpy(float))
                image = ax.imshow(matrix, origin='lower', interpolation='nearest', aspect='auto', cmap=SHARE_CMAP,
                                  norm=PowerNorm(0.5, vmin=0, vmax=1),
                                  extent=(-0.5, x_max + 0.5, -0.5, len(outcomes) - 0.5))
                ax.axhline(1.5, linewidth=0.6, color='0.6')
                ax.set_yticks([0, 1, *(ticks + 2)], ['none', 'other', *[f'{front.J0.iloc[i]:g}' for i in ticks]])
                ax.tick_params(labelsize=7, length=2)
                ax.get_yticklabels()[0].set_verticalalignment('top')
                ax.get_yticklabels()[1].set_verticalalignment('bottom')
                ax.set_xticks(np.linspace(0, x_max, 5))
                if r < len(rows) - 1 or b < len(blocks) - 1:
                    ax.tick_params(labelbottom=False)
                if r == 0:
                    ax.set_title(env.upper(), fontsize=10)
                if c == 0:
                    ax.set_ylabel(f'{row_label}\n{axis_name}' if axis_name else row_label, fontsize=8)
    fig.text((left + 0.99) / 2, 0.68 / height, xlabel, ha='center', va='center', fontsize=9)
    bar = fig.colorbar(image, cax=fig.add_axes([0.30, 0.40 / height, 0.48, 0.11 / height]), orientation='horizontal',
                       ticks=[0, 0.1, 0.25, 0.5, 1])
    bar.set_label('Mean share of actors (square-root color scale)', fontsize=8)
    bar.ax.tick_params(labelsize=7)
    return fig


def save(fig: plt.Figure, path: Path) -> None:
    """Saves the figure with the calling script as creator and closes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'Creator': Path(sys.argv[0]).name})
    plt.close(fig)


# Tables

def cell(mean: float | None,
         sd: float | None = None,
         scale: float = 100.0,
         digits: int = 1) -> str:
    """Table entry in the table's unit (percent by default): the mean, or mean +- standard deviation over seeds."""
    if mean is None or not np.isfinite(mean):
        return '--'
    text = f'{scale * mean:.{digits}f}'
    if sd is None or not np.isfinite(sd):
        return text
    return text + f'{{\\scriptsize$\\pm${scale * sd:.{digits}f}}}'


def sd_of(row: pd.Series) -> float:
    """Standard deviation over seeds stored with a summary row."""
    return float(row['std'])


def head(*cells: str) -> str:
    """Header line of a table."""
    return ' & '.join(cells) + r' \\'


def table_tex(label_name: str,
              caption: str,
              align: str,
              header: Sequence[str],
              rows: Sequence,
              wide: bool = False,
              note: str = '') -> str:
    """booktabs table with rows as cell lists or literal lines; wide selects the smaller font, note goes below."""
    body = [row if isinstance(row, str) else ' & '.join(map(str, row)) + r' \\' for row in rows]
    below = [r'\par\smallskip', r'\begin{minipage}{\linewidth}', note, r'\end{minipage}'] if note else []
    return '\n'.join([f'% Generated by {Path(sys.argv[0]).name}. Do not edit by hand. Requires booktabs.',
                      r'\begin{table}[t]', r'\centering', f'\\caption{{{caption}}}', f'\\label{{{label_name}}}',
                      r'\footnotesize' if wide else r'\small', r'\setlength{\tabcolsep}{3pt}',
                      f'\\begin{{tabular}}{{{align}}}', r'\toprule', *header, r'\midrule', *body, r'\bottomrule',
                      r'\end{tabular}', *below, r'\end{table}', ''])


def figure_tex(path: str,
               caption: str,
               label_name: str,
               wide: bool = True,
               description: str = '') -> str:
    """Figure at text width (wide) or natural size with the plain-text \\Description that acmart expects."""
    size = r'[width=\textwidth]' if wide else ''
    alt = f'\\Description{{{description}}}' if description else '% TODO: \\Description{plain-text description}'
    return (f'% Generated by {Path(sys.argv[0]).name}. Do not edit by hand. Paths are relative to the paper folder.\n'
            f'\\begin{{figure}}[t]\n\\centering\n\\includegraphics{size}{{{path}}}\n'
            f'\\caption{{{caption}}}\n\\label{{{label_name}}}\n{alt}\n\\end{{figure}}\n')