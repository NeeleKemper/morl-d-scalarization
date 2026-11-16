"""Exact Pareto fronts of the six DST/LDST environments in one row at the JAIR text width.

Supported returns (on the upper convex hull, optimal for some positive weighted sum) are filled, unsupported returns
open; the dashed line is the hull. Writes the figure and, next to it, <stem>_figure.tex with caption and \\Description.

  python plot_dst_fronts.py --out fig_dst_fronts.pdf
"""
import argparse

import numpy as np
import mo_gymnasium as mo_gym
import matplotlib.pyplot as plt

from typing import Sequence
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from envs import dst_envs
from paper_style import WIDTH, figure_tex, save


# Front geometry

def pareto_filter(points: np.ndarray) -> np.ndarray:
    """Nondominated subset of the points (maximization)."""
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not keep[i]:
            continue
        dominates = np.all(points >= p - 1e-9, axis=1) & np.any(points > p + 1e-9, axis=1)
        if np.any(dominates):
            keep[i] = False
    return points[keep]


def upper_hull(points: np.ndarray) -> np.ndarray:
    """Upper-right convex hull of a 2-D front by increasing first objective; collinear points are kept."""
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    hull = []
    for p in ordered:
        while len(hull) >= 2:
            a = hull[-2]
            b = hull[-1]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross > 1e-9:
                hull.pop()
            else:
                break
        hull.append(p)
    return np.asarray(hull)


def classify_front(front: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supported points, unsupported points and the hull polyline of a 2-D front."""
    hull = upper_hull(front)
    on_hull = np.zeros(len(front), dtype=bool)
    for i, p in enumerate(front):
        on_hull[i] = np.any(np.all(np.isclose(hull, p, atol=1e-8), axis=1))
    return front[on_hull], front[~on_hull], hull


def load_fronts(env_ids: Sequence[str], gamma: float, max_episode_steps: int) -> list[np.ndarray]:
    """Exact nondominated fronts of the environments."""
    dst_envs.register_envs(max_episode_steps=max_episode_steps)
    fronts = []
    for env_id in env_ids:
        env = mo_gym.make(env_id)
        front = np.asarray(env.unwrapped.pareto_front(gamma=gamma), dtype=np.float64)
        env.close()
        fronts.append(pareto_filter(front))
    return fronts


# Figure

def make_figure(fronts: list[np.ndarray],
                titles: Sequence[str],
                annotate_counts: bool,
                width: float,
                height: float) -> plt.Figure:
    """One panel per front in a row; a panel with the time range of its left neighbor shares its y tick labels."""
    left = 0.47
    right = 0.06
    top = 0.39
    bottom = 0.72
    marker_color = '0.15'
    hull_color = '0.45'
    ranges = [(front[:, 1].min(), front[:, 1].max()) for front in fronts]
    shared = [i > 0 and np.allclose(ranges[i], ranges[i - 1]) for i in range(len(fronts))]
    gaps = [0.0 if i == 0 else 0.10 if shared[i] else 0.36 for i in range(len(fronts))]
    panel_width = (width - left - right - sum(gaps)) / len(fronts)
    panel_height = height - top - bottom
    fig = plt.figure(figsize=(width, height))
    axes = []
    x = left
    for gap in gaps:
        x += gap
        axes.append(fig.add_axes([x / width, bottom / height, panel_width / width, panel_height / height]))
        x += panel_width
    for i, (ax, front, title) in enumerate(zip(axes, fronts, titles)):
        supported, unsupported, hull = classify_front(front)
        ax.plot(hull[:, 0], hull[:, 1], linestyle=(0, (4, 3)), linewidth=0.9, color=hull_color, zorder=1)
        ax.scatter(supported[:, 0], supported[:, 1], s=13, facecolors=marker_color, edgecolors=marker_color,
                   linewidths=0.8, zorder=3)
        if len(unsupported):
            ax.scatter(unsupported[:, 0], unsupported[:, 1], s=13, facecolors='white', edgecolors=marker_color,
                       linewidths=0.8, zorder=3)
        ax.set_title(title, fontsize=10, pad=13.0 if annotate_counts else 3.0)
        ax.tick_params(labelsize=7, pad=1.5, length=2.5, labelleft=not shared[i])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0.06, y=0.12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if annotate_counts:
            ax.text(0.0, 1.03, f'{len(unsupported)} of {len(front)} unsupported', transform=ax.transAxes, fontsize=7,
                    color='0.25')
    axes[0].set_ylabel(r'Time $v_2$', fontsize=9, labelpad=2.0)
    center = (left + (width - right)) / 2 / width
    fig.text(center, 0.40 / height, r'Treasure $v_1$', ha='center', va='center', fontsize=9)
    handles = [Line2D([], [], marker='o', linestyle='none', markerfacecolor=marker_color,
                      markeredgecolor=marker_color, markersize=4, label='Supported return'),
               Line2D([], [], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor=marker_color,
                      markersize=4, label='Unsupported return'),
               Line2D([], [], linestyle=(0, (4, 3)), linewidth=0.9, color=hull_color, label='Upper convex hull')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(center, 0), ncol=3, frameon=False, fontsize=8,
               handletextpad=0.4, columnspacing=1.4)
    return fig


def main() -> None:
    """Draws the fronts, writes the figure environment and prints the counts of supported returns."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='results/evaluation/scalarization/fig_dst_fronts.pdf')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--no-counts', action='store_true')
    parser.add_argument('--width', type=float, default=WIDTH, help='figure width in inches (default: JAIR text width)')
    parser.add_argument('--height', type=float, default=1.9,
                        help='figure height in inches; smaller means flatter panels')
    args = parser.parse_args()

    # Figure
    env_ids = ('DST1-v0', 'DST2-v0', 'DST3-v0', 'DST4-v0', 'LDST1-v0', 'LDST2-v0')
    fronts = load_fronts(env_ids, args.gamma, args.max_episode_steps)
    out_path = Path(args.out)
    titles = [env_id.split('-')[0] for env_id in env_ids]
    save(make_figure(fronts, titles, not args.no_counts, args.width, args.height), out_path)

    # LaTeX
    counts = '' if args.no_counts else ('The label above each panel gives the number of unsupported returns among all '
                                        'Pareto returns. ')
    caption = (f'Deterministic Pareto fronts at $\\gamma={args.gamma:g}$. Filled and open markers indicate supported '
               f'and unsupported returns, respectively. Dashed lines show the upper convex hull. {counts}These fronts '
               'define the oracle targets and exact-front recall.')
    description = ('Six scatter plots in a row, one per environment, with treasure on the horizontal axis and time on '
                   'the vertical axis. Filled circles mark supported Pareto returns and open circles unsupported ones. '
                   'A dashed line connects the supported returns along the upper convex hull.')
    wrapper = out_path.with_name(f'{out_path.stem}_figure.tex')
    wrapper.write_text(figure_tex(out_path.name, caption, 'fig:dst_fronts', description=description), encoding='utf-8')
    for env_id, front in zip(env_ids, fronts):
        supported, unsupported, _ = classify_front(front)
        print(f'{env_id:<12} |F*|={len(front):>3}  supported={len(supported):>3}  unsupported={len(unsupported):>3}')
    print(f'wrote {out_path} and {wrapper}')


if __name__ == '__main__':
    main()