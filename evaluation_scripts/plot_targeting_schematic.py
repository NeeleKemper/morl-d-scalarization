"""Schematic figure for the introduction: which returns a finite preference grid targets, and where policies end up.

Two panels show the same toy front with ten Pareto returns and five evenly spaced preferences; straight arrows connect
every preference with the return it targets under a weighted sum (left) and Tchebycheff scalarization (right). The
curved learning paths and policy returns are NOT computed: they are set by hand in make_figure to illustrate that a
policy may or may not reach its target. With --no-policies, the figure shows the preferences and their targets only.

  python plot_targeting_schematic.py
  python plot_targeting_schematic.py --no-policies --out fig_targeting_plain.pdf
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.path import Path as CurvePath
from matplotlib.legend import Legend
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Transform
from matplotlib.legend_handler import HandlerBase

from paper_style import FAMILY_COLORS, WIDTH, figure_tex, save


# Geometry

def supported_mask(front: np.ndarray) -> np.ndarray:
    """True for returns on the upper convex hull of a two-objective maximization front."""
    hull = []
    for i in np.argsort(front[:, 0]):
        while len(hull) >= 2:
            a = front[hull[-2]]
            b = front[hull[-1]]
            p = front[i]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9:
                hull.pop()
            else:
                break
        hull.append(i)
    mask = np.zeros(len(front), dtype=bool)
    mask[hull] = True
    return mask


def scores(front: np.ndarray, weight: np.ndarray, scalarizer: str) -> np.ndarray:
    """Scalarized scores of all returns (higher is better), as defined in the paper."""
    if scalarizer == 'ws':
        return front @ weight
    ideal = front.max(0)
    active = weight > 0
    return -np.max(weight[active] * (ideal[active] - front[:, active]), axis=1)


def targets(front: np.ndarray, weights: np.ndarray, scalarizer: str) -> list[int]:
    """Index of the return every preference targets."""
    return [int(np.argmax(scores(front, weight, scalarizer))) for weight in weights]


def policy_return(end: str | int | tuple, front: np.ndarray, target: int) -> tuple[np.ndarray, bool]:
    """Illustrative return of a policy ('target', an index of the front, or a dominated pair) and whether it is on
    target."""
    if isinstance(end, str):
        return front[target], True
    if isinstance(end, int):
        return front[end], end == target
    return np.asarray(end, dtype=float), False


def learning_path(ax: plt.Axes,
                  points: list,
                  clear_start_pt: float,
                  clear_end_pt: float) -> np.ndarray:
    """Catmull-Rom curve through the points in display space, stopping short of both ends by the given points."""
    scale = ax.figure.dpi / 72.0
    p = ax.transData.transform(np.asarray(points, dtype=float))
    padded = np.vstack([p[0], p, p[-1]])
    t = np.linspace(0.0, 1.0, 60, endpoint=False)[:, None]
    pieces = []
    for i in range(len(p) - 1):
        p0 = padded[i]
        p1 = padded[i + 1]
        p2 = padded[i + 2]
        p3 = padded[i + 3]
        pieces.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                             + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    curve = np.vstack([*pieces, p[-1]])
    keep = ((np.hypot(*(curve - p[0]).T) >= clear_start_pt * scale)
            & (np.hypot(*(curve - p[-1]).T) >= clear_end_pt * scale))
    return ax.transData.inverted().transform(curve[keep])


# Legend

class WavyArrowHandler(HandlerBase):
    def create_artists(self,
                       legend: Legend,
                       orig_handle: object,
                       xdescent: float,
                       ydescent: float,
                       width: float,
                       height: float,
                       fontsize: float,
                       trans: Transform) -> list[Line2D]:
        """Legend entry of a learning path: a short curved line with an arrow head."""
        t = np.linspace(0.0, 1.0, 60)
        x = -xdescent + t * (width - 3.0)
        y = -ydescent + height / 2 + 0.30 * height * np.sin(3 * np.pi * t) * np.sin(np.pi * t) ** 0.8
        line = Line2D(x, y, color='0.25', linewidth=1.0, transform=trans)
        head = Line2D([-xdescent + width - 1.5], [-ydescent + height / 2], marker='>', markersize=3.8,
                      color='0.25', linestyle='none', transform=trans)
        return [line, head]


class LearningPath:
    """Placeholder handle for the legend entry drawn by WavyArrowHandler."""


# Figure

def make_figure(show_policies: bool, preferences: int) -> tuple[plt.Figure, dict]:
    """Both panels with front, hull, preferences and targets, optionally with the hand-set learning paths."""
    front = np.array([(1.0, 9.6), (2.2, 9.0), (3.0, 7.2), (3.7, 5.8), (4.5, 5.3),
                      (5.7, 4.85), (6.9, 4.0), (8.0, 3.4), (8.8, 2.3), (9.5, 0.8)])
    policies = {
        'ws': {1: dict(via=[(0.55, 3.0), (0.38, 6.2)], end='target'),
               2: dict(via=[(2.58, 4.42)], end=1),
               3: dict(via=[(2.92, 4.08), 5, 6], end='target'),
               4: dict(via=[(3.65, 3.35)], end=(6.4, 2.7)),
               5: dict(via=[(3.2, 0.75), (6.4, 0.35)], end='target')},
        'tch': {1: dict(via=[(0.55, 3.0), (0.38, 6.2)], end='target'),
                2: dict(via=[(1.75, 5.25)], end=1),
                3: dict(via=[(2.75, 4.25), 4], end='target'),
                4: dict(via=[(4.7, 2.3)], end=(7.2, 2.4)),
                5: dict(via=[(3.2, 0.75), (6.4, 0.35)], end='target')}}
    label_offsets = {('ws', 4): (0.38, 0.0, 'left', 'center'), ('tch', 3): (0.05, -0.42, 'center', 'top'),
                     ('tch', 4): (0.38, -0.10, 'left', 'top')}
    marker_color = '0.15'
    hull_color = '0.45'
    path_color = '0.25'
    text_color = plt.rcParams['text.color']
    first = np.linspace(0.0, 1.0, preferences)
    weights = np.column_stack([first, 1.0 - first])
    anchors = np.linspace((1.4, 5.6), (5.0, 2.0), preferences)
    supported = supported_mask(front)
    hull = front[supported][np.argsort(front[supported][:, 0])]
    height = 3.10 if show_policies else 2.85
    bottom = 0.84 if show_policies else 0.62
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.35 / WIDTH, right=1 - 0.10 / WIDTH, top=1 - 0.46 / height, bottom=bottom / height,
                        wspace=0.10)
    counts = {}
    for ax, (scalarizer, title) in zip(axes, (('ws', '(a) Weighted sum'), ('tch', '(b) Tchebycheff'))):
        color = FAMILY_COLORS[scalarizer]
        chosen = targets(front, weights, scalarizer)
        ringed = sorted(set(chosen))
        counts[scalarizer] = len(ringed)
        ax.set(xlim=(0, 10.8), ylim=(0, 10.8), xticks=[], yticks=[])
        ax.plot(hull[:, 0], hull[:, 1], linestyle=(0, (4, 3)), linewidth=0.9, color=hull_color, zorder=1)
        ax.plot(anchors[:, 0], anchors[:, 1], linewidth=0.6, color='0.6', zorder=1)
        ax.scatter(anchors[:, 0], anchors[:, 1], s=7, color=marker_color, zorder=3)
        for k, (anchor, index) in enumerate(zip(anchors, chosen), start=1):
            ax.annotate('', xy=front[index], xytext=anchor, zorder=2,
                        arrowprops=dict(arrowstyle='-|>', color=color, linewidth=0.9, mutation_scale=7,
                                        shrinkA=2.5, shrinkB=4.6))
            ax.text(anchor[0] - 0.28, anchor[1] - 0.28, f'$\\mathbf{{w}}_{k}$', ha='right', va='top', fontsize=9)
        ax.scatter(*front[ringed].T, s=60, facecolors='none', edgecolors=color, linewidths=1.0, zorder=4)
        ax.scatter(*front[supported].T, s=13, facecolors=marker_color, edgecolors=marker_color, linewidths=0.8,
                   zorder=5)
        ax.scatter(*front[~supported].T, s=13, facecolors='white', edgecolors=marker_color, linewidths=0.8, zorder=5)
        if show_policies:
            groups = {}
            for k, target in enumerate(chosen, start=1):
                point, _ = policy_return(policies[scalarizer][k]['end'], front, target)
                groups.setdefault(tuple(point), []).append(k)
                on_front = bool(np.any(np.all(np.isclose(front, point), axis=1)))
                has_ring = bool(np.any(np.all(np.isclose(front[ringed], point), axis=1)))
                if not on_front:
                    ax.scatter(*point, s=13, facecolors='0.65', edgecolors=marker_color, linewidths=0.8, zorder=5)
                via = [front[v] if isinstance(v, int) else v for v in policies[scalarizer][k]['via']]
                curve = learning_path(ax, [(0.0, 0.0), *via, point], clear_start_pt=2.0,
                                      clear_end_pt=5.6 if has_ring else 3.4)
                ax.add_patch(FancyArrowPatch(path=CurvePath(curve), arrowstyle='-|>', mutation_scale=7,
                                             color=path_color, linewidth=1.0, fill=False, zorder=3))
            for point, members in groups.items():
                dx, dy, ha, va = label_offsets.get((scalarizer, members[0]), (0.42, 0.30, 'left', 'bottom'))
                text = '$' + ',\\,'.join(f'\\pi_{{\\mathbf{{w}}_{k}}}' for k in members) + '$'
                ax.text(point[0] + dx, point[1] + dy, text, ha=ha, va=va, fontsize=9)
        ax.set_title(title, fontsize=10, pad=15)
        ax.text(0.5, 1.02, f'{preferences} preferences target {counts[scalarizer]} of {len(front)} Pareto returns',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.plot(1, 0, '>', color=text_color, markersize=4, transform=ax.transAxes, clip_on=False)
        ax.plot(0, 1, '^', color=text_color, markersize=4, transform=ax.transAxes, clip_on=False)
        ax.set_xlabel('Objective 1', fontsize=9, labelpad=3)
    axes[0].set_ylabel('Objective 2', fontsize=9, labelpad=2)
    marker = dict(marker='o', linestyle='none', markersize=4, markeredgewidth=0.8, markeredgecolor=marker_color)
    handles = [Line2D([], [], markerfacecolor=marker_color, label='Supported Pareto return', **marker),
               Line2D([], [], markerfacecolor='white', label='Unsupported Pareto return', **marker),
               Line2D([], [], linestyle=(0, (4, 3)), linewidth=0.9, color=hull_color, label='Upper convex hull'),
               Line2D([], [], marker='o', linestyle='none', markersize=7.5, markeredgewidth=1.0, markerfacecolor='none',
                      markeredgecolor=marker_color, label='Targeted by a preference')]
    labels = [handle.get_label() for handle in handles]
    if show_policies:
        handles.insert(2, Line2D([], [], markerfacecolor='0.65', **marker))
        labels.insert(2, 'Dominated return')
        handles.append(LearningPath())
        labels.append('Learning path of policy $\\pi_{\\mathbf{w}_k}$ to its return (illustrative)')
    center = (fig.subplotpars.left + fig.subplotpars.right) / 2
    fig.legend(handles, labels, ncol=3 if show_policies else len(handles), loc='lower center',
               bbox_to_anchor=(center, 0.0), frameon=False, fontsize=8, handlelength=2.2,
               columnspacing=1.2 if show_policies else 0.9, handletextpad=0.5,
               handler_map={LearningPath: WavyArrowHandler()})
    return fig, counts


def main() -> None:
    """Draws the schematic and writes it with its figure environment."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, default=Path('results/evaluation/scalarization/fig_targeting.pdf'))
    parser.add_argument('--no-policies', action='store_true',
                        help='show only the preferences and their targets, without the illustrative learning paths')
    parser.add_argument('--preview', type=Path, default=None, help='optional PNG copy for a quick look')
    args = parser.parse_args()

    # Figure
    preferences = 5
    show_policies = not args.no_policies
    fig, counts = make_figure(show_policies, preferences)
    if args.preview:
        fig.savefig(args.preview, dpi=200)
    save(fig, args.out)

    # LaTeX
    numbers = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')
    words = {name: numbers[n] if n < len(numbers) else str(n) for name, n in counts.items()}
    k_word = numbers[preferences] if preferences < len(numbers) else str(preferences)
    policies = ('Curved arrows sketch illustrative learning paths of the policies $\\pi_{\\mathbf w_k}$. A path can '
                'pass through other Pareto returns, as reported for weighted sums~\\cite{van2014learning}, and can '
                'end away from its target. ') if show_policies else ''
    caption = ('Schematic example of a finite preference grid with two maximized objectives. Filled and open markers '
               'indicate supported and unsupported Pareto returns, and the dashed line shows the upper convex hull. '
               f'Straight arrows connect {k_word} evenly spaced preferences with the returns they target. {policies}'
               '(a) A weighted sum targets only supported returns, and several preferences share a target. '
               '(b) Tchebycheff scalarization also targets unsupported returns, but other returns remain untargeted.')
    description = ('Two scatter plots of the same front with ten returns that falls from the upper left to the lower '
                   'right. Five labelled preferences on a diagonal line below the front are connected by straight '
                   f"arrows with the returns they target. With the weighted sum, the arrows end at {words['ws']} "
                   f"supported returns. With Tchebycheff scalarization, they end at {words['tch']} returns, including "
                   'unsupported ones.')
    if show_policies:
        description += (' In each plot, five curved arrows lead from the origin to the returns of the policies '
                        'trained with the five preferences. Three arrows end at the targeted return. The other two '
                        'end at another return of the front or at a dominated return below the front. In each plot, '
                        'one curved arrow passes through other returns of the front on its way.')
    wrapper = args.out.with_name(args.out.stem + '_figure.tex')
    wrapper.write_text(figure_tex(args.out.name, caption, 'fig:targeting', description=description), encoding='utf-8')
    print(f'wrote {args.out} and {wrapper}: ' + ', '.join(f'{name} targets {n} returns' for name, n in counts.items()))


if __name__ == '__main__':
    main()