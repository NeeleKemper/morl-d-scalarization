"""Paper figure: final returns of one seed on the additional environments without known fronts (MO-Reacher left out),
with the reference front, the HV reference point and the region counted by the HV. MO-PPO and MO-SAC are shown with
the scalarizer of the baseline comparison. Reads the output of build_reference_fronts.py and writes
<figure_output>/paper/appendix/fronts_seed<seed>.pdf and its figure environment fronts_seed<seed>_figure.tex.

  python plot_fronts_paper.py
  python plot_fronts_paper.py --seed 3 --figure_output some/folder
  python plot_fronts_paper.py --ppo_variant tch_norm
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from build_reference_fronts import nondominated
from compute_metrics_baseline import constructed_reference
from paper_style import WIDTH, figure_tex, label, save


# Data

def points_of(run: dict) -> np.ndarray:
    """The run's final front as the metrics evaluate it: rounded and non-dominated."""
    return nondominated(np.round(run['front'], 6) + 0.0) if len(run['front']) else np.empty((0, 0))


def groups(runs: list[dict], variants: dict) -> dict:
    """Points per legend entry: MO-PPO and MO-SAC with the scalarizer of the baseline comparison, and each baseline."""
    out = {}
    for r in runs:
        if r['algorithm'] in variants and r['variant'] != variants[r['algorithm']]:
            continue
        if len(points := points_of(r)):
            out.setdefault(r['algorithm'], []).append(points)
    return {k: np.vstack(v) for k, v in out.items()}


# Figure

def method_styles() -> tuple[dict, dict]:
    """Colors and markers of our methods (grey) and of the baselines, as in plot_learning_curves.py."""
    ours = {'MO-PPO': dict(color='0.45', marker='o'),
            'MO-SAC': dict(color='0.72', marker='s')}
    baselines = {'GPI-LS': dict(color='#e34948', marker='^'),
                 'PCN': dict(color='#6250d6', marker='D'),
                 'IPRO': dict(color='#008300', marker='X'),
                 'C-MORL': dict(color='#2a78d6', marker='v'),
                 'MORL/D': dict(color='#1baf7a', marker='P')}
    return ours, baselines


def panel(ax: plt.Axes,
          data: dict,
          front: np.ndarray,
          ref: np.ndarray,
          x: int,
          y: int,
          title: str) -> None:
    """One projection: box, reference front, our group, baselines and the HV reference point."""
    ours, baselines = method_styles()
    ideal = front.max(0)
    ax.fill_between([ref[x], ideal[x]], ref[y], ideal[y], color='0.35', alpha=0.08, lw=0, zorder=0)
    ax.axvline(ref[x], color='0.35', lw=0.7, ls='--', zorder=1)
    ax.axhline(ref[y], color='0.35', lw=0.7, ls='--', zorder=1)
    for method, kw in ours.items():
        if method in data:
            ax.scatter(data[method][:, x], data[method][:, y], s=6, marker=kw['marker'], color=kw['color'], lw=0,
                       zorder=2)
    ax.scatter(front[:, x], front[:, y], s=9, facecolors='none', edgecolors='black', linewidths=0.5, zorder=3)
    for method, kw in baselines.items():
        if method in data:
            ax.scatter(data[method][:, x], data[method][:, y], s=11, marker=kw['marker'], color=kw['color'],
                       linewidths=0, zorder=4)
    ax.scatter([ref[x]], [ref[y]], s=55, marker='*', color='black', lw=0, zorder=5)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlabel(f'$v_{x + 1}$', fontsize=9, labelpad=1)
    ax.set_ylabel(f'$v_{y + 1}$', fontsize=9, labelpad=1)
    ax.tick_params(labelsize=7, length=2, pad=1.5)
    ax.spines[['top', 'right']].set_visible(False)


def legend(ax: plt.Axes, methods: list[str], variants: dict) -> None:
    """Legend in the free panel of the last row."""
    ours, baselines = method_styles()
    handles = [Line2D([], [], ls='', marker='o', ms=4, markerfacecolor='none', markeredgecolor='black',
                      markeredgewidth=0.6, label='Reference front'),
               Line2D([], [], ls='', marker='*', ms=8, color='black', label='HV reference point'),
               Patch(facecolor='0.35', alpha=0.15, edgecolor='0.35', ls='--', lw=0.7,
                     label='Region counted by the HV')]
    handles += [Line2D([], [], ls='', marker=ours[m]['marker'], ms=3.5, color=ours[m]['color'],
                       label=f'{m} ({label(variants[m])})') for m in ours if m in methods]
    handles += [Line2D([], [], ls='', marker=baselines[m]['marker'], ms=4, color=baselines[m]['color'], label=m)
                for m in baselines if m in methods]
    ax.axis('off')
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(-0.22, 0.5), frameon=False, fontsize=8,
              handletextpad=0.5, labelspacing=0.38)


def figure(references: dict,
           runs: dict,
           variants: dict,
           names: dict) -> plt.Figure:
    """Pairwise projections of every three-objective environment, then the two-objective ones and the legend."""
    envs_3d = [e for e in names if e in references and references[e]['front'].shape[1] == 3]
    envs_2d = [e for e in names if e in references and references[e]['front'].shape[1] == 2]
    rows = len(envs_3d) + 1
    fig, axes = plt.subplots(rows, 3, figsize=(WIDTH, 1.55 * rows + 0.1), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.06, wspace=0.42, hspace=0.62)
    methods = set()
    for r, env in enumerate(envs_3d):
        data = groups(runs[env], variants)
        methods |= set(data)
        for c, (x, y) in enumerate(((0, 1), (0, 2), (1, 2))):
            panel(axes[r, c], data, references[env]['front'], references[env]['ref'], x, y, names[env])
    for c, env in enumerate(envs_2d):
        data = groups(runs[env], variants)
        methods |= set(data)
        panel(axes[-1, c], data, references[env]['front'], references[env]['ref'], 0, 1, names[env])
    for c in range(len(envs_2d), 3):
        if c == 2:
            legend(axes[-1, c], sorted(methods), variants)
        else:
            axes[-1, c].axis('off')
    return fig


def macro(variant: str) -> str:
    """The paper's macro of a scalarizer, e.g. ws_raw -> \\wsraw{}."""
    return '\\' + variant.replace('_', '') + '{}'


def caption(seed: int, variants: dict) -> str:
    """Figure caption for the seed and the shown scalarizers."""
    return (f'Final returns of seed {seed} on the additional environments without known fronts, except MO-Reacher '
            'with four objectives. Open circles show the reference front, the star shows the HV reference point, and '
            'dashed lines mark its coordinates. A return contributes to the HV only if it lies beyond the dashed lines '
            'in every objective. For environments with three objectives, each row shows the three pairwise '
            'projections, so a contributing return lies inside the shaded region in all three panels of its row. '
            f"Grey circles and squares show the archives of MO-PPO with {macro(variants['MO-PPO'])} and MO-SAC "
            f"with {macro(variants['MO-SAC'])}, as in the comparison with established methods. The reference front "
            'also contains the returns of the other configurations. The seed was fixed before the results were '
            'examined.')


def main() -> None:
    """Draws the front figure of one seed and writes it with its figure environment."""
    # Arguments
    names = dict(minecart='Minecart', ant='MO-Ant', hopper='MO-Hopper', halfcheetah='MO-HalfCheetah',
                 humanoid='MO-Humanoid')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ppo_variant', default='ws_raw',
                        choices=('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm'))
    parser.add_argument('--sac_variant', default='ws_raw', choices=('ws_raw', 'ws_norm'))
    parser.add_argument('--figure_output', type=Path, default=Path('results/evaluation/scalarization/front_plots'))
    parser.add_argument('--envs', nargs='+', default=list(names),
                        help="Default: all environments of the figure; 'mujoco' stands for its variants")
    parser.add_argument('--reference_fronts', type=Path, nargs='+',
                        default=[Path('results/evaluation/scalarization/reference_fronts'),
                                 Path('results/evaluation/scalarization/reference_fronts_mujoco')],
                        help='Output folders of build_reference_fronts.py, searched in this order')
    args = parser.parse_args()
    mujoco = ('ant', 'halfcheetah', 'hopper', 'humanoid')
    args.envs = [env for name in args.envs for env in (mujoco if name == 'mujoco' else (name,))]
    args.seeds = [args.seed]
    variants = {'MO-PPO': args.ppo_variant, 'MO-SAC': args.sac_variant}

    # Data
    references = {}
    runs = {}
    for env in args.envs:
        if env not in names:
            print(f'{env}: not part of this figure, skipped')
            continue
        reference, runs[env] = constructed_reference(args, env)
        references[env] = dict(front=reference['front'], ref=np.asarray(reference['hv_reference_point'], float))

    # Output
    appendix = args.figure_output / 'paper' / 'appendix'
    save(figure(references, runs, variants, names), appendix / f'fronts_seed{args.seed}.pdf')
    description = ('Grid of scatter plots of two objectives each. Every panel shows the reference front, the final '
                   'returns of all methods for one seed, the HV reference point, and the region in which returns '
                   'contribute to the hypervolume.')
    (appendix / f'fronts_seed{args.seed}_figure.tex').write_text(
        figure_tex(f'appendix/fronts_seed{args.seed}.pdf', caption(args.seed, variants), 'fig:fronts',
                   description=description), encoding='utf-8')
    print(f'Output: {appendix}')


if __name__ == '__main__':
    main()