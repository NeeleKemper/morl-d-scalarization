"""Fixed E2 paper outputs: one main table and one main figure, five appendix tables and two appendix figures. The paper
says "deterministic execution" for greedy action selection; captions use that term and G/S as column abbreviations.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D

from paper_style import (APPENDIX_ENVS, FAMILY_COLORS, FIGURE_STATS, MAIN_ENVS, POOLED_STATS, SEED_STATS, WIDTH, cell,
                         draw, figure_tex, finish, label, mark, panels, save, sd_of, table_tex, versions)
from reconstruct_e2 import DATA_DIR, DEFAULT_ENTROPY, ENVS, ENTROPIES, H, METHODS, read_json


# Data

class Summary:
    def __init__(self, directory: Path) -> None:
        """Summary measurements at h of every entropy, environment, scalarizer, collection and execution rule."""
        frame = pd.read_csv(directory / 'e2_summary.csv', float_precision='round_trip')
        measures = frame[frame.kind.eq('measures') & frame.h.eq(H)]
        self.data = measures.set_index(['entropy_coef', 'environment', 'scalarizer', 'collection', 'execution',
                                        'metric'])

    def get(self,
            method: str,
            collection: str,
            execution: str,
            metric: str,
            environment: str = 'ALL',
            entropy: float = DEFAULT_ENTROPY) -> pd.Series:
        """Summary row of one measurement."""
        return self.data.loc[(entropy, environment, method, collection, execution, metric)]

    def cell(self,
             method: str,
             collection: str,
             execution: str,
             metric: str,
             environment: str = 'ALL',
             entropy: float = DEFAULT_ENTROPY) -> str:
        """Mean +- standard deviation over the ten seeds, in the table's unit."""
        row = self.get(method, collection, execution, metric, environment, entropy)
        scale, digits = {'hv': (1.0, 2), 'checkpoint_age_mean': (1.0, 1)}.get(metric, (100.0, 1))
        return cell(row['mean'], sd_of(row), scale, digits)


class Diagnostics:
    def __init__(self, directory: Path) -> None:
        """Active actors at h with their mean sampled return: beyond (not weakly dominated by the deterministic
        front), single (equal to one Pareto return) and interior (both preference weights positive)."""
        manifest = read_json(directory / 'e2_manifest.json')
        self.fronts = {env: np.asarray(ref['front'], float) for env, ref in manifest['references'].items()}
        atol = float(manifest['protocol']['match_atol'])
        frame = pd.read_csv(directory / 'e2_policy_diagnostics.csv', float_precision='round_trip')
        actors = frame[frame.collection.eq('population') & frame.h.eq(H)].copy()
        actors['interior'] = (actors.actor_row > 0) & (actors.actor_row < actors.K - 1)
        means = actors[['mean_J0', 'mean_J1']].to_numpy(float)
        single = np.zeros(len(actors), bool)
        beyond = np.zeros(len(actors), bool)
        for env, front in self.fronts.items():
            rows = actors.environment.eq(env).to_numpy()
            gap = front[None, :, :] - means[rows, None, :]
            single[rows] = (np.abs(gap) <= atol).all(axis=2).any(axis=1)
            beyond[rows] = ~(gap >= -atol).all(axis=2).any(axis=1)
        actors['single'] = single
        actors['beyond'] = beyond
        self.actors = actors

    def select(self, entropy: float = DEFAULT_ENTROPY) -> pd.DataFrame:
        """Actors with interior preferences that were trained with one entropy coefficient."""
        return self.actors[self.actors.interior & np.isclose(self.actors.entropy_coef, entropy)]

    def cell(self, method: str, share: str, entropy: float) -> str:
        """Share of actors in percent: environments weighted equally within seed, mean +- sd over the seed means."""
        actors = self.select(entropy)
        runs = actors[actors.scalarizer.eq(method)].groupby(['seed', 'environment'])[share].mean()
        seeds = runs.groupby('seed').mean()
        return cell(seeds.mean(), seeds.std(ddof=1), 100.0, 1)


# Tables

def execution_table(summary: Summary, diagnostics: Diagnostics) -> str:
    """Main-text table: the active actors under both execution rules, without and with the default entropy bonus."""
    entropies = (0.0, DEFAULT_ENTROPY)
    rows = [[label(method)] + [text for entropy in entropies for text in (
                summary.cell(method, 'population', 'deterministic', 'coverage_q050', entropy=entropy),
                summary.cell(method, 'population', 'sampled', 'coverage_q050', entropy=entropy),
                summary.cell(method, 'population', 'sampled', 'greedy_exact_rate', entropy=entropy),
                diagnostics.cell(method, 'beyond', entropy))]
            for method in METHODS]
    groups = ' & '.join(rf'\multicolumn{{4}}{{c}}{{Entropy coefficient ${entropy:g}$' +
                        (' (default)' if entropy == DEFAULT_ENTROPY else '') + '}' for entropy in entropies)
    return table_tex('tab:e2-execution',
                     'Active policy sets at $h=100$ under deterministic and sampled execution (\\%), trained '
                     f'without an entropy bonus and with the default coefficient ${DEFAULT_ENTROPY:g}$. Det.\\ shows '
                     'Pareto returns produced by the final actors under deterministic execution (active recall). '
                     'Sampled shows Pareto returns that some actor produces in at least half of 100 sampled episodes. '
                     'Own shows the sampled episodes in which an actor reproduces its own deterministic return, for '
                     'actors whose deterministic return is Pareto-optimal. Beyond shows the actors with an interior '
                     'preference whose expected return (mean of the 100 episodes) is not weakly dominated by any '
                     f'deterministic Pareto return. {POOLED_STATS}',
                     'lrrrrrrrr', [rf'& {groups} \\', r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
                                   'Scalarizer' + ' & Det. & Sampled & Own & Beyond' * len(entropies) + r' \\'],
                     rows, wide=True)


def overview(summary: Summary) -> str:
    """Recall and HV ratio of both collections under both execution rules."""
    cases = (('deterministic', 'coverage_q050'), ('sampled', 'coverage_q050'), ('deterministic', 'hv'),
             ('sampled', 'hv'))
    rows = [[label(method)] + [summary.cell(method, collection, execution, metric)
                               for collection in ('population', 'archive') for execution, metric in cases]
            for method in METHODS]
    return table_tex('tab:e2-coverage',
                     'Exact-front recall (\\%) and hypervolume ratio of both policy collections under both execution '
                     'rules at $h=100$ and entropy coefficient $0.05$. G denotes deterministic (greedy) execution and '
                     'S sampled execution. Recall under S counts the Pareto returns that some policy produces in at '
                     'least half of 100 sampled episodes. Table~\\ref{tab:e2-check} adds the threshold $0.9$. HV is '
                     'the hypervolume ratio of one deterministic return or one mean of the sampled episodes per '
                     'policy, relative to the deterministic front. It may exceed one under S. The archive holds, for '
                     'each archived return, the checkpoint that first produced it under deterministic execution. '
                     f'{POOLED_STATS} Per-environment values are in Table~\\ref{{tab:e2-environments}}.',
                     'lrrrrrrrr', [r'& \multicolumn{4}{c}{Active policy set} '
                                   r'& \multicolumn{4}{c}{First-discovery archive} \\',
                                   r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
                                   'Scalarizer' + ' & Recall, G & Recall, S & HV, G & HV, S' * 2 + r' \\'],
                     rows, wide=True)


def environment_table(summary: Summary) -> str:
    """Coverage and HV ratio per environment for both collections and execution rules."""
    cases = (('population', 'deterministic'), ('population', 'sampled'), ('archive', 'deterministic'),
             ('archive', 'sampled'))
    rows = []
    for env in ENVS:
        if rows:
            rows.append(r'\midrule')
        rows += [[env.upper() if i == 0 else '', label(method)] +
                 [summary.cell(method, c, e, metric, env) for metric in ('coverage_q050', 'hv') for c, e in cases]
                 for i, method in enumerate(METHODS)]
    return table_tex('tab:e2-environments',
                     'Deterministic and sampled execution per environment at $h=100$ and entropy coefficient $0.05$. '
                     'A denotes the active policy set and R the first-discovery archive. G and S denote deterministic '
                     '(greedy) and sampled execution. Coverage uses $q=0.5$ and is in percent. HV uses one '
                     'deterministic return or one mean of the sampled episodes per policy, relative to the same '
                     f'deterministic front. A sampled HV ratio may exceed one. {SEED_STATS}',
                     'llrrrrrrrr', [r'& & \multicolumn{4}{c}{Exact-return coverage (\%)} '
                                    r'& \multicolumn{4}{c}{HV ratio} \\',
                                    r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
                                    r'Env. & Scalarizer & A, G & A, S & R, G & R, S & A, G & A, S & R, G & R, S \\'],
                     rows, wide=True)


def checks_table(summary: Summary) -> str:
    """Sampled coverage at q=0.5 and q=0.9, on all episodes and held out."""
    rows = []
    for collection, name in (('population', 'Active policy set'), ('archive', 'First-discovery archive')):
        if rows:
            rows.append(r'\midrule')
        rows.append(r'\multicolumn{5}{l}{\emph{' + name + r'}} \\')
        rows += [[label(method)] + [summary.cell(method, collection, 'sampled', metric) for metric in
                                    ('coverage_q050', 'coverage_heldout_q050', 'coverage_q090',
                                     'coverage_heldout_q090')]
                 for method in METHODS]
    return table_tex('tab:e2-check',
                     'Sampled exact-return coverage (\\%) at $h=100$ and entropy coefficient $0.05$. Full shows the '
                     'best available frequency over all 100 episodes. Held shows the frequency on episodes 51--100 for '
                     f'the policy selected using episodes 1--50. {POOLED_STATS} These are empirical frequencies, not '
                     'probability guarantees.',
                     'lrrrr', [r'& \multicolumn{2}{c}{$q=0.5$} & \multicolumn{2}{c}{$q=0.9$} \\',
                               r'Scalarizer & Full & Held & Full & Held \\'], rows)


def diagnostic_table(summary: Summary) -> str:
    """Sampled behavior of the selected checkpoints."""
    rows = []
    metrics = ('greedy_exact_rate', 'greedy_treasure_rate', 'pareto_episode_rate', 'detour_episode_rate',
               'no_treasure_episode_rate', 'checkpoint_age_mean')
    for collection, name in (('population', 'Active policy set'), ('archive', 'First-discovery archive')):
        if rows:
            rows.append(r'\midrule')
        rows.append(r'\multicolumn{7}{l}{\emph{' + name + r'}} \\')
        rows += [[label(method)] + [summary.cell(method, collection, 'sampled', metric) for metric in metrics]
                 for method in METHODS]
    return table_tex('tab:e2-diagnostics',
                     'Sampled behavior of the selected checkpoints at $h=100$ and entropy coefficient $0.05$. Greedy '
                     'exact and Greedy treasure show how often a policy reproduces its own deterministic (greedy) '
                     'return or treasure, conditional on that return lying on the exact front. Their difference '
                     'concerns extra steps for the same treasure, not a comparison of independently selected policies. '
                     'Pareto, detour, and none are episode shares over all selected policies. A detour reaches a '
                     'reference treasure with extra steps. All rates are in percent and averaged within run first. '
                     'Undefined conditional rates are omitted (denominators are recorded in the CSV). Age is $100$ '
                     f'minus the checkpoint iteration. {POOLED_STATS} Neither age nor these frequencies establish a '
                     'benefit from checkpoint replacement.',
                     'lrrrrrr', [r'Scalarizer & Greedy exact & Greedy treasure & Pareto & Detour & None & Age \\'],
                     rows, wide=True)


def mixture_table(diagnostics: Diagnostics) -> str:
    """Shares of actors beyond the deterministic front and with a single Pareto return."""
    entropies = (0.0, DEFAULT_ENTROPY)
    shares = (('beyond', 'Beyond'), ('single', 'Single'))
    rows = [[label(method)] + [diagnostics.cell(method, share, entropy)
                               for entropy in entropies for share, _ in shares]
            for method in METHODS]
    names = ' & '.join(name for _ in entropies for _, name in shares)
    return table_tex('tab:e2-mixtures',
                     'Expected returns of the active actors relative to the deterministic Pareto front at $h=100$, '
                     'for training without an entropy bonus and with the default coefficient. Each expected return is '
                     'the mean of 100 sampled episodes. Actors with a boundary preference are excluded. Beyond shows '
                     'the share of actors (\\%) whose expected return is not weakly dominated by any deterministic '
                     'Pareto return. Single shows the share of actors that produce the same Pareto return in every '
                     f'sampled episode. {POOLED_STATS}',
                     'lrrrr', [r'& \multicolumn{2}{c}{Entropy $0$} & '
                               rf'\multicolumn{{2}}{{c}}{{Entropy ${DEFAULT_ENTROPY:g}$}} \\',
                               r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
                               rf'Scalarizer & {names} \\'], rows)


# Figures

def entropy_figure(summary: Summary) -> plt.Figure:
    """Active recall over the entropy coefficient in two blocks of three environments, deterministic and sampled."""
    rows = (('deterministic', 'Active recall,\ndeterministic (%)'), ('sampled', 'Active recall,\nsampled (%)'))
    blocks = (MAIN_ENVS, APPENDIX_ENVS)
    ylabels = [ylabel for _ in blocks for _, ylabel in rows]
    fig, axes = panels(len(ylabels), MAIN_ENVS, ylabels)
    fig.subplots_adjust(hspace=0.45)
    for b, envs in enumerate(blocks):
        for col, env in enumerate(envs):
            for row, (execution, _) in enumerate(rows):
                ax = axes[2 * b + row, col]
                mark(ax, DEFAULT_ENTROPY)
                for method in METHODS:
                    values = 100 * np.array([summary.get(method, 'population', execution, 'coverage_q050', env, e)
                                             [['mean', 'ci_low', 'ci_high']].to_numpy(float) for e in ENTROPIES])
                    draw(ax, ENTROPIES, values[:, 0], values[:, 1], values[:, 2], method)
                ax.set(xticks=ENTROPIES, xlim=(ENTROPIES[0] - 0.006, ENTROPIES[-1] + 0.006))
                ax.set_xticklabels([f'{e:g}' for e in ENTROPIES])
    finish(fig, axes, MAIN_ENVS, ylabels, 'Entropy coefficient in training', METHODS)
    for col, env in enumerate(APPENDIX_ENVS):
        axes[2, col].set_title(env.upper(), fontsize=10)
    return fig


def upper_hull(front: np.ndarray) -> np.ndarray:
    """Vertices of the upper convex hull of a front, ordered by the first objective."""
    hull = []
    for point in front[np.argsort(front[:, 0])]:
        while len(hull) >= 2:
            u = hull[-1] - hull[-2]
            v = point - hull[-2]
            if u[0] * v[1] - u[1] * v[0] < 0:
                break
            hull.pop()
        hull.append(point)
    return np.array(hull)


def return_figure(diagnostics: Diagnostics, envs: tuple[str, ...]) -> plt.Figure:
    """Mean sampled return of every interior active actor over the deterministic front, per normalized scalarizer."""
    methods = ('ws_norm', 'tch_norm', 'stch_norm')
    actors = diagnostics.select(DEFAULT_ENTROPY)
    fig, axes = plt.subplots(len(methods), len(envs), figsize=(WIDTH, 1.8 * len(methods) + 0.7), sharex='col',
                             sharey='col', squeeze=False)
    for col, env in enumerate(envs):
        front = diagnostics.fronts[env]
        hull = upper_hull(front)
        low = front.min(axis=0)
        span = np.ptp(front, axis=0)
        floor = low[1] - 0.3 * span[1]
        for row, method in enumerate(methods):
            ax = axes[row, col]
            color = FAMILY_COLORS[method.split('_')[0]]
            data = actors[actors.environment.eq(env) & actors.scalarizer.eq(method)]
            inside = data.mean_J1 >= floor
            ax.plot(hull[:, 0], hull[:, 1], color='0.45', linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
            ax.scatter(front[:, 0], front[:, 1], s=26, facecolor='white', edgecolor='black', linewidth=0.9, zorder=2)
            ax.scatter(data.mean_J0[inside], data.mean_J1[inside], s=13, color=color, edgecolor='0.2',
                       linewidth=0.4, alpha=0.9, zorder=3)
            ax.scatter(data.mean_J0[~inside], np.full((~inside).sum(), floor), s=13, marker='v', facecolor='white',
                       edgecolor='0.2', linewidth=0.8, zorder=3, clip_on=False)
            ax.text(0.0, 1.03, f'{100 * data.beyond.mean():.0f}% beyond, {100 * data.single.mean():.0f}% single',
                    transform=ax.transAxes, fontsize=8, color='0.25')
            ax.set(xlim=(low[0] - 0.05 * span[0], low[0] + 1.05 * span[0]), ylim=(floor, low[1] + 1.08 * span[1]))
            ax.grid(True, color='0.9', linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
        axes[0, col].set_title(env.upper(), fontsize=10, pad=16)
        axes[-1, col].set_xlabel('Treasure')
    for row, method in enumerate(methods):
        axes[row, 0].set_ylabel(f'{label(method)}\nTime')
    handles = [Line2D([], [], linestyle='', marker='o', markersize=5, markerfacecolor='white',
                      markeredgecolor='black'),
               Line2D([], [], color='0.45', linestyle=(0, (4, 3)), linewidth=0.9),
               Line2D([], [], linestyle='', marker='o', markersize=4.5, color='0.35'),
               Line2D([], [], linestyle='', marker='v', markersize=4.5, markerfacecolor='white',
                      markeredgecolor='0.35')]
    fig.legend(handles, ['Deterministic Pareto front', 'Upper convex hull', 'Expected return of one actor',
                         'Actor below the plotted range'], loc='lower center', ncol=4, frameon=False, fontsize=8,
               handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.1, w_pad=0.8)
    return fig


def return_caption(envs: tuple[str, ...]) -> str:
    """Caption of the return figure for its environments."""
    names = ', '.join(env.upper() for env in envs[:-1]) + f', and {envs[-1].upper()}'
    return (f'Expected returns of the active actors under sampled execution on {names} at $h=100$ and entropy '
            f'coefficient ${DEFAULT_ENTROPY:g}$, for the normalized scalarizers. Each dot is one actor with an '
            'interior preference from one of ten training seeds, placed at the mean return of its 100 sampled '
            'episodes. Open circles show the deterministic Pareto front, and the dashed line shows its upper convex '
            'hull. Triangles mark actors whose mean time lies below the plotted range. The values above each panel '
            'give the percentage of actors whose expected return is not weakly dominated by any deterministic Pareto '
            'return (beyond) and the percentage that produce the same Pareto return in every sampled episode '
            '(single).')


def main() -> None:
    """Writes the E2 tables and figures to <input>/paper."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Data
    summary = Summary(args.input)
    diagnostics = Diagnostics(args.input)
    output = args.input / 'paper'
    appendix = output / 'appendix'
    appendix.mkdir(parents=True, exist_ok=True)

    # Tables
    (output / 'e2_execution.tex').write_text(execution_table(summary, diagnostics), encoding='utf-8')
    (appendix / 'e2_tables.tex').write_text('\n'.join((overview(summary), mixture_table(diagnostics),
                                                     environment_table(summary), checks_table(summary),
                                                     diagnostic_table(summary))), encoding='utf-8')

    # Figures
    caption = ('Execution rule and entropy bonus in all six environments. The panels show the active policy sets at '
               '$h=100$ for all six scalarizers, trained with the entropy coefficient on the horizontal axis. DST1, '
               'DST4, and LDST2 are shown on top, DST2, DST3, and LDST1 below. The upper row of each block shows '
               'active recall, the percentage of Pareto returns that some actor produces under deterministic '
               'execution. The lower row shows the same actors with sampled actions, where a Pareto return counts '
               'only if some actor produces it in at least half of 100 episodes ($q=0.5$). '
               f'{FIGURE_STATS} The dotted line marks the default coefficient $0.05$. Removing the entropy bonus does '
               'not make sampled execution deterministic.')
    entropy_description = ('Line charts in two blocks of three environments. In each block, the upper row shows '
                           'active recall under deterministic execution and the lower row under sampled execution, '
                           'plotted against the entropy coefficient for the six scalarizers.')
    return_description = ('Scatter plots, one row per normalized scalarizer and one column per environment. Each '
                          'panel shows the deterministic Pareto front as open circles, its upper convex hull as a '
                          'dashed line, and one dot per actor at its mean sampled return.')
    save(entropy_figure(summary), appendix / 'e2_entropy.pdf')
    (appendix / 'e2_entropy_figure.tex').write_text(
        figure_tex('appendix/e2_entropy.pdf', caption, 'fig:e2-entropy', description=entropy_description),
        encoding='utf-8')
    for envs, path, tag, other in versions('e2_returns', 'fig:e2-returns'):
        save(return_figure(diagnostics, envs), output / path)
        (output / path).with_name(f'{Path(path).stem}_figure.tex').write_text(
            figure_tex(path, f'{return_caption(envs)} {other}', tag, description=return_description),
            encoding='utf-8')
    print(f'E2: one table and the return figure in {output}; five tables and two figures in {appendix}')


if __name__ == '__main__':
    main()