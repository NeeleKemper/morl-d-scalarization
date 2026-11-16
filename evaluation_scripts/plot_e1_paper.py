"""Fixed E1 publication outputs: one main table, four appendix tables and four figures with all six scalarizers, each
in a paper version (DST1, DST4, LDST2) and an appendix version (DST2, DST3, LDST1), and a DST1 actor-history case
study (lowest protocol seed in the paper, the other seeds in the appendix). Requires the dense NPZ histories and saved
run configurations named in e1_manifest.json.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any, Sequence
from pathlib import Path
from functools import partial
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.font_manager import FontProperties, findfont

from paper_style import (FIGURE_STATS, OTHER_HATCH, POOLED_STATS, WIDTH, cell, draw, envs_text, figure_tex, finish,
                         head, label, mark, ordered_colors, outcome_map, panels, save, style, table_tex, versions)
from reconstruct_e1 import DATA_DIR, ENVS, H, HORIZONS, METHODS, boolean, read_json, standard_k
from reconstruct_e1 import Reference, load_history, match_matrix
from summarize_e1 import ALLOCATIONS


# Curves

def band(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise intervals from 2,000 resamples of training seeds, not actors."""
    n = matrix.shape[1]
    if n < 2:
        return matrix[:, 0], matrix[:, 0]
    samples = np.random.default_rng(2027).integers(0, n, size=(2000, n))
    return np.quantile(matrix[:, samples].mean(2), [0.025, 0.975], axis=1)


def curves(frame: pd.DataFrame,
           x: str,
           metrics: Sequence[str],
           envs: Sequence[str],
           methods: Sequence[str],
           xlabel: str) -> plt.Figure:
    """Rows: metrics in percent; columns: environments; one line per scalarizer with its seed-bootstrap band."""
    labels = {'oracle_recall': 'Oracle (%)', 'archive_recall': 'Archive (%)', 'active_recall': 'Active (%)',
              'complete_archive': 'Complete (%)', 'targets_active': 'Targets\nactive (%)'}
    ylabels = [labels[metric] for metric in metrics]
    fig, axes = panels(len(metrics), envs, ylabels)
    for c, env in enumerate(envs):
        for r, metric in enumerate(metrics):
            ax = axes[r, c]
            for method in methods:
                g = frame.loc[frame.environment.eq(env) & frame.scalarizer.eq(method)]
                values = 100 * g.pivot(index=x, columns='seed', values=metric).sort_index()
                xx = values.index.to_numpy(float)
                yy = values.to_numpy(float)
                if metric == 'targets_active':
                    kw = style(method)
                    ax.axhline(100 * g.oracle_recall.mean(), color=kw['color'], linestyle=kw['linestyle'],
                               linewidth=0.8, alpha=0.35, zorder=1)
                if metric == 'oracle_recall':
                    low = yy.mean(1)
                    high = yy.mean(1)
                else:
                    low, high = band(yy)
                draw(ax, xx, yy.mean(1), low, high, method)
            ax.set_xticks({'K': [2, 10, 20, 30, 50], 'cadence': [1, 10, 25, 50, 100]}.get(x, list(HORIZONS)))
            if x == 'K':
                mark(ax, standard_k(env))
    finish(fig, axes, envs, ylabels, xlabel, methods)
    return fig


# Actor histories

def run_temperature(history: Path, method: str) -> float:
    """STCH temperature of a run from the saved configuration next to its NPZ; NaN for the other scalarizers."""
    if not method.startswith('stch_'):
        return np.nan
    root = history.parent.parent
    for path in sorted({*root.glob('*.json'), *root.glob('config/*.json'), *root.glob('configs/*.json')}):
        config = read_json(path)
        if {'num_steps', 'scalarization_method', 'num_subproblems'} <= set(config):
            return float(config.get('mu') or np.nan)


def load_actor_histories(input_dir: Path, manifest: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Dense saved actor histories on DST1 at the standard population: outcomes, weights and targets per actor."""
    env = 'dst1'
    k = standard_k(env)
    seeds = sorted(int(seed) for seed in manifest['protocol']['seeds'])
    atol = float(manifest['protocol']['match_atol'])
    front = np.asarray(manifest['references'][env]['front'], float)
    point_ids = np.asarray(manifest['references'][env]['point_ids'], int)
    points = pd.read_csv(input_dir / 'e1_points.csv')
    points = points.loc[points.environment.eq(env) & points.K.eq(k)]
    records = {(r['scalarizer'], int(r['seed'])): r for r in manifest['runs']
               if r['environment'] == env and int(r['K']) == k and r['scalarizer'] in METHODS}
    histories = {}
    supported = None
    for seed in seeds:
        histories[seed] = {}
        for method in METHODS:
            path = Path(records[method, seed]['history']).expanduser()
            if not path.is_file() and not path.is_absolute():
                path = input_dir / path
            p = points.loc[points.scalarizer.eq(method) & points.seed.eq(seed)].set_index('pf_id').loc[point_ids]
            supported = p.supported.map(boolean).to_numpy(bool)
            ref = Reference(front=front, supported=supported, targeted=p.targeted.map(boolean).to_numpy(bool),
                            targets_by_weight=None, effective_mu=run_temperature(path, method), point_ids=point_ids)
            hist = load_history(path, method, k, ref)
            matches = match_matrix(hist.returns, front, atol)
            outcomes = np.where(np.abs(hist.returns[..., 0]) <= atol, 0, 1)
            on_front = matches.any(-1)
            outcomes[on_front] = 2 + matches.argmax(-1)[on_front]
            order = np.argsort(hist.weights[:, 0], kind='stable')
            histories[seed][method] = dict(
                outcomes=outcomes[:, order].T, weights=hist.weights[order],
                targets=np.asarray(hist.assigned_targets, bool)[order], policy_index=order,
                policy_ids=hist.policy_ids[order] if hist.policy_ids is not None else None,
                identity_verified=hist.policy_index_source == 'agent_ids')
    return front, supported, histories


def actor_history_map(histories: dict,
                      front: np.ndarray,
                      supported: np.ndarray,
                      seed: int) -> plt.Figure:
    """Six panels (rows WS, TCH, STCH; columns raw, norm) with one fixed row per actor and its return per iteration."""
    front = np.asarray(front, float)
    by_treasure = np.argsort(front[:, 0], kind='stable')
    rank = np.argsort(by_treasure)
    shades = ordered_colors(len(front))
    colors = np.vstack(([[1, 1, 1, 1], [1, 1, 1, 1]], shades[rank]))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors))
    hatched = dict(facecolor='white', edgecolor='0.25', hatch=OTHER_HATCH, linewidth=0)
    height = 6.2
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, height))
    fig.subplots_adjust(left=0.70 / WIDTH, right=1 - 0.52 / WIDTH, bottom=0.82 / height, top=1 - 0.30 / height,
                        wspace=0.47, hspace=0.24)
    for n, (ax, method) in enumerate(zip(axes.flat, METHODS)):
        data = histories[seed][method]
        k = len(data['weights'])
        ax.imshow(data['outcomes'], cmap=cmap, norm=norm, aspect='auto', interpolation='nearest',
                  origin='upper', extent=(-0.5, H + 0.5, k - 0.5, -0.5))
        for row, outcomes in enumerate(data['outcomes']):
            edges = np.diff(np.concatenate(([0], (np.asarray(outcomes) == 1).astype(int), [0])))
            for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                ax.add_patch(Rectangle((first - 0.5, row - 0.5), last - first, 1, **hatched))
        names = data['policy_ids'] if data['policy_ids'] is not None else data['policy_index']
        ax.set_yticks(range(k), [f'{i} : {w[0]:.3f}' for i, w in zip(names, data['weights'])])
        ax.set_xticks([0, 25, 50, 75, H])
        ax.set_xlim(-0.5, H + 0.5)
        ax.tick_params(labelsize=7, length=2)
        ax.set_yticks(np.arange(k - 1) + 0.5, minor=True)
        ax.grid(axis='y', which='minor', linewidth=0.7, color='white')
        ax.tick_params(axis='y', which='minor', length=0)
        if n < len(METHODS) - 2:
            ax.tick_params(labelbottom=False)
        ax.set_title(label(method), fontsize=10, pad=12 if n < 2 else 3)
        if n == 0:
            ax.text(-0.03, 1.02, r'Actor : $w_0$', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        if n == 1:
            ax.text(1.025, 1.02, 'Target(s)', transform=ax.transAxes, ha='left', va='bottom', fontsize=7)
        for i, targets in enumerate(data['targets']):
            text = ', '.join(f'{x:g}' for x in front[targets, 0])
            ax.text(1.025, i, text, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=7,
                    clip_on=False)
        for spine in ax.spines.values():
            spine.set(linewidth=0.5, edgecolor='0.45')
    center = (fig.subplotpars.left + fig.subplotpars.right) / 2
    fig.text(center, 0.55 / height, 'Training iteration $h$', ha='center', va='center', fontsize=9)
    handles = [Patch(facecolor='white', edgecolor='0.6', linewidth=0.4, label='none'),
               Patch(label='other', **hatched | dict(linewidth=0.4)),
               *[Patch(facecolor=shades[position], edgecolor='0.6', linewidth=0.4,
                       label=f"({front[i, 0]:g}, {front[i, 1]:g}) {'S' if supported[i] else 'U'}")
                 for position, i in enumerate(by_treasure)]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(center, 0.0), ncol=6, fontsize=7,
               frameon=False, columnspacing=1.3, handlelength=1.2)
    return fig


def make_actor_figures(front: np.ndarray,
                       supported: np.ndarray,
                       histories: dict,
                       out: Path) -> int:
    """One fixed-seed paper example and every other protocol seed in the appendix."""
    description = ('Six panels, one per scalarizer. Each panel has one row per actor and one cell per training '
                   'iteration. The shade of a cell identifies the return the actor produces at that iteration, from '
                   'light for the smallest treasure to dark for the largest.')
    seeds = sorted(histories)
    main_wrappers = []
    appendix_wrappers = []
    for seed in seeds:
        is_main = seed == seeds[0]
        name = ('' if is_main else 'appendix/') + f'e1_actor_histories_dst1_seed_{seed:04d}.pdf'
        tag = 'fig:e1-actor-histories' if is_main else f'fig:e1-actor-histories-seed-{seed:04d}'
        fig = actor_history_map(histories, front, supported, seed)
        fig.savefig(out / name)
        plt.close(fig)
        verified = all(data['identity_verified'] for data in histories[seed].values())
        identity = ('Saved actor IDs are aligned across checkpoints. ' if verified else
                    'Where actor IDs were not saved, identity assumes stable logging row order. ')
        caption = (
            f'Individual greedy-return histories on DST1 at $K=10$, seed {seed}, all six '
            'scalarizers. Each row follows one learner, ordered once by its fixed treasure weight $w_0$ '
            '($w_1=1-w_0$). The shade of a cell identifies the greedy return. Pareto returns range from light '
            '(smallest treasure) to dark (largest treasure) and are given in the legend as (treasure, time), with S '
            'for supported and U for unsupported. Hatched cells mark other, a treasure return off the front, and '
            'white cells mark none, no treasure. Shades do not encode actor shares. Right-hand annotations give the '
            'treasure values of all tied oracle targets. No averaging or interpolation is used. '
            + identity + 'Consecutive matching evaluations do not establish convergence or behavior '
            'between checkpoints. ')
        if is_main:
            caption += 'The lowest numbered protocol seed is selected independently of the observed trajectories. '
            if len(seeds) > 1:
                caption += (f'Other seeds: Appendix Figs.~\\ref{{fig:e1-actor-histories-seed-{seeds[1]:04d}}}'
                            f'--\\ref{{fig:e1-actor-histories-seed-{seeds[-1]:04d}}}.')
        else:
            caption += 'Same fixed layout as Fig.~\\ref{fig:e1-actor-histories}.'
        wrapper = figure_tex(name, caption, tag, description=description)
        (main_wrappers if is_main else appendix_wrappers).append(wrapper)
    for relative, wrappers in (('e1_actor_histories.tex', main_wrappers),
                               ('appendix/e1_actor_histories.tex', appendix_wrappers)):
        (out / relative).write_text('\n'.join(wrappers), encoding='utf-8')
    for relative, wrappers in (('e1_figures.tex', main_wrappers),
                               ('appendix/e1_figures.tex', appendix_wrappers)):
        with (out / relative).open('a', encoding='utf-8') as handle:
            handle.write('\n' + '\n'.join(wrappers))
    return len(seeds)


# Tables

def table(headers: Sequence[str],
          rows: list,
          caption: str,
          label_name: str,
          wide: bool = False,
          note: str = '') -> str:
    """E1 table in the shared format with one header row; a string row is written as is."""
    n = 2 if headers[0] in ('Environment', 'Group') else 1
    return table_tex(label_name, caption, 'l' * n + 'r' * (len(headers) - n), [head(*headers)], rows, wide, note)


def norm_key(values: Sequence) -> tuple:
    """Summary lookup key with missing entries as -1."""
    return tuple(-1 if pd.isna(v) else v for v in values)


def make_tables(summary: pd.DataFrame, runs: pd.DataFrame) -> tuple[str, str]:
    """Main table and the four appendix tables."""
    setup = ('after $h=100$ iterations at the standard populations ($K{=}10$ preferences on DST, $K{=}20$ on LDST, '
             'one actor each)')
    columns = ('Oracle: optimal for at least one grid preference. Archive: found by any checkpoint, observed every '
               'iteration ($C{{=}}1$). Complete: {complete}. Active: returned by the actors at $h$. {extra}'
               '$C{{=}}10$: archive recall when only every tenth iteration is observed. Oracle is fixed by the grid '
               'and Complete is a count, so neither has a deviation.')
    keys = ['source', 'scope', 'environment', 'scalarizer', 'K', 'horizon', 'cadence', 'preference_group', 'metric']
    lookup = {norm_key(row[keys]): row for _, row in summary.iterrows()}

    def stat(method: str,
             metric: str,
             *,
             env: str = 'ALL',
             k: int | None = None,
             h: int = H,
             c: int = 1,
             source: str = 'returns',
             part: str = 'all',
             scope: str | None = None) -> pd.Series:
        """Summary row of one cell."""
        scope = scope or ('standard' if env == 'ALL' else 'condition')
        k = standard_k(env) if k is None and env != 'ALL' else k
        return lookup[norm_key([source, scope, env, method, k, h, c, part, metric])]

    def pct(method: str, metric: str, **kw: Any) -> str:
        """Mean in percent without deviation."""
        value = stat(method, metric, **kw)['mean']
        return f'{100 * value:.1f}' if np.isfinite(value) else '--'

    def complete(method: str, **kw: Any) -> str:
        """Complete runs as a count."""
        row = stat(method, 'complete_archive', **kw)
        return f'{int(row.numerator)}/{int(row.denominator)}'

    def seed_values(method: str,
                    metric: str,
                    env: str,
                    c: int,
                    k: int | None,
                    h: int) -> pd.Series:
        """Per-seed values behind a mean; for ALL the environment-equal mean within each seed."""
        g = runs.loc[runs.scalarizer.eq(method) & runs.cadence.eq(c) & runs.horizon.eq(h)
                     & runs.K.eq(runs.environment.map(standard_k) if k is None else k)]
        if env != 'ALL':
            return g.loc[g.environment.eq(env)].set_index('seed')[metric].astype(float)
        return g.groupby('seed')[metric].mean()

    def pct_sd(method: str,
               metric: str,
               env: str = 'ALL',
               c: int = 1,
               k: int | None = None,
               h: int = H,
               **kw: Any) -> str:
        """Mean from the summary and the standard deviation over seeds from e1_runs.csv or the summary's std."""
        row = stat(method, metric, env=env, k=k, h=h, c=c, **kw)
        if not np.isfinite(row['mean']):
            return '--'
        if metric in runs.columns and not set(kw) - {'scope'}:
            sd = seed_values(method, metric, env, c, k, h).std(ddof=1)
        else:
            sd = row['std']
        return cell(row['mean'], sd)

    standard = runs.loc[runs.K.eq(runs.environment.map(standard_k)) & runs.horizon.eq(H)]
    main = table(['Scalarizer', 'Oracle', 'Archive', 'Complete', 'Active', '$C=10$'],
                 [[label(m), pct(m, 'oracle_recall'), pct_sd(m, 'archive_recall'), complete(m),
                   pct_sd(m, 'active_recall'), pct_sd(m, 'archive_recall', c=10)] for m in METHODS],
                 f"Exact-front recall {setup}. Recalls are percentages of each environment's Pareto returns under "
                 'greedy evaluation; means weight the six environments equally within each of ten seeds, and '
                 '$\\pm$ gives the standard deviation of these ten seed means. '
                 + columns.format(complete='runs, of 60 environment--seed runs, whose archive contains the whole front',
                                  extra=''),
                 'tab:e1-summary')
    rows = []
    for e in ENVS:
        if rows:
            rows.append('\\midrule')
        rows += [[e.upper() if i == 0 else '', label(m), pct(m, 'oracle_recall', env=e),
                  pct_sd(m, 'archive_recall', env=e), complete(m, env=e), pct_sd(m, 'active_recall', env=e),
                  pct_sd(m, 'assigned_hit_active', env=e), pct_sd(m, 'active_unsupported_recall', env=e),
                  pct_sd(m, 'archive_recall', env=e, c=10)] for i, m in enumerate(METHODS)]
    detailed = table(
        ['Environment', 'Scalarizer', 'Oracle', 'Archive', 'Complete', 'Active', 'Own-target hit', 'Unsupp.', '$C=10$'],
        rows,
        f"Exact-front recall by environment {setup}. Recalls are percentages of the environment's Pareto returns "
        'under greedy evaluation; means $\\pm$ standard deviations over ten seeds. '
        + columns.format(complete='runs whose archive contains the whole front',
                         extra='Own-target hit: percent of actors, not returns, whose return is an optimum of their '
                               "own preference. Unsupp.: active recall within the environment's unsupported returns; "
                               '--: the environment has none. '),
        'tab:e1-scalarizers', True)
    rows = []
    for metric, title in (('archive_recall', 'Archive recall'), ('active_recall', 'Active recall')):
        if rows:
            rows.append('\\midrule')
        rows.append(f'\\multicolumn{{{len(ALLOCATIONS) + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\')
        rows += [[label(m), *[pct_sd(m, metric, k=k, h=h, scope='budget') for k, h in ALLOCATIONS]] for m in METHODS]
    budget = table(['Scalarizer', *[f'${k}/{h}$' for k, h in ALLOCATIONS]], rows,
                   'Recall (percent) at equal training allocations $K/h$ with $Kh=500$: $K$ preferences with one '
                   'actor each, evaluated after $h$ iterations under greedy evaluation. Archive: Pareto returns found '
                   'by any checkpoint up to $h$, observed every iteration ($C=1$). Active: returned by the actors at '
                   f'$h$. {POOLED_STATS} Evaluation costs are not matched.',
                   'tab:e1-budget')
    base = standard.loc[standard.cadence.eq(1)].set_index(['environment', 'scalarizer', 'K', 'seed'])
    rows = []
    for c, g in standard.groupby('cadence'):
        g = g.set_index(base.index.names).reindex(base.index)
        dense = base.complete_archive.to_numpy(bool)
        common = dense & g.complete_archive.to_numpy(bool)
        delay = (g.first_complete_iteration.to_numpy(float)[common]
                 - base.first_complete_iteration.to_numpy(float)[common])
        rows.append([c, f'{common.sum()}/{dense.sum()}',
                     f'{100 * common.sum() / dense.sum():.1f}' if dense.any() else '--',
                     f'{np.median(delay):g}' if len(delay) else '--'])
    cadence = table(['$C$', 'Still complete', 'Percent', 'Delay'], rows,
                    'Archive completion at the standard populations and $h=100$, conditional on complete dense '
                    'recovery. Delay is the median paired increase in first observed completion among common-success '
                    'runs, in iterations. Schedules reuse the same recorded histories; they do not save evaluations '
                    'already performed.',
                    'tab:e1-cadence')
    metrics = ('own_target_rate', 'off_front_rate', 'no_treasure_rate', 'redundant_pareto_actor_rate')
    diagnostics = table(
        ['Scalarizer', 'Own-target hit', 'Off front', 'No treasure', 'Duplicates', 'Tied misses'],
        [[label(m), *[pct_sd(m, metric, source='actors') for metric in metrics],
          pct(m, 'score_tied_miss_rate', source='actors')] for m in METHODS],
        f'Actor diagnostics {setup}, under greedy evaluation (percent). {POOLED_STATS} Tied misses are reported as '
        'means; runs without scored misses have no tied-miss rate and are left out of its average.',
        'tab:e1-policy-diagnostics', True,
        note='\\textit{Definitions.} Own-target hit: the actor\'s return is a Pareto target of its own preference. '
             'Off front: the return is not an exact Pareto return; this includes No treasure, an episode without '
             'treasure. Duplicates: actors producing an already represented Pareto return, after counting one actor '
             'per distinct return, so four actors with the same return contribute three. These four columns are '
             'percentages of all actors. Tied misses: percentage of the scored exact-target misses whose scalarized '
             'score equals the target\'s score within numerical tolerance. The columns are not mutually exclusive. '
             'Corner and interior subsets, counts, and denominators are in \\texttt{e1\\_summary.csv}.')
    return main, '\n'.join([detailed, budget, cadence, diagnostics])


# Figures

def make_figures(runs: pd.DataFrame, train: pd.DataFrame, out: Path) -> None:
    """Every figure twice: the paper version (MAIN_ENVS) and the appendix version (APPENDIX_ENVS)."""
    iteration_label = 'Training iteration $h$'
    lines = ('Line charts with one column per environment and one row each for {rows}, plotted against {x} for the '
             'six scalarizers.')
    standard = runs.loc[runs.K.eq(runs.environment.map(standard_k))]
    standard = standard.assign(targets_active=standard.active_targeted_count / standard.num_pf_points)
    plots = [
        ('e1_weight_resolution', 'fig:e1-weight-resolution',
         partial(curves, runs.loc[runs.horizon.eq(H) & runs.cadence.eq(1)], 'K',
                 ['oracle_recall', 'archive_recall', 'active_recall'], methods=METHODS, xlabel='Preferences $K$'),
         'Oracle, archive, and active recall over the number of preferences $K$ (one actor each) on {envs} at '
         "$h=100$, under greedy evaluation. Oracle: percentage of the environment's Pareto returns that are optimal "
         'for at least one grid preference; it is fixed by the grid. Archive: percentage found by any checkpoint, '
         'observed every iteration ($C=1$). Active: percentage returned by the actors at $h$. Increasing $K$ also '
         'increases training interactions, and the grids are not nested. Dotted lines mark the standard '
         f'populations. {FIGURE_STATS} {{other}}',
         lines.format(rows='oracle, archive, and active recall', x='the number of preferences')),
        ('e1_training', 'fig:e1-training',
         partial(outcome_map, train, rows=[(label(m), {'scalarizer': m}) for m in METHODS], x='iteration',
                 xlabel=iteration_label, x_max=H),
         'Greedy-return distributions at the standard populations on {envs}: mean share of actors over ten seeds. '
         'Pareto rows are ordered by treasure; other means a treasure return outside the exact front, none means '
         'no treasure. Square-root color scale. These are population distributions, not individual actor '
         'trajectories. {other}',
         'Heatmaps with one column per environment and one row per scalarizer. Each heatmap shows, for every training '
         'iteration, the share of actors that produce each return, with rows for no treasure, other returns, and the '
         'Pareto returns ordered by treasure. Darker cells mean larger shares.'),
        ('e1_horizon_resolution', 'fig:e1-horizon',
         partial(curves, standard.loc[standard.cadence.eq(1)], 'horizon',
                 ['archive_recall', 'active_recall', 'targets_active'], methods=METHODS, xlabel=iteration_label),
         'Archive recall, active recall, and targets active over training on {envs} at the standard populations '
         '($K=10$ preferences on DST, $K=20$ on LDST), under greedy evaluation. Archive: percentage of the '
         "environment's Pareto returns found by any checkpoint up to iteration $h$, observed every iteration "
         '($C=1$); it cannot decrease. Active: percentage returned by the actors at iteration $h$. Targets active: '
         'percentage of the front that is an oracle target, optimal for at least one grid preference, and returned by the actors '
         "at $h$; faint horizontal lines mark each scalarizer's oracle recall, the percentage of the front its grid "
         'targets and thus the ceiling of this row. Each $h\\in\\{{10,25,50,75,100\\}}$ is a cutoff within '
         f'the same 100-iteration runs. {FIGURE_STATS} {{other}}',
         lines.format(rows='archive recall, active recall, and targets active', x='the training iteration')),
        ('e1_cadence_resolution', 'fig:e1-cadence',
         partial(curves, standard.loc[standard.horizon.eq(H)], 'cadence', ['archive_recall', 'complete_archive'],
                 methods=METHODS, xlabel='Interval $C$'),
         'Archive recall and complete recovery on {envs} when the checkpoints are observed only every $C$ '
         "iterations, at the standard populations and $h=100$. Archive: percentage of the environment's Pareto "
         'returns found by an observed checkpoint. Complete: percentage of runs whose archive contains the whole '
         'front. Initialization and the last checkpoint are always observed. The same recorded histories are '
         f'subsampled, so no evaluation is saved. {FIGURE_STATS} {{other}}',
         lines.format(rows='archive recall and complete recovery', x='the checkpoint evaluation interval')),
    ]
    wrappers = {'main': [], 'appendix': []}
    for stem, tag, plot, caption, description in plots:
        for part, (envs, name, name_tag, other) in zip(('main', 'appendix'), versions(stem, tag)):
            save(plot(envs), out / name)
            wrappers[part].append(figure_tex(name, caption.format(envs=envs_text(envs), other=other), name_tag,
                                             description=description))
    (out / 'e1_figures.tex').write_text('\n'.join(wrappers['main']))
    (out / 'appendix/e1_figures.tex').write_text('\n'.join(wrappers['appendix']))


def main() -> None:
    """Writes the E1 tables, figures and actor-history case study to <input>/paper."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR, help='Directory produced by the lean E1 scripts')
    args = parser.parse_args()

    # Data
    manifest = read_json(args.input / 'e1_manifest.json')
    runs = pd.read_csv(args.input / 'e1_runs.csv')
    for col in ('complete_archive', 'dense_complete_archive'):
        runs[col] = runs[col].map(boolean)
    summary = pd.read_csv(args.input / 'e1_summary.csv')
    train = pd.read_csv(args.input / 'e1_training.csv', dtype={'outcome': str})
    front, supported, histories = load_actor_histories(args.input, manifest)

    # Output
    main_table, appendix = make_tables(summary, runs)
    out = args.input / 'paper'
    (out / 'appendix').mkdir(parents=True, exist_ok=True)
    (out / 'e1_summary_table.tex').write_text(main_table)
    (out / 'appendix/e1_tables.tex').write_text(appendix)
    make_figures(runs, train, out)
    n_actor_seeds = make_actor_figures(front, supported, histories, out)
    font = Path(findfont(FontProperties(family=plt.rcParams['font.family']))).stem
    print(f'Saved 1 main table, 5 main figures, 4 appendix tables and '
          f'{4 + n_actor_seeds - 1} appendix figures to {out} (font: {font})')


if __name__ == '__main__':
    main()