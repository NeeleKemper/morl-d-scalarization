"""Learning curves of the baseline comparison: the HV ratio of every stored front over training.

Every front in a run's front store is scored with the reference front and HV reference point of
compute_metrics_baseline.py, so a curve ends at the HV ratio of the metric tables (compared with metrics_runs.csv if
it exists). IPRO's front at a step is the non-dominated set of its oracle answers up to that step
(oracle_queries.jsonl); with --ipro_final_only IPRO is drawn as one marker. The share figures place every run on its
own budget; the pair figure shows MO-PPO with C-MORL and MO-SAC with MORL/D over environment steps (8 parallel
environments for MO-PPO and C-MORL), cut at 3 million counted steps.

Outputs (in --output): learning_curves.csv and paper/appendix/baseline_curves.pdf, baseline_curves_mujoco.pdf,
baseline_pairs.pdf and baseline_curves_figures.tex.

  python plot_learning_curves.py
  python plot_learning_curves.py --envs mujoco
"""
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any, Iterator, Sequence
from pathlib import Path
from matplotlib.lines import Line2D

from build_reference_fronts import front_stores, ipro_records, nondominated, run_dirs, step_keys
from check_ipro_runs import newest
from compute_metrics_baseline import constructed_reference, exact_reference, hypervolume
from paper_style import FIGURE_STATS, WIDTH, figure_tex, label, save


def style(method: str, **extra: Any) -> dict:
    """Line style of a method: our learners in neutral ink and solid, the baselines colored and dashed."""
    styles = {'MO-PPO': dict(color='black', marker='o', linestyle='-'),
              'MO-SAC': dict(color='0.5', marker='s', linestyle='-'),
              'GPI-LS': dict(color='#e34948', marker='^', linestyle='--'),
              'PCN': dict(color='#6250d6', marker='D', linestyle='--'),
              'IPRO': dict(color='#008300', marker='X', linestyle='--'),
              'C-MORL': dict(color='#2a78d6', marker='v', linestyle='--'),
              'MORL/D': dict(color='#1baf7a', marker='P', linestyle='--')}
    return dict(linewidth=1.0, markersize=3.5) | styles[method] | extra


def name(env: str) -> str:
    """Display name of an environment; environments without one in upper case."""
    names = dict(minecart='Minecart', reacher='MO-Reacher', ant='MO-Ant', halfcheetah='MO-HalfCheetah',
                 hopper='MO-Hopper', humanoid='MO-Humanoid')
    return names.get(env, env.upper())


# Reading runs

def stored_fronts(path: Path, limit: int) -> Iterator[tuple[int, np.ndarray]]:
    """(step, front) of at most limit evenly spaced evaluations of a front store, always with the last one."""
    with np.load(path) as data:
        keys = step_keys(data)
        if len(keys) > limit:
            keys = [keys[i] for i in np.unique(np.linspace(0, len(keys) - 1, limit).round().astype(int))]
        for step, key in keys:
            front = np.atleast_2d(np.asarray(data[key], float)) if data[key].size else np.empty((0, 0))
            yield step, front[np.isfinite(front).all(1)]


def hv_ratio(front: np.ndarray, reference: dict) -> float:
    """Hypervolume of the rounded non-dominated points over that of the reference front."""
    if not len(front):
        return 0.0
    points = nondominated(np.round(front, 6) + 0.0)
    return hypervolume(points, reference['hv_reference_point']) / reference['hypervolume']


def ipro_fronts(run_dir: Path, record: dict, limit: int) -> list[tuple[int, np.ndarray]]:
    """(step, front) of one IPRO run: empty at step 0, the oracle answers so far after each call, the final front."""
    path = newest(run_dir, 'oracle_queries.jsonl')
    calls = [json.loads(line) for line in path.read_text().split('\n')[:-1] if line]
    steps = np.array([c['training_steps'] for c in calls], float)
    answers = np.array([c['returned_vector'] for c in calls], float) if calls else np.empty((0, 0))
    finite = np.isfinite(answers).all(1) if len(answers) else np.zeros(0, bool)
    steps = steps[finite]
    answers = answers[finite]
    at = np.unique(steps[steps < record['step']])
    if len(at) > limit:
        at = at[np.unique(np.linspace(0, len(at) - 1, limit).round().astype(int))]
    return ([(0, answers[:0])] + [(int(step), answers[steps <= step]) for step in at if step > 0]
            + [(record['step'], record['front'])])


def curves_of(args: argparse.Namespace, env: str, reference: dict) -> list[dict]:
    """One row per run and evaluation of one environment; stops if a run is missing."""
    algorithms = {'ppo': ('MO-PPO', 'mo_ppo_nl_baselines', True),
                  'sac': ('MO-SAC', 'mo_sac_nl_baselines', True),
                  'gpi_ls': ('GPI-LS', 'gpi_ls', False),
                  'pcn': ('PCN', 'pcn', False),
                  'c_morl': ('C-MORL', 'c_morl', False),
                  'morl_d': ('MORL/D', 'morl_d', False)}
    if env in ('ant', 'halfcheetah', 'hopper', 'humanoid'):
        sources = ('ppo', 'sac', 'c_morl', 'morl_d', 'gpi_ls')
    else:
        sources = ('ppo', 'sac', 'gpi_ls', 'pcn', 'ipro')
    rows = []
    problems = []
    for algo in sources:
        if algo == 'ipro':
            records, missing = ipro_records(args, env)
            problems += missing
            for r in records:
                if args.ipro_final_only:
                    fronts = [(r['step'], r['front'])]
                else:
                    base = args.results_root / args.prefix / 'ipro' / env
                    found = sorted(p for p in base.rglob(f"s_{r['seed']:04d}") if p.is_dir())
                    if len(found) != 1:
                        problems.append(f"IPRO seed {r['seed']}: {len(found)} run folders s_{r['seed']:04d} "
                                        f'under {base}')
                        continue
                    fronts = ipro_fronts(found[0], r, args.max_evaluations)
                rows += [dict(environment=env, method='IPRO', label='IPRO', seed=r['seed'], step=step,
                              hv_ratio=hv_ratio(front, reference)) for step, front in fronts]
            continue
        method, directory, archive = algorithms[algo]
        variant = dict(ppo=args.ppo_variant, sac=args.sac_variant).get(algo, '')
        for seed in args.seeds:
            dirs = run_dirs(args, directory, env, variant, seed)
            if len(dirs) != 1:
                problems.append(f'{method} {variant} seed {seed}: {len(dirs)} run folders s_{seed:04d} under '
                                f'{args.results_root / args.prefix / directory / env}')
                continue
            stores = front_stores(dirs[0], archive)
            if len(stores) != 1:
                problems.append(f'{method} {variant} seed {seed}: {len(stores)} front stores in {dirs[0]}')
                continue
            rows += [dict(environment=env, method=method, label=f'{method} {variant}'.strip(), seed=seed, step=step,
                          hv_ratio=hv_ratio(front, reference))
                     for step, front in stored_fronts(stores[0], args.max_evaluations)]
    if problems:
        sys.exit(f'STOP: {env}: {len(problems)} runs missing:\n  ' + '\n  '.join(problems[:40]))
    return rows


# Statistics

def held(steps: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The value of the last evaluation at or before each position; NaN before the first evaluation."""
    i = np.searchsorted(steps, at, side='right') - 1
    return np.where(i >= 0, values[np.maximum(i, 0)], np.nan)


def runs_of(table: pd.DataFrame, env: str, method: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], float]:
    """Per seed the evaluation steps and HV ratios of one method in one environment, and the method's budget."""
    data = table[table.environment.eq(env) & table.method.eq(method)].sort_values('step')
    runs = [(g.step.to_numpy(float), g.hv_ratio.to_numpy(float)) for _, g in data.groupby('seed')]
    return runs, (float(data.step.max()) if len(data) else np.nan)


def mean_band(curves: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean over seeds (rows) and the pointwise 95% seed-bootstrap interval; NaN where a seed has no value yet."""
    mean = np.full(curves.shape[1], np.nan)
    low = np.full(curves.shape[1], np.nan)
    high = np.full(curves.shape[1], np.nan)
    ok = np.isfinite(curves).all(0)
    if ok.any():
        resampled = curves[:, ok][rng.integers(0, len(curves), (2000, len(curves)))].mean(1)
        mean[ok] = curves[:, ok].mean(0)
        low[ok], high[ok] = np.percentile(resampled, [2.5, 97.5], axis=0)
    return mean, low, high


# Figures

def layout(n_rows: int, n_cols: int, **shared: Any) -> tuple[plt.Figure, np.ndarray]:
    """Panels at text width, each with its own tick labels, and room for one x label and one legend row."""
    height = 1.32 * n_rows + 0.92
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH, height), squeeze=False, **shared)
    fig.subplots_adjust(left=0.075, right=0.99, top=1 - 0.24 / height, bottom=0.80 / height, wspace=0.30, hspace=0.42)
    return fig, axes


def finish(fig: plt.Figure,
           axes: np.ndarray,
           xlabel: str,
           ylabel: str,
           methods: Sequence[str],
           names: dict,
           markers_only: Sequence[str] = ()) -> None:
    """Shared y axes, one x label and one legend row; methods in markers_only get a legend entry without a line."""
    for (r, c), ax in np.ndenumerate(axes):
        ax.set(ylim=(0, 1.02), yticks=[0, 0.5, 1])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=9)
    center = (fig.subplotpars.left + fig.subplotpars.right) / 2
    height = fig.get_figheight()
    end = dict(linestyle='', markersize=5.5, markeredgecolor='black', markeredgewidth=0.5)
    fig.text(center, 0.40 / height, xlabel, ha='center', va='center', fontsize=9)
    fig.legend(handles=[Line2D([], [], label=names.get(m, m), **style(m, **end if m in markers_only else {}))
                        for m in methods], ncol=len(methods),
               loc='lower center', bbox_to_anchor=(center, 0), frameon=False, fontsize=8)


def draw(ax: plt.Axes,
         x: np.ndarray,
         curves: np.ndarray,
         method: str,
         order: int,
         rng: np.random.Generator) -> None:
    """Mean curve of one method with its interval; the markers of different methods are offset against each other."""
    mean, low, high = mean_band(curves, rng)
    kw = style(method, markevery=(8 * order + 4, 40))
    ax.fill_between(x, low, high, color=kw['color'], alpha=0.15, linewidth=0)
    ax.plot(x, mean, **kw)


def end_marker(ax: plt.Axes,
               x: float,
               finals: np.ndarray,
               method: str,
               rng: np.random.Generator) -> None:
    """A method with a final front only: its mean final value with the seed-bootstrap interval."""
    mean, low, high = mean_band(finals[:, None], rng)
    kw = style(method, linestyle='', markersize=5.5, markeredgecolor='black', markeredgewidth=0.5)
    ax.errorbar([x], mean, yerr=[mean - low, high - mean], ecolor='black', elinewidth=0.8, capsize=2, zorder=3, **kw)


def methods_in(table: pd.DataFrame, envs: Sequence[str]) -> list[str]:
    """Methods with runs in the environments, in legend order."""
    present = set(table[table.environment.isin(envs)].method)
    return [m for m in ('MO-PPO', 'MO-SAC', 'GPI-LS', 'PCN', 'IPRO', 'C-MORL', 'MORL/D') if m in present]


def share_figure(table: pd.DataFrame,
                 envs: Sequence[str],
                 names: dict,
                 columns: int = 4) -> plt.Figure:
    """HV ratio over the share of each run's own budget, one panel per environment."""
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 201)
    columns = min(columns, len(envs))
    fig, axes = layout(-(-len(envs) // columns), columns, sharex=True)
    methods = methods_in(table, envs)
    markers_only = set()
    for ax, env in zip(axes.flat, envs):
        for order, method in enumerate(methods):
            runs, _ = runs_of(table, env, method)
            if not runs:
                continue
            if all(len(steps) == 1 for steps, _ in runs):
                markers_only.add(method)
                end_marker(ax, 100, np.array([values[-1] for _, values in runs]), method, rng)
            else:
                draw(ax, 100 * grid, np.array([held(s, v, grid * s[-1]) for s, v in runs]), method, order, rng)
        ax.set_title(name(env), fontsize=10)
        ax.set(xlim=(-3, 104), xticks=[0, 25, 50, 75, 100])
    for i in range(len(envs), axes.size):
        axes.flat[i].set_visible(False)
        axes.flat[i - columns].tick_params(labelbottom=True)
    finish(fig, axes, "Share of each run's own training budget (%)", 'HV ratio', methods, names, markers_only)
    return fig


def pair_figure(table: pd.DataFrame, envs: Sequence[str], names: dict) -> plt.Figure:
    """HV ratio over environment steps for the pairs with the same learner, each row cut at 3 million counted steps."""
    rng = np.random.default_rng(0)
    pairs = (('MO-PPO', 'C-MORL'), ('MO-SAC', 'MORL/D'))
    transitions = {'MO-PPO': 8, 'C-MORL': 8}
    fig, axes = layout(len(pairs), len(envs), sharey='col')
    for r, pair in enumerate(pairs):
        end = 3_000_000 * max(transitions.get(m, 1) for m in pair) / 1e6
        for c, env in enumerate(envs):
            for order, method in enumerate(pair):
                runs, budget = runs_of(table, env, method)
                if runs:
                    steps = np.linspace(0, min(budget, 3_000_000), 201)
                    x = steps * transitions.get(method, 1) / 1e6
                    draw(axes[r, c], x, np.array([held(s, v, steps) for s, v in runs]), method, order, rng)
            axes[r, c].set(xlim=(0, end), xticks=np.linspace(0, end, 4))
            if r == 0:
                axes[r, c].set_title(name(env), fontsize=10)
    finish(fig, axes, 'Environment steps (millions)', 'HV ratio', [m for pair in pairs for m in pair], names)
    return fig


def captions(names: dict, ipro_curve: bool) -> dict:
    """Figure captions in the style of the paper."""
    ours = f"{names['MO-PPO']} and {names['MO-SAC']} archives"
    shared = (f'Each curve shows the HV ratio of the front stored at each evaluation, with the reference front and HV '
              f"reference point of the final metrics. The horizontal axis gives the share of each run's own training "
              f'budget, which ends at its final step. Solid lines show the {ours}, and dashed lines show the '
              f'baselines. {FIGURE_STATS}')
    ipro = (' The front of IPRO at a step is the nondominated set of the oracle answers returned up to that step. '
            'Before the first answer, the front is empty and its HV ratio is 0. The IPRO runs stop at different steps.'
            if ipro_curve else ' IPRO stores only a final front and is shown as a single marker.')
    return dict(discrete='Learning curves of the baseline comparison on the environments with discrete actions. '
                         + shared + ipro,
                mujoco='Learning curves of the baseline comparison on the MuJoCo environments. ' + shared,
                pairs=(f'Learning curves over environment steps of the two pairs that share a learner on the MuJoCo '
                       f"environments. The top row shows {names['MO-PPO']} and C-MORL, and the bottom row shows "
                       f"{names['MO-SAC']} and MORL/D. MO-PPO and C-MORL collect transitions in 8 parallel "
                       f'environments, so the top row spans {3_000_000 * 8 / 1e6:g} million and the bottom row '
                       f'{3_000_000 / 1e6:g} million environment steps. {FIGURE_STATS}'))


# Main

def parse_args() -> argparse.Namespace:
    """Command line arguments; 'dst' and 'mujoco' in --envs expand to their variants."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--envs', nargs='+', default=['dst', 'minecart', 'reacher', 'mujoco'],
                        help="Default: all environments; 'dst' and 'mujoco' stand for their variants")
    parser.add_argument('--ppo_variant', default='ws_raw',
                        choices=('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm'))
    parser.add_argument('--sac_variant', default='ws_raw', choices=('ws_raw', 'ws_norm'))
    parser.add_argument('--reference_fronts', type=Path, nargs='+',
                        default=[Path('results/evaluation/scalarization/reference_fronts'),
                                 Path('results/evaluation/scalarization/reference_fronts_mujoco')],
                        help='Output folders of build_reference_fronts.py, searched in this order')
    parser.add_argument('--results_root', type=Path, default=Path('results'))
    parser.add_argument('--prefix', default='scalarization/baselines',
                        help="Launchers' path prefix below --results_root")
    parser.add_argument('--ipro_fronts', type=Path,
                        default=Path('results/evaluation/scalarization/ipro_checks/ipro_fronts.csv'))
    parser.add_argument('--ipro_final_only', action='store_true',
                        help="Do not read IPRO's oracle_queries.jsonl: draw IPRO as one marker at its final front")
    parser.add_argument('--metrics', type=Path,
                        default=Path('results/evaluation/scalarization/metrics/metrics_runs.csv'),
                        help='metrics_runs.csv of compute_metrics_baseline.py; if present, the curve ends are compared')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--max_evaluations', type=int, default=200, help='Evaluations scored per run (evenly spaced)')
    parser.add_argument('--output', type=Path, default=Path('results/evaluation/scalarization/learning_curves'))
    args = parser.parse_args()
    groups = dict(dst=('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2'),
                  mujoco=('ant', 'halfcheetah', 'hopper', 'humanoid'))
    args.envs = [env for group in args.envs for env in groups.get(group, (group,))]
    args.calibration = False
    return args


def compare_with_metrics(table: pd.DataFrame, path: Path) -> None:
    """Largest difference between the curve ends and the HV ratios of compute_metrics_baseline.py, if present."""
    if not path.is_file():
        print(f'\n{path} not found: curve ends not compared with compute_metrics_baseline.py')
        return
    ends = table.sort_values('step').groupby(['environment', 'label', 'seed'], as_index=False).last()
    both = ends.merge(pd.read_csv(path), on=['environment', 'label', 'seed'], suffixes=('', '_metrics'))
    if both.empty:
        print(f'\nNo runs in common with {path}')
        return
    worst = both.loc[(both.hv_ratio - both.hv_ratio_metrics).abs().idxmax()]
    print(f'\nCurve ends against {path}: {len(both)} runs, largest difference in HV ratio '
          f'{abs(worst.hv_ratio - worst.hv_ratio_metrics):.2e} ({worst.environment}, {worst.label}, seed {worst.seed})')


def main() -> None:
    """Scores every stored front, draws the learning curve figures and compares the curve ends with the metrics."""
    # Arguments
    args = parse_args()

    # Curves
    rows = []
    for env in args.envs:
        if env.startswith(('dst', 'ldst')):
            reference, _ = exact_reference(args, env)
        else:
            reference, _ = constructed_reference(args, env)
        reference['hypervolume'] = hypervolume(reference['front'], reference['hv_reference_point'])
        rows += curves_of(args, env, reference)
    table = pd.DataFrame(rows)
    names = {'MO-PPO': f'MO-PPO ({label(args.ppo_variant)})', 'MO-SAC': f'MO-SAC ({label(args.sac_variant)})'}
    appendix = args.output / 'paper' / 'appendix'
    appendix.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / 'learning_curves.csv', index=False, float_format='%.10g')

    # Figures
    discrete = [e for e in ('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2', 'minecart', 'reacher') if e in args.envs]
    mujoco = [e for e in ('ant', 'halfcheetah', 'hopper', 'humanoid') if e in args.envs]
    ipro = table[table.method.eq('IPRO')]
    ipro_curve = len(ipro) > ipro.groupby(['environment', 'seed']).ngroups
    text = captions(names, ipro_curve)
    figures = []
    for envs, stem, tag, group in ((discrete, 'baseline_curves', 'fig:baseline-curves', 'discrete'),
                                   (mujoco, 'baseline_curves_mujoco', 'fig:baseline-curves-mujoco', 'mujoco')):
        if envs:
            save(share_figure(table, envs, names), appendix / f'{stem}.pdf')
            figures.append(figure_tex(f'appendix/{stem}.pdf', text[group], tag))
    if mujoco and all(m in set(table.method) for m in ('MO-PPO', 'C-MORL', 'MO-SAC', 'MORL/D')):
        save(pair_figure(table, mujoco, names), appendix / 'baseline_pairs.pdf')
        figures.append(figure_tex('appendix/baseline_pairs.pdf', text['pairs'], 'fig:baseline-pairs'))
    (appendix / 'baseline_curves_figures.tex').write_text('\n'.join(figures), encoding='utf-8')

    # Summary
    finals = table.sort_values('step').groupby(['environment', 'method', 'seed'], sort=False).last().reset_index()
    print('Final HV ratio (mean over seeds) and final step:')
    for env, group in finals.groupby('environment', sort=False):
        print(f'\n{name(env)}')
        for method in methods_in(group, [env]):
            runs = group[group.method.eq(method)]
            print(f'  {names.get(method, method):18s} HV ratio {runs.hv_ratio.mean():.4f}   '
                  f'final step {runs.step.mean():14,.0f}')
    compare_with_metrics(table, args.metrics)
    print(f'\nOutput: {args.output}')


if __name__ == '__main__':
    main()