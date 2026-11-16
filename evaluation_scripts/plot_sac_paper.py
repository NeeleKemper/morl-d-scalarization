"""Fixed MO-SAC versus MO-PPO paper outputs for the weighted sum: one main table, three appendix tables and one
appendix training figure with all six environments. Run reconstruct_sac.py and summarize_sac.py first. The paper says
"deterministic execution" for greedy action selection, and captions follow the wording of the paper.
"""
import argparse
import json

import numpy as np
import pandas as pd

from pathlib import Path

from paper_style import (APPENDIX_ENVS, MAIN_ENVS, POOLED_STATS, SEED_STATS, cell, envs_text, figure_tex, head, label,
                         outcome_map, save, sd_of, table_tex)
from reconstruct_sac import DATA_DIR, ENVS


class Summary:
    def __init__(self, frame: pd.DataFrame) -> None:
        """SAC summary indexed by kind, environment, algorithm, scalarizer and metric."""
        self.index = frame.set_index(['kind', 'environment', 'algorithm', 'scalarizer', 'metric'])

    def get(self,
            metric: str,
            algorithm: str,
            scalarizer: str,
            env: str = 'ALL',
            kind: str = 'measures') -> pd.Series:
        """Summary row of one measurement or comparison."""
        return self.index.loc[(kind, env, algorithm, scalarizer, metric)]

    def cell(self,
             metric: str,
             algorithm: str,
             scalarizer: str,
             env: str = 'ALL') -> str:
        """Percent, mean +- standard deviation over seeds; complete runs as a count k/n, as in E1's tables."""
        r = self.get(metric, algorithm, scalarizer, env)
        if metric == 'complete':
            n = int(r['n_seeds']) * int(r['n_environments'])
            return f"{round(float(r['mean']) * n)}/{n}"
        return cell(float(r['mean']), sd_of(r))

    def contrast(self,
                 metric: str,
                 algorithm: str,
                 scalarizer: str,
                 env: str = 'ALL') -> str:
        """Paired difference in percentage points with its 95% seed-bootstrap interval, as in E5's table."""
        r = self.get(metric, algorithm, scalarizer, env, kind='comparison')
        m = 100 * float(r['mean'])
        lo = 100 * float(r['ci_low'])
        hi = 100 * float(r['ci_high'])
        return f'{m:+.1f} [{lo:+.1f}, {hi:+.1f}]' if np.isfinite(m) else '--'


def budgets(manifest: dict) -> str:
    """Training budget of each algorithm as caption sentences."""
    runs = sorted({(s['algorithm'], s['horizon'], s['cadence'], s['transitions_per_policy'])
                   for s in manifest['schedules']})
    return ' '.join(f'MO-{a} runs for {h:,} iterations with {t:,} transitions per actor'
                    + ('.' if c == 1 else f' and is evaluated every {c} iterations.') for a, h, c, t in runs)


def tables(summary: Summary, train: pd.DataFrame, setup: str) -> tuple[str, str]:
    """Main comparison table and the three appendix tables."""
    rows_of = (('ws_raw', 'PPO'), ('ws_raw', 'SAC'), ('ws_norm', 'PPO'), ('ws_norm', 'SAC'))
    metrics = (('archive_recall', 'Archive'), ('complete', 'Complete'), ('active_recall', 'Active'),
               ('unsupported_archive_recall', r'\shortstack[r]{Unsupp.\\archive}'),
               ('unsupported_active_recall', r'\shortstack[r]{Unsupp.\\active}'))
    contrasts_of = (('PPO', 'ws_norm - ws_raw'), ('SAC', 'ws_norm - ws_raw'), ('SAC - PPO', 'ws_raw'),
                    ('SAC - PPO', 'ws_norm'))
    columns = ['Scalarizer', 'Algorithm', *(h for _, h in metrics)]
    rows = []
    for s in ('ws_raw', 'ws_norm'):
        if rows:
            rows.append(r'\midrule')
        rows += [[label(s) if a == 'PPO' else '', a] + [summary.cell(m, a, s) for m, _ in metrics]
                 for a in ('PPO', 'SAC')]
    main = table_tex('tab:sac-comparison',
                     'Recall (\\%) for weighted-sum MO-PPO and MO-SAC at the end of their respective training runs. '
                     'Archive shows Pareto returns recovered from all evaluated checkpoints ($C=1$), while Active '
                     'shows returns represented by the final actors. Complete counts runs in which the archive '
                     'recovers the entire front. The Unsupp.\\ columns report recall for unsupported returns only, '
                     f'excluding DST4, which has none. {POOLED_STATS}',
                     'll' + 'r' * len(metrics), [head(*columns)], rows, wide=True)
    rows = []
    for env in ENVS:
        if rows:
            rows.append(r'\midrule')
        rows += [[env.upper() if j == 0 else '', label(s) if a == 'PPO' else '', a]
                 + [summary.cell(m, a, s, env) for m, _ in metrics] for j, (s, a) in enumerate(rows_of)]
    environments = table_tex('tab:sac-environments',
                             'Recall (\\%) by environment for weighted-sum MO-PPO and MO-SAC at the end of their '
                             'runs, with columns defined as in Table~\\ref{tab:sac-comparison}. Dashes mark the '
                             f'environment without unsupported returns. {SEED_STATS}',
                             'lll' + 'r' * len(metrics), [head('Env.', *columns)], rows, wide=True)
    rows = []
    for metric, title in (('archive_recall', 'Archive recall'), ('active_recall', 'Active recall'),
                          ('unsupported_active_recall', 'Unsupported active recall')):
        if rows:
            rows.append(r'\midrule')
        rows.append(rf'\multicolumn{{{1 + len(contrasts_of)}}}{{l}}{{\emph{{{title}}}}} \\')
        rows += [[name] + [summary.contrast(metric, a, s, env) for a, s in contrasts_of]
                 for env, name in (*((e, e.upper()) for e in ENVS), ('ALL', 'All'))]
    raw = label('ws_raw')
    norm = label('ws_norm')
    contrasts = table_tex('tab:sac-contrasts',
                          'Paired differences in recall (percentage points) between the weighted-sum configurations. '
                          'The first two columns give normalized minus raw for each algorithm, and the last two give '
                          'MO-SAC minus MO-PPO for each scalarizer. Runs are paired by environment and seed. Entries '
                          'are mean differences over ten seeds, with 95\\% seed-bootstrap intervals in brackets. All '
                          'weights the environments equally within each seed. Unsupported active recall covers the '
                          'five environments with unsupported returns.',
                          'l' + 'r' * len(contrasts_of),
                          [rf'& \multicolumn{{2}}{{c}}{{{norm} $-$ {raw}}} & \multicolumn{{2}}{{c}}{{SAC $-$ PPO}} '
                           r'\\',
                           r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', head('Env.', 'PPO', 'SAC', raw, norm)],
                          rows, wide=True)
    zero = train.loc[train.outcome.eq('zero')].set_index(['progress', 'environment', 'scalarizer', 'algorithm']).share
    none_at = (20, 100)
    rows = []
    for p in none_at:
        if rows:
            rows.append(r'\midrule')
        title = 'At the end of the run' if p == 100 else f'At {p}\\% of the run'
        rows.append(rf'\multicolumn{{{1 + len(rows_of)}}}{{l}}{{\emph{{{title}}}}} \\')
        rows += [[name] + [cell(np.mean([zero.loc[(p, e, s, a)] for e in envs])) for s, a in rows_of]
                 for envs, name in (*(((e,), e.upper()) for e in ENVS), (ENVS, 'All'))]
    none = table_tex('tab:sac-none',
                     'Actors that collect no treasure under deterministic execution (\\%), at '
                     f"{none_at[0]}\\% of each algorithm's run and at its end. {setup} Values are means over ten "
                     'seeds. The training summary stores mean shares only, so no deviation is given. All weights the '
                     'environments equally.',
                     'l' + 'r' * len(rows_of),
                     [rf'& \multicolumn{{2}}{{c}}{{{raw}}} & \multicolumn{{2}}{{c}}{{{norm}}} \\',
                      r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', head('Env.', *(a for _, a in rows_of))], rows)
    return main, '\n'.join([environments, contrasts, none])


def figure_caption(setup: str) -> str:
    """Caption of the training figure."""
    return ('Share of actors producing each return over the progress of each run, for the weighted-sum populations '
            f'of MO-PPO and MO-SAC on all six environments. {envs_text(MAIN_ENVS)} are shown on top, '
            f'{envs_text(APPENDIX_ENVS)} below. {setup} Rows and colors are as in Fig.~\\ref{{fig:e1-training}}.')


def main() -> None:
    """Writes the SAC comparison tables and the training figure to <input>/paper."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Data
    manifest = json.loads((args.input / 'sac_manifest.json').read_text())
    summary = Summary(pd.read_csv(args.input / 'sac_summary.csv', low_memory=False))
    train = pd.read_csv(args.input / 'sac_training.csv', low_memory=False, dtype={'outcome': str})
    setup = budgets(manifest)
    out = args.input / 'paper'
    (out / 'appendix').mkdir(parents=True, exist_ok=True)

    # Tables
    main_table, appendix = tables(summary, train, setup)
    (out / 'sac_comparison.tex').write_text(main_table, encoding='utf-8')
    (out / 'appendix/sac_tables.tex').write_text(appendix, encoding='utf-8')

    # Figure
    rows = [(f'{label(s)}\n{a}', {'scalarizer': s, 'algorithm': a})
            for s, a in (('ws_raw', 'PPO'), ('ws_raw', 'SAC'), ('ws_norm', 'PPO'), ('ws_norm', 'SAC'))]
    description = ('Heatmaps in two blocks of three environments. Each block has four rows for the raw and the '
                   'normalized weighted sum under PPO and SAC. Each heatmap shows the share of actors producing each '
                   'return over training progress, with darker cells for larger shares.')
    save(outcome_map(train, (MAIN_ENVS, APPENDIX_ENVS), rows, 'progress', 'Training progress (% of the run)',
                     row_height=0.62, axis_name=''), out / 'appendix/sac_training.pdf')
    (out / 'appendix/sac_training_figure.tex').write_text(
        figure_tex('appendix/sac_training.pdf', figure_caption(setup), 'fig:sac-training', description=description),
        encoding='utf-8')
    print(f'SAC paper outputs: {out}')


if __name__ == '__main__':
    main()