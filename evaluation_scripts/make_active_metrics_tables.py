"""Appendix table: HV ratio of the final archive and the final active population of MO-PPO on the additional
environments, mean +- sample standard deviation over seeds. Reads active_runs.csv of compute_metrics_active.py and
writes <input>/paper/active_population.tex (label tab:active-additional).

  python table_active_paper.py
  python table_active_paper.py --input results/evaluation/scalarization/metrics_active
"""
import argparse

import pandas as pd

from pathlib import Path

from paper_style import SEED_STATS, head, table_tex


def main() -> None:
    """Builds the archive and active population table and writes it as LaTeX."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=Path('results/evaluation/scalarization/metrics_active'),
                        help='Output folder of compute_metrics_active.py')
    args = parser.parse_args()

    # Table
    env_names = {'minecart': 'Minecart', 'reacher': 'Reacher', 'ant': 'Ant', 'halfcheetah': 'HalfCheetah',
                 'hopper': 'Hopper', 'humanoid': 'Humanoid'}
    scalarizer_names = {'ws_raw': r'\wsraw{}', 'ws_norm': r'\wsnorm{}', 'tch_raw': r'\tchraw{}',
                        'tch_norm': r'\tchnorm{}', 'stch_raw': r'\stchraw{}', 'stch_norm': r'\stchnorm{}'}
    runs = pd.read_csv(args.input / 'active_runs.csv')
    envs = [e for e in env_names if e in set(runs.environment)]
    rows = []
    for column, title in (('archive_hv_ratio', '(a) Archive'), ('active_hv_ratio', '(b) Active population')):
        stats = runs.groupby(['scalarizer', 'environment'])[column].agg(['mean', 'std'])
        best = stats['mean'].groupby('environment').max()
        if rows:
            rows.append(r'\midrule')
        rows.append(f'\\multicolumn{{{len(envs) + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\')
        for name in (s for s in scalarizer_names if s in stats.index.get_level_values('scalarizer')):
            cells = []
            for e in envs:
                if (name, e) not in stats.index:
                    cells.append('--')
                    continue
                mean, sd = stats.loc[(name, e)]
                value = f'\\textbf{{{mean:.3f}}}' if mean == best[e] else f'{mean:.3f}'
                cells.append(f'{value}{{\\scriptsize$\\pm${sd:.3f}}}')
            rows.append([scalarizer_names[name], *cells])
    table = table_tex(
        'tab:active-additional',
        'Hypervolume ratio of (a) the final archive and (b) the final active population for MO-PPO on the additional '
        'environments. The active population consists of the returns of the actors at the end of training. '
        f'{SEED_STATS[:-1]}, with the best mean for each environment in bold.',
        'l' + 'r' * len(envs), [head('Scalarizer', *(env_names[e] for e in envs))], rows, wide=True)

    # Output
    output = args.input / 'paper'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'active_population.tex').write_text(table, encoding='utf-8')
    print(f'Active paper outputs: {output}')


if __name__ == '__main__':
    main()