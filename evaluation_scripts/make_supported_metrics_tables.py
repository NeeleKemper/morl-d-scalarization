"""Table of the unsupported returns within every run's own front: mean +- sample standard deviation over seeds.

Reads supported_own_runs.csv of supported_share.py and writes <input>/paper/supported_own.tex (label
tab:unsupported-own).

  python table_supported_paper.py
  python table_supported_paper.py --input results/evaluation/scalarization/supported --resolution 0.01
"""
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from paper_style import SEED_STATS, cell, head, table_tex


def cells(stats: pd.DataFrame, envs: list[str], label: str) -> list[str]:
    """One entry per environment: mean +- standard deviation in percent, or a dash if the method has no runs."""
    return [cell(*stats.loc[(label, e)]) if (label, e) in stats.index else '--' for e in envs]


def main() -> None:
    """Builds the table of unsupported returns within the own fronts and writes it as LaTeX."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=Path('results/evaluation/scalarization/supported'),
                        help='Output folder of supported_share.py')
    parser.add_argument('--resolution', type=float, default=0.01,
                        help='Resolution eps of supported_share.py to report')
    args = parser.parse_args()

    # Table
    env_names = {'minecart': 'Minecart', 'reacher': 'Reacher', 'ant': 'Ant', 'halfcheetah': 'HalfCheetah',
                 'hopper': 'Hopper', 'humanoid': 'Humanoid'}
    scalarizer_names = {'ws_raw': r'\wsraw{}', 'ws_norm': r'\wsnorm{}', 'tch_raw': r'\tchraw{}',
                        'tch_norm': r'\tchnorm{}', 'stch_raw': r'\stchraw{}', 'stch_norm': r'\stchnorm{}'}
    runs = pd.read_csv(args.input / 'supported_own_runs.csv')
    runs = runs[np.isclose(runs.eps, args.resolution) & runs.environment.isin(env_names)]
    stats = runs.groupby(['label', 'environment']).own_unsupported_share.agg(['mean', 'std'])
    labels = set(stats.index.get_level_values('label'))
    envs = [e for e in env_names if e in set(runs.environment)]
    rows = []
    for learner in ('MO-PPO', 'MO-SAC'):
        used = [s for s in scalarizer_names if f'{learner} {s}' in labels]
        rows += [[learner if n == 0 else '', scalarizer_names[s], *cells(stats, envs, f'{learner} {s}')]
                 for n, s in enumerate(used)]
        if used:
            rows.append(r'\addlinespace')
    rows += [[f'\\multicolumn{{2}}{{l}}{{{b}}}', *cells(stats, envs, b)]
             for b in ('GPI-LS', 'PCN', 'IPRO', 'C-MORL', 'MORL/D') if b in labels]
    percent = f'{100 * args.resolution:g}\\%'
    table = table_tex(
        'tab:unsupported-own',
        'Unsupported returns (\\%) within the own front of each run on the additional environments, at a resolution '
        f'of {percent}. Returns within this distance of each other in every objective count once. {SEED_STATS} '
        'A dash indicates that the method was not run on the environment.',
        'll' + 'r' * len(envs), [head('Algorithm', 'Scalarizer', *(env_names[e] for e in envs))], rows, wide=True)

    # Output
    output = args.input / 'paper'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'supported_own.tex').write_text(table, encoding='utf-8')
    print(f'Supported paper outputs: {output}')


if __name__ == '__main__':
    main()