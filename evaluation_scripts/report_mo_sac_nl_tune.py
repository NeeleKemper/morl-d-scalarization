import csv
import json
import argparse

from pathlib import Path


def load_summaries(root: Path, scalarization_method: str, seeds: list[int] | None) -> list:
    """Every candidate summary written by tune_mo_sac_nl.py below results/<path_prefix>."""
    summaries = []
    for path in sorted(root.glob(f'*/{scalarization_method}/*/summary_seeds_*.json')):
        summary = json.loads(path.read_text(encoding='utf-8'))
        if seeds is None or summary['seeds'] == seeds:
            summary['summary_path'] = str(path)
            summaries.append(summary)
    return summaries


def candidate_label(summary: dict) -> str:
    """Learning rate, batch size and alpha of a candidate; 'config' where the configuration value is used."""
    candidate = summary['candidate']
    return (f"lr={candidate['learning_rate'] if candidate['learning_rate'] is not None else 'config':<8} "
            f"batch={candidate['batch_size'] if candidate['batch_size'] is not None else 'config':<6} "
            f"alpha={candidate['alpha'] if candidate['alpha'] is not None else 'config'}")


def ranked(rows: list, metric: str, top: int) -> list:
    """Best candidates by the mean of the metric (lower is better for IGD+); candidates without it are dropped."""
    scored = [row for row in rows if row['mean'] is not None]
    scored.sort(key=lambda row: row['mean'], reverse=metric not in ('igd_plus_raw', 'igd_plus_normalized'))
    return scored[:top]


def print_table(title: str,
                rows: list,
                metric: str,
                top: int) -> None:
    """Ranking of the best candidates with mean, standard deviation, seeds and protocol."""
    print(f'\n{title}')
    best = ranked(rows, metric, top)
    if not best:
        print(f'  no candidate has {metric}.')
        return
    for rank, row in enumerate(best, start=1):
        std = 'n/a' if row['std'] is None else f"{row['std']:.4f}"
        print(f"  {rank}. {row['label']}  {metric}={row['mean']:.4f} +/- {std} "
              f"(seeds={row['seeds']}, {row['protocol_id']})")


def main() -> None:
    """Reports the best MO-SAC-NL tuning candidates per environment and aggregated over DST/LDST."""
    # Arguments
    metrics_all = ('hv_ratio', 'hv_raw', 'hv_unit_box', 'igd_plus_normalized', 'igd_plus_raw')
    parser = argparse.ArgumentParser(
        description='Report the best MO-SAC-NL tuning configurations: the top candidates per '
                    'environment and aggregated over all DST/LDST variants.')
    parser.add_argument('--metric', choices=metrics_all, default='hv_ratio',
                        help='Ranking metric; hv_ratio needs an exact reference front (DST/LDST only).')
    parser.add_argument('--fallback_metric', choices=metrics_all, default='hv_raw',
                        help='Used for environments without the ranking metric (Minecart/Reacher).')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Only summaries for exactly these training seeds; default: every summary found.')
    parser.add_argument('--scalarization_method', choices=('ws_raw', 'ws_norm'), default='ws_raw')
    parser.add_argument('--path_prefix', type=str, default='scalarization/mo_sac_tune')
    parser.add_argument('--csv', type=str, default=None,
                        help='Optional path for the full per-environment ranking as CSV.')
    args = parser.parse_args()

    # Summaries
    dst_envs = ('dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2')
    root = Path('results') / args.path_prefix
    summaries = load_summaries(root, args.scalarization_method, args.seeds)
    print(f'{len(summaries)} candidate summaries below {root}.')
    per_environment = {}
    aggregate_rows = []
    csv_rows = []
    for summary in summaries:
        base = {'label': candidate_label(summary), 'protocol_id': summary['protocol_id'],
                'seeds': ','.join(map(str, summary['seeds']))}
        for name, metrics in summary['per_environment'].items():
            metric = args.metric if metrics[args.metric]['mean'] is not None else args.fallback_metric
            per_environment.setdefault(name, {'metric': metric, 'rows': []})
            per_environment[name]['rows'].append({**base, **metrics[metric]})
            csv_rows.append({**base, 'environment': name, **{
                f'{key}_{field}': metrics[key][field] for key in metrics_all for field in ('mean', 'std', 'n')}})
        if tuple(summary['environments']) == dst_envs and args.metric in summary['aggregate']:
            aggregate_rows.append({**base, **summary['aggregate'][args.metric]})

    # Rankings
    for name in sorted(per_environment):
        group = per_environment[name]
        note = '' if group['metric'] == args.metric else f" (no {args.metric}; ranked by {group['metric']})"
        print_table(f'Top {args.top} for {name}{note}', group['rows'], group['metric'], args.top)
    print_table(f"Top {args.top} aggregated over {', '.join(dst_envs)}", aggregate_rows, args.metric, args.top)
    if not aggregate_rows:
        print('  run the sweep with --env all_dst to get this aggregate.')

    # Output
    if args.csv is not None:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in csv_rows for k in r)))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f'\nFull per-environment table: {path}')


if __name__ == '__main__':
    main()