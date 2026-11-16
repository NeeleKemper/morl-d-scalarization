"""One tidy E1 summary CSV; rates are calculated within runs before seed averaging."""
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from reconstruct_e1 import DATA_DIR, MATCH_ATOL, QUALITY_COLUMNS, RUN_KEYS, boolean, standard_k

ALLOCATIONS = ((5, 100), (10, 50), (20, 25), (50, 10))


# Data

def load_evidence(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run and policy evidence of reconstruct_e1.py with nullable Boolean flags."""
    runs = pd.read_csv(directory / 'e1_runs.csv')
    policies = pd.read_csv(directory / 'e1_policy_diagnostics.csv')
    for frame, columns in ((runs, ('include_in_pooled', 'complete_archive', 'dense_complete_archive')),
                           (policies, ('include_in_pooled', 'identity_verified', 'matches_assigned_target',
                                       'ever_matched_assigned_target', 'zero_treasure', 'own_target_treasure_reached',
                                       'score_tied'))):
        for col in columns:
            frame[col] = frame[col].map(boolean, na_action='ignore').astype('boolean')
    return runs, policies


# Rates

def return_rates(g: pd.DataFrame) -> dict:
    """Numerator and denominator of every return-based rate per run row."""
    n = g['num_pf_points'].to_numpy(float)
    t = g['num_oracle_targets'].to_numpy(float)
    s = g['num_supported_points'].to_numpy(float)
    a = g.archive_count.to_numpy(float)
    p = g.active_count.to_numpy(float)
    ones = np.ones(len(g))
    parts = {
        'oracle_recall': (t, n), 'archive_recall': (a, n), 'active_recall': (p, n),
        'archive_target_coverage': (g.archive_targeted_count, t),
        'archive_off_target_recall': (g.archive_off_target_count, n - t),
        'archive_unsupported_recall': (g.archive_unsupported_count, n - s),
        'active_unsupported_recall': (g.active_unsupported_count, n - s),
        'active_target_coverage': (g.active_targeted_count, t),
        'discovered_target_absence_rate': (g.discovered_target_absent_count, g.archive_targeted_count),
        'discovered_off_target_absence_rate': (g.discovered_off_target_absent_count, g.archive_off_target_count),
        'assigned_hit_ever': (g.assigned_hit_ever_count, g.assigned_policy_count),
        'assigned_hit_active': (g.assigned_hit_active_count, g.assigned_policy_count),
        'complete_archive': (g.complete_archive.astype(int), ones),
        'complete_retention_of_dense': ((g.complete_archive & g.dense_complete_archive).astype(int),
                                        g.dense_complete_archive.astype(int)),
        'cadence_missed_recall': (g.dense_archive_count - g.archive_count, n),
    }
    parts.update({name: (g[name], ones) for name in QUALITY_COLUMNS})
    return parts


def flag(actors: pd.DataFrame, col: str) -> np.ndarray:
    """Boolean column with missing values as False."""
    return actors[col].fillna(False).to_numpy(bool)


def actor_rates(policies: pd.DataFrame) -> pd.DataFrame:
    """Per-run actor counts for all, corner and interior preferences."""
    rows = []
    for key, actors in policies.groupby(RUN_KEYS + ['horizon'], sort=False):
        corner = actors.w0.notna() & actors.w1.notna() & (actors.w0.eq(0) | actors.w1.eq(0))
        known_weights = actors.w0.notna() & actors.w1.notna()
        for part, p in (('all', actors), ('corner', actors.loc[corner]),
                        ('interior', actors.loc[known_weights & ~corner])):
            assigned = p.matches_assigned_target.notna().to_numpy()
            own = flag(p, 'matches_assigned_target')
            ever = flag(p, 'ever_matched_assigned_target')
            scored = p.scalarized_gap.notna().to_numpy()
            miss = scored & assigned & ~own
            pf = p.matched_pf_id.notna().to_numpy()
            extra = flag(p, 'own_target_treasure_reached') & p.excess_steps.gt(MATCH_ATOL).fillna(False).to_numpy(bool)
            rates = {
                'own_target_rate': (own.sum(), assigned.sum()),
                'own_target_ever_rate': (ever.sum(), p.ever_matched_assigned_target.notna().sum()),
                'absent_after_own_hit_rate': ((ever & ~own).sum(), ever.sum()),
                'other_grid_target_rate': (p.front_relation.eq('other_grid_target').sum(), assigned.sum()),
                'off_target_pareto_rate': (p.front_relation.eq('off_target_pareto').sum(), len(p)),
                'off_front_rate': ((~pf).sum(), len(p)),
                'no_treasure_rate': (flag(p, 'zero_treasure').sum(), len(p)),
                'own_target_treasure_extra_rate': (extra.sum(), assigned.sum()),
                'redundant_pareto_actor_rate': (pf.sum() - p.loc[pf, 'matched_pf_id'].nunique(), len(p)),
                'score_tied_miss_rate': ((miss & flag(p, 'score_tied')).sum(), miss.sum()),
                'scored_actor_rate': (scored.sum(), len(p)),
                'identity_verified_rate': (flag(p, 'identity_verified').sum(), len(p)),
                'scalarized_gap': (p.scalarized_gap.sum(), scored.sum()),
            }
            base = dict(zip(RUN_KEYS + ['horizon'], key)) | {'preference_group': part}
            for metric, (num, den) in rates.items():
                rows.append(base | dict(metric=metric, numerator=float(num), denominator=int(den)))
    return pd.DataFrame(rows)


def aggregate_rates(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average run rates within seed; keep count ratios as separate statistics."""
    data = data.copy()
    data['value'] = np.divide(data.numerator, data.denominator,
                              out=np.full(len(data), np.nan), where=data.denominator > 0)
    blocks = data.groupby(keys + ['seed'], observed=True)['value'].mean()
    estimates = blocks.groupby(level=keys, observed=True).agg(mean='mean', std='std', n_seeds='count')
    counts = data.groupby(keys, observed=True).agg(n_runs=('value', 'size'), n_defined=('value', 'count'),
                                                 numerator=('numerator', 'sum'), denominator=('denominator', 'sum'))
    result = estimates.join(counts).reset_index()
    result['pooled'] = result.numerator.div(result.denominator.where(result.denominator > 0))
    return result


def make_summary(runs: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    """Rates per condition, pooled over the standard populations, over all K and at the equal budgets."""
    keys = ['source', 'scope', 'environment', 'scalarizer', 'K', 'horizon', 'cadence', 'preference_group', 'metric']
    parts = []
    for metric, (num, den) in return_rates(runs).items():
        part = runs[RUN_KEYS + ['horizon', 'cadence', 'include_in_pooled']].copy()
        part['metric'] = metric
        part['numerator'] = np.asarray(num, float)
        part['denominator'] = np.asarray(den, float)
        parts.append(part)
    returns = pd.concat(parts, ignore_index=True).assign(source='returns', preference_group='all')
    standard = returns.loc[returns.K.eq(returns.environment.map(standard_k))]
    budget = returns.loc[returns.cadence.eq(1) &
                         pd.MultiIndex.from_frame(returns[['K', 'horizon']]).isin(ALLOCATIONS)]
    actors = actor_rates(policies).assign(source='actors', cadence=1)
    active_standard = actors.loc[actors.K.eq(actors.environment.map(standard_k)) & actors.metric.ne('scalarized_gap')]
    data = pd.concat([
        returns.assign(scope='condition'),
        standard.assign(scope='standard', environment='ALL', K=-1),
        returns.loc[returns.cadence.eq(1) & returns.include_in_pooled].assign(
            scope='all_K', environment='ALL', scalarizer='ALL', K=-1),
        budget.assign(scope='budget', environment='ALL'),
        actors.assign(scope='condition'),
        active_standard.assign(scope='standard', environment='ALL', K=-1),
    ], ignore_index=True)
    result = aggregate_rates(data, keys)
    result.loc[result.metric.isin(QUALITY_COLUMNS), ['numerator', 'denominator', 'pooled']] = np.nan
    result.loc[result.K.eq(-1), 'K'] = np.nan
    return result


def main() -> None:
    """Writes e1_summary.csv next to the evidence of reconstruct_e1.py."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR, help='Directory produced by reconstruct_e1.py')
    args = parser.parse_args()

    # Summary
    runs, policies = load_evidence(args.input)
    summary = make_summary(runs, policies)
    summary.to_csv(args.input / 'e1_summary.csv', index=False, float_format='%.17g')
    print(f"Saved {len(summary)} rows to {args.input / 'e1_summary.csv'}")


if __name__ == '__main__':
    main()