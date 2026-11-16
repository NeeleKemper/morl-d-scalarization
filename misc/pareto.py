import numpy as np

from copy import deepcopy
from typing import Any

from misc.metric import filter_pareto_dominated


def calculate_crowding_distance(evals: list[np.ndarray]) -> np.ndarray:
    """Return the normalized crowding distance of every evaluation; boundary points get infinity."""
    arr = np.vstack(evals)
    n, m = arr.shape
    distances = np.zeros(n, dtype=float)
    for obj in range(m):
        vals = arr[:, obj]
        order = np.argsort(vals)
        norm = vals[order[-1]] - vals[order[0]] or 1.0
        for i in range(1, n - 1):
            distances[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / norm
        if n > 1:
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
    return distances


class ParetoArchive:
    def __init__(self, max_size: int = None):
        """Archive of unique non-dominated individuals, truncated by crowding distance beyond max_size."""
        self.max_size = max_size
        self.individuals: list = []
        self.evaluations: list[np.ndarray] = []

    def _truncate_by_crowding_distance(self) -> None:
        """Keep the boundary points and then the interior points with the largest crowding distance."""
        dists = calculate_crowding_distance(self.evaluations)
        extreme_idx = [i for i, d in enumerate(dists) if np.isinf(d)]
        interior_idx = sorted((i for i, d in enumerate(dists) if np.isfinite(d)), key=lambda i: dists[i], reverse=True)
        keep_idx = (extreme_idx + interior_idx)[:self.max_size]
        self.evaluations = [self.evaluations[i] for i in keep_idx]
        self.individuals = [self.individuals[i] for i in keep_idx]

    def _is_duplicate(self, evaluation: np.ndarray) -> bool:
        """Return whether the evaluation is already in the archive."""
        return any(np.array_equal(evaluation, e) for e in self.evaluations)

    def add(self, candidate: Any, evaluation: np.ndarray) -> None:
        """Add a copy of the candidate unless it is a duplicate, then keep only the non-dominated individuals."""
        if self._is_duplicate(evaluation):
            return
        self.evaluations.append(evaluation)
        self.individuals.append(deepcopy(candidate))

        # Pareto filtering (first occurrence of every non-dominated evaluation)
        nd_candidates = {tuple(x) for x in filter_pareto_dominated(self.evaluations)}
        non_dominated_evals = []
        non_dominated_tuples = []
        non_dominated_individuals = []
        for e, ind in zip(self.evaluations, self.individuals):
            t = tuple(e)
            if t in nd_candidates and t not in non_dominated_tuples:
                non_dominated_evals.append(e)
                non_dominated_tuples.append(t)
                non_dominated_individuals.append(ind)
        self.evaluations = non_dominated_evals
        self.individuals = non_dominated_individuals

        if self.max_size is not None and len(self.evaluations) > self.max_size:
            self._truncate_by_crowding_distance()
