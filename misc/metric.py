import numpy as np

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus


# -------------------- REFERENCE-FREE METRICS --------------------

def hypervolume(ref_point: np.ndarray, points: list[np.ndarray] | np.ndarray) -> float:
    """Hypervolume of the maximized points with respect to ref_point."""
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def sparsity(front: list[np.ndarray] | np.ndarray) -> float:
    """MORL sparsity: summed squared consecutive spacing per objective, divided by n - 1 (0 for < 2 points)."""
    if len(front) < 2:
        return 0.0
    front = np.array(front)
    per_dim = np.array([np.sum(np.square(np.diff(np.sort(front.T[dim])))) for dim in range(front.shape[1])])
    return float(np.sum(per_dim / (len(front) - 1)))


def expected_utility(front: list[np.ndarray] | np.ndarray, weights_set: np.ndarray) -> float:
    """Mean over the weights of the best linear utility in the front."""
    return np.mean(np.array([np.max([np.dot(weights, point) for point in front]) for weights in weights_set]), axis=0)


def cardinality(front: list[np.ndarray] | np.ndarray) -> int:
    """Number of points in the front."""
    return len(front)


def maximum_spread(front: list[np.ndarray] | np.ndarray) -> float:
    """Diagonal of the front's bounding box."""
    F = np.asarray(front, dtype=float)
    ranges = F.max(axis=0) - F.min(axis=0)
    return float(np.sqrt(np.sum(ranges ** 2)))


def maximum_gap(front: list[np.ndarray] | np.ndarray) -> float:
    """Largest gap between consecutive sorted values over all objectives (0 for < 2 points)."""
    F = np.asarray(front, dtype=float)
    if len(F) < 2:
        return 0.0
    return float(np.max([np.max(np.diff(np.sort(F[:, d]))) for d in range(F.shape[1])]))


# -------------------- REFERENCE-BASED METRICS --------------------

def igd(known_front: list[np.ndarray], current_estimate: list[np.ndarray]) -> float:
    """Inverted generational distance of the estimate to the known front."""
    return IGD(np.array(known_front))(np.array(current_estimate))


def igd_plus(known_front: list[np.ndarray], current_estimate: list[np.ndarray]) -> float:
    """IGD+ of the estimate to the known front, adapted to maximization by a sign flip."""
    return float(IGDPlus(np.array(known_front) * -1)(np.array(current_estimate) * -1))


def maximum_utility_loss(front: list[np.ndarray], reference_set: list[np.ndarray], weights_set: np.ndarray) -> float:
    """Largest drop of the best linear utility from the reference set to the front over all weights."""
    best_reference = [np.max([np.dot(weight, point) for point in reference_set]) for weight in weights_set]
    best_front = [np.max([np.dot(weight, point) for point in front]) for weight in weights_set]
    return np.max([ref - est for ref, est in zip(best_reference, best_front)])


# -------------------- PARETO FILTERING --------------------

def get_non_pareto_dominated_inds(candidates: np.ndarray | list) -> np.ndarray:
    """Return a mask of the first occurrences of the non-dominated rows (maximization)."""
    P = np.asarray(candidates, dtype=float)
    dominated = ((P[None, :, :] >= P[:, None, :]).all(axis=2) & (P[None, :, :] > P[:, None, :]).any(axis=2)).any(axis=1)
    _, first_idx = np.unique(P, axis=0, return_index=True)
    first = np.zeros(P.shape[0], dtype=bool)
    first[first_idx] = True
    return ~dominated & first


def filter_pareto_dominated(candidates: np.ndarray | list) -> np.ndarray:
    """Return the unique non-dominated rows (maximization)."""
    P = np.asarray(candidates)
    return P[get_non_pareto_dominated_inds(P)]
