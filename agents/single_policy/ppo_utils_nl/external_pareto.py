import numpy as np

from copy import deepcopy
from typing import Optional


def check_dominated(obj_batch: np.ndarray, obj: np.ndarray) -> bool:
    """Return whether any row of obj_batch Pareto-dominates obj."""
    return np.logical_and((obj_batch >= obj).all(axis=1), (obj_batch > obj).any(axis=1)).any()


def get_ep_indices(obj_batch_input: np.ndarray) -> list:
    """Return the indices of the unique non-dominated rows, sorted by the first objective."""
    obj_batch = np.array(obj_batch_input)
    _, unique_indices = np.unique(obj_batch.round(decimals=6), axis=0, return_index=True)
    sorted_indices = np.argsort(obj_batch[unique_indices].T[0])
    return [idx for idx in unique_indices[sorted_indices] if not check_dominated(obj_batch, obj_batch[idx])]


class ExternalPareto:
    def __init__(self, archive_size: Optional[int]):
        """Archive of non-dominated samples, truncated by crowding distance when it exceeds archive_size."""
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])
        self.archive_size = archive_size

    def index(self, indices: list) -> None:
        """Keep only the samples and objectives at the given indices."""
        indices = np.array(indices, dtype=int)
        self.obj_batch = self.obj_batch[indices]
        self.sample_batch = self.sample_batch[indices]

    def update(self, sample_batch: list) -> None:
        """Add samples, keep the non-dominated ones and truncate to archive_size by crowding distance."""
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))
        for sample in sample_batch:
            if len(self.obj_batch) > 0:
                self.obj_batch = np.vstack([self.obj_batch, sample.objs])
            else:
                self.obj_batch = np.array([sample.objs])

        self.index(get_ep_indices(self.obj_batch))
        if self.archive_size is not None and len(self.sample_batch) > self.archive_size:
            crowding_distances = self.calculate_crowding_distance(self.obj_batch, True)
            self.index(np.argsort(-crowding_distances)[:self.archive_size])

    @staticmethod
    def calculate_crowding_distance(obj_batch: np.ndarray, extreme: bool) -> np.ndarray:
        """Return the crowding distance of every row; with extreme, boundary points get infinity."""
        num_samples = obj_batch.shape[0]
        crowding_distances = np.zeros(num_samples)
        for i in range(obj_batch.shape[1]):
            obj_values = obj_batch[:, i]
            sorted_indices = np.argsort(obj_values)
            distances = np.zeros(num_samples)
            distances[1:-1] = ((obj_values[sorted_indices[2:]] - obj_values[sorted_indices[:-2]])
                               / (obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]))
            if extreme:
                distances[0] = np.inf
                distances[-1] = np.inf
            crowding_distances[sorted_indices] += distances
        return crowding_distances