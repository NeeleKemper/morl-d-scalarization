import numpy as np

from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from misc.utils import ParetoFrontStore
from misc.metric import (hypervolume, sparsity, expected_utility, cardinality, maximum_spread, maximum_gap, igd,
                         igd_plus, maximum_utility_loss)


def compute_metrics(points: list,
                    hv_ref_point: np.ndarray,
                    reward_dim: int,
                    num_sample_weights: int,
                    ref_front: Optional[list[np.ndarray]]) -> tuple:
    """Pareto-filter the points and return HV, sparsity, EU, cardinality, spread, gap, front and IGD, IGD+, MUL."""
    P = np.asarray(points, dtype=np.float64)
    pareto_front = P[NonDominatedSorting().do(-P, only_non_dominated_front=True)]
    weights_set = get_reference_directions('energy', reward_dim, num_sample_weights)
    hv = hypervolume(hv_ref_point, pareto_front)
    sp = sparsity(pareto_front)
    eu = expected_utility(pareto_front, weights_set=weights_set)
    card = cardinality(pareto_front)
    max_spread = maximum_spread(pareto_front)
    max_gap = maximum_gap(pareto_front)
    if ref_front is None:
        return hv, sp, eu, card, max_spread, max_gap, pareto_front, None, None, None
    ref_front_arr = np.asarray(ref_front, dtype=np.float64)
    return (hv, sp, eu, card, max_spread, max_gap, pareto_front,
            igd(known_front=ref_front_arr, current_estimate=pareto_front),
            igd_plus(known_front=ref_front_arr, current_estimate=pareto_front),
            maximum_utility_loss(front=pareto_front, reference_set=ref_front_arr, weights_set=weights_set))


def log_metrics(front: list,
                ref_point: np.ndarray,
                known_pareto_front: Optional[list[np.ndarray]],
                reward_dim: int,
                num_sample_weights: int,
                global_step: int,
                writer: SummaryWriter,
                log: bool,
                save_fronts: bool = False,
                store: Optional[ParetoFrontStore] = None,
                tag_prefix: str = 'archive',
                weights: Optional[np.ndarray] = None,
                global_iteration: Optional[int] = None,
                agent_ids: Optional[np.ndarray] = None,
                weight_ids: Optional[np.ndarray] = None,
                replica_ids: Optional[np.ndarray] = None) -> float:
    """Compute the front metrics, log them to TensorBoard, optionally store the snapshot and return the HV."""
    if tag_prefix == 'archive':
        print(f'{tag_prefix}: {np.round(front, 2)}')
    hv, sp, eu, card, max_spread, max_gap, filtered_front, gd, gd_plus, mul = compute_metrics(
        points=front, hv_ref_point=ref_point, reward_dim=reward_dim, num_sample_weights=num_sample_weights,
        ref_front=known_pareto_front)

    # TensorBoard
    if log:
        scalars = {'hypervolume': hv, 'sparsity': sp, 'eu': eu, 'cardinality': card, 'max_spread': max_spread,
                   'max_gap': max_gap}
        if known_pareto_front is not None:
            scalars.update({'igd': gd, 'igd_plus': gd_plus, 'mul': mul})
        for name, value in scalars.items():
            writer.add_scalar(f'{tag_prefix}/{name}', value, global_step=global_step)
        if known_pareto_front is not None and tag_prefix == 'archive':
            print(f'@step={global_step} IGD={round(gd, 2)}, IGD+={round(gd_plus, 2)}')
        writer.flush()

    # NPZ snapshot: every current policy with its metadata, or the Pareto-filtered archive
    if save_fronts:
        if tag_prefix == 'policy':
            store.add(np.asarray(front, dtype=np.float32), global_step=global_step, weights=weights,
                      global_iteration=global_iteration, agent_ids=agent_ids, weight_ids=weight_ids,
                      replica_ids=replica_ids)
        else:
            store.add(filtered_front, global_step=global_step, global_iteration=global_iteration)
        store.flush()
    return hv
