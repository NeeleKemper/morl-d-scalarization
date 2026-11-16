import numpy as np

from pymoo.util.ref_dirs import get_reference_directions


def generate_layer_energy_weights(layers: tuple, reward_dim: int = 2) -> np.ndarray:
    """Generate layer-energy reference directions for the given layer sizes."""
    return get_reference_directions('layer-energy', reward_dim, layers)


def generate_das_dennis_weights(num_weights: int, reward_dim: int = 2) -> np.ndarray:
    """Generate a Das-Dennis grid of num_weights preference vectors."""
    return get_reference_directions('das-dennis', reward_dim, n_points=num_weights)


def generate_dirichlet_weights(num_weights: int, reward_dim: int, alpha: float) -> list:
    """Sample num_weights preference vectors from a symmetric Dirichlet(alpha) distribution."""
    return np.random.dirichlet(alpha * np.ones(reward_dim), size=num_weights).tolist()
