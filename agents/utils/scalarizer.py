import torch

import numpy as np

from typing import Optional

from misc.weights import generate_das_dennis_weights


class Scalarizer:
    def __init__(self, method: str = 'ws_raw', mu: float = 0.1) -> None:
        """Scalarization functions for maximization; mu is the STCH smoothing (normalized or raw units)."""
        method = method.lower()
        self.method = 'ws_raw' if method == 'ws' else method
        self.mu = mu
        self.objective_scale = None
        self._impls = {
            'ws_raw': self._ws_raw,
            'ws_norm': self._ws_norm,
            'tch_raw': self._tch_raw,
            'tch_norm': self._tch_norm,
            'stch_raw': self._stch_raw,
            'stch_norm': self._stch_norm,
        }

    def __call__(self,
                 V: torch.Tensor,
                 w: torch.Tensor,
                 z_star: Optional[torch.Tensor] = None,
                 z_nad: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the scalarized utility of the value vectors V for the weights w."""
        w = torch.as_tensor(w, device=V.device, dtype=V.dtype)
        z_star = None if z_star is None else torch.as_tensor(z_star, device=V.device, dtype=V.dtype)
        z_nad = None if z_nad is None else torch.as_tensor(z_nad, device=V.device, dtype=V.dtype)
        return self._impls[self.method](V, w, z_star, z_nad)

    def _scale(self, V: torch.Tensor, z_star: torch.Tensor, z_nad: torch.Tensor) -> torch.Tensor:
        """Return the normalization range: objective_scale if set, otherwise z_star - z_nad."""
        if self.objective_scale is None:
            return z_star - z_nad
        return torch.as_tensor(self.objective_scale, device=V.device, dtype=V.dtype)

    @staticmethod
    def _ideal_gap(V: torch.Tensor, z_star: torch.Tensor) -> torch.Tensor:
        """Return the gap between the ideal point and V."""
        return z_star - V

    # -------------------- WEIGHTED SUM --------------------

    @staticmethod
    def _ws_raw(V: torch.Tensor,
                w: torch.Tensor,
                z_star: Optional[torch.Tensor],
                z_nad: Optional[torch.Tensor]) -> torch.Tensor:
        """Weighted sum of the raw values."""
        return (w * V).sum(dim=-1)

    def _ws_norm(self,
                 V: torch.Tensor,
                 w: torch.Tensor,
                 z_star: torch.Tensor,
                 z_nad: torch.Tensor) -> torch.Tensor:
        """Weighted sum of the nadir-shifted, range-normalized values."""
        return (w * ((V - z_nad) / self._scale(V, z_star, z_nad))).sum(dim=-1)

    # -------------------- WEIGHTED TCHEBYCHEFF --------------------

    def _tch_raw(self,
                 V: torch.Tensor,
                 w: torch.Tensor,
                 z_star: torch.Tensor,
                 z_nad: Optional[torch.Tensor]) -> torch.Tensor:
        """Negative maximum weighted raw gap to the ideal point; zero weights are ignored."""
        weighted_gap = (w * self._ideal_gap(V, z_star)).masked_fill(w == 0, -torch.inf)
        return -weighted_gap.amax(dim=-1)

    def _tch_norm(self,
                  V: torch.Tensor,
                  w: torch.Tensor,
                  z_star: torch.Tensor,
                  z_nad: torch.Tensor) -> torch.Tensor:
        """Negative maximum weighted normalized gap to the ideal point; zero weights are ignored."""
        weighted_gap = (w * self._ideal_gap(V, z_star) / self._scale(V, z_star, z_nad)).masked_fill(w == 0, -torch.inf)
        return -weighted_gap.amax(dim=-1)

    # -------------------- SMOOTH TCHEBYCHEFF --------------------

    def _stch_raw(self,
                  V: torch.Tensor,
                  w: torch.Tensor,
                  z_star: torch.Tensor,
                  z_nad: Optional[torch.Tensor]) -> torch.Tensor:
        """Negative log-sum-exp smoothed maximum of the weighted raw gaps."""
        weighted_gap = (w * self._ideal_gap(V, z_star)).masked_fill(w == 0, -torch.inf)
        return -self.mu * torch.logsumexp(weighted_gap / self.mu, dim=-1)

    def _stch_norm(self,
                   V: torch.Tensor,
                   w: torch.Tensor,
                   z_star: torch.Tensor,
                   z_nad: torch.Tensor) -> torch.Tensor:
        """Negative log-sum-exp smoothed maximum of the weighted normalized gaps."""
        weighted_gap = (w * self._ideal_gap(V, z_star) / self._scale(V, z_star, z_nad)).masked_fill(w == 0, -torch.inf)
        return -self.mu * torch.logsumexp(weighted_gap / self.mu, dim=-1)


def calibrate_stch_raw_mu(mu: float,
                          objective_ranges: np.ndarray,
                          num_weights: int,
                          reward_dim: int) -> float:
    """Convert a normalized STCH mu into raw return units via the geometric mean scale over a Das-Dennis set."""
    preference_set = np.asarray(generate_das_dennis_weights(num_weights=num_weights, reward_dim=reward_dim),
                                dtype=np.float64)
    normalized_gaps = np.max(preference_set, axis=1)
    raw_gaps = np.max(preference_set * objective_ranges[None, :], axis=1)
    environment_scale = float(np.exp(np.mean(np.log(raw_gaps / normalized_gaps))))
    return mu * environment_scale