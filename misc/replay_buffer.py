import torch

import numpy as np

from typing import Any, Optional


def _to_np(x: Any) -> np.ndarray:
    """Convert a tensor or array-like to a NumPy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


class ReplayBuffer:
    def __init__(self,
                 obs_shape: tuple[int, ...],
                 action_dim: int,
                 rew_dim: int = 1,
                 max_size: int = 100000,
                 seed: int = 42):
        """Ring buffer of (obs, action, vector reward, next_obs, done) transitions."""
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size,) + obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, obs: Any, action: Any, reward: Any, next_obs: Any, done: Any) -> None:
        """Store one transition, overwriting the oldest one when full."""
        self.obs[self.ptr] = _to_np(obs)
        self.actions[self.ptr] = _to_np(action)
        self.next_obs[self.ptr] = _to_np(next_obs)
        self.rewards[self.ptr] = _to_np(reward)
        self.dones[self.ptr] = _to_np(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, to_tensor: bool = False, device: Optional[str] = None) -> tuple:
        """Sample a batch with replacement, optionally as tensors."""
        inds = self.rng.choice(self.size, batch_size, replace=True)
        batch = (self.obs[inds], self.actions[inds], self.rewards[inds], self.next_obs[inds], self.dones[inds])
        if to_tensor:
            return tuple(torch.tensor(x, device=device) for x in batch)
        return batch
