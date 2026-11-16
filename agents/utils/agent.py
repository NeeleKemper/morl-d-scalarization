import torch

import numpy as np
import gymnasium as gym

from typing import Union
from abc import ABC, abstractmethod

from misc.utils import seed_everything


class Agent(ABC):
    def __init__(self, env: gym.Env, seed: int, device: Union[torch.device, str], name: str):
        """Base agent: seeds everything and derives observation, action and reward dimensions from the env."""
        seed_everything(seed)
        self.seed = seed
        self.env = env
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        observation_space = env.unwrapped.observation_space
        self.action_space = env.unwrapped.action_space
        self.reward_dim = env.unwrapped.reward_space.shape[0]

        # --- Observation space ---
        if isinstance(observation_space, gym.spaces.Discrete):
            self.obs_shape = (1,)
            self.obs_dim = 1
        else:
            self.obs_shape = observation_space.shape
            self.obs_dim = observation_space.shape[0]

        # --- Action space (storage shape and network output dimension) ---
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_shape = (1,)
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
            self.action_shape = (self.action_space.n,)
            self.action_dim = self.action_space.n
        else:
            self.action_shape = self.action_space.shape
            self.action_dim = self.action_space.shape[0]

        # --- Device & run state ---
        self.device = torch.device(
            device if device in ['cpu', 'cuda'] else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.name = name
        self.global_step = 0
        self.num_episodes = 0
        self.np_random = np.random.default_rng(self.seed)

    @abstractmethod
    def eval(self, obs: Union[np.ndarray, torch.Tensor], w: Union[np.ndarray, torch.Tensor]) -> Union[
        np.ndarray, torch.Tensor]:
        """Return the action for an observation and preference weight."""

    @abstractmethod
    def save(self, path: str, file_name: str, save_replay_buffer: bool = True) -> None:
        """Save the agent."""

    @abstractmethod
    def load(self, path: str, file_name: str) -> None:
        """Load the agent."""

    @abstractmethod
    def save_config(self, path: str, file_name: str) -> None:
        """Save the agent configuration as JSON."""