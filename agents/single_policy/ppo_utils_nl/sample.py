import os
import torch
import pickle

import numpy as np
import torch.optim as optim

from copy import deepcopy
from typing import Union, Optional, Any

from agents.single_policy.ppo_utils_nl.ppo_nl import PPONL
from agents.single_policy.ppo_utils_nl.a2c_ppo.model_nl import PolicyNL, PolicyNLDiscrete


class Sample:
    def __init__(self,
                 env_params: dict[str, Any],
                 actor_critic: Union[PolicyNL, PolicyNLDiscrete],
                 agent: PPONL,
                 weights: torch.Tensor,
                 learning_rate: float,
                 eps: float,
                 objs: Optional[np.ndarray] = None):
        """One population member: policy, PPO agent, env normalisation statistics, weight and objective vector."""
        self.env_params = env_params
        self.actor_critic = actor_critic
        self.agent = agent
        self.learning_rate = learning_rate
        self.weights = weights
        self.eps = eps
        self.link_policy_agent()
        self.objs = objs

    @classmethod
    def copy_from(cls, sample: 'Sample') -> 'Sample':
        """Return an independent deep copy of a sample."""
        return cls(
            env_params=deepcopy(sample.env_params),
            actor_critic=deepcopy(sample.actor_critic),
            agent=deepcopy(sample.agent),
            learning_rate=sample.learning_rate,
            eps=sample.eps,
            weights=deepcopy(sample.weights),
            objs=deepcopy(sample.objs),
        )

    def set_weights(self, weights: torch.Tensor) -> None:
        """Replace the preference weight."""
        self.weights = weights

    def link_policy_agent(self) -> None:
        """Bind the agent to this policy and rebuild its optimizer on the policy parameters, keeping the state."""
        self.agent.actor_critic = self.actor_critic
        optim_state_dict = deepcopy(self.agent.optimizer.state_dict())
        self.agent.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate, eps=self.eps)
        self.agent.optimizer.load_state_dict(optim_state_dict)

    def save(self, path: str, file_name: str) -> None:
        """Save policy, optimizer state, env statistics and objective vector."""
        torch.save(self.actor_critic.state_dict(), os.path.join(path, f'{file_name}_actor_critic.pt'))
        torch.save(self.agent.optimizer.state_dict(), os.path.join(path, f'{file_name}_optimizer.pt'))
        with open(os.path.join(path, f'{file_name}_env_params.pkl'), 'wb') as f:
            pickle.dump(self.env_params, f)
        np.savetxt(os.path.join(path, f'{file_name}_objs.txt'), self.objs, delimiter=',', fmt='%.6f')