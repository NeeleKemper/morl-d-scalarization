import os
import copy
import torch

import numpy as np
import gymnasium as gym
import torch.nn.functional as F

from torch import optim
from dataclasses import dataclass
from typing import Union, Optional, Iterable

from misc.network import polyak_update
from misc.replay_buffer import ReplayBuffer
from agents.single_policy.sac_utils_nl.sac_net import ActorDiscrete, SoftQNetworkDiscrete


def actor_probabilities(actor: ActorDiscrete, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the floored, renormalized action probabilities and their logarithms."""
    _, _, probabilities = actor.sample(observations)
    probabilities = probabilities.clamp_min(1e-8)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return probabilities, probabilities.log()


def select_action(actor: ActorDiscrete, x: torch.Tensor, deterministic: bool) -> int:
    """Return the greedy or a sampled action for a single observation."""
    with torch.no_grad():
        probabilities, _ = actor_probabilities(actor, x)
        action = probabilities.argmax(-1) if deterministic else torch.distributions.Categorical(probabilities).sample()
    return action.item()


def linear_coefficients(weights: Union[np.ndarray, torch.Tensor],
                        objective_scale: Optional[np.ndarray]) -> torch.Tensor:
    """Return the WS projection coefficients: w for ws_raw, (w / Delta) normalized to sum 1 for ws_norm."""
    w = torch.as_tensor(weights, dtype=torch.float32).flatten().clone()
    if objective_scale is None:
        return w
    delta = torch.as_tensor(objective_scale, dtype=torch.float32).flatten()
    c = w / delta
    return c / c.sum()


@dataclass
class SACPolicySnapshot:
    """CPU inference copy of a discrete SAC actor, without environment, replay buffer or optimizers."""
    actor: torch.nn.Module
    weights: np.ndarray
    id: int
    seed: int
    global_step: int
    actor_net_arch: list[int]
    obs_dim: int
    action_dim: int
    env_id: str
    deterministic: bool = True
    objs: Optional[np.ndarray] = None

    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None, deterministic: Optional[bool] = None) -> int:
        """Return the greedy or a sampled action for one observation."""
        parameter = next(self.actor.parameters())
        x = torch.as_tensor(obs, dtype=parameter.dtype, device=parameter.device).reshape(1, -1)
        return select_action(self.actor, x, self.deterministic if deterministic is None else deterministic)

    def save(self, path: str, file_name: str) -> None:
        """Save the actor weights and all metadata needed for inference."""
        os.makedirs(path, exist_ok=True)
        payload = {
            'format_version': 1,
            'kind': 'independent_sac_actor',
            'actor_state_dict': {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
            'weights': self.weights.tolist(),
            'id': self.id,
            'seed': self.seed,
            'global_step': self.global_step,
            'actor_net_arch': list(self.actor_net_arch),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'env_id': self.env_id,
            'deterministic': self.deterministic,
            'objs': None if self.objs is None else self.objs.tolist(),
            'observation_preprocessing': 'none',
        }
        torch.save(payload, os.path.join(path, f'{file_name}.pt'))

    @classmethod
    def load(cls, path: str, file_name: str) -> 'SACPolicySnapshot':
        """Load a saved snapshot for inference without touching the default RNG."""
        payload = torch.load(os.path.join(path, f'{file_name}.pt'), map_location='cpu', weights_only=True)
        with torch.random.fork_rng(devices=[]):
            actor = ActorDiscrete(payload['obs_dim'], payload['action_dim'], net_arch=payload['actor_net_arch']).float()
        actor.load_state_dict(payload['actor_state_dict'])
        actor.eval().requires_grad_(False)
        objs = payload['objs']
        return cls(
            actor=actor,
            weights=np.asarray(payload['weights'], dtype=np.float64),
            objs=None if objs is None else np.asarray(objs, dtype=np.float64),
            **{key: payload[key] for key in ('id', 'seed', 'global_step', 'actor_net_arch',
                                             'obs_dim', 'action_dim', 'env_id', 'deterministic')},
        )


class SACDiscreteNL:
    def __init__(
            self,
            id: int,
            env: gym.Env,
            obs_shape: tuple[int, ...],
            action_dim: int,
            reward_dim: int,
            actor_net_arch: list[int],
            critic_net_arch: list[int],
            actor_lr: float = 3e-4,
            critic_lr: float = 1e-3,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: float = 0.2,
            batch_size: int = 256,
            learning_starts: int = 1000,
            policy_freq: int = 1,
            target_net_freq: int = 1,
            clip_grad_norm: bool = True,
            actor_clip_norm: float = 2.0,
            critic_clip_norm: float = 5.0,
            seed: int = 42,
            objective_scale: Optional[np.ndarray] = None,
    ):
        """Discrete SAC for one fixed linear WS preference with twin vector critics and exact action expectations."""
        # --- Environment ---
        self.id = id
        self.env = env
        self.seed = seed
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        # --- SAC hyperparameters ---
        self.actor_net_arch = list(actor_net_arch)
        self.critic_net_arch = list(critic_net_arch)
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq
        self.clip_grad_norm = clip_grad_norm
        self.actor_clip_norm = actor_clip_norm
        self.critic_clip_norm = critic_clip_norm

        # --- Scalarization & counters ---
        self.objective_scale = objective_scale
        self.weights = None
        self.coefficients = None
        self.global_step = 0
        self.num_episodes = 0
        self.update_step = 0
        self.actor_update_count = 0

        # --- Networks & optimizers (construction order fixes the RNG stream) ---
        self.actor = ActorDiscrete(self.obs_dim, action_dim, net_arch=self.actor_net_arch).float()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.qf1 = self._make_critic()
        self.qf2 = self._make_critic()
        self.qf1_target = copy.deepcopy(self.qf1).requires_grad_(False)
        self.qf2_target = copy.deepcopy(self.qf2).requires_grad_(False)
        self._critic_parameters = [*self.qf1.parameters(), *self.qf2.parameters()]
        self.q_optimizer = optim.Adam(self._critic_parameters, lr=critic_lr)

        # --- Seeding & first observation ---
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self.obs, _ = self.env.reset(seed=seed)

    # -------------------- SETUP --------------------

    def _make_critic(self) -> SoftQNetworkDiscrete:
        """Build a critic with one Q output per action and objective."""
        return SoftQNetworkDiscrete(self.obs_dim, self.action_dim, output_dim=self.reward_dim,
                                    net_arch=self.critic_net_arch).float()

    def set_weights(self, weights: np.ndarray) -> None:
        """Set the preference weight and the resulting linear projection coefficients."""
        self.weights = torch.as_tensor(weights, dtype=torch.float32).flatten().clone()
        self.coefficients = linear_coefficients(self.weights, self.objective_scale)

    def _projected_all_q(self, network: SoftQNetworkDiscrete, observations: torch.Tensor) -> torch.Tensor:
        """Return the projected scalar Q value of every action."""
        return (network.forward_all(observations) * self.coefficients).sum(dim=-1)

    # -------------------- ACTING --------------------

    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None, deterministic: bool = True) -> int:
        """Return the greedy or a sampled action for one observation."""
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        return select_action(self.actor, x, deterministic)

    def snapshot(self,
                 env_id: str,
                 deterministic: bool = True,
                 objs: Optional[np.ndarray] = None) -> SACPolicySnapshot:
        """Return a frozen inference copy of the current actor."""
        return SACPolicySnapshot(
            actor=copy.deepcopy(self.actor).eval().requires_grad_(False),
            weights=self.weights.numpy().astype(np.float64, copy=True),
            id=self.id,
            seed=self.seed,
            global_step=self.global_step,
            actor_net_arch=self.actor_net_arch.copy(),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            env_id=env_id,
            deterministic=deterministic,
            objs=None if objs is None else np.array(objs, dtype=np.float64),
        )

    def collect_sample(self, replay_buffer: ReplayBuffer) -> None:
        """Take one environment step (random before learning_starts) and store it; truncations are bootstrapped."""
        if self.global_step < self.learning_starts:
            action = self.env.action_space.sample()
        else:
            x = torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self.actor.sample(x)
            action = action.reshape(-1)[0].item()
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        replay_buffer.add(self.obs, action, reward, next_obs, terminated)
        self.obs = next_obs
        self.global_step += 1
        if terminated or truncated:
            self.num_episodes += 1
            self.obs, _ = self.env.reset(seed=self.seed + self.num_episodes)

    # -------------------- LEARNING --------------------

    def _clip(self, parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
        """Clip the gradient norm (if enabled) and return it; non-finite gradients raise."""
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm if self.clip_grad_norm else torch.inf,
                                              error_if_nonfinite=True)

    def update(self, batch: tuple) -> Optional[dict]:
        """Run one critic update and, every policy_freq steps, one actor update; return the training metrics."""
        if self.global_step < self.learning_starts:
            return None
        obs, actions, rewards, next_obs, dones = batch
        obs = torch.as_tensor(obs, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long).reshape(-1, 1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, self.reward_dim)
        dones = torch.as_tensor(dones, dtype=torch.float32).reshape(-1)

        # Critic: projected soft-Q backup with the exact expectation over all actions
        with torch.no_grad():
            p, logp = actor_probabilities(self.actor, next_obs)
            q_next = torch.minimum(self._projected_all_q(self.qf1_target, next_obs),
                                   self._projected_all_q(self.qf2_target, next_obs))
            soft_value = (p * (q_next - self.alpha * logp)).sum(-1)
            target = (rewards * self.coefficients).sum(-1) + (1.0 - dones) * self.gamma * soft_value
        q1_a = self._projected_all_q(self.qf1, obs).gather(1, actions).flatten()
        q2_a = self._projected_all_q(self.qf2, obs).gather(1, actions).flatten()
        loss1 = F.mse_loss(q1_a, target)
        loss2 = F.mse_loss(q2_a, target)
        critic_loss = loss1 + loss2
        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = self._clip(self._critic_parameters, self.critic_clip_norm)
        self.q_optimizer.step()
        self.update_step += 1

        # Actor (delayed by policy_freq)
        actor_loss_value = np.nan
        entropy = np.nan
        entropy_norm = np.nan
        actor_norm_value = np.nan
        if self.update_step % self.policy_freq == 0:
            p, logp = actor_probabilities(self.actor, obs)
            with torch.no_grad():
                q_min = torch.minimum(self._projected_all_q(self.qf1, obs), self._projected_all_q(self.qf2, obs))
            actor_loss = (p * (self.alpha * logp - q_min)).sum(-1).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_norm_value = self._clip(self.actor.parameters(), self.actor_clip_norm).item()
            self.actor_optimizer.step()
            self.actor_update_count += 1
            actor_loss_value = actor_loss.item()
            entropy = -(p.detach() * logp.detach()).sum(-1).mean().item()
            entropy_norm = entropy / np.log(self.action_dim)

        # Target networks
        if self.update_step % self.target_net_freq == 0:
            polyak_update(self.qf1.parameters(), self.qf1_target.parameters(), self.tau)
            polyak_update(self.qf2.parameters(), self.qf2_target.parameters(), self.tau)

        return {
            'critic_loss': critic_loss.item(),
            'qf1_loss': loss1.item(),
            'qf2_loss': loss2.item(),
            'actor_loss': actor_loss_value,
            'alpha': self.alpha,
            'q1_mean': q1_a.mean().item(),
            'q2_mean': q2_a.mean().item(),
            'target_q_mean': target.mean().item(),
            'target_q_std': target.std(unbiased=False).item(),
            'policy_entropy': entropy,
            'policy_entropy_norm': entropy_norm,
            'actor_grad_norm': actor_norm_value,
            'critic_grad_norm': critic_norm.item(),
        }