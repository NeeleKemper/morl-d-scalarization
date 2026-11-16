import os
import copy
import torch

import numpy as np
import gymnasium as gym
import torch.nn.functional as F

from torch import optim
from dataclasses import dataclass
from typing import Optional, Iterable

from misc.network import polyak_update
from misc.replay_buffer import ReplayBuffer
from agents.single_policy.sac_utils_nl.sac_net import Actor, SoftQNetwork
from agents.single_policy.sac_utils_nl.sac_discrete_action_nl import linear_coefficients


def _environment_action(action: torch.Tensor,
                        shape: tuple[int, ...],
                        low: np.ndarray,
                        high: np.ndarray) -> np.ndarray:
    """Convert an actor output into a clipped float32 environment action of the given shape."""
    result = action.detach().numpy().reshape(shape).astype(np.float32, copy=False)
    return np.clip(result, low, high).astype(np.float32, copy=False)


@dataclass
class SACContinuousPolicySnapshot:
    """CPU inference copy of a continuous SAC actor, without replay buffer or optimizers."""
    actor: torch.nn.Module
    weights: np.ndarray
    id: int
    seed: int
    global_step: int
    actor_net_arch: list[int]
    obs_dim: int
    action_dim: int
    env_id: str
    action_shape: tuple[int, ...]
    action_low: np.ndarray
    action_high: np.ndarray
    deterministic: bool = True
    objs: Optional[np.ndarray] = None

    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None, deterministic: Optional[bool] = None) -> np.ndarray:
        """Return the mean (deterministic) or a sampled action for one observation."""
        parameter = next(self.actor.parameters())
        observations = torch.as_tensor(obs, dtype=parameter.dtype, device=parameter.device).reshape(1, -1)
        use_mean = self.deterministic if deterministic is None else deterministic
        with torch.no_grad():
            action = self.actor.get_action(observations) if use_mean else self.actor.sample(observations)[0]
        return _environment_action(action, self.action_shape, self.action_low, self.action_high)

    def save(self, path: str, file_name: str) -> None:
        """Save the actor weights and all metadata needed for inference."""
        os.makedirs(path, exist_ok=True)
        payload = {
            'format_version': 2,
            'kind': 'independent_sac_continuous_actor',
            'actor_network': 'agents.single_policy.sac_utils_nl.sac_net.Actor',
            'actor_state_dict': {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
            'weights': self.weights.tolist(),
            'id': int(self.id),
            'seed': int(self.seed),
            'global_step': int(self.global_step),
            'actor_net_arch': list(self.actor_net_arch),
            'obs_dim': int(self.obs_dim),
            'action_dim': int(self.action_dim),
            'env_id': self.env_id,
            'action_shape': list(self.action_shape),
            'action_low': self.action_low.tolist(),
            'action_high': self.action_high.tolist(),
            'deterministic': bool(self.deterministic),
            'objs': None if self.objs is None else self.objs.tolist(),
            'observation_preprocessing': 'none',
            'deterministic_action': 'affine-scaled tanh of Gaussian mean (not expected squashed action)',
        }
        torch.save(payload, os.path.join(path, f'{file_name}.pt'))

    @classmethod
    def from_payload(cls, payload: dict) -> 'SACContinuousPolicySnapshot':
        """Rebuild a frozen snapshot from a saved payload without touching the default RNG."""
        shape = tuple(int(v) for v in payload['action_shape'])
        low = np.asarray(payload['action_low'], dtype=np.float64).reshape(shape)
        high = np.asarray(payload['action_high'], dtype=np.float64).reshape(shape)
        space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        with torch.random.fork_rng(devices=[]):
            actor = Actor(payload['obs_dim'], payload['action_dim'], space, net_arch=payload['actor_net_arch']).float()
        actor.load_state_dict(payload['actor_state_dict'])
        actor.eval().requires_grad_(False)
        objs = payload['objs']
        return cls(
            actor=actor,
            weights=np.asarray(payload['weights'], dtype=np.float64),
            actor_net_arch=list(payload['actor_net_arch']),
            action_shape=shape,
            action_low=low,
            action_high=high,
            objs=None if objs is None else np.asarray(objs, dtype=np.float64),
            **{key: payload[key] for key in ('id', 'seed', 'global_step', 'obs_dim', 'action_dim',
                                             'env_id', 'deterministic')},
        )

    @classmethod
    def load(cls, path: str, file_name: str) -> 'SACContinuousPolicySnapshot':
        """Load a saved snapshot for inference."""
        payload = torch.load(os.path.join(path, f'{file_name}.pt'), map_location='cpu', weights_only=True)
        return cls.from_payload(payload)


class SACContinuousNL:
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
        """SAC learner for one fixed linear WS preference with twin vector critics and a tanh-squashed Box actor."""
        space = env.action_space

        # --- Environment & action space ---
        self.id = int(id)
        self.env = env
        self.seed = int(seed)
        self.obs_dim = int(obs_shape[0])
        self.action_dim = int(action_dim)
        self.reward_dim = int(reward_dim)
        self.action_shape = tuple(space.shape)
        self.action_low = space.low.copy()
        self.action_high = space.high.copy()

        # --- SAC hyperparameters ---
        self.actor_net_arch = list(actor_net_arch)
        self.critic_net_arch = list(critic_net_arch)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.policy_freq = int(policy_freq)
        self.target_net_freq = int(target_net_freq)
        self.clip_grad_norm = bool(clip_grad_norm)
        self.actor_clip_norm = float(actor_clip_norm)
        self.critic_clip_norm = float(critic_clip_norm)

        # --- Scalarization & counters ---
        self.objective_scale = objective_scale
        self.weights = None
        self.coefficients = None
        self.global_step = 0
        self.num_episodes = 0
        self.update_step = 0
        self.actor_update_count = 0

        # --- Networks & optimizers (construction order fixes the RNG stream) ---
        self.actor = Actor(self.obs_dim, self.action_dim, space, net_arch=self.actor_net_arch).float()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.qf1 = self._make_critic()
        self.qf2 = self._make_critic()
        self.qf1_target = copy.deepcopy(self.qf1).requires_grad_(False)
        self.qf2_target = copy.deepcopy(self.qf2).requires_grad_(False)
        self._critic_parameters = [*self.qf1.parameters(), *self.qf2.parameters()]
        self.q_optimizer = optim.Adam(self._critic_parameters, lr=critic_lr)

        # --- Seeding & first observation ---
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)
        self.obs, _ = self.env.reset(seed=self.seed)

    # -------------------- SETUP --------------------

    def _make_critic(self) -> SoftQNetwork:
        """Build a critic with one Q output per objective."""
        return SoftQNetwork(self.obs_dim, self.action_dim, self.reward_dim, self.critic_net_arch).float()

    def set_weights(self, weights: np.ndarray) -> None:
        """Set the preference weight and the resulting linear projection coefficients."""
        self.weights = torch.as_tensor(weights, dtype=torch.float32).flatten().clone()
        self.coefficients = linear_coefficients(self.weights, self.objective_scale)

    def _project(self, vectors: torch.Tensor) -> torch.Tensor:
        """Project vector values onto the scalar WS objective."""
        return (vectors * self.coefficients).sum(dim=-1)

    # -------------------- ACTING --------------------

    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None, deterministic: bool = True) -> np.ndarray:
        """Return the mean (deterministic) or a sampled action for one observation."""
        observations = torch.as_tensor(obs, dtype=torch.float32).reshape(1, self.obs_dim)
        with torch.no_grad():
            actions = self.actor.get_action(observations) if deterministic else self.actor.sample(observations)[0]
        return _environment_action(actions, self.action_shape, self.action_low, self.action_high)

    def snapshot(self,
                 env_id: str,
                 deterministic: bool = True,
                 objs: Optional[np.ndarray] = None) -> SACContinuousPolicySnapshot:
        """Return a frozen inference copy of the current actor."""
        return SACContinuousPolicySnapshot(
            actor=copy.deepcopy(self.actor).eval().requires_grad_(False),
            weights=self.weights.numpy().astype(np.float64, copy=True),
            id=self.id,
            seed=self.seed,
            global_step=self.global_step,
            actor_net_arch=self.actor_net_arch.copy(),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            env_id=env_id,
            action_shape=self.action_shape,
            action_low=self.action_low.copy(),
            action_high=self.action_high.copy(),
            deterministic=deterministic,
            objs=None if objs is None else np.array(objs, dtype=np.float64),
        )

    def collect_sample(self, replay_buffer: ReplayBuffer) -> None:
        """Take one environment step (random before learning_starts) and store it; truncations are bootstrapped."""
        if self.global_step < self.learning_starts:
            action = self.env.action_space.sample()
        else:
            action = self.eval(self.obs, deterministic=False)
        old_obs = np.array(self.obs)
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        replay_buffer.add(old_obs, np.asarray(action).reshape(-1), np.asarray(reward, dtype=np.float32),
                          next_obs, terminated)
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

    def _set_critic_trainable(self, trainable: bool) -> None:
        """Enable or disable gradients for the critic parameters."""
        for parameter in self._critic_parameters:
            parameter.requires_grad_(trainable)

    def update(self, batch: tuple) -> Optional[dict]:
        """Run one critic update and, every policy_freq steps, one actor update; return the training metrics."""
        if self.global_step < self.learning_starts:
            return None
        obs, actions, rewards, next_obs, dones = batch
        obs = torch.as_tensor(obs, dtype=torch.float32).reshape(-1, self.obs_dim)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32).reshape(-1, self.obs_dim)
        actions = torch.as_tensor(actions).float().reshape(-1, self.action_dim)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, self.reward_dim)
        dones = torch.as_tensor(dones, dtype=torch.float32).reshape(-1)

        # Critic: projected clipped double-Q soft backup
        with torch.no_grad():
            next_actions, next_logp, _ = self.actor.sample(next_obs)
            q_next = torch.minimum(self._project(self.qf1_target(next_obs, next_actions)),
                                   self._project(self.qf2_target(next_obs, next_actions)))
            soft_value = q_next - self.alpha * next_logp.reshape(-1)
            target = self._project(rewards) + self.gamma * (1.0 - dones) * soft_value
        q1_a = self._project(self.qf1(obs, actions))
        q2_a = self._project(self.qf2(obs, actions))
        loss1 = F.mse_loss(q1_a, target)
        loss2 = F.mse_loss(q2_a, target)
        critic_loss = loss1 + loss2
        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = self._clip(self._critic_parameters, self.critic_clip_norm)
        self.q_optimizer.step()
        self.update_step += 1

        # Actor (delayed by policy_freq, critic parameters frozen)
        actor_loss_value = np.nan
        entropy = np.nan
        actor_norm_value = np.nan
        if self.update_step % self.policy_freq == 0:
            self._set_critic_trainable(False)
            sampled_actions, log_pi, _ = self.actor.sample(obs)
            q_min = torch.minimum(self._project(self.qf1(obs, sampled_actions)),
                                  self._project(self.qf2(obs, sampled_actions)))
            actor_loss = (self.alpha * log_pi.reshape(-1) - q_min).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_norm_value = self._clip(self.actor.parameters(), self.actor_clip_norm).item()
            self.actor_optimizer.step()
            self._set_critic_trainable(True)
            self.actor_update_count += 1
            actor_loss_value = actor_loss.item()
            entropy = -log_pi.detach().mean().item()

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
            'policy_entropy_norm': np.nan,
            'policy_entropy_per_action_dim': entropy / self.action_dim,
            'actor_grad_norm': actor_norm_value,
            'critic_grad_norm': critic_norm.item(),
        }