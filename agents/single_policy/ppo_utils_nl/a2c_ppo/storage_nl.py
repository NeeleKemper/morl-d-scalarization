import torch

import gymnasium as gym

from typing import Generator

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorageNL:
    def __init__(
            self,
            num_steps: int,
            num_processes: int,
            obs_shape: tuple[int, ...],
            action_space: gym.Space,
            recurrent_hidden_state_size: int,
            reward_dim: int = 1,
    ) -> None:
        """Rollout buffer for PPO with vector-valued rewards, values and returns."""
        action_shape = action_space.shape[0] if action_space.shape else 1
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, reward_dim)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, reward_dim)
        self.returns = torch.zeros(num_steps + 1, num_processes, reward_dim)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device: torch.device) -> None:
        """Move all storage tensors to the given device."""
        for name in ('obs', 'recurrent_hidden_states', 'rewards', 'value_preds', 'returns', 'action_log_probs',
                     'actions', 'masks', 'bad_masks'):
            setattr(self, name, getattr(self, name).to(device))

    def insert(
            self,
            obs: torch.Tensor,
            recurrent_hidden_states: torch.Tensor,
            actions: torch.Tensor,
            action_log_probs: torch.Tensor,
            value_preds: torch.Tensor,
            rewards: torch.Tensor,
            masks: torch.Tensor,
            bad_masks: torch.Tensor,
    ) -> None:
        """Insert one timestep at index self.step."""
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self) -> None:
        """Copy the last timestep to index 0 so the next rollout continues from it."""
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
            self,
            next_value: torch.Tensor,
            use_gae: bool,
            gamma: float,
            gae_lambda: float,
            use_proper_time_limits: bool = True,
    ) -> None:
        """Compute discounted or GAE returns per objective, optionally bootstrapping at time-limit truncations."""
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.num_steps)):
                delta = (self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                         - self.value_preds[step])
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                if use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.num_steps)):
                ret = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
                if use_proper_time_limits:
                    ret = ret * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
                self.returns[step] = ret

    def feed_forward_generator(self, advantages: torch.Tensor, num_mini_batch: int) -> Generator[tuple, None, None]:
        """Yield num_mini_batch random minibatches of the flattened rollout."""
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), batch_size // num_mini_batch, drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, *self.value_preds.size()[2:])[indices]
            return_batch = self.returns[:-1].view(-1, *self.returns.size()[2:])[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, advantages.size(-1))[indices]
            yield (obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch,
                   masks_batch, old_action_log_probs_batch, adv_targ)