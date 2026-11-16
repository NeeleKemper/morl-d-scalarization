import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import mo_gymnasium as mo_gym

from typing import Optional, Any

from agents.utils.scalarizer import Scalarizer
from agents.single_policy.ppo_utils_nl.a2c_ppo.model_nl import PolicyNL
from agents.single_policy.ppo_utils_nl.a2c_ppo.storage_nl import RolloutStorageNL


def _empty_scalarization_info(num_objectives: int) -> dict:
    """Return the scalarization diagnostics with NaN defaults."""
    nan = float('nan')
    nan_vec = [nan] * int(num_objectives)
    return {
        'v0': nan_vec.copy(),
        'loss_weights': nan_vec.copy(),
        'loss_weight_norm': nan,
        'loss_weights_finite': nan,
    }


class PPONL:
    def __init__(self,
                 actor_critic: PolicyNL,
                 clip_param: float,
                 ppo_epoch: int,
                 num_mini_batches: int,
                 value_loss_coef: float,
                 entropy_coef: float,
                 lr: float,
                 eps: float,
                 max_grad_norm: float,
                 use_clipped_value_loss: bool,
                 scalarizer: Scalarizer,
                 mc_k: int = 256,
                 advantage_normalization: str = 'global'):
        """PPO with non-linear scalarization; advantages are normalized 'global', per 'minibatch' or 'none'."""
        # --- PPO parameters ---
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        # --- Scalarization ---
        self.scalarizer = scalarizer
        self.mc_k = mc_k
        self.advantage_normalization = advantage_normalization
        self.init_obs = None
        self.z_star = None
        self.z_nad = None
        self.last_v0 = None

    def collect_init_obs(self, env_id: str, seed: int, max_episode_steps: int) -> None:
        """Collect mc_k initial-state observations from freshly seeded environments."""
        obs_list = []
        for i in range(self.mc_k):
            env = mo_gym.make(env_id, max_episode_steps=max_episode_steps)
            obs, _ = env.reset(seed=seed + i)
            obs_list.append(torch.tensor(obs, dtype=torch.double).unsqueeze(0))
            env.close()
        self.init_obs = torch.cat(obs_list, dim=0)

    def set_z_bounds(self, ideal: Optional[np.ndarray], nadir: Optional[np.ndarray]) -> None:
        """Store the ideal and nadir reference points on the policy's device."""
        device = next(self.actor_critic.parameters()).device
        if ideal is not None:
            self.z_star = torch.tensor(ideal, dtype=torch.double, device=device)
        if nadir is not None:
            self.z_nad = torch.tensor(nadir, dtype=torch.double, device=device)

    @staticmethod
    def _standardize(advantages: torch.Tensor) -> torch.Tensor:
        """Standardize advantages to zero mean and unit variance."""
        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def _compute_loss_weights(self,
                              weights: torch.Tensor,
                              ob_rms: Optional[Any] = None,
                              obj_rms_var: Optional[np.ndarray] = None) -> tuple[torch.Tensor, dict]:
        """Compute the per-objective loss weights df/dv at the critic's initial-state estimate v0 and diagnostics."""
        device = next(self.actor_critic.parameters()).device

        # Raw WS: the loss weights are the preference weights
        if self.scalarizer.method == 'ws_raw':
            w_linear = weights.to(device=device, dtype=torch.double).detach()
            info = _empty_scalarization_info(weights.numel())
            info['loss_weights'] = w_linear.cpu().tolist()
            info['loss_weight_norm'] = float(torch.linalg.vector_norm(w_linear).item())
            info['loss_weights_finite'] = float(torch.isfinite(w_linear).all().item())
            self.last_v0 = None
            return w_linear, info

        # Estimate v0 in raw-return coordinates from the vector critic at the initial states
        init_obs = self.init_obs.to(device=device, dtype=torch.double)
        if ob_rms is not None:
            mean = torch.as_tensor(ob_rms.mean, device=device, dtype=torch.double)
            var = torch.as_tensor(ob_rms.var, device=device, dtype=torch.double)
            init_obs = torch.clamp((init_obs - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)
        with torch.no_grad():
            v0 = self.actor_critic.get_value(init_obs).mean(dim=0)
        if obj_rms_var is not None:
            v0 = v0 * torch.as_tensor(np.sqrt(obj_rms_var + 1e-8), device=device, dtype=torch.double)

        # Loss weights df/dv0
        v0 = v0.detach().requires_grad_(True)
        weights = weights.to(device=device, dtype=torch.double)
        u = self.scalarizer(v0.unsqueeze(0), weights, self.z_star, self.z_nad).mean()
        (loss_weights,) = torch.autograd.grad(u, v0)
        loss_weights = loss_weights.detach()
        self.last_v0 = v0.detach().clone()

        # Diagnostics
        info = {
            'v0': v0.detach().cpu().tolist(),
            'loss_weights': loss_weights.cpu().tolist(),
            'loss_weight_norm': float(torch.linalg.vector_norm(loss_weights).item()),
            'loss_weights_finite': float(torch.isfinite(loss_weights).all().item()),
        }
        return loss_weights, info

    def evaluation_diagnostics(self, evaluated_return: np.ndarray) -> dict:
        """Return the range-normalized L2 distance between the last critic estimate v0 and the evaluated return."""
        if self.last_v0 is None or self.z_star is None or self.z_nad is None:
            return {'v0_eval_error_normalized_l2': float('nan')}
        evaluated = torch.as_tensor(evaluated_return, device=self.last_v0.device, dtype=self.last_v0.dtype)
        width = (self.z_star - self.z_nad).abs().clamp_min(1e-6)
        normalized_difference = (self.last_v0 - evaluated) / width
        return {'v0_eval_error_normalized_l2': float(torch.linalg.vector_norm(normalized_difference).item())}

    def update(
            self,
            rollouts: RolloutStorageNL,
            weights: torch.Tensor,
            obj_var: Optional[np.ndarray] = None,
            ob_rms: Optional[Any] = None
    ) -> tuple[float, float, float, float, float, float, dict]:
        """Run ppo_epoch PPO epochs on the rollout with the scalarized advantage and return the mean statistics."""
        # --- Raw-scale returns and value predictions ---
        returns = rollouts.returns
        value_preds = rollouts.value_preds
        if obj_var is not None:
            objective_scale = torch.Tensor(np.sqrt(obj_var + 1e-8))
            returns = returns * objective_scale
            value_preds = value_preds * objective_scale

        # --- Scalarized advantages ---
        loss_weights, info = self._compute_loss_weights(weights, ob_rms, obj_rms_var=obj_var)
        vec_advantages = returns[:-1] - value_preds[:-1]
        global_normalization = self.advantage_normalization in ('global', 'none')
        if global_normalization:
            advantages = (vec_advantages * loss_weights).sum(dim=-1, keepdim=True)
            if self.advantage_normalization == 'global':
                advantages = self._standardize(advantages)
        else:
            advantages = vec_advantages

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        clip_fraction_epoch = 0
        approx_kl_epoch = 0
        explained_var_epoch = 0

        for _ in range(self.ppo_epoch):
            for sample in rollouts.feed_forward_generator(advantages, self.num_mini_batches):
                (obs_batch, _, actions_batch, value_preds_batch, return_batch, _,
                 old_action_log_probs_batch, adv_targ) = sample
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                # Diagnostics
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    approx_kl = (old_action_log_probs_batch - action_log_probs).mean().item()
                    returns_flat = return_batch.view(-1)
                    values_flat = values.view(-1).detach()
                    explained_var = (1.0 - (returns_flat - values_flat).var() / (returns_flat.var() + 1e-8)).item()

                # Policy loss
                if global_normalization:
                    adv_scalar = adv_targ
                else:
                    adv_scalar = self._standardize((adv_targ * loss_weights).sum(dim=-1, keepdim=True))
                surr1 = ratio * adv_scalar
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_scalar
                action_loss = -(torch.min(surr1, surr2)).mean()

                # Value loss (per objective, then averaged)
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Optimization step
                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                clip_fraction_epoch += clip_fraction
                approx_kl_epoch += approx_kl
                explained_var_epoch += explained_var

        num_updates = self.ppo_epoch * self.num_mini_batches
        return (value_loss_epoch / num_updates,
                action_loss_epoch / num_updates,
                dist_entropy_epoch / num_updates,
                clip_fraction_epoch / num_updates,
                approx_kl_epoch / num_updates,
                explained_var_epoch / num_updates,
                info)