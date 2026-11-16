import torch

import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym

from copy import deepcopy
from typing import Union, Optional, Any

from agents.single_policy.ppo_utils_nl.sample import Sample
from agents.single_policy.ppo_utils_nl.a2c_ppo.envs import make_vec_envs
from agents.single_policy.ppo_utils_nl.a2c_ppo.utils import update_linear_schedule
from agents.single_policy.ppo_utils_nl.a2c_ppo.storage_nl import RolloutStorageNL
from agents.single_policy.ppo_utils_nl.a2c_ppo.model_nl import PolicyNL, PolicyNLDiscrete


def _evaluation_episode(actor_critic: Union[PolicyNL, PolicyNLDiscrete],
                        env: gym.Env,
                        ob_rms: Optional[Any],
                        model_parameter: torch.Tensor,
                        seed: int,
                        eval_gamma: float,
                        reward_dim: int,
                        deterministic: bool,
                        total_obj: np.ndarray) -> tuple[np.ndarray, int, bool, float]:
    """Run one seeded episode, add its discounted return to total_obj and return return, length, truncation, -log pi."""
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    episode_obj = np.zeros(reward_dim, dtype=float)
    done = False
    truncated = False
    discounted_gamma = 1.0
    length = 0
    neg_log_prob = 0.0

    while not done:
        if ob_rms is not None:
            obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
        obs_tensor = torch.as_tensor(obs, dtype=model_parameter.dtype, device=model_parameter.device).unsqueeze(0)
        _, action_tensor, action_log_prob = actor_critic.act(obs_tensor, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action_tensor.cpu().numpy().squeeze())
        done = terminated or truncated

        total_obj += discounted_gamma * reward
        episode_obj += discounted_gamma * reward
        discounted_gamma *= eval_gamma
        neg_log_prob -= float(action_log_prob.sum())
        length += 1

    return episode_obj, length, bool(truncated), neg_log_prob


def evaluation_exp_stochastic(
        sample: Sample,
        env_id: str,
        reward_dim: int,
        eval_num: int,
        eval_seed: int,
        eval_gamma: float,
        max_episode_steps: int,
        det_eval_num: int = 1,
) -> dict:
    """Evaluate one frozen actor with deterministic and with sampled actions on the same episode seeds."""
    actor_critic = sample.actor_critic
    was_training = actor_critic.training
    ob_rms = sample.env_params.get('ob_rms', None)
    model_parameter = next(actor_critic.parameters())

    det_total = np.zeros(reward_dim, dtype=float)
    stoch_total = np.zeros(reward_dim, dtype=float)
    det_returns = np.zeros((det_eval_num, reward_dim), dtype=float)
    det_lengths = np.zeros(det_eval_num, dtype=np.int64)
    det_truncated = np.zeros(det_eval_num, dtype=bool)
    episode_returns = np.zeros((eval_num, reward_dim), dtype=float)
    episode_lengths = np.zeros(eval_num, dtype=np.int64)
    episode_truncated = np.zeros(eval_num, dtype=bool)
    total_neg_log_prob = 0.0

    with torch.random.fork_rng(), torch.no_grad():
        env = mo_gym.make(env_id, max_episode_steps=max_episode_steps)
        actor_critic.eval()
        for i in range(det_eval_num):
            det_returns[i], det_lengths[i], det_truncated[i], _ = _evaluation_episode(
                actor_critic, env, ob_rms, model_parameter, eval_seed + i, eval_gamma, reward_dim, True, det_total)
        for i in range(eval_num):
            episode_returns[i], episode_lengths[i], episode_truncated[i], neg_log_prob = _evaluation_episode(
                actor_critic, env, ob_rms, model_parameter, eval_seed + i, eval_gamma, reward_dim, False, stoch_total)
            total_neg_log_prob += neg_log_prob
        actor_critic.train(was_training)
        env.close()

    return {
        'stoch_mean': stoch_total / eval_num,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'episode_truncated': episode_truncated,
        'det_return': det_total / det_eval_num,
        'det_episode_returns': det_returns,
        'det_episode_lengths': det_lengths,
        'det_episode_truncated': det_truncated,
        'mean_action_entropy': total_neg_log_prob / max(int(episode_lengths.sum()), 1),
        'eval_rep': int(eval_num),
        'eval_rep_deterministic': int(det_eval_num),
        'eval_seed': int(eval_seed),
    }


def evaluation(
        sample: Sample,
        env_id: str,
        reward_dim: int,
        eval_num: int,
        eval_seed: int,
        eval_gamma: float,
        max_episode_steps: int,
        deterministic: bool = True,
) -> np.ndarray:
    """Return the mean discounted vector return of sample.actor_critic over eval_num seeded episodes."""
    actor_critic = sample.actor_critic
    was_training = actor_critic.training
    ob_rms = sample.env_params.get('ob_rms', None)
    model_parameter = next(actor_critic.parameters())
    total_obj = np.zeros(reward_dim, dtype=float)

    with torch.random.fork_rng(), torch.no_grad():
        env = mo_gym.make(env_id, max_episode_steps=max_episode_steps)
        actor_critic.eval()
        for i in range(eval_num):
            _evaluation_episode(actor_critic, env, ob_rms, model_parameter, eval_seed + i, eval_gamma, reward_dim,
                                deterministic, total_obj)
        actor_critic.train(was_training)
        env.close()

    return total_obj / eval_num


def ppo_worker_nl(sample_id: int,
                  sample: Sample,
                  device: torch.device,
                  start_iteration: int,
                  end_iteration: int,
                  total_iteration: int,
                  env_id: str,
                  seed: int,
                  num_processes: int,
                  num_steps: int,
                  gamma: float,
                  obj_rms: bool,
                  ob_rms: bool,
                  reward_dim: int,
                  use_linear_lr_decay: bool,
                  lr_decay_ratio: float,
                  lr: float,
                  use_gae: bool,
                  gae_lambda: float,
                  use_proper_time_limits: bool,
                  max_episode_steps: int,
                  eval_rep: int,
                  eval_seed: int,
                  eval_gamma: float,
                  eval_deterministic: bool = True,
                  do_training: bool = True,
                  eval_rep_deterministic: int = 1) -> dict:
    """Train one sample for the given PPO iterations, evaluate it and return the result dict for Pool.starmap."""
    env_params = sample.env_params
    actor_critic = sample.actor_critic
    agent = sample.agent
    weights = sample.weights

    # ── Environment setup ─────────────────────────────────────────────────
    envs = make_vec_envs(
        env_name=env_id,
        seed=seed,
        num_processes=num_processes,
        gamma=gamma,
        log_dir=None,
        device=device,
        allow_early_resets=False,
        obj_rms=obj_rms,
        ob_rms=ob_rms,
        max_episode_steps=max_episode_steps,
        multiprocessing_envs=False,
    )
    for key in ('ob_rms', 'ret_rms', 'obj_rms'):
        if env_params[key] is not None:
            setattr(envs.venv, key, deepcopy(env_params[key]))

    rollouts = RolloutStorageNL(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=1,
        reward_dim=reward_dim,
    )
    rollouts.obs[0].copy_(envs.reset())
    rollouts.to(device)

    # ── Training ──────────────────────────────────────────────────────────
    nan = float('nan')
    nan_vector = [nan] * reward_dim
    stats = {
        'dist_entropy': nan,
        'mean_value_loss': nan,
        'mean_action_loss': nan,
        'mean_clip_fraction': nan,
        'mean_approx_kl': nan,
        'mean_explained_var': nan,
        'mean_lr': nan,
        'v0': nan_vector.copy(),
        'loss_weights': nan_vector.copy(),
        'loss_weight_norm': nan,
        'loss_weights_finite': nan,
    }

    if do_training:
        scalar_hist = {key: [] for key in ('dist_entropy', 'mean_value_loss', 'mean_action_loss', 'mean_clip_fraction',
                                           'mean_approx_kl', 'mean_explained_var', 'mean_lr')}

        for j in range(start_iteration, end_iteration):
            torch.manual_seed(start_iteration + j + sample_id)
            if use_linear_lr_decay:
                update_linear_schedule(agent.optimizer, j * lr_decay_ratio, total_iteration, lr)

            # Rollout
            for step in range(num_steps):
                with torch.no_grad():
                    value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
                obs, reward, done, infos = envs.step(action)
                obj_tensor = torch.zeros([num_processes, reward_dim])
                for idx, info in enumerate(infos):
                    obj_tensor[idx] = torch.from_numpy(np.asarray(info['obj'], dtype=np.float64))
                masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info else [1.0] for info in infos])
                rollouts.insert(obs, 1, action, action_log_prob, value, obj_tensor, masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
            rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)

            # PPO update
            obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None
            value_loss, action_loss, dist_entropy, clip_frac, approx_kl, expl_var, scalarization_info = agent.update(
                rollouts, weights, obj_rms_var, deepcopy(envs.ob_rms))
            iteration_stats = {
                'dist_entropy': dist_entropy,
                'mean_value_loss': value_loss,
                'mean_action_loss': action_loss,
                'mean_clip_fraction': clip_frac,
                'mean_approx_kl': approx_kl,
                'mean_explained_var': expl_var,
                'mean_lr': agent.optimizer.param_groups[0]['lr'],
            }
            for key, stat in iteration_stats.items():
                scalar_hist[key].append(float(stat))

            rollouts.after_update()
            env_params = {key: deepcopy(getattr(envs, key)) for key in ('ob_rms', 'ret_rms', 'obj_rms')}

        # Iteration means and last scalarization diagnostics
        stats.update({key: float(np.mean(values)) for key, values in scalar_hist.items()})
        stats['v0'] = list(scalarization_info['v0'])
        stats['loss_weights'] = list(scalarization_info['loss_weights'])
        stats['loss_weight_norm'] = float(scalarization_info['loss_weight_norm'])
        stats['loss_weights_finite'] = float(scalarization_info['loss_weights_finite'])

    # ── Evaluation ────────────────────────────────────────────────────────
    sample_out = Sample(env_params, deepcopy(actor_critic), deepcopy(agent), deepcopy(weights),
                        sample.learning_rate, sample.eps)
    stoch_vs_deter = None
    if eval_deterministic:
        disc_obj = evaluation(sample_out, env_id, reward_dim, eval_rep, eval_seed, eval_gamma, max_episode_steps,
                              eval_deterministic)
    else:
        stoch_vs_deter = evaluation_exp_stochastic(sample_out, env_id, reward_dim, eval_rep, eval_seed, eval_gamma,
                                                   max_episode_steps, det_eval_num=eval_rep_deterministic)
        disc_obj = stoch_vs_deter['stoch_mean']
    sample_out.objs = disc_obj
    if np.isclose(eval_gamma, gamma):
        eval_error = agent.evaluation_diagnostics(disc_obj)['v0_eval_error_normalized_l2']
    else:
        eval_error = nan
    envs.close()

    result = {
        'task_id': sample_id,
        'offspring_batch': np.array([sample_out]),
        **stats,
        'z_objs': disc_obj,
        'v0_eval_error_normalized_l2': eval_error,
        'done': True,
    }
    if stoch_vs_deter is not None:
        result['stoch_vs_deter'] = stoch_vs_deter
    return result