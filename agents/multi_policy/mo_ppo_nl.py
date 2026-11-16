import os
import json
import glob
import torch
import pickle

import numpy as np
import gymnasium as gym

from copy import deepcopy
from typing import Union, Optional, Any
from typing_extensions import override
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.pool import Pool

from agents.utils.agent import Agent
from agents.utils.scalarizer import Scalarizer
from misc.evaluation import log_metrics
from misc.utils import setup_directories, BASE_DIR
from misc.weights import generate_das_dennis_weights, generate_layer_energy_weights, generate_dirichlet_weights

from agents.single_policy.ppo_utils_nl.sample import Sample
from agents.single_policy.ppo_utils_nl.ppo_nl import PPONL
from agents.single_policy.ppo_utils_nl.ppo_worker_nl import ppo_worker_nl
from agents.single_policy.ppo_utils_nl.a2c_ppo.model_nl import PolicyNL, PolicyNLDiscrete
from agents.single_policy.ppo_utils_nl.a2c_ppo.envs import make_vec_envs
from agents.single_policy.ppo_utils_nl.external_pareto import ExternalPareto


def _available_cpus():
    """Return the number of CPUs available to this process."""
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    return len(os.sched_getaffinity(0))


class StochVsDeterStore:
    FILE_NAME = 'stoch_vs_deter.npz'

    def __init__(self, directory: str):
        """Collect paired stochastic/deterministic evaluation records of one run in an NPZ file."""
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, self.FILE_NAME)
        self.arrays = {}

    def add(self,
            global_step: int,
            global_iteration: int,
            weights: np.ndarray,
            records: list) -> None:
        """Add the paired evaluation of one checkpoint and rewrite the file."""
        prefix = f't_{int(global_step)}'
        eval_rep = int(records[0]['eval_rep'])
        det_rep = int(records[0]['eval_rep_deterministic'])
        eval_seed = int(records[0]['eval_seed'])

        def stack(name: str, dtype: type) -> np.ndarray:
            return np.stack([np.asarray(record[name], dtype=dtype) for record in records])

        self.arrays.update({
            f'{prefix}_iteration': np.asarray(global_iteration, dtype=np.int64),
            f'{prefix}_weights': np.asarray(weights, dtype=np.float64),
            f'{prefix}_det_returns': stack('det_return', np.float64),
            f'{prefix}_det_episode_returns': stack('det_episode_returns', np.float64),
            f'{prefix}_det_episode_lengths': stack('det_episode_lengths', np.int64),
            f'{prefix}_det_episode_truncated': stack('det_episode_truncated', bool),
            f'{prefix}_stoch_mean_returns': stack('stoch_mean', np.float64),
            f'{prefix}_episode_returns': stack('episode_returns', np.float64),
            f'{prefix}_episode_lengths': stack('episode_lengths', np.int64),
            f'{prefix}_episode_truncated': stack('episode_truncated', bool),
            f'{prefix}_episode_seeds': eval_seed + np.arange(eval_rep, dtype=np.int64),
            f'{prefix}_mean_action_entropy': stack('mean_action_entropy', np.float64),
            f'{prefix}_eval_rep': np.asarray(eval_rep, dtype=np.int64),
            f'{prefix}_eval_rep_deterministic': np.asarray(det_rep, dtype=np.int64),
        })
        self._write()

    def _write(self) -> None:
        """Write all arrays atomically via a temporary file."""
        temporary = self.path[:-len('.npz')] + f'.tmp.{os.getpid()}.npz'
        np.savez(temporary, **self.arrays)
        os.replace(temporary, self.path)


class MOPPONL(Agent):
    def __init__(
            self,
            env_id: str,
            tmp_env: gym.Env,
            num_subproblems: int = 6,
            replicas_per_weight: int = 1,
            num_processes: int = 8,
            archive_size: Optional[int] = None,
            num_steps: int = 512,
            gamma: float = 0.99,
            obj_rms: bool = True,
            ob_rms: bool = True,
            use_proper_time_limits: bool = True,
            net_arch: list[int] = [64, 64],
            layernorm: bool = False,
            learning_rate: float = 3e-4,
            eps: float = 1e-4,
            clip_param: float = 0.2,
            ppo_epoch: int = 10,
            num_mini_batches: int = 32,
            entropy_coef: float = 0.0,
            value_loss_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_clipped_value_loss: bool = True,
            use_linear_lr_decay: bool = True,
            lr_decay_ratio: float = 1.0,
            use_gae: bool = True,
            gae_lambda: float = 0.95,
            init_w_sampling: str = 'uniform',
            dirichlet_alpha: float = 1.0,
            scalarization_method: str = 'ws_raw',
            mu: float = 0.1,
            log: bool = True,
            seed: int = 42,
            max_episode_steps: int = 500,
            device: Union[torch.device, str] = 'cpu',
            name: str = 'mo_ppo_nl',
    ):
        """Multi-objective PPO with a population of P policies on K = P / R distinct weights."""
        super().__init__(tmp_env, device=device, seed=seed, name=name)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # --- Environment & seeding ---
        self.env_id = env_id
        self.seed = seed
        self.device = device

        # --- Population parameters ---
        self.num_subproblems = int(num_subproblems)
        self.replicas_per_weight = int(replicas_per_weight)
        self.population_size = self.num_subproblems
        self.num_distinct_weights = self.population_size // self.replicas_per_weight
        self.num_processes = num_processes
        self.archive_size = archive_size

        # --- Rollout parameters ---
        self.num_steps = num_steps
        self.gamma = gamma
        self.obj_rms = obj_rms
        self.ob_rms = ob_rms
        self.use_proper_time_limits = use_proper_time_limits

        # --- PPO network / optimizer ---
        self.net_arch = net_arch
        self.layernorm = layernorm
        self.learning_rate = learning_rate
        self.eps = eps
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batches = num_mini_batches
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.max_episode_steps = max_episode_steps

        # --- Learning-rate schedule & GAE ---
        self.use_linear_lr_decay = use_linear_lr_decay
        self.lr_decay_ratio = lr_decay_ratio
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        # --- Weights & scalarization ---
        self.init_w_sampling = init_w_sampling
        self.dirichlet_alpha = dirichlet_alpha
        self.scalarizer = Scalarizer(method=scalarization_method, mu=mu)
        self.log = log

        # --- Global counters ---
        self.global_iterations = 0
        self.global_timesteps = 0
        self.global_step = 0

        self.ep = ExternalPareto(self.archive_size)

        # --- Weight population & initial samples ---
        self.distinct_weights = self._sample_weights(num_weights=self.num_distinct_weights)
        self.initial_weights, self.agent_weight_ids, self.agent_replica_ids = self._expand_weight_replicas(
            self.distinct_weights)
        print(f'[INFO] Initialised weight population: {self.num_distinct_weights} distinct weights x '
              f'{self.replicas_per_weight} replicas = {self.population_size} policies.')
        self.initial_samples = self._build_initial_samples()

        # --- Run state ---
        self.total_iterations = None
        self.eval_iterations = None
        self._stoch_vs_deter_config = None

        # --- Reference points ---
        self.ideal = None
        self.nadir = None

    # -------------------- SETUP --------------------

    def _sample_weights(self, num_weights: int) -> np.ndarray:
        """Generate K distinct decomposition weights with the configured sampling method."""
        if self.init_w_sampling == 'dirichlet':
            weights = generate_dirichlet_weights(num_weights=num_weights,
                                                 reward_dim=self.reward_dim,
                                                 alpha=self.dirichlet_alpha)
        elif self.init_w_sampling == 'das_dennis' or self.reward_dim == 2:
            weights = generate_das_dennis_weights(num_weights=num_weights,
                                                  reward_dim=self.reward_dim)
        else:
            layers = {3: [1],
                      4: [1, 0],
                      6: [2],
                      7: [2, 0],
                      9: [3],
                      10: [3, 0],
                      12: [4],
                      13: [4, 0],
                      15: [4, 1],
                      16: [4, 0, 1],
                      18: [5, 1],
                      19: [5, 0, 1],
                      21: [5, 2]}
            weights = generate_layer_energy_weights(layers=layers[num_weights],
                                                    reward_dim=self.reward_dim)
        return np.asarray(weights, dtype=np.float64)

    def _expand_weight_replicas(self, distinct_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Repeat every distinct weight R times in contiguous order and return weights, weight IDs and replica IDs."""
        weights = np.asarray(distinct_weights, dtype=np.float64)
        expanded_weights = np.repeat(weights, repeats=self.replicas_per_weight, axis=0)
        weight_ids = np.repeat(np.arange(self.num_distinct_weights, dtype=np.int64), repeats=self.replicas_per_weight)
        replica_ids = np.tile(np.arange(self.replicas_per_weight, dtype=np.int64), reps=self.num_distinct_weights)
        return expanded_weights, weight_ids, replica_ids

    def _build_policy(self) -> Union[PolicyNL, PolicyNLDiscrete]:
        """Build a discrete or continuous NL policy for the environment's action space."""
        policy_cls = PolicyNLDiscrete if isinstance(self.env.action_space, gym.spaces.Discrete) else PolicyNL
        return policy_cls(
            self.obs_shape,
            self.action_space,
            net_arch=self.net_arch,
            reward_dim=self.reward_dim,
            layernorm=self.layernorm,
        ).to(self.device).double()

    def _build_agent(self, policy: Union[PolicyNL, PolicyNLDiscrete], scalarizer: Scalarizer) -> PPONL:
        """Build the PPO-NL optimizer for a policy."""
        return PPONL(
            actor_critic=policy,
            clip_param=self.clip_param,
            ppo_epoch=self.ppo_epoch,
            num_mini_batches=self.num_mini_batches,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef,
            lr=self.learning_rate,
            eps=self.eps,
            max_grad_norm=self.max_grad_norm,
            use_clipped_value_loss=self.use_clipped_value_loss,
            scalarizer=scalarizer,
        )

    def _collect_env_params(self) -> dict:
        """Create a throwaway VecEnv and copy its normalisation statistics."""
        venv = make_vec_envs(
            env_name=self.env_id,
            seed=self.seed,
            num_processes=self.num_processes,
            gamma=self.gamma,
            log_dir=None,
            device=self.device,
            allow_early_resets=True,
            obj_rms=self.obj_rms,
            ob_rms=self.ob_rms,
            max_episode_steps=self.max_episode_steps,
            multiprocessing_envs=False,
        )
        env_params = {key: deepcopy(getattr(venv, key)) for key in ('ob_rms', 'ret_rms', 'obj_rms')}
        venv.close()
        return env_params

    def _build_initial_samples(self) -> list:
        """Build one untrained sample per initial weight."""
        initial_samples = []
        for w in self.initial_weights:
            policy = self._build_policy()
            agent = self._build_agent(policy, self.scalarizer)
            env_params = self._collect_env_params()
            agent.collect_init_obs(env_id=self.env_id, seed=self.seed, max_episode_steps=self.max_episode_steps)
            sample = Sample(env_params, policy, agent, weights=torch.Tensor(w),
                            learning_rate=self.learning_rate, eps=self.eps)
            sample.objs = -np.inf
            initial_samples.append(sample)
        return initial_samples

    @override
    def eval(self, obs: Union[np.ndarray, torch.Tensor], w: Union[np.ndarray, torch.Tensor]) -> Union[
        np.ndarray, torch.Tensor]:
        """Not used: MOPPONL returns a policy collection."""
        pass

    # -------------------- PERSISTENCE --------------------

    @override
    def load(
            self,
            path: str,
            file_name: str,
            load_replay_buffer: bool = True,
            load_archive: bool = False
    ) -> None:
        """Load all saved policies, optimizers, env statistics and objectives into the archive."""
        suffix = '_actor_critic.pt'
        paths = glob.glob(os.path.join(path, f'{glob.escape(file_name)}_*{suffix}'))
        indices = sorted(int(os.path.basename(p)[len(file_name) + 1:-len(suffix)]) for p in paths)

        loaded_samples = []
        for idx in indices:
            prefix = os.path.join(path, f'{file_name}_{idx}')

            policy = self._build_policy()
            policy.load_state_dict(torch.load(prefix + suffix, map_location=self.device))

            agent = self._build_agent(policy, self.scalarizer)
            opt_path = prefix + '_optimizer.pt'
            if os.path.exists(opt_path):
                agent.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))

            env_path = prefix + '_env_params.pkl'
            env_params = None
            if os.path.exists(env_path):
                with open(env_path, 'rb') as f:
                    env_params = pickle.load(f)

            objs_path = prefix + '_objs.txt'
            objs = np.loadtxt(objs_path, delimiter=',', ndmin=1) if os.path.exists(objs_path) else None

            sample = Sample(env_params, policy, agent, weights=torch.Tensor(self.initial_weights[idx]),
                            learning_rate=self.learning_rate, eps=self.eps)
            sample.objs = objs
            loaded_samples.append(sample)

        self.ep.sample_batch = loaded_samples
        if loaded_samples:
            self.ep.obj_batch = np.vstack([s.objs for s in loaded_samples])
        else:
            self.ep.obj_batch = np.empty((0,))
        print(f'[INFO] Loaded {len(loaded_samples)} MO-PPO-NL agents from {path}')

    @override
    def save(self, path: str, file_name: str, save_replay_buffer: bool = True) -> None:
        """Save every sample of the archive."""
        for i, sample in enumerate(self.ep.sample_batch):
            sample.save(path, f'{file_name}_{i}')
        print(f'[INFO] All MO-PPO-NL agents saved to {path}.')

    @override
    def save_config(self, path: str, file_name: str):
        """Write hyperparameters, weight population and reference points as JSON."""
        def to_list(value: Optional[np.ndarray]) -> Optional[list]:
            return None if value is None else np.asarray(value).tolist()

        config = {
            'num_subproblems': self.num_subproblems,
            'replicas_per_weight': self.replicas_per_weight,
            'num_distinct_weights': self.num_distinct_weights,
            'population_size': self.population_size,
            'distinct_weights': self.distinct_weights.tolist(),
            'initial_weights': self.initial_weights.tolist(),
            'agent_weight_ids': self.agent_weight_ids.tolist(),
            'agent_replica_ids': self.agent_replica_ids.tolist(),
            'num_processes': self.num_processes,
            'archive_size': self.archive_size,
            'num_steps': self.num_steps,
            'gamma': self.gamma,
            'obj_rms': self.obj_rms,
            'ob_rms': self.ob_rms,
            'use_proper_time_limits': self.use_proper_time_limits,
            'net_arch': self.net_arch,
            'layernorm': self.layernorm,
            'learning_rate': self.learning_rate,
            'eps': self.eps,
            'clip_param': self.clip_param,
            'ppo_epoch': self.ppo_epoch,
            'num_mini_batches': self.num_mini_batches,
            'entropy_coef': self.entropy_coef,
            'value_loss_coef': self.value_loss_coef,
            'max_grad_norm': self.max_grad_norm,
            'use_clipped_value_loss': self.use_clipped_value_loss,
            'use_linear_lr_decay': self.use_linear_lr_decay,
            'lr_decay_ratio': self.lr_decay_ratio,
            'use_gae': self.use_gae,
            'gae_lambda': self.gae_lambda,
            'max_episode_steps': self.max_episode_steps,
            'total_iterations': self.total_iterations,
            'eval_iterations': self.eval_iterations,
            'logging_x_axis': 'global_timesteps',
            'global_timestep_formula': 'global_iteration * num_subproblems * num_steps',
            'evaluation_history_storage': 'ParetoFrontStore NPZ',
            'evaluation_history_x_axis': 'global_step',
            'evaluation_history_iteration_key': 't_<global_step>_iteration',
            'ideal': to_list(self.ideal),
            'nadir': to_list(self.nadir),
            'mu': self.scalarizer.mu,
            'init_w_sampling': self.init_w_sampling,
            'dirichlet_alpha': self.dirichlet_alpha,
            'scalarization_method': self.scalarizer.method,
            'seed': self.seed,
        }
        if self._stoch_vs_deter_config:
            config.update(self._stoch_vs_deter_config)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{file_name}.json'), 'w') as f:
            json.dump(config, f, indent=4)
        print('[INFO] MO-PPO-NL configuration saved')

    # -------------------- TRAINING HELPERS --------------------

    @staticmethod
    def _weights_array(samples: list) -> np.ndarray:
        """Stack the weights of all samples into one array."""
        return np.stack([
            sample.weights.detach().cpu().numpy()
            if torch.is_tensor(sample.weights)
            else np.asarray(sample.weights)
            for sample in samples
        ])

    def _launch_workers(self,
                        pool: Pool,
                        selected_samples: list,
                        current_iteration: int,
                        end_iteration: int,
                        total_iterations: int,
                        eval_rep: int,
                        eval_seed: int,
                        eval_gamma: float,
                        eval_deterministic: bool,
                        do_training: bool,
                        eval_rep_deterministic: int = 1) -> dict:
        """Dispatch one task per subproblem to the persistent pool and return the results by task ID."""
        task_args = [
            (sample_id, sample, self.device,
             current_iteration, end_iteration, total_iterations,
             self.env_id, self.seed, self.num_processes,
             self.num_steps, self.gamma,
             self.obj_rms, self.ob_rms, self.reward_dim,
             self.use_linear_lr_decay, self.lr_decay_ratio, self.learning_rate,
             self.use_gae, self.gae_lambda,
             self.use_proper_time_limits, self.max_episode_steps,
             eval_rep, eval_seed, eval_gamma, eval_deterministic,
             do_training, eval_rep_deterministic)
            for sample_id, sample in enumerate(selected_samples)
        ]
        return {r['task_id']: r for r in pool.starmap(ppo_worker_nl, task_args)}

    def _collect_samples_and_stats(self, results_by_id: dict) -> tuple:
        """Copy the returned samples and gather per-agent training statistics."""
        scalar_keys = {
            'entropy': 'dist_entropy',
            'value_loss': 'mean_value_loss',
            'action_loss': 'mean_action_loss',
            'clip_frac': 'mean_clip_fraction',
            'approx_kl': 'mean_approx_kl',
            'expl_var': 'mean_explained_var',
            'lr': 'mean_lr',
            'loss_weight_norm': 'loss_weight_norm',
            'loss_weights_finite': 'loss_weights_finite',
            'v0_eval_error_normalized_l2': 'v0_eval_error_normalized_l2',
        }
        vector_keys = ('v0', 'loss_weights')
        all_sample_batch = []
        stats = {key: {} for key in (*scalar_keys, *vector_keys)}
        for tid in sorted(results_by_id):
            result = results_by_id[tid]
            all_sample_batch += [Sample.copy_from(sample) for sample in result['offspring_batch']]
            for key, result_key in scalar_keys.items():
                stats[key][tid] = float(result[result_key])
            for key in vector_keys:
                stats[key][tid] = list(result[key])
        return all_sample_batch, stats

    def _record_stoch_vs_deter(self,
                               results_by_id: dict,
                               selected_samples: list,
                               store: StochVsDeterStore,
                               writer: SummaryWriter) -> None:
        """Store and log the paired evaluation of the current checkpoint."""
        records = [results_by_id[tid]['stoch_vs_deter'] for tid in sorted(results_by_id)]
        store.add(self.global_step, self.global_iterations, self._weights_array(selected_samples), records)
        self._log_stoch_vs_deter(writer, records, self.global_step)

    # -------------------- LOGGING --------------------

    def _log_stoch_vs_deter(self, writer: SummaryWriter, records: list, global_step: int) -> None:
        """Log deterministic return, stochastic mean/spread and return distribution per policy."""
        timeout_rates = []
        entropies = []
        for agent_idx, record in enumerate(records):
            episodes = np.asarray(record['episode_returns'], dtype=np.float64)
            spread = episodes.std(axis=0, ddof=1) if len(episodes) > 1 else np.zeros(self.reward_dim)
            tag = f'stoch_vs_deter/agent_{agent_idx:02d}'
            for objective in range(self.reward_dim):
                writer.add_scalar(f'{tag}/det_obj_{objective}', float(record['det_return'][objective]), global_step)
                writer.add_scalar(f'{tag}/stoch_mean_obj_{objective}', float(record['stoch_mean'][objective]),
                                  global_step)
                writer.add_scalar(f'{tag}/stoch_std_obj_{objective}', float(spread[objective]), global_step)
                writer.add_histogram(f'stoch_vs_deter_dist/agent_{agent_idx:02d}/obj_{objective}',
                                     episodes[:, objective], global_step)
            timeout_rates.append(float(np.mean(record['episode_truncated'])))
            entropies.append(float(record['mean_action_entropy']))
            writer.add_scalar(f'{tag}/stoch_timeout_rate', timeout_rates[-1], global_step)
            writer.add_scalar(f'{tag}/det_timeout_rate', float(np.mean(record['det_episode_truncated'])), global_step)
            writer.add_scalar(f'{tag}/action_entropy', entropies[-1], global_step)
        writer.add_scalar('stoch_vs_deter/stoch_timeout_rate_mean', float(np.mean(timeout_rates)), global_step)
        writer.add_scalar('stoch_vs_deter/action_entropy_mean', float(np.mean(entropies)), global_step)

    def _log_progress(self, writer: SummaryWriter) -> None:
        """Log the global iteration against the global timestep axis."""
        writer.add_scalar('progress/global_iteration', self.global_iterations, self.global_step)

    def _log_entropy(self, writer: SummaryWriter, stats: dict, global_step: int) -> None:
        """Log mean, per-agent and corner/interior action entropy."""
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        max_entropy = np.log(self.env.action_space.n) if discrete else None
        valid = {tid: e for tid, e in stats['entropy'].items() if not np.isnan(e)}
        if not valid:
            return

        mean_e = float(np.mean(list(valid.values())))
        writer.add_scalar('entropy/mean', mean_e, global_step)
        if max_entropy:
            writer.add_scalar('entropy/mean_normalised', mean_e / max_entropy, global_step)

        # Per-agent curves
        for agent_idx, e in valid.items():
            writer.add_scalar(f'entropy/agent_{agent_idx:02d}', e, global_step)
            if max_entropy:
                writer.add_scalar(f'entropy/agent_{agent_idx:02d}_normalised', e / max_entropy, global_step)

        # Corner (max(w) > 0.99) vs interior split
        corner_e = []
        interior_e = []
        for agent_idx, e in valid.items():
            (corner_e if max(self.initial_weights[agent_idx]) > 0.99 else interior_e).append(e)
        if corner_e:
            writer.add_scalar('entropy/corner_agents_mean', float(np.mean(corner_e)), global_step)
        if interior_e:
            writer.add_scalar('entropy/interior_agents_mean', float(np.mean(interior_e)), global_step)

    def _log_train_stats(self, writer: SummaryWriter, stats: dict, global_step: int) -> None:
        """Log PPO health metrics and scalarization diagnostics."""
        def _mean_valid(key: str) -> Optional[float]:
            values = [value for value in stats[key].values() if not np.isnan(value)]
            return float(np.mean(values)) if values else None

        # Standard PPO metrics
        standard_metrics = [
            ('train/value_loss_mean', 'value_loss'),
            ('train/action_loss_mean', 'action_loss'),
            ('train/clip_fraction_mean', 'clip_frac'),
            ('train/approx_kl_mean', 'approx_kl'),
            ('train/explained_var_mean', 'expl_var'),
            ('train/lr', 'lr'),
        ]
        for tag, key in standard_metrics:
            value = _mean_valid(key)
            if value is not None:
                writer.add_scalar(tag, value, global_step)

        # Per-agent scalarization diagnostics
        for agent_idx in sorted(stats['v0']):
            tag = f'scalarisation/agent_{agent_idx:02d}'
            for objective, value in enumerate(stats['v0'][agent_idx]):
                if not np.isnan(value):
                    writer.add_scalar(f'{tag}/v0_obj_{objective}', value, global_step)
            for objective, value in enumerate(stats['loss_weights'][agent_idx]):
                if not np.isnan(value):
                    writer.add_scalar(f'{tag}/loss_weight_obj_{objective}', value, global_step)
            for key in ('loss_weight_norm', 'loss_weights_finite', 'v0_eval_error_normalized_l2'):
                value = stats[key][agent_idx]
                if not np.isnan(value):
                    writer.add_scalar(f'{tag}/{key}', value, global_step)

        discrepancy = _mean_valid('v0_eval_error_normalized_l2')
        if discrepancy is not None:
            writer.add_scalar('scalarisation/v0_eval_error_normalized_l2_mean', discrepancy, global_step)

    def _eval_and_log_fronts(self,
                             selected_samples: list,
                             ref_point: np.ndarray,
                             known_pareto_front: Optional[list[np.ndarray]],
                             num_eval_weights: int,
                             save_fronts: bool,
                             pf_store: Optional[Any],
                             policy_store: Optional[Any],
                             writer: SummaryWriter,
                             global_step: int) -> None:
        """Log metrics of the archive front and the current policy front and store both snapshots."""
        common = dict(
            ref_point=ref_point,
            known_pareto_front=known_pareto_front,
            reward_dim=self.reward_dim,
            num_sample_weights=num_eval_weights,
            global_step=global_step,
            writer=writer,
            log=self.log,
            save_fronts=save_fronts,
            global_iteration=self.global_iterations,
        )

        # Archive front (cumulative non-dominated)
        hv_archive = log_metrics(
            self.ep.obj_batch,
            store=pf_store,
            tag_prefix='archive',
            **common,
        )

        # Current-policy front (one point per policy)
        hv_policy = log_metrics(
            [s.objs for s in selected_samples],
            store=policy_store,
            tag_prefix='policy',
            weights=self._weights_array(selected_samples),
            agent_ids=np.arange(len(selected_samples), dtype=np.int64),
            weight_ids=self.agent_weight_ids.copy(),
            replica_ids=self.agent_replica_ids.copy(),
            **common,
        )

        print(f'HV archive @ iteration {self.global_iterations}, step {global_step} '
              f'({len(self.ep.obj_batch)}): {round(hv_archive, 2)}')
        print(f'HV policy  @ iteration {self.global_iterations}, step {global_step} '
              f'({len(selected_samples)}): {round(hv_policy, 2)}\n')

    # -------------------- TRAINING --------------------

    def train(
            self,
            total_iterations: int,  # per agent
            ref_point: np.ndarray,
            eval_iterations: int = 1,  # per agent
            known_pareto_front: Optional[list[np.ndarray]] = None,
            ideal: Optional[np.ndarray] = None,
            nadir: Optional[np.ndarray] = None,
            num_eval_weights: int = 100,
            eval_rep: int = 5,
            eval_seed: int = 43,
            eval_gamma: float = 0.99,
            eval_deterministic: bool = True,
            eval_rep_deterministic: int = 1,
            save_fronts: bool = False,
            save_models: bool = False,
            log_dir: str = 'mo_ppo_nl',
            file_name: str = 'mo_ppo_nl',
    ) -> list:
        """Train all policies in a persistent worker pool and evaluate the fronts every eval_iterations."""
        # ── Directories and stores ────────────────────────────────────────────
        runs_path, model_path, config_path, pf_store, policy_store = setup_directories(
            log_dir, file_name, save_fronts, save_models)
        writer = SummaryWriter(log_dir=runs_path, filename_suffix=f'{self.seed:04d}')
        fronts = []

        # ── Scalarization references ──────────────────────────────────────────
        if self.scalarizer.method in {'ws', 'ws_raw'}:
            print('[INFO] Raw WS does not require reference points.')
        else:
            self.ideal = np.asarray(ideal, dtype=np.float64)
            self.nadir = np.asarray(nadir, dtype=np.float64)
            print(f'[INFO] Using provided ideal={self.ideal.round(4)}, nadir={self.nadir.round(4)}.')
        objective_scale = self.ideal - self.nadir if self.scalarizer.method.endswith('_norm') else None
        self.scalarizer.objective_scale = objective_scale
        for sample in self.initial_samples:
            sample.agent.set_z_bounds(ideal=self.ideal, nadir=self.nadir)
            sample.agent.scalarizer.objective_scale = objective_scale

        # ── Paired stochastic/deterministic evaluation ────────────────────────
        stoch_vs_deter_store = None
        self._stoch_vs_deter_config = None
        if not eval_deterministic:
            stoch_vs_deter_store = StochVsDeterStore(os.path.join(BASE_DIR, log_dir, 'policies'))
            self._stoch_vs_deter_config = {
                'eval_deterministic': False,
                'paired_evaluation': True,
                'eval_rep': int(eval_rep),
                'eval_rep_deterministic': int(eval_rep_deterministic),
                'eval_seed': int(eval_seed),
                'eval_gamma': float(eval_gamma),
                'policy_objective_vector': 'stochastic_mean',
                'paired_evaluation_file': StochVsDeterStore.FILE_NAME,
            }
            print(f'[INFO] Paired evaluation: {eval_rep} stochastic + {eval_rep_deterministic} deterministic '
                  f'episode(s) per policy and checkpoint -> {stoch_vs_deter_store.path}')

        self.total_iterations = total_iterations
        self.eval_iterations = eval_iterations
        self.save_config(config_path, file_name)

        # ── Iteration bookkeeping ─────────────────────────────────────────────
        timesteps_per_iteration = self.num_subproblems * self.num_steps
        current_iteration = 0
        self.global_iterations = 0
        self.global_timesteps = 0
        self.global_step = 0

        cpus_per_task = _available_cpus()
        pool_size = min(cpus_per_task, self.num_subproblems)
        print(f'[INFO] Using persistent worker pool of size {pool_size} '
              f'(cpus={cpus_per_task}, num_processes={self.num_processes}, '
              f'K={self.num_distinct_weights}, R={self.replicas_per_weight}, '
              f'population={self.population_size}).')

        # ── Persistent pool for the entire training run ───────────────────────
        with Pool(processes=pool_size) as pool:
            # Initial evaluation (step 0, no training)
            results_by_id = self._launch_workers(
                pool, self.initial_samples, current_iteration, 0, total_iterations,
                eval_rep, eval_seed, eval_gamma, eval_deterministic,
                do_training=False, eval_rep_deterministic=eval_rep_deterministic)

            selected_samples, _ = self._collect_samples_and_stats(results_by_id)
            for s in selected_samples:
                self.ep.update([s])

            self._log_progress(writer)
            self._eval_and_log_fronts(
                selected_samples, ref_point, known_pareto_front,
                num_eval_weights, save_fronts, pf_store, policy_store, writer, self.global_step)
            fronts.append(self.ep.obj_batch.copy())
            if stoch_vs_deter_store is not None:
                self._record_stoch_vs_deter(results_by_id, selected_samples, stoch_vs_deter_store, writer)

            # ── Main training loop ────────────────────────────────────────────
            while current_iteration < total_iterations:
                end_iteration = min(current_iteration + eval_iterations, total_iterations)
                results_by_id = self._launch_workers(
                    pool, selected_samples, current_iteration, end_iteration, total_iterations,
                    eval_rep, eval_seed, eval_gamma, eval_deterministic,
                    do_training=True, eval_rep_deterministic=eval_rep_deterministic)

                selected_samples, stats = self._collect_samples_and_stats(results_by_id)
                for s in selected_samples:
                    self.ep.update([s])
                current_iteration = end_iteration
                self.global_iterations = current_iteration
                self.global_timesteps = self.global_iterations * timesteps_per_iteration
                self.global_step = self.global_timesteps

                self._log_entropy(writer, stats, self.global_step)
                self._log_train_stats(writer, stats, self.global_step)
                self._log_progress(writer)
                self._eval_and_log_fronts(
                    selected_samples, ref_point, known_pareto_front,
                    num_eval_weights, save_fronts, pf_store, policy_store, writer, self.global_step)
                fronts.append(self.ep.obj_batch.copy())
                if stoch_vs_deter_store is not None:
                    self._record_stoch_vs_deter(results_by_id, selected_samples, stoch_vs_deter_store, writer)

        # ── Cleanup ───────────────────────────────────────────────────────────
        if save_models:
            self.save(model_path, file_name)
        self.env.close()
        writer.close()

        return fronts