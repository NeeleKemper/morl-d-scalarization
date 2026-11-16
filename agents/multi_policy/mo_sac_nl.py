import os
import copy
import json
import torch
import pickle

import numpy as np
import gymnasium as gym

from typing import Union, Optional, Any
from typing_extensions import override
from torch.utils.tensorboard import SummaryWriter

from agents.utils.agent import Agent
from misc.pareto import ParetoArchive
from misc.evaluation import log_metrics
from misc.utils import setup_directories
from misc.weights import generate_das_dennis_weights, generate_layer_energy_weights, generate_dirichlet_weights

from agents.single_policy.sac_utils_nl.sac_worker_nl import SACWorkerPool, isolated_rng, learner_seed


def available_cpus() -> int:
    """Return the smaller of the Slurm CPU allocation and the CPU affinity mask."""
    counts = [len(os.sched_getaffinity(0))]
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        counts.append(int(os.environ['SLURM_CPUS_PER_TASK']))
    return max(1, min(counts))


class MOSACNL(Agent):
    def __init__(
            self,
            env_id: str,
            env: gym.Env,
            num_subproblems: int = 6,
            replicas_per_weight: int = 1,
            num_steps: int = 512,
            archive_size: Optional[int] = None,
            gamma: float = 0.99,
            actor_lr: float = 3e-4,
            critic_lr: float = 1e-3,
            tau: float = 0.005,
            alpha: float = 0.2,
            actor_net_arch: Optional[list[int]] = None,
            critic_net_arch: Optional[list[int]] = None,
            buffer_size: int = 1_000_000,
            batch_size: int = 256,
            learning_starts: int = 1000,
            gradient_updates: int = 1,
            policy_freq: int = 1,
            target_net_freq: int = 1,
            clip_grad_norm: bool = True,
            actor_clip_norm: float = 2.0,
            critic_clip_norm: float = 5.0,
            init_w_sampling: str = 'uniform',
            dirichlet_alpha: float = 1.0,
            scalarization_method: str = 'ws_raw',
            log: bool = True,
            seed: int = 42,
            max_episode_steps: int = 500,
            name: str = 'mo_sac_nl',
            max_workers: Optional[int] = None,
            worker_timeout: float = 3600.0,
    ):
        """Multi-objective SAC with a population of P independent learners on K = P / R distinct weights."""
        super().__init__(env, device='cpu', seed=seed, name=name)

        # --- Environment & seeding ---
        self.env_id = env_id
        self._env_spec = copy.deepcopy(env.spec)
        self.seed = seed

        # --- Action space ---
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.sac_implementation = 'SACDiscreteNL' if discrete else 'SACContinuousNL'

        # --- Population parameters ---
        self.num_subproblems = int(num_subproblems)
        self.replicas_per_weight = int(replicas_per_weight)
        self.population_size = self.num_subproblems
        self.num_distinct_weights = self.population_size // self.replicas_per_weight
        self.archive_size = archive_size
        self.max_workers = max_workers
        self.worker_timeout = worker_timeout

        # --- Rollout & replay parameters ---
        self.num_steps = num_steps
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gradient_updates = gradient_updates
        self.max_episode_steps = max_episode_steps

        # --- SAC network / optimizer ---
        self.actor_net_arch = list(actor_net_arch or [256, 256])
        self.critic_net_arch = list(critic_net_arch or [256, 256])
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.alpha = alpha
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq
        self.clip_grad_norm = clip_grad_norm
        self.actor_clip_norm = actor_clip_norm
        self.critic_clip_norm = critic_clip_norm

        # --- Weights & scalarization ---
        self.init_w_sampling = init_w_sampling
        self.dirichlet_alpha = dirichlet_alpha
        method = scalarization_method.lower()
        self.scalarization_method = 'ws_raw' if method == 'ws' else method
        self.objective_scale = None
        self.log = log

        # --- Global counters ---
        self.global_iterations = 0
        self.global_timesteps = 0
        self.global_step = 0

        self.archive = ParetoArchive(max_size=archive_size)

        # --- Weight population ---
        with isolated_rng():
            np.random.seed(seed % 2 ** 32)
            self.distinct_weights = self._sample_weights(num_weights=self.num_distinct_weights)
        self.initial_weights, self.agent_weight_ids, self.agent_replica_ids = self._expand_weight_replicas(
            self.distinct_weights)
        self.agent_ids = np.arange(self.population_size, dtype=np.int64)
        self.agent_env_seeds = np.asarray([learner_seed(seed, i) for i in self.agent_ids], dtype=np.int64)
        print(f'[INFO] Initialised weight population: {self.num_distinct_weights} distinct weights x '
              f'{self.replicas_per_weight} replicas = {self.population_size} policies.')

        # --- Run state ---
        self.agents = []
        self.total_iterations = None
        self.eval_iterations = None
        self.eval_rep = None
        self.eval_seed = None
        self.eval_gamma = None
        self.eval_deterministic = None

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

    def _learner_kwargs(self) -> dict:
        """Return the hyperparameters forwarded to every SAC learner."""
        return {
            'actor_net_arch': list(self.actor_net_arch),
            'critic_net_arch': list(self.critic_net_arch),
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'learning_starts': self.learning_starts,
            'policy_freq': self.policy_freq,
            'target_net_freq': self.target_net_freq,
            'clip_grad_norm': self.clip_grad_norm,
            'actor_clip_norm': self.actor_clip_norm,
            'critic_clip_norm': self.critic_clip_norm,
        }

    def _worker_config(self) -> dict:
        """Build the shared worker configuration."""
        return {
            'env_id': self.env_id,
            'obs_shape': self.obs_shape,
            'reward_dim': self.reward_dim,
            'max_episode_steps': self.max_episode_steps,
            'learner_kwargs': self._learner_kwargs(),
            'buffer_size': self.buffer_size,
            'gradient_updates': self.gradient_updates,
            'objective_scale': self.objective_scale,
        }

    def _worker_count(self, population: int) -> int:
        """Return the number of worker processes for a population."""
        return min(population, available_cpus(), self.max_workers or available_cpus())

    @override
    def eval(self, obs: Union[np.ndarray, torch.Tensor], w: Union[np.ndarray, torch.Tensor]) -> Union[
        np.ndarray, torch.Tensor]:
        """Not available: a population has no single default policy."""
        raise NotImplementedError('Evaluate a selected SAC policy snapshot; a population has no single default policy.')

    # -------------------- PERSISTENCE --------------------

    @override
    def load(self, path: str, file_name: str, load_replay_buffer: bool = True) -> None:
        """Not implemented: population training cannot be resumed."""
        raise NotImplementedError('Population training resume is not implemented. '
                                  'Use the snapshot classes\' load() for deployment.')

    @override
    def save(self, path: str, file_name: str, save_replay_buffer: bool = False) -> None:
        """Save every snapshot of the archive."""
        for i, snapshot in enumerate(self.archive.individuals):
            snapshot.save(path, f'{file_name}_{i}')
        print(f'[INFO] All MO-SAC-NL archive snapshots saved to {path}.')

    @override
    def save_config(self, path: str, file_name: str) -> None:
        """Write hyperparameters, weight population and reference points as JSON."""

        def to_list(value: Optional[Any]) -> Optional[list]:
            return None if value is None else np.asarray(value).tolist()

        total = self.total_iterations
        every = self.eval_iterations
        config = {
            'num_subproblems': self.num_subproblems,
            'replicas_per_weight': self.replicas_per_weight,
            'num_distinct_weights': self.num_distinct_weights,
            'population_size': self.population_size,
            'distinct_weights': self.distinct_weights.tolist(),
            'initial_weights': self.initial_weights.tolist(),
            'agent_ids': self.agent_ids.tolist(),
            'agent_weight_ids': self.agent_weight_ids.tolist(),
            'agent_replica_ids': self.agent_replica_ids.tolist(),
            'agent_env_seeds': self.agent_env_seeds.tolist(),
            'worker_processes': self._worker_count(self.population_size),
            'archive_size': self.archive_size,
            'num_steps': self.num_steps,
            'gamma': self.gamma,
            'buffer_size': self.buffer_size,
            'total_replay_capacity': self.population_size * self.buffer_size,
            'batch_size': self.batch_size,
            'learning_starts': self.learning_starts,
            'gradient_updates': self.gradient_updates,
            'max_episode_steps': self.max_episode_steps,
            'actor_net_arch': self.actor_net_arch,
            'critic_net_arch': self.critic_net_arch,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'tau': self.tau,
            'alpha': self.alpha,
            'policy_freq': self.policy_freq,
            'target_net_freq': self.target_net_freq,
            'clip_grad_norm': self.clip_grad_norm,
            'actor_clip_norm': self.actor_clip_norm,
            'critic_clip_norm': self.critic_clip_norm,
            'sac_implementation': self.sac_implementation,
            'total_iterations': total,
            'eval_iterations': every,
            'total_population_env_transitions': (None if total is None
                                                 else total * self.population_size * self.num_steps),
            'total_env_transitions_per_policy': None if total is None else total * self.num_steps,
            'evaluation_interval_per_policy': None if every is None else every * self.num_steps,
            'eval_rep': self.eval_rep,
            'eval_seed': self.eval_seed,
            'eval_gamma': self.eval_gamma,
            'eval_deterministic': self.eval_deterministic,
            'ideal': to_list(self.ideal),
            'nadir': to_list(self.nadir),
            'init_w_sampling': self.init_w_sampling,
            'dirichlet_alpha': self.dirichlet_alpha,
            'scalarization_method': self.scalarization_method,
            'env_id': self.env_id,
            'seed': self.seed,
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{file_name}.json'), 'w') as f:
            json.dump(config, f, indent=4)
        print('[INFO] MO-SAC-NL configuration saved')

    # -------------------- TRAINING HELPERS --------------------

    def _collect_samples_and_stats(self, results: dict) -> tuple[list, np.ndarray, dict]:
        """Unpickle the evaluated snapshots and gather their returns and training metrics."""
        snapshots = [pickle.loads(results[i]['snapshot']) for i in self.agent_ids]
        points = np.stack([results[i]['returns'] for i in self.agent_ids])
        metrics = {i: results[i]['metrics'] for i in self.agent_ids}
        return snapshots, points, metrics

    def _update_archive(self, snapshots: list, points: np.ndarray) -> None:
        """Add every evaluated snapshot with its return vector to the Pareto archive."""
        for snapshot, point in zip(snapshots, points):
            self.archive.add(snapshot, point.copy())

    def _archive_front(self) -> np.ndarray:
        """Return a copy of the archive's objective vectors as an array."""
        return np.array(self.archive.evaluations, dtype=np.float64).reshape(-1, self.reward_dim)

    # -------------------- LOGGING --------------------

    def _log_progress(self, writer: SummaryWriter) -> None:
        """Log the global iteration against the global timestep axis."""
        writer.add_scalar('progress/global_iteration', self.global_iterations, self.global_step)

    def _log_entropy(self, writer: SummaryWriter, metrics: dict, global_step: int) -> None:
        """Log mean, per-agent and corner/interior policy entropy."""
        valid = {i: row['policy_entropy'] for i, row in metrics.items()
                 if np.isfinite(row.get('policy_entropy', np.nan))}
        if not valid:
            return
        normalised = {i: metrics[i]['policy_entropy_norm'] for i in valid
                      if np.isfinite(metrics[i].get('policy_entropy_norm', np.nan))}

        writer.add_scalar('entropy/mean', float(np.mean(list(valid.values()))), global_step)
        if normalised:
            writer.add_scalar('entropy/mean_normalised', float(np.mean(list(normalised.values()))), global_step)

        # Per-agent curves
        for agent_idx, e in valid.items():
            writer.add_scalar(f'entropy/agent_{agent_idx:02d}', e, global_step)
            if agent_idx in normalised:
                writer.add_scalar(f'entropy/agent_{agent_idx:02d}_normalised', normalised[agent_idx], global_step)

        # Corner (max(w) > 0.99) vs interior split
        corner_e = []
        interior_e = []
        for agent_idx, e in valid.items():
            (corner_e if max(self.initial_weights[agent_idx]) > 0.99 else interior_e).append(e)
        if corner_e:
            writer.add_scalar('entropy/corner_agents_mean', float(np.mean(corner_e)), global_step)
        if interior_e:
            writer.add_scalar('entropy/interior_agents_mean', float(np.mean(interior_e)), global_step)

    def _log_train_stats(self, writer: SummaryWriter, metrics: dict, global_step: int) -> None:
        """Log the population mean of every SAC health metric."""
        keys = {key for row in metrics.values() for key in row} - {'policy_entropy', 'policy_entropy_norm'}
        for key in sorted(keys):
            values = [row[key] for row in metrics.values() if key in row and np.isfinite(row[key])]
            if values:
                writer.add_scalar(f'train/{key}_mean', float(np.mean(values)), global_step)

    def _eval_and_log_fronts(self,
                             points: np.ndarray,
                             ref_point: np.ndarray,
                             known_pareto_front: Optional[list[np.ndarray]],
                             num_eval_weights: int,
                             save_fronts: bool,
                             pf_store: Optional[Any],
                             policy_store: Optional[Any],
                             writer: SummaryWriter,
                             global_step: int) -> None:
        """Log metrics of the archive front and the current policy front and store both snapshots."""
        archive_front = self._archive_front()
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
            archive_front,
            store=pf_store,
            tag_prefix='archive',
            **common,
        )

        # Current-policy front (one point per policy)
        hv_policy = log_metrics(
            points,
            store=policy_store,
            tag_prefix='policy',
            weights=self.initial_weights.copy(),
            agent_ids=self.agent_ids.copy(),
            weight_ids=self.agent_weight_ids.copy(),
            replica_ids=self.agent_replica_ids.copy(),
            **common,
        )

        print(f'HV archive @ iteration {self.global_iterations}, step {global_step} '
              f'({len(archive_front)}): {round(hv_archive, 2)}')
        print(f'HV policy  @ iteration {self.global_iterations}, step {global_step} '
              f'({len(points)}): {round(hv_policy, 2)}\n')

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
            eval_gamma: Optional[float] = None,
            eval_deterministic: bool = True,
            save_fronts: bool = False,
            save_models: bool = False,
            log_dir: str = 'mo_sac_nl',
            file_name: str = 'mo_sac_nl',
    ) -> list:
        """Train all SAC learners in persistent workers and evaluate the fronts every eval_iterations."""
        # ── Directories and stores ────────────────────────────────────────────
        runs_path, model_path, config_path, pf_store, policy_store = setup_directories(
            log_dir, file_name, save_fronts, save_models)
        writer = SummaryWriter(log_dir=runs_path, filename_suffix=f'{self.seed:04d}')
        fronts = []

        # ── Scalarization references ──────────────────────────────────────────
        if self.scalarization_method == 'ws_raw':
            print('[INFO] Raw WS does not require reference points.')
        else:
            self.ideal = np.asarray(ideal, dtype=np.float64)
            self.nadir = np.asarray(nadir, dtype=np.float64)
            self.objective_scale = self.ideal - self.nadir
            print(f'[INFO] Using provided ideal={self.ideal.round(4)}, nadir={self.nadir.round(4)}.')

        # ── Run settings ──────────────────────────────────────────────────────
        ref_point = np.asarray(ref_point, dtype=np.float64)
        eval_gamma = self.gamma if eval_gamma is None else eval_gamma
        self.total_iterations = total_iterations
        self.eval_iterations = eval_iterations
        self.eval_rep = eval_rep
        self.eval_seed = eval_seed
        self.eval_gamma = eval_gamma
        self.eval_deterministic = eval_deterministic
        self.save_config(config_path, file_name)

        # ── Iteration bookkeeping ─────────────────────────────────────────────
        timesteps_per_iteration = self.population_size * self.num_steps
        current_iteration = 0
        self.global_iterations = 0
        self.global_timesteps = 0
        self.global_step = 0

        specs = [{'id': i, 'seed': self.agent_env_seeds[i], 'weight': self.initial_weights[i]}
                 for i in self.agent_ids]
        workers_count = self._worker_count(self.population_size)
        print(f'[INFO] Using {workers_count} persistent worker processes for {self.population_size} independent '
              f'{self.sac_implementation} learners (K={self.num_distinct_weights}, R={self.replicas_per_weight}); '
              f'{self.num_steps} actual transitions/policy/iteration.')

        # ── Persistent workers for the entire training run ────────────────────
        with SACWorkerPool(specs, self._env_spec, self._worker_config(), workers_count,
                           self.worker_timeout) as workers:
            # Initial evaluation (iteration 0, no training)
            results = workers.evaluate(eval_rep, eval_seed, eval_gamma, eval_deterministic)
            self.agents, points, metrics = self._collect_samples_and_stats(results)
            self._update_archive(self.agents, points)

            self._log_entropy(writer, metrics, self.global_step)
            self._log_train_stats(writer, metrics, self.global_step)
            self._log_progress(writer)
            self._eval_and_log_fronts(points, ref_point, known_pareto_front,
                                      num_eval_weights, save_fronts, pf_store,
                                      policy_store, writer, self.global_step)
            fronts.append(self._archive_front())

            # ── Main training loop ────────────────────────────────────────────
            while current_iteration < total_iterations:
                end_iteration = min(current_iteration + eval_iterations, total_iterations)
                workers.advance((end_iteration - current_iteration) * self.num_steps)

                current_iteration = end_iteration
                self.global_iterations = current_iteration
                self.global_timesteps = self.global_iterations * timesteps_per_iteration
                self.global_step = self.global_timesteps

                # Eval checkpoint
                results = workers.evaluate(eval_rep, eval_seed, eval_gamma, eval_deterministic)
                self.agents, points, metrics = self._collect_samples_and_stats(results)
                self._update_archive(self.agents, points)

                self._log_entropy(writer, metrics, self.global_step)
                self._log_train_stats(writer, metrics, self.global_step)
                self._log_progress(writer)
                self._eval_and_log_fronts(points, ref_point, known_pareto_front,
                                          num_eval_weights, save_fronts, pf_store,
                                          policy_store, writer, self.global_step)
                fronts.append(self._archive_front())

        # ── Cleanup ───────────────────────────────────────────────────────────
        if save_models:
            self.save(model_path, file_name)
        self.env.close()
        writer.close()

        return fronts