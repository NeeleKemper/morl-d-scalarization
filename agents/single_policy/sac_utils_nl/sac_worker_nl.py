import copy
import time
import torch
import pickle
import random

import numpy as np
import gymnasium as gym
import multiprocessing as mp

from types import TracebackType
from contextlib import contextmanager
from typing import Union, Optional, Any, Iterator
from multiprocessing.connection import Connection
from gymnasium.envs.registration import EnvSpec

from misc.replay_buffer import ReplayBuffer
from agents.single_policy.sac_utils_nl.sac_discrete_action_nl import SACDiscreteNL
from agents.single_policy.sac_utils_nl.sac_continues_action_nl import SACContinuousNL


def learner_seed(population_seed: int, agent_id: int) -> int:
    """Return a learner seed that is stable across process counts."""
    sequence = np.random.SeedSequence([population_seed, agent_id, 0, 0x534143])
    return sequence.generate_state(1, dtype=np.uint32)[0].item()


# -------------------- RNG ISOLATION --------------------

def _rng_state() -> tuple:
    """Return the Python, NumPy and PyTorch default RNG states."""
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _set_rng_state(state: tuple) -> None:
    """Restore the Python, NumPy and PyTorch default RNG states."""
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


@contextmanager
def isolated_rng() -> Iterator[None]:
    """Restore the Python, NumPy and PyTorch default RNG states after the block."""
    state = _rng_state()
    yield
    _set_rng_state(state)


# -------------------- HELPERS --------------------

def learner_class_for(space: gym.Space) -> tuple[type, int, int]:
    """Return learner class, action dimension and replay action width for a Discrete or Box action space."""
    if isinstance(space, gym.spaces.Discrete):
        return SACDiscreteNL, int(space.n), 1
    dim = int(np.prod(space.shape))
    return SACContinuousNL, dim, dim


def make_env(env_spec: EnvSpec, max_episode_steps: int) -> gym.Env:
    """Create an environment from a serialized EnvSpec."""
    return gym.make(copy.deepcopy(env_spec), max_episode_steps=max_episode_steps, disable_env_checker=True)


def evaluate_actor(agent: Union[SACDiscreteNL, SACContinuousNL],
                   env: gym.Env,
                   eval_rep: int,
                   eval_seed: int,
                   eval_gamma: float,
                   deterministic: bool,
                   max_episode_steps: int) -> dict:
    """Return the mean discounted return over eval_rep seeded episodes without changing the default RNG."""
    returns = np.zeros((eval_rep, agent.reward_dim), dtype=np.float64)
    was_training = agent.actor.training
    agent.actor.eval()
    with isolated_rng(), torch.no_grad():
        for episode in range(eval_rep):
            seed = eval_seed + episode
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            obs, _ = env.reset(seed=seed)
            factor = 1.0
            for _ in range(max_episode_steps):
                obs, reward, terminated, truncated, _ = env.step(agent.eval(obs, deterministic=deterministic))
                returns[episode] += factor * np.asarray(reward, dtype=np.float64)
                factor *= eval_gamma
                if terminated or truncated:
                    break
    agent.actor.train(was_training)
    return {'returns': returns.mean(axis=0)}


# -------------------- LEARNER --------------------

class _LearnerState:
    def __init__(self, spec: dict, env_spec: EnvSpec, config: dict):
        """One learner inside a worker process with its own environment, replay buffer and RNG state."""
        self.agent_id = int(spec['id'])
        self.seed = int(spec['seed'])
        self.env_id = config['env_id']
        self.max_episode_steps = config['max_episode_steps']
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # --- Environment, learner & replay buffer ---
        self.env = make_env(env_spec, self.max_episode_steps)
        learner_class, action_dim, storage_dim = learner_class_for(self.env.action_space)
        self.agent = learner_class(
            id=self.agent_id,
            env=self.env,
            obs_shape=config['obs_shape'],
            action_dim=action_dim,
            reward_dim=config['reward_dim'],
            seed=self.seed,
            objective_scale=config['objective_scale'],
            **config['learner_kwargs'],
        )
        self.agent.set_weights(spec['weight'])
        self.replay_buffer = ReplayBuffer(tuple(config['obs_shape']), storage_dim, rew_dim=config['reward_dim'],
                                          max_size=config['buffer_size'])
        self.gradient_updates = config['gradient_updates']

        # --- Evaluation & metrics ---
        self.last_metrics = {}
        with isolated_rng():
            self.eval_env = make_env(env_spec, self.max_episode_steps)
        self.rng_state = _rng_state()

    def advance(self, transitions: int) -> dict:
        """Collect the given number of transitions with updates and store the finite round-mean metrics."""
        _set_rng_state(self.rng_state)
        sums = {}
        counts = {}
        for _ in range(transitions):
            self.agent.collect_sample(self.replay_buffer)
            if self.agent.global_step < self.agent.learning_starts or self.replay_buffer.size < self.agent.batch_size:
                continue
            for _ in range(self.gradient_updates):
                batch = self.replay_buffer.sample(self.agent.batch_size, to_tensor=True, device='cpu')
                info = self.agent.update(batch)
                for key, value in (info or {}).items():
                    if np.isfinite(value):
                        sums[key] = sums.get(key, 0.0) + value
                        counts[key] = counts.get(key, 0) + 1
        self.last_metrics = {key: sums[key] / counts[key] for key in sums}
        self.rng_state = _rng_state()
        return {'id': self.agent_id, 'global_step': self.agent.global_step}

    def evaluate(self, request: dict) -> dict:
        """Evaluate the learner and return its returns, training metrics and a pickled inference snapshot."""
        _set_rng_state(self.rng_state)
        result = evaluate_actor(self.agent, self.eval_env, eval_rep=request['eval_rep'],
                                eval_seed=request['eval_seed'], eval_gamma=request['eval_gamma'],
                                deterministic=request['deterministic'], max_episode_steps=self.max_episode_steps)
        self.rng_state = _rng_state()
        result.update(id=self.agent_id, global_step=self.agent.global_step, metrics=self.last_metrics.copy())
        snapshot = self.agent.snapshot(self.env_id, request['deterministic'], result['returns'])
        result['snapshot'] = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    def close(self) -> None:
        """Close the training and evaluation environments."""
        self.env.close()
        self.eval_env.close()


# -------------------- WORKER PROCESS --------------------

def _send(connection: Connection, message: Any) -> None:
    """Send a pickled message through the pipe."""
    connection.send_bytes(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))


def _worker_main(connection: Connection,
                 specs: list[dict],
                 env_spec: EnvSpec,
                 config: dict) -> None:
    """Run the learners of one worker process and answer advance/evaluate commands until close."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    states = [_LearnerState(spec, env_spec, config) for spec in specs]
    _send(connection, [state.agent_id for state in states])
    while True:
        command, payload = pickle.loads(connection.recv_bytes())
        if command == 'close':
            break
        if command == 'advance':
            _send(connection, [state.advance(payload['transitions']) for state in states])
        else:
            _send(connection, [state.evaluate(payload) for state in states])
    for state in states:
        state.close()


# -------------------- POOL --------------------

class SACWorkerPool:
    def __init__(self,
                 specs: list[dict],
                 env_spec: EnvSpec,
                 config: dict,
                 num_workers: int,
                 timeout: float = 3600.0):
        """Start persistent spawn workers that each own a fixed shard of learners."""
        self.processes = []
        self.connections = []
        self.timeout = timeout
        context = mp.get_context('spawn')
        for group in np.array_split(np.arange(len(specs)), min(num_workers, len(specs))):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(target=_worker_main, daemon=True,
                                      args=(child, [specs[i] for i in group], env_spec, config))
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        for index in range(len(self.processes)):
            self._receive(index)

    def _receive(self, index: int) -> list:
        """Wait for the answer of one worker; raise if it dies or exceeds the timeout."""
        connection = self.connections[index]
        process = self.processes[index]
        deadline = time.monotonic() + self.timeout
        while not connection.poll(0.1):
            if not process.is_alive():
                raise RuntimeError(f'SAC worker exited unexpectedly (exitcode={process.exitcode}).')
            if time.monotonic() >= deadline:
                raise TimeoutError('SAC worker timed out; the run is incomplete and must not be '
                                   'aggregated as complete.')
        return pickle.loads(connection.recv_bytes())

    def _broadcast(self, command: str, payload: dict) -> dict:
        """Send a command to all workers and return their results by learner ID."""
        for connection in self.connections:
            _send(connection, (command, payload))
        return {result['id']: result for index in range(len(self.processes)) for result in self._receive(index)}

    def advance(self, transitions: int) -> dict:
        """Advance every policy by the given number of environment transitions."""
        return self._broadcast('advance', {'transitions': transitions})

    def evaluate(self,
                 eval_rep: int,
                 eval_seed: int,
                 eval_gamma: float,
                 deterministic: bool = True) -> dict:
        """Evaluate every policy and return the results with snapshots by learner ID."""
        return self._broadcast('evaluate', {'eval_rep': eval_rep, 'eval_seed': eval_seed, 'eval_gamma': eval_gamma,
                                            'deterministic': deterministic})

    def close(self) -> None:
        """Stop all workers, terminating those that do not exit in time."""
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                _send(connection, ('close', {}))
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for connection in self.connections:
            connection.close()

    def __enter__(self) -> 'SACWorkerPool':
        """Return the pool for use in a with block."""
        return self

    def __exit__(self,
                 exc_type: Optional[type],
                 exc_value: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        """Close the pool when the with block ends."""
        self.close()