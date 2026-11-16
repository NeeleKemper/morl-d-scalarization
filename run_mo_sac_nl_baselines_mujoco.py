import os

# Thread limits must be set before NumPy/PyTorch are imported; the spawned SAC workers inherit them
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse

import numpy as np
import mo_gymnasium as mo_gym

from agents.multi_policy.mo_sac_nl import MOSACNL
from misc.utils import read_algo_config, read_env_config, seed_everything


def float_token(value: float) -> str:
    """Format a float for use in a directory name (6 significant digits, '-' -> 'm', '.' -> 'p')."""
    return f'{value:.6g}'.replace('-', 'm').replace('.', 'p')


def main() -> None:
    """Parse the arguments, build MO-SAC-NL for a MuJoCo task and run one training with frozen ideal/nadir."""
    # --- Arguments ---
    parser = argparse.ArgumentParser(description='Run MO-SAC-NL on the MuJoCo tasks.')
    parser.add_argument('--env', choices=('ant', 'hopper', 'halfcheetah', 'humanoid'), default='halfcheetah')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_subproblems', type=int, default=None,
                        help='Total number P of independently trained policies.')
    parser.add_argument('--replicas_per_weight', type=int, default=1,
                        help='Number R of policies per distinct weight; K=P/R.')
    parser.add_argument('--scalarization_method', choices=('ws_raw', 'ws_norm'), default='ws_raw')
    parser.add_argument('--init_w_sampling', choices=('das_dennis', 'uniform', 'dirichlet'), default='das_dennis')
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None, help='Fixed entropy temperature; never autotuned.')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Actual transitions per policy and iteration; for a PPO rollout of L steps in N vector '
                             'environments pass L*N.')
    parser.add_argument('--max_workers', type=int, default=None)
    parser.add_argument('--max_episode_steps', type=int, default=None)
    parser.add_argument('--worker_timeout', type=float, default=3600.0,
                        help='Seconds one training chunk (eval_iteration iterations) may take before the run aborts.')
    parser.add_argument('--eval_rep', type=int, default=5)
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--num_eval_weights', type=int, default=100)
    parser.add_argument('--save_fronts', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_models', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--path_prefix', type=str, default='scalarization/baselines')
    args = parser.parse_args()
    seed_everything(args.seed)

    # --- Configs (command-line values override the JSON algorithm config) ---
    env_config = read_env_config('configs/environment_configs.json')
    env_id = env_config[args.env]['env_id']
    ref_point = np.asarray(env_config[args.env]['ref_point'], dtype=np.float64)
    config = read_algo_config('configs/mo_sac_nl_mujoco.json')

    num_subproblems = int(config['num_subproblems']) if args.num_subproblems is None else args.num_subproblems
    num_steps = int(config['num_steps']) if args.num_steps is None else args.num_steps
    gamma = float(config['gamma']) if args.gamma is None else args.gamma
    alpha = float(config['alpha']) if args.alpha is None else args.alpha
    max_episode_steps = int(config['max_episode_steps']) if args.max_episode_steps is None else args.max_episode_steps
    num_distinct_weights = num_subproblems // args.replicas_per_weight

    # --- Environment & frozen reference points (only ws_norm uses them) ---
    env = mo_gym.make(env_id, max_episode_steps=max_episode_steps)
    ideal = None
    nadir = None
    if args.scalarization_method == 'ws_norm':
        ideal = np.asarray(env_config[args.env]['ideal'], dtype=np.float64)
        nadir = np.asarray(env_config[args.env]['nadir'], dtype=np.float64)

    # --- Agent ---
    agent = MOSACNL(
        env_id=env_id,
        env=env,
        num_subproblems=num_subproblems,
        replicas_per_weight=args.replicas_per_weight,
        num_steps=num_steps,
        archive_size=config['archive_size'],
        gamma=gamma,
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        tau=config['tau'],
        alpha=alpha,
        actor_net_arch=config['actor_net_arch'],
        critic_net_arch=config['critic_net_arch'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        learning_starts=config['learning_starts'],
        gradient_updates=config['gradient_updates'],
        policy_freq=config['policy_freq'],
        target_net_freq=config['target_net_freq'],
        clip_grad_norm=config['clip_grad_norm'],
        actor_clip_norm=config['actor_clip_norm'],
        critic_clip_norm=config['critic_clip_norm'],
        init_w_sampling=args.init_w_sampling,
        dirichlet_alpha=args.dirichlet_alpha,
        scalarization_method=args.scalarization_method,
        log=True,
        seed=args.seed,
        max_episode_steps=max_episode_steps,
        max_workers=args.max_workers,
        worker_timeout=args.worker_timeout,
        name='mo_sac_nl_baselines'
    )

    # --- Training ---
    log_dir = (f'{args.path_prefix}/{agent.name}/{args.env}/gamma_{float_token(gamma)}/'
               f'{args.init_w_sampling}/P_{num_subproblems:04d}/'
               f'K_{num_distinct_weights:04d}/R_{args.replicas_per_weight:04d}/'
               f'H_{config["total_iterations"]:05d}/eval_{config["eval_iterations"]:04d}/'
               f'lr_{float_token(config["actor_lr"])}/alpha_{float_token(alpha)}/{args.scalarization_method}/'
               f's_{args.seed:04d}')

    agent.train(
        total_iterations=config['total_iterations'],
        eval_iterations=config['eval_iterations'],
        ref_point=ref_point,
        known_pareto_front=None,
        ideal=ideal,
        nadir=nadir,
        num_eval_weights=args.num_eval_weights,
        eval_rep=args.eval_rep,
        eval_seed=args.eval_seed,
        eval_gamma=gamma,
        eval_deterministic=True,
        save_fronts=args.save_fronts,
        save_models=args.save_models,
        log_dir=log_dir,
        file_name='mo_sac_nl',
    )


if __name__ == '__main__':
    main()
