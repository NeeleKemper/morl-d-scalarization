import os

# Thread limits must be set before NumPy/PyTorch are imported
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import torch
import argparse

import numpy as np
import mo_gymnasium as mo_gym

from agents.multi_policy.mo_ppo_nl import MOPPONL
from agents.utils.scalarizer import calibrate_stch_raw_mu
from misc.utils import read_algo_config, read_env_config, seed_everything


def float_token(value: float) -> str:
    """Format a float for use in a directory name (6 significant digits, '-' -> 'm', '.' -> 'p')."""
    return f'{value:.6g}'.replace('-', 'm').replace('.', 'p')


def main() -> None:
    """Parse the arguments, build MO-PPO-NL from the JSON configs and run one training."""
    torch.set_default_dtype(torch.float64)

    # --- Arguments ---
    parser = argparse.ArgumentParser(description='Run the reduced DST/LDST learning-rate and entropy axis-sweep '
                                                 'experiment.')
    parser.add_argument('--env', choices=['dst1', 'dst2', 'dst3', 'dst4', 'ldst1', 'ldst2', 'minecart', 'reacher'],
                        default='dst1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_subproblems', type=int, default=None,
                        help='Total number P of independently trained policies.')
    parser.add_argument('--replicas_per_weight', type=int, default=1,
                        help='Number R of policies per distinct weight; K=P/R.')
    parser.add_argument('--scalarization_method', default='ws_raw',
                        choices=('ws_raw', 'ws_norm', 'tch_raw', 'tch_norm', 'stch_raw', 'stch_norm'))
    parser.add_argument('--init_w_sampling', choices=('das_dennis', 'uniform', 'dirichlet'), default='das_dennis')
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--entropy_coef', type=float, default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--num_processes', type=int, default=None)
    parser.add_argument('--max_episode_steps', type=int, default=None)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--eval_rep', type=int, default=1)
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
    if args.env in ('minecart', 'reacher'):
        config_name = args.env
    else:
        config_name = 'ldst' if args.env.startswith('ldst') else 'dst'
    config = read_algo_config(f'configs/mo_ppo_nl_{config_name}.json')

    num_subproblems = int(config['num_subproblems']) if args.num_subproblems is None else args.num_subproblems
    num_processes = int(config['num_processes']) if args.num_processes is None else args.num_processes
    num_steps = int(config['num_steps']) if args.num_steps is None else args.num_steps
    gamma = float(config['gamma']) if args.gamma is None else args.gamma
    learning_rate = float(config['learning_rate']) if args.learning_rate is None else args.learning_rate
    entropy_coef = float(config['entropy_coef']) if args.entropy_coef is None else args.entropy_coef
    max_episode_steps = int(config['max_episode_steps']) if args.max_episode_steps is None else args.max_episode_steps
    num_distinct_weights = num_subproblems // args.replicas_per_weight

    # --- Environment & reference points ---
    if args.env in ('minecart', 'reacher'):
        env = mo_gym.make(env_id, max_episode_steps=max_episode_steps)
        known_pf = None
        ideal = np.asarray(env_config[args.env]['ideal'], dtype=np.float64)
        nadir = np.asarray(env_config[args.env]['nadir'], dtype=np.float64)
    else:
        from envs import dst_envs
        dst_envs.register_envs(max_episode_steps=max_episode_steps)
        env = mo_gym.make(env_id)
        known_pf = np.asarray(env.unwrapped.pareto_front(gamma=gamma), dtype=np.float64)
        ideal = known_pf.max(axis=0)
        nadir = known_pf.min(axis=0)

    # --- Agent ---
    if args.scalarization_method == 'stch_raw':
        mu = calibrate_stch_raw_mu(mu=args.mu, objective_ranges=ideal - nadir, num_weights=config['num_subproblems'],
                                   reward_dim=env_config[args.env]['num_obj'])
    else:
        mu = args.mu

    agent = MOPPONL(
        env_id=env_id,
        tmp_env=env,
        num_subproblems=num_subproblems,
        replicas_per_weight=args.replicas_per_weight,
        init_w_sampling=args.init_w_sampling,
        dirichlet_alpha=args.dirichlet_alpha,
        num_processes=num_processes,
        archive_size=config['archive_size'],
        num_steps=num_steps,
        gamma=gamma,
        obj_rms=config['obj_rms'],
        ob_rms=config['ob_rms'],
        use_proper_time_limits=config['use_proper_time_limits'],
        net_arch=config['net_arch'],
        layernorm=config['layernorm'],
        learning_rate=learning_rate,
        eps=args.eps,
        clip_param=config['clip_param'],
        ppo_epoch=config['ppo_epoch'],
        num_mini_batches=config['num_mini_batches'],
        entropy_coef=entropy_coef,
        value_loss_coef=config['value_loss_coef'],
        max_grad_norm=config['max_grad_norm'],
        use_clipped_value_loss=config['use_clipped_value_loss'],
        use_linear_lr_decay=config['use_linear_lr_decay'],
        lr_decay_ratio=config['lr_decay_ratio'],
        use_gae=config['use_gae'],
        gae_lambda=config['gae_lambda'],
        scalarization_method=args.scalarization_method,
        mu=mu,
        log=True,
        seed=args.seed,
        max_episode_steps=max_episode_steps,
        device='cpu',
        name='mo_ppo_nl_baselines',
    )

    # --- Training ---
    scalarizer_path = args.scalarization_method
    if args.scalarization_method == 'stch_raw':
        scalarizer_path += f'/mu_{float_token(args.mu)}'
    log_dir = (f'{args.path_prefix}/{agent.name}/{args.env}/gamma_{float_token(gamma)}/'
               f'{args.init_w_sampling}/P_{num_subproblems:04d}/'
               f'K_{num_distinct_weights:04d}/R_{args.replicas_per_weight:04d}/'
               f'H_{config["total_iterations"]:05d}/eval_{config["eval_iterations"]:04d}/'
               f'lr_{float_token(learning_rate)}/ec_{float_token(entropy_coef)}/{scalarizer_path}/s_{args.seed:04d}')

    agent.train(
        total_iterations=config['total_iterations'],
        eval_iterations=config['eval_iterations'],
        ref_point=ref_point,
        known_pareto_front=known_pf,
        ideal=ideal,
        nadir=nadir,
        num_eval_weights=args.num_eval_weights,
        eval_rep=args.eval_rep,
        eval_seed=args.eval_seed,
        eval_gamma=gamma,
        save_fronts=args.save_fronts,
        save_models=args.save_models,
        log_dir=log_dir,
        file_name='mo_ppo_nl',
    )


if __name__ == '__main__':
    main()
