# Targeted, Discovered, Retained: Separating the Effects of Scalarization, Archiving, and Execution in Multi-Policy Multi-Objective Reinforcement Learning

**Neele Kemper, Helena Stegherr, Jonathan Wurth, Michael Heider, Jörg Hähner**  
*Submitted to the Journal of Artificial Intelligence Research (JAIR)*  
[Paper](#)

---

## Abstract

**Background:** Decomposition-based multi-objective reinforcement learning trains one policy per preference by scalarizing vector-valued returns. Weighted sums can only target supported Pareto returns on the convex hull of the front, whereas nonlinear scalarizers such as the Tchebycheff function can also target unsupported returns in its concave regions. However, training also produces Pareto-optimal returns that no preference targets, and a checkpoint archive can keep them. It is therefore unclear what nonlinear scalarization adds.

**Objectives:** We examine to what extent checkpoint archiving compensates for the limited targeting of weighted sums and which advantages of nonlinear scalarization then remain. We also examine how the answers depend on training and evaluation budgets and on deterministic or sampled execution.

**Methods:** We distinguish four categories of returns, namely those that a preference grid targets, that training discovers, that an archive keeps, and that the final actors retain. We compare weighted-sum, Tchebycheff (TCH), and smooth Tchebycheff (STCH) scalarization with raw and normalized objectives in populations of independent proximal policy optimization actors. The main experiments use six Deep Sea Treasure environments with known Pareto fronts, five of which have concave regions. Further experiments cover soft actor-critic, six environments without known fronts, and established methods.

**Results:** On Deep Sea Treasure, raw weighted sums target only a fraction of the Pareto returns, but their archives recover nearly all of them, including every unsupported return. This compensation requires frequent checkpoint evaluation and deterministic execution, and it does not extend to the final actors, which represent far fewer returns and often miss their own targets. Normalized TCH and STCH recover the complete front in every run and keep the broadest sets of final actors, but this advantage shrinks or disappears with sampled actions. Without known fronts, where reference points must be estimated, no nonlinear configuration is consistently better than raw weighted sums. With an archive, fixed-preference populations trained with raw weighted sums are competitive with established methods in most environments.

**Conclusions:** Checkpoint archiving partially compensates for the limited targeting of weighted sums, but only in the archive. With known fronts and exact reference points, nonlinear scalarization improves recovery reliability rather than discovery, but this advantage does not carry over consistently to unknown fronts. Evaluations should therefore report separately which returns are targeted, discovered, archived, and represented by the final actors, and under which execution rule.

---

## Installation

**Requirements:** Python 3.11, Conda

**1. Create and activate a conda environment:**
```bash
conda create -n morl-scalarization python=3.11
conda activate morl-scalarization
```

**2. Install PyTorch** (select your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/)):
```bash
pip install torch
```

**3. Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

**4. Optional, for the evaluation scripts:**
```bash
pip install pandas matplotlib scipy
```

---

## Repository Structure
```
morl-d-scalarization/
├── agents/
│   ├── multi_policy/
│   │   ├── mo_ppo_nl.py                    # MO-PPO: population of independent PPO actors with fixed preferences
│   │   └── mo_sac_nl.py                    # MO-SAC: population of independent SAC actors with fixed preferences
│   ├── single_policy/
│   │   ├── ppo_utils_nl/
│   │   │   ├── ppo_nl.py                   # PPO update with nonlinear scalarization
│   │   │   ├── ppo_worker_nl.py            # Parallel training and evaluation worker for MO-PPO
│   │   │   ├── external_pareto.py          # Checkpoint archive of nondominated returns
│   │   │   └── ...                         # Policy networks, rollout storage, vectorized envs
│   │   └── sac_utils_nl/
│   │       ├── sac_continues_action_nl.py  # SAC learner for continuous action spaces
│   │       ├── sac_discrete_action_nl.py   # SAC learner for discrete action spaces
│   │       ├── sac_net.py                  # Actor and critic network definitions for SAC
│   │       └── sac_worker_nl.py            # Parallel training and evaluation worker for MO-SAC
│   └── utils/
│       ├── agent.py                        # Abstract base class for all agents
│       └── scalarizer.py                   # WS, TCH and STCH with raw or normalized objectives
├── configs/
│   ├── environment_configs.json            # Environment IDs, hypervolume reference points, ideal and nadir points
│   ├── mo_ppo_nl_{dst,ldst,minecart,reacher,mujoco}.json   # MO-PPO hyperparameters
│   └── mo_sac_nl_{dst,ldst,minecart,reacher,mujoco}.json   # MO-SAC hyperparameters
├── envs/
│   └── dst_envs.py                         # DST-1 to DST-4, LDST-1 and LDST-2 with exact Pareto fronts
├── evaluation_scripts/                     # Metrics, tables and figures of the paper (see Evaluation)
├── misc/                                   # Shared utilities: Pareto archive, metrics, evaluation, weight generation
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── run_mo_ppo_nl_baselines.py              # MO-PPO on DST/LDST, Minecart and Reacher
├── run_mo_ppo_nl_baselines_mujoco.py       # MO-PPO on the MuJoCo environments
├── run_mo_sac_nl_baselines.py              # MO-SAC on DST/LDST, Minecart and Reacher
├── run_mo_sac_nl_baselines_mujoco.py       # MO-SAC on the MuJoCo environments
└── run_mo_ppo_nl_dst_*.py                  # Exact-front MO-PPO experiments on DST/LDST (see Usage)

```
---

## Usage

All commands are run from the repository root. Settings that are not passed on the command line are read from the JSON files in `configs/`.

### MO-PPO

```bash
python run_mo_ppo_nl_baselines.py --env ldst2 --seed 1 --scalarization_method tch_norm
python run_mo_ppo_nl_baselines_mujoco.py --env halfcheetah --seed 1 --scalarization_method ws_raw
```

### MO-SAC

```bash
python run_mo_sac_nl_baselines.py --env minecart --seed 1 --scalarization_method ws_raw
python run_mo_sac_nl_baselines_mujoco.py --env halfcheetah --seed 1 --scalarization_method ws_raw
```

### Exact-Front MO-PPO Experiments

The experiments on the Deep Sea Treasure environments (`dst1` to `dst4`, `ldst1`, `ldst2`) have one script each.

| Experiment | Script | Varied setting |
| --- | --- | --- |
| E1: preferences, checkpoint evaluation, training duration | `run_mo_ppo_nl_dst_weight_resolution.py` | `--num_subproblems` |
| E2: deterministic versus sampled execution | `run_mo_ppo_nl_dst_stoch_vs_deter.py` | `--entropy_coef` |
| E3: preferences versus replicas | `run_mo_ppo_nl_dst_weights_vs_replicas.py` | `--num_subproblems`, `--replicas_per_weight` |
| E4: learning-rate sensitivity | `run_mo_ppo_nl_dst_optimizer_axis_sweeps.py` | `--learning_rate` |
| E5: longer training | `run_mo_ppo_nl_dst_long_horizon.py` | 500 instead of 100 iterations |

```bash
python run_mo_ppo_nl_dst_weight_resolution.py --env ldst2 --seed 1 --num_subproblems 50 --scalarization_method tch_norm
python run_mo_ppo_nl_dst_stoch_vs_deter.py --env ldst2 --seed 1 --entropy_coef 0.1 --scalarization_method stch_norm
python run_mo_ppo_nl_dst_weights_vs_replicas.py --env ldst2 --seed 1 --num_subproblems 20 --replicas_per_weight 2 \
    --scalarization_method ws_raw --path_prefix scalarization/dst_weights_vs_replicas_default_budget
python run_mo_ppo_nl_dst_optimizer_axis_sweeps.py --env ldst2 --seed 1 --learning_rate 1e-3 \
    --scalarization_method ws_raw --path_prefix scalarization/dst_optimizer_axis_sweeps
python run_mo_ppo_nl_dst_long_horizon.py --env ldst2 --seed 1 --scalarization_method ws_raw
```

The evaluation scripts read E3 from `scalarization/dst_weights_vs_replicas_default_budget` (population sizes 10 and 20) and `scalarization/dst_weights_vs_replicas` (population size 50), and E4 from `scalarization/dst_optimizer_axis_sweeps`. These prefixes differ from the script defaults and must be passed with `--path_prefix`.

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--env` | script-specific | Environment: `dst1`, `dst2`, `dst3`, `dst4`, `ldst1`, `ldst2`, `minecart`, `reacher` or, in the `*_mujoco.py` scripts, `ant`, `halfcheetah`, `hopper`, `humanoid` |
| `--seed` | `1` | Random seed |
| `--scalarization_method` | `ws_raw` | Scalarizer: `ws_raw`, `ws_norm`, `tch_raw`, `tch_norm`, `stch_raw`, `stch_norm` (MO-SAC: `ws_raw`, `ws_norm`) |
| `--num_subproblems` | from config | Population size $P$ |
| `--replicas_per_weight` | `1` | Actors per preference $R$, with $K = P / R$ distinct preferences |
| `--init_w_sampling` | `das_dennis` | Preference generation: `das_dennis`, `uniform`, `dirichlet` |
| `--mu` | `0.1` | Smoothing constant $\kappa_\mu$ of STCH (MO-PPO only) |
| `--gamma` | from config | Discount factor |
| `--learning_rate` | from config | Learning rate (not in `run_mo_sac_nl_baselines_mujoco.py`) |
| `--entropy_coef` | from config | Entropy coefficient (MO-PPO only) |
| `--alpha` | from config | Fixed entropy temperature (MO-SAC only) |
| `--total_iterations` | `100` | Training iterations $H$, only in the script for E1. The scripts for E2 to E5 name it `--total_iteration` (E5: `500`). All other scripts read $H$ from the config |
| `--eval_iterations` | `1` | Checkpoint-evaluation interval $C$, only in the script for E1. The scripts for E2 to E5 name it `--eval_iteration`. All other scripts read $C$ from the config |
| `--eval_rep` | `1` | Evaluation episodes per actor (MuJoCo: `5`, E2: `100` with sampled actions) |
| `--save_fronts` / `--no-save_fronts` | enabled | Store the archive and the returns of the current actors at every evaluation |
| `--save_models` | disabled | Store the trained models |
| `--path_prefix` | script-specific | Folder below `results/` |

The default settings per environment group are:

| Config | $P$ | $H$ | $C$ | $\gamma$ | Episode limit |
| --- | --- | --- | --- | --- | --- |
| `dst` (`dst1` to `dst4`) | 10 | 100 | 1 | 1.0 | 50 |
| `ldst` (`ldst1`, `ldst2`) | 20 | 100 | 1 | 1.0 | 100 |
| `minecart` | 15 | 1000 | 10 | 0.98 | 1000 |
| `reacher` | 20 | 500 | 5 | 0.99 | 50 |
| `mujoco` | 6 | 1000 | 10 | 0.99 | 500 |

### Outputs

Every run writes to:
```
results/{path_prefix}/{agent_name}/{env}/gamma_{gamma}/{init_w_sampling}/P_{P:04d}/K_{K:04d}/R_{R:04d}/H_{H:05d}/eval_{C:04d}/lr_{learning_rate}/ec_{entropy_coef}/{scalarization_method}/s_{seed:04d}/
```
MO-SAC uses `alpha_{alpha}` instead of `ec_{entropy_coef}`, and `stch_raw` adds the level `mu_{mu}` below the scalarizer. Floats are written with `p` for the decimal point, for example `lr_0p0003`.

| Folder | Content |
| --- | --- |
| `runs/` | TensorBoard logs with the metrics of the archive (`archive/...`) and of the current actors (`policy/...`) |
| `fronts/` | Archive at every evaluation (NPZ, one array `t_{global_step}` per checkpoint) |
| `policies/` | Returns of the current actors at every evaluation with their preferences. E2 also writes `stoch_vs_deter.npz` here |
| `config/` | Configuration of the run |
| `models/` | Trained models (only with `--save_models`) |

To monitor training:
```bash
tensorboard --logdir results/scalarization/baselines/
```

### Evaluation

The scripts in `evaluation_scripts/` compute all tables and figures of the paper from the stored returns. They do not train or execute actors. They are run from the repository root with the root on the Python path, for example:
```bash
PYTHONPATH=. python evaluation_scripts/compute_metrics_baseline.py --envs dst
```

**Exact-front study (DST/LDST).** `compute_dst_targability.py` computes the oracle targets of every scalarizer and preference grid. Each experiment then has three steps, `reconstruct_<id>.py`, `summarize_<id>.py` and `plot_<id>_paper.py`, with `<id>` in `e1` to `e5` and `sac` (MO-SAC versus MO-PPO with weighted sums). These scripts read the exact fronts from `results/evaluation/scalarization/dst/geometry/dst_points.csv` and write to `results/evaluation/scalarization/dst/`.

**Environments without known fronts and baseline comparison.**

| Script | Purpose |
| --- | --- |
| `check_ipro_runs.py` | Final fronts and run checks of IPRO |
| `build_reference_fronts.py` | Reference fronts and hypervolume reference points. With `--calibration`, the ideal and nadir points for `configs/environment_configs.json` |
| `compute_metrics_baseline.py`, `make_baseline_metics_tables.py` | HV ratio and IGD+ of every final front and the comparison tables |
| `compute_metrics_active.py`, `make_active_metrics_tables.py` | Final archive versus final actors of MO-PPO |
| `supported_share.py`, `make_supported_metrics_tables.py` | Supported and unsupported returns of the reference fronts and of every run |
| `prepare_bayes_tests.py`, `run_bayes_tests.py` | Bayesian t-tests of raw weighted sums against the other scalarizers |
| `plot_baseline_fronts.py`, `plot_baseline_learning_curve.py` | Final fronts and learning curves |
| `plot_dst_fronts.py`, `plot_targeting_schematic.py` | Exact DST/LDST fronts and the schematic figure of the introduction |

The runs of GPI-LS, PCN, IPRO, C-MORL and MORL/D are not part of this repository. The scripts expect them below `results/scalarization/baselines/` in the folders `gpi_ls`, `pcn`, `ipro`, `c_morl` and `morl_d`. `run_bayes_tests.py` additionally needs `arviz`, `cmdstanpy` and a local copy of the `bayesian-model-comparison` repository (`--repo` or `BAYES_REPO`).

---

## Acknowledgements

This repository builds on the following codebases:
- [MORL/D](https://github.com/LucasAlegre/morl-baselines) — MORL/D, GPI-LS and PCN baselines
- [C-MORL](https://github.com/RuohLiuq/C-MORL) — C-MORL baseline
- [PG-MORL](https://github.com/mit-gfx/PGMORL) — PPO implementation underlying MO-PPO
- [IPRO](https://arxiv.org/abs/2402.07182) — IPRO baseline

The Deep Sea Treasure environments follow T. Li, Y. Meng, and L. Tang, "A Decomposition Optimization-Based Multiobjective Reinforcement Learning Algorithm for Obtaining Nonconvex Pareto Fronts," IEEE TNNLS, 2026.

---

## Citation

```bibtex
@unpublished{kemper2026targeted,
  title  = {Targeted, Discovered, Retained: Separating the Effects of Scalarization, Archiving, 
            and Execution in Multi-Policy Multi-Objective Reinforcement Learning},
  author = {Kemper, Neele and Stegherr, Helena and Wurth, Jonathan and Heider, Michael and H{\"a}hner, J{\"o}rg},
  note   = {Submitted to the Journal of Artificial Intelligence Research (JAIR)},
  year   = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
