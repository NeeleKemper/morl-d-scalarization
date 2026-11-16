import os
import json
import torch
import random

import numpy as np

from typing import Optional
from dataclasses import dataclass, field

BASE_DIR = 'results'


@dataclass
class ParetoFrontStore:
    """Collects front snapshots under t_<global_step> keys with row-aligned metadata and writes them as NPZ."""
    path: str
    buf: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self,
            front: np.ndarray,
            global_step: int,
            weights: Optional[np.ndarray] = None,
            global_iteration: Optional[int] = None,
            agent_ids: Optional[np.ndarray] = None,
            weight_ids: Optional[np.ndarray] = None,
            replica_ids: Optional[np.ndarray] = None) -> None:
        """Store one front snapshot and the given metadata under suffix keys."""
        key = f't_{int(global_step):08d}'
        self.buf[key] = np.asarray(front, dtype=np.float32)
        if global_iteration is not None:
            self.buf[f'{key}_iteration'] = np.asarray(int(global_iteration), dtype=np.int64)
        metadata = {
            'w': (weights, np.float32),
            'agent_id': (agent_ids, np.int64),
            'weight_id': (weight_ids, np.int64),
            'replica_id': (replica_ids, np.int64),
        }
        for suffix, (values, dtype) in metadata.items():
            if values is not None:
                self.buf[f'{key}_{suffix}'] = np.asarray(values, dtype=dtype)

    def flush(self) -> None:
        """Write all snapshots to the compressed NPZ file."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez_compressed(self.path, **self.buf)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and enable deterministic algorithms."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_env_config(path: str) -> dict:
    """Read the environment configuration JSON."""
    with open(path) as f:
        return json.load(f)


def read_algo_config(path: str) -> dict:
    """Read an algorithm configuration JSON."""
    with open(path) as f:
        return json.load(f)


def setup_directories(log_dir: str,
                      file_name: str,
                      save_fronts: bool,
                      save_models: bool) -> tuple[str, str, str, Optional[ParetoFrontStore],
                                                  Optional[ParetoFrontStore]]:
    """Create the run, model, front, policy and config directories and return their paths and the front stores."""
    runs_path = f'{BASE_DIR}/{log_dir}/runs'
    model_path = f'{BASE_DIR}/{log_dir}/models'
    front_path = f'{BASE_DIR}/{log_dir}/fronts'
    config_path = f'{BASE_DIR}/{log_dir}/config'
    policy_path = f'{BASE_DIR}/{log_dir}/policies'
    pf_store = None
    policy_store = None
    os.makedirs(runs_path, exist_ok=True)
    if save_models:
        os.makedirs(model_path, exist_ok=True)
    if save_fronts:
        os.makedirs(front_path, exist_ok=True)
        pf_store = ParetoFrontStore(path=f'{front_path}/{file_name}.npz')
        os.makedirs(policy_path, exist_ok=True)
        policy_store = ParetoFrontStore(path=f'{policy_path}/{file_name}.npz')
    os.makedirs(config_path, exist_ok=True)
    return runs_path, model_path, config_path, pf_store, policy_store
