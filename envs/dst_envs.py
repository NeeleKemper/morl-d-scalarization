"""
Deep Sea Treasure benchmark environments from:

T. Li, Y. Meng, and L. Tang,
"A Decomposition Optimization-Based Multiobjective Reinforcement Learning Algorithm for Obtaining Nonconvex Pareto Fronts,"
IEEE TNNLS, vol. 37, no. 1, pp. 219-232, Jan. 2026.
DOI: 10.1109/TNNLS.2025.3603165
"""

import numpy as np
import gymnasium as gym

from collections import deque
from gymnasium.utils import EzPickle
from typing import List, Optional, Tuple, Any
from gymnasium.spaces import Box, Discrete

__all__ = [
    'DeepSeaTreasure',
    'DST1_MAP', 'DST2_MAP', 'DST3_MAP', 'DST4_MAP', 'LDST1_MAP', 'LDST2_MAP',
    'ALL_MAPS', 'TRUE_FRONTS', 'register_envs', 'pareto_front_of_map',
    'convex_hull_mask',
]


# -------------------- MAPS --------------------

def _staircase_map(shape: Tuple[int, int], treasures: List[Tuple[int, int, float]]) -> np.ndarray:
    """Build a DST map: water, the given treasure cells and seafloor (-10) below every treasure."""
    m = np.zeros(shape, dtype=np.float64)
    for r, c, v in treasures:
        m[r, c] = v
    for r, c, _ in treasures:
        m[r + 1:, c] = -10.0
    return m


# --- DST-1 to DST-4 (11x11, shared treasure cells) ---
_DST_CELLS = [(1, 0), (2, 1), (3, 2), (4, 3), (4, 4), (4, 5), (7, 6), (7, 7), (9, 8), (10, 9)]
_DST1_VALUES = [1.0, 2.0, 3.0, 5.0, 8.0, 16.0, 24.0, 50.0, 74.0, 124.0]
_DST2_VALUES = [1.0, 2.0, 3.0, 5.0, 8.0, 16.0, 100.0, 50.0, 74.0, 124.0]
_DST3_VALUES = [5.0, 25.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0, 92.0, 100.0]
_DST4_VALUES = [100.0, 925.0, 1231.0, 1442.0, 1525.0, 1597.0, 1797.0, 1829.0, 1889.0, 1900.0]
DST1_MAP = _staircase_map((11, 11), [(r, c, v) for (r, c), v in zip(_DST_CELLS, _DST1_VALUES)])
DST2_MAP = _staircase_map((11, 11), [(r, c, v) for (r, c), v in zip(_DST_CELLS, _DST2_VALUES)])
DST3_MAP = _staircase_map((11, 11), [(r, c, v) for (r, c), v in zip(_DST_CELLS, _DST3_VALUES)])
DST4_MAP = _staircase_map((11, 11), [(r, c, v) for (r, c), v in zip(_DST_CELLS, _DST4_VALUES)])

# --- LDST-1 (21x21) ---
_LDST1_CELLS = [(1, 0), (2, 1), (3, 2), (4, 3), (4, 4), (6, 5), (7, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11),
                (8, 12), (9, 13), (11, 14), (13, 15), (16, 16), (18, 17), (19, 18), (20, 19)]
_LDST1_VALUES = [1.0, 2.0, 15.0, 35.0, 58.0, 142.0, 314.0, 702.0, 1174.0, 1724.0, 2214.0, 2517.0, 2706.0, 2911.0,
                 3147.0, 3256.0, 3351.0, 3404.0, 3446.0, 3477.0]
LDST1_MAP = _staircase_map((21, 21), [(r, c, v) for (r, c), v in zip(_LDST1_CELLS, _LDST1_VALUES)])

# --- LDST-2 (21x21) ---
_LDST2_CELLS = [(1, 0), (2, 1), (3, 2), (4, 3), (4, 4), (4, 5), (7, 6), (7, 7), (9, 8), (10, 9), (12, 10), (14, 11),
                (17, 12), (17, 13), (19, 14), (19, 15), (20, 16), (20, 17), (20, 18), (20, 19)]
_LDST2_VALUES = [1.0, 1501.0, 2723.0, 3654.0, 4042.0, 4316.0, 5106.0, 5211.0, 5423.0, 5520.0, 5632.0, 5888.0,
                 6239.0, 6792.0, 7358.0, 8096.0, 8487.0, 9560.0, 10700.0, 14890.0]
LDST2_MAP = _staircase_map((21, 21), [(r, c, v) for (r, c), v in zip(_LDST2_CELLS, _LDST2_VALUES)])

ALL_MAPS = {
    'DST-1': DST1_MAP,
    'DST-2': DST2_MAP,
    'DST-3': DST3_MAP,
    'DST-4': DST4_MAP,
    'LDST-1': LDST1_MAP,
    'LDST-2': LDST2_MAP,
}


# -------------------- PARETO-FRONT UTILITIES --------------------

def _bfs_steps(sea_map: np.ndarray, start: Tuple[int, int]) -> dict:
    """Return the minimum number of steps from start to every reachable (absorbing) treasure cell."""
    nrow, ncol = sea_map.shape
    dist = {tuple(start): 0}
    q = deque([tuple(start)])
    steps = {}
    while q:
        r, c = q.popleft()
        if sea_map[r, c] > 0:
            steps[(r, c)] = dist[(r, c)]
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < nrow and 0 <= nc < ncol and sea_map[nr, nc] != -10 and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return steps


def pareto_front_of_map(sea_map: np.ndarray, start: Tuple[int, int] = (0, 0)) -> List[np.ndarray]:
    """Return the undiscounted Pareto front (treasure value, -steps) of a map, sorted by treasure value."""
    pts = [(sea_map[cell], -s) for cell, s in _bfs_steps(sea_map, start).items()]
    front = [p for p in pts if not any((q[0] >= p[0] and q[1] >= p[1]) and q != p for q in pts)]
    front.sort(key=lambda p: p[0])
    return [np.array(p, dtype=np.float64) for p in front]


def convex_hull_mask(front: List[np.ndarray]) -> List[bool]:
    """Mark the front points on the convex part, i.e. those reachable with a positive weighted sum."""
    pts = [tuple(map(float, p)) for p in front]
    order = sorted(range(len(pts)), key=lambda i: pts[i])
    hull: List[int] = []
    for i in order:
        while len(hull) >= 2:
            x1, y1 = pts[hull[-2]]
            x2, y2 = pts[hull[-1]]
            x3, y3 = pts[i]
            if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    on_hull = set(hull)
    return [i in on_hull for i in range(len(pts))]


# -------------------- ENVIRONMENT --------------------

class DeepSeaTreasure(gym.Env, EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 4}
    _COLORS = {'water': (48, 172, 228), 'floor': (26, 26, 26), 'sub': (220, 40, 40), 'grid': (200, 60, 40)}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 dst_map: np.ndarray = DST1_MAP,
                 float_state: bool = False,
                 start_pos: Tuple[int, int] = (0, 0)):
        """Deep Sea Treasure on any DST map; actions up/down/left/right, reward [treasure value, -1]."""
        EzPickle.__init__(self, render_mode, dst_map, float_state, start_pos)
        self.render_mode = render_mode
        self.float_state = float_state

        # --- Map & true front ---
        self.sea_map = np.array(dst_map, dtype=np.float64)
        self.start_pos = np.array(start_pos, dtype=np.int32)
        self._treasure_steps = _bfs_steps(self.sea_map, tuple(start_pos))
        self._pareto_front = pareto_front_of_map(self.sea_map, tuple(start_pos))
        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),
            1: np.array([1, 0], dtype=np.int32),
            2: np.array([0, -1], dtype=np.int32),
            3: np.array([0, 1], dtype=np.int32),
        }

        # --- Spaces ---
        nrow, ncol = self.sea_map.shape
        if self.float_state:
            self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.observation_space = Box(low=np.zeros(2, dtype=np.int32),
                                         high=np.array([nrow - 1, ncol - 1], dtype=np.int32),
                                         dtype=np.int32)
        self.action_space = Discrete(4)
        self.reward_space = Box(low=np.array([0.0, -1.0], dtype=np.float32),
                                high=np.array([float(self.sea_map.max()), -1.0], dtype=np.float32),
                                dtype=np.float32)
        self.reward_dim = 2

        # --- State & rendering ---
        self.current_state = self.start_pos.copy()
        self._window = None
        self._clock = None

    # -------------------- API --------------------

    def pareto_front(self, gamma: float) -> List[np.ndarray]:
        """Return the discounted true front: treasure v after n steps gives [v * gamma^(n-1), -sum_i gamma^i]."""
        out = []
        for v, neg_n in self._pareto_front:
            n = int(-neg_n)
            out.append(np.array([v * gamma ** (n - 1), -float(np.sum(gamma ** np.arange(n)))]))
        return out

    def treasure_specs(self) -> List[Tuple[Tuple[int, int], float, int]]:
        """Return (cell, value, min_steps) for all reachable treasures, sorted by steps."""
        return sorted(((cell, float(self.sea_map[cell]), s) for cell, s in self._treasure_steps.items()),
                      key=lambda t: t[2])

    # -------------------- MECHANICS --------------------

    def _is_valid_state(self, state: np.ndarray) -> bool:
        """Return whether a cell lies inside the grid and is not seafloor."""
        nrow, ncol = self.sea_map.shape
        return 0 <= state[0] < nrow and 0 <= state[1] < ncol and self.sea_map[state[0], state[1]] != -10

    def _get_state(self) -> np.ndarray:
        """Return the (row, col) observation, scaled to [0, 1] with float_state."""
        if self.float_state:
            scale = np.array(self.sea_map.shape, dtype=np.float32) - 1.0
            return (self.current_state.astype(np.float32) / scale).astype(np.float32)
        return self.current_state.copy()

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> tuple[np.ndarray, dict]:
        """Place the submarine at the start position."""
        super().reset(seed=seed)
        self.current_state = self.start_pos.copy()
        if self.render_mode == 'human':
            self.render()
        return self._get_state(), {}

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """Move (invalid moves are no-ops); entering a treasure ends the episode with its value."""
        nxt = self.current_state + self.dir[int(action)]
        if self._is_valid_state(nxt):
            self.current_state = nxt
        treasure_value = self.sea_map[self.current_state[0], self.current_state[1]]
        terminal = bool(treasure_value > 0)
        if not terminal:
            treasure_value = 0.0
        vec_reward = np.array([treasure_value, -1.0], dtype=np.float32)
        if self.render_mode == 'human':
            self.render()
        return self._get_state(), vec_reward, terminal, False, {}

    # -------------------- RENDERING --------------------

    def _frame(self, cell_px: int = 24) -> np.ndarray:
        """Draw the map and submarine as an RGB image; higher treasures are deeper gold."""
        nrow, ncol = self.sea_map.shape
        img = np.zeros((nrow * cell_px, ncol * cell_px, 3), dtype=np.uint8)
        vmax = self.sea_map.max()
        for r in range(nrow):
            for c in range(ncol):
                v = self.sea_map[r, c]
                if v == -10:
                    col = self._COLORS['floor']
                elif v > 0:
                    t = 0.25 + 0.75 * (v / vmax)
                    col = (255, int(230 - 120 * t), int(120 - 120 * t))
                else:
                    col = self._COLORS['water']
                img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px] = col
        r, c = self.current_state
        pad = max(2, cell_px // 5)
        img[r * cell_px + pad:(r + 1) * cell_px - pad, c * cell_px + pad:(c + 1) * cell_px - pad] = self._COLORS['sub']
        img[::cell_px, :, :] = self._COLORS['grid']
        img[:, ::cell_px, :] = self._COLORS['grid']
        return img

    def render(self) -> Optional[Any]:
        """Render as ANSI text, RGB array or a pygame window, depending on render_mode."""
        if self.render_mode == 'ansi':
            chars = {0.0: '..', -10.0: '##'}
            rows = []
            for r in range(self.sea_map.shape[0]):
                row = ''
                for c in range(self.sea_map.shape[1]):
                    row += '@@' if (r, c) == tuple(self.current_state) else chars.get(self.sea_map[r, c], '$$')
                rows.append(row)
            return '\n'.join(rows)
        frame = self._frame()
        if self.render_mode == 'rgb_array':
            return frame

        # Human mode: pygame window drawn from the same frame
        import pygame
        if self._window is None:
            pygame.init()
            pygame.display.set_caption('Deep Sea Treasure')
            self._window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            self._clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(self.metadata['render_fps'])

    def close(self) -> None:
        """Close the pygame window if one is open."""
        if self._window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None


# -------------------- REGISTRATION --------------------

TRUE_FRONTS = {name: pareto_front_of_map(m) for name, m in ALL_MAPS.items()}


def register_envs(max_episode_steps: Optional[int] = None) -> None:
    """Register DST1-v0 to LDST2-v0 with Gymnasium, optionally with a TimeLimit."""
    env_maps = {
        'DST1-v0': DST1_MAP,
        'DST2-v0': DST2_MAP,
        'DST3-v0': DST3_MAP,
        'DST4-v0': DST4_MAP,
        'LDST1-v0': LDST1_MAP,
        'LDST2-v0': LDST2_MAP,
    }
    for env_id, m in env_maps.items():
        if env_id in gym.registry:
            continue
        gym.register(id=env_id, entry_point='envs.dst_envs:DeepSeaTreasure', kwargs={'dst_map': m},
                     max_episode_steps=max_episode_steps)


if __name__ == '__main__':
    register_envs()
    for name, m in ALL_MAPS.items():
        front = pareto_front_of_map(m)
        hull = convex_hull_mask(front)
        print(f'{name}: grid {m.shape}, {int((m > 0).sum())} treasures, |PF| = {len(front)}, '
              f'convex points = {sum(hull)}')