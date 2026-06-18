"""
datacenter_env.py
─────────────────
Replay-based Gymnasium environment for datacenter cooling control.

State  : Full 43-dim standardized feature vector from the dataset.
Action : 16-dim delta vector for P_ac (8) and P_cu (8) — Box(−1, 1).
Reward : −Σ(applied_power) − λ · Σ max(0, T_celCC − T_max)²

Temperature dynamics follow real dataset transitions. The agent's delta
actions shift the current power readings before the environment advances
to the next real data row, keeping power continuity across steps.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Column definitions (must match the CSV header order) ─────────────────────
P_AC_COLS   = [f"P_ac-{i}"   for i in range(8)]
P_CU_COLS   = [f"P_cu-{i}"   for i in range(8)]
POWER_COLS  = P_AC_COLS + P_CU_COLS              # 16 controllable inputs
T_OUT_COLS  = [f"T_out-{i}"  for i in range(8)]
T_MEAS_COLS = [f"T_MEAS-{i}" for i in range(8)]
T_CELCC_COLS= [f"T_celCC-{i}"for i in range(8)]  # safety-critical rack temps
TIME_COLS   = ["TLHC", "DoW", "WeH"]

ALL_COLS = (
    P_AC_COLS + P_CU_COLS
    + T_OUT_COLS + T_MEAS_COLS + T_CELCC_COLS
    + TIME_COLS
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
T_CELCC_MAX      = 2.0   # upper safety bound (standardized, ≈ +2σ)
LAMBDA_CONSTRAINT= 5.0   # penalty weight per unit of squared violation
CLIP_POWER       = 3.0   # hard clip on applied power (standardized)
EPISODE_LEN      = 500   # timesteps per episode
TRAIN_FRAC       = 0.80  # 80 / 20 train-test split


class DatacenterEnv(gym.Env):
    """
    Parameters
    ----------
    data      : float32 array of shape (T, 43) — pre-loaded standardized CSV.
    col_names : list of 43 column name strings matching `data` column order.
    split     : 'train' or 'test' — which portion of data to sample from.
    episode_len: number of steps per episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: np.ndarray,
        col_names: list[str],
        split: str = "train",
        episode_len: int = EPISODE_LEN,
    ):
        super().__init__()
        self.data        = data.astype(np.float32)
        self.col_names   = col_names
        self.episode_len = episode_len

        # Build index maps
        idx = {c: i for i, c in enumerate(col_names)}
        self.power_idx = np.array([idx[c] for c in POWER_COLS], dtype=int)
        self.celcc_idx = np.array([idx[c] for c in T_CELCC_COLS], dtype=int)

        # Compute valid start indices for the chosen split
        n = len(data)
        split_pt = int(TRAIN_FRAC * n)
        if split == "train":
            lo, hi = 0, split_pt - episode_len - 1
        else:
            lo, hi = split_pt, n - episode_len - 1
        self._lo, self._hi = lo, hi

        n_feat = data.shape[1]
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(n_feat,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(POWER_COLS),), dtype=np.float32
        )

        self._ptr: int = 0
        self._step: int = 0
        self._state: np.ndarray | None = None

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ptr   = int(self.np_random.integers(self._lo, self._hi))
        self._step  = 0
        self._state = self.data[self._ptr].copy()
        return self._state.copy(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # 1. Apply delta to current power readings
        applied_power = np.clip(
            self._state[self.power_idx] + action, -CLIP_POWER, CLIP_POWER
        )
        self._state[self.power_idx] = applied_power

        # 2. Reward: minimise energy, penalise thermal constraint violations
        energy_cost       = float(np.sum(applied_power))
        t_celcc           = self._state[self.celcc_idx]
        viol              = np.maximum(0.0, t_celcc - T_CELCC_MAX)
        constraint_penalty= float(np.sum(viol ** 2))
        reward = -energy_cost - LAMBDA_CONSTRAINT * constraint_penalty

        # 3. Advance: take next real data row, carry applied power forward
        self._ptr  += 1
        self._step += 1
        next_row = self.data[self._ptr].copy()
        next_row[self.power_idx] = applied_power
        self._state = next_row

        truncated  = self._step >= self.episode_len
        terminated = False
        info = {
            "energy_cost"        : energy_cost,
            "constraint_penalty" : constraint_penalty,
            "constraint_violation": float(np.any(viol > 0)),
        }
        return self._state.copy(), reward, terminated, truncated, info
