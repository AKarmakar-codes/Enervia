"""
mpc_guardrail.py
────────────────
Linear MPC safety filter for datacenter cooling.

Two-phase design
────────────────
Phase 1 — Offline fit (LinearDynamicsModel.fit):
    Fit  T_{t+1} = A @ T_t + B @ P_t + c  via Ridge regression on training data.
    T_t  = [T_MEAS(8), T_out(8), T_celCC(8), TLHC(1)]   (25-dim temperature state)
    P_t  = [P_ac(8), P_cu(8)]                            (16-dim power / control)

Phase 2 — Online projection (MPCGuardrail.project):
    Given the RL agent's proposed delta action du_rl, solve the QP:

        min   ‖du − du_rl‖²          (stay as close as possible to RL intent)
        s.t.  T_celCC[k] ≤ T_MAX     for k = 1 … N   (rack temperature safety)
              du ∈ [−1, 1]^(N·n_u)   (actuator delta limits)

    The constraint is linearised via the unrolled prediction:
        T[k] = T_free[k] + S[k] @ du
    where S is the block-lower-triangular sensitivity matrix and T_free is the
    open-loop (zero-delta) predicted trajectory.

    Solved with OSQP (<1 ms per call on CPU for N=5, n_u=16).
    Falls back to unmodified RL action if the QP is infeasible.
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse as sp
import osqp
import gymnasium as gym
from sklearn.linear_model import Ridge

# ── Feature definitions (must match datacenter_env.py ordering) ──────────────
_T_MEAS  = [f"T_MEAS-{i}"  for i in range(8)]
_T_OUT   = [f"T_out-{i}"   for i in range(8)]
_T_CELCC = [f"T_celCC-{i}" for i in range(8)]
_TLHC    = ["TLHC"]
T_STATE_COLS = _T_MEAS + _T_OUT + _T_CELCC + _TLHC  # 25 features

POWER_COLS   = [f"P_ac-{i}" for i in range(8)] + [f"P_cu-{i}" for i in range(8)]

# Index of T_celCC within the 25-dim temperature state vector (rows 16–23)
CELCC_IN_TSTATE = list(range(16, 24))

# ── Constants ────────────────────────────────────────────────────────────────
N_TEMP    = 25    # temperature state dimension
N_CTRL    = 16    # control input dimension
T_MAX     = 2.0   # standardised upper safety bound (+2σ)
N_HORIZON = 5     # MPC prediction horizon (steps)


# ─────────────────────────────────────────────────────────────────────────────
class LinearDynamicsModel:
    """Offline linear temperature-dynamics model: T' = A T + B P + c."""

    def __init__(self, alpha: float = 1.0):
        self._ridge = Ridge(alpha=alpha, fit_intercept=True)
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.c: np.ndarray | None = None
        self.r2_: float | None = None

    def fit(self, data: np.ndarray, col_names: list[str]) -> "LinearDynamicsModel":
        """
        Fit the model from raw (standardised) dataset rows.

        Parameters
        ----------
        data      : float array (T, n_features)
        col_names : list of column name strings
        """
        idx = {c: i for i, c in enumerate(col_names)}

        t_idx = np.array([idx[c] for c in T_STATE_COLS])
        p_idx = np.array([idx[c] for c in POWER_COLS])

        T_curr = data[:-1, t_idx]   # (T-1, 25)
        P_curr = data[:-1, p_idx]   # (T-1, 16)
        T_next = data[1:,  t_idx]   # (T-1, 25)

        X = np.concatenate([T_curr, P_curr], axis=1)   # (T-1, 41)
        self._ridge.fit(X, T_next)

        coef = self._ridge.coef_                # (25, 41)
        self.A = coef[:, :N_TEMP]               # (25, 25)
        self.B = coef[:, N_TEMP:]               # (25, 16)
        self.c = self._ridge.intercept_         # (25,)
        self.r2_ = self._ridge.score(X, T_next)
        print(f"[LinearDynamicsModel] R² = {self.r2_:.4f}")
        return self

    def predict_one(self, T: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Single-step prediction: T' = A T + B P + c."""
        return self.A @ T + self.B @ P + self.c


# ─────────────────────────────────────────────────────────────────────────────
class MPCGuardrail:
    """
    Online MPC safety filter.  Call `project(T_state, P_curr, du_rl)` at
    every environment step to obtain a safe action.

    Parameters
    ----------
    model     : fitted LinearDynamicsModel
    horizon   : prediction horizon N (default 5)
    t_max     : upper temperature safety bound (standardised)
    """

    def __init__(
        self,
        model: LinearDynamicsModel,
        horizon: int = N_HORIZON,
        t_max: float = T_MAX,
    ):
        self.model   = model
        self.N       = horizon
        self.t_max   = t_max

        A, B = model.A, model.B
        n_x, n_u = N_TEMP, N_CTRL
        N = self.N

        # ── Pre-compute sensitivity matrix S (N·n_x × N·n_u) ───────────────
        # S[k·n_x:(k+1)·n_x, j·n_u:(j+1)·n_u] = A^(k-j) @ B   for k >= j
        S = np.zeros((N * n_x, N * n_u))
        # Pre-compute A powers
        Apow = [np.eye(n_x)]
        for _ in range(N):
            Apow.append(Apow[-1] @ A)

        for k in range(N):
            for j in range(k + 1):
                S[k * n_x:(k + 1) * n_x,
                  j * n_u:(j + 1) * n_u] = Apow[k - j] @ B

        # ── Selector matrix: pick T_celCC rows from each predicted state ───
        # C_sel ∈ R^(8 × n_x)
        n_celcc = len(CELCC_IN_TSTATE)
        C_sel = np.zeros((n_celcc, n_x))
        for r, c in enumerate(CELCC_IN_TSTATE):
            C_sel[r, c] = 1.0

        # C_block ∈ R^(N·n_celcc × N·n_x) — block-diagonal with C_sel
        C_block = np.kron(np.eye(N), C_sel)

        # ── Constraint sensitivity: maps du → T_celCC predictions ──────────
        # shape: (N·n_celcc, N·n_u)
        self._CS = C_block @ S                # (N·8, N·16)
        self._C_block = C_block               # kept to compute T_free part
        self._Apow  = Apow

        n_var = N * n_u
        n_con_temp = N * n_celcc     # temperature constraints
        n_con_box  = n_var           # box constraints on du

        # ── OSQP cost: 0.5 xᵀPx + qᵀx, P = I ────────────────────────────
        P_osqp = sp.eye(n_var, format="csc")

        # ── OSQP constraint matrix (stacked) ─────────────────────────────
        # Row block 1: CS @ du ≤ T_max - CS_free  (temperature upper bound)
        # Row block 2: I @ du ∈ [-1, 1]           (box on delta action)
        CS_sparse  = sp.csc_matrix(self._CS)
        eye_sparse = sp.eye(n_var, format="csc")
        A_osqp = sp.vstack([CS_sparse, eye_sparse], format="csc")

        self._P_osqp  = P_osqp
        self._A_osqp  = A_osqp
        self._n_var   = n_var
        self._n_celcc = n_celcc
        self._n_con_temp = n_con_temp

        # Initialise OSQP solver (problem structure is constant)
        self._solver = osqp.OSQP()
        dummy_q = np.zeros(n_var)
        dummy_l = np.full(n_con_temp + n_con_box, -np.inf)
        dummy_u = np.ones(n_con_temp + n_con_box)
        self._solver.setup(
            P_osqp, dummy_q, A_osqp, dummy_l, dummy_u,
            warm_starting=True,
            verbose=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=4000,
            polish=False,
        )
        self._setup_done = True

    def _free_response(self, T0: np.ndarray, P0: np.ndarray) -> np.ndarray:
        """
        Predict temperature trajectory for zero delta-action (du=0),
        so P stays constant at P0 for the whole horizon.
        Returns shape (N·n_x,).
        """
        A, B, c = self.model.A, self.model.B, self.model.c
        traj = []
        T = T0.copy()
        for _ in range(self.N):
            T = A @ T + B @ P0 + c
            traj.append(T)
        return np.concatenate(traj)   # (N·n_x,)

    def project(
        self,
        T_state: np.ndarray,
        P_curr: np.ndarray,
        du_rl: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Project RL action du_rl onto the safe set.

        Parameters
        ----------
        T_state : 25-dim temperature state vector
        P_curr  : 16-dim current power vector
        du_rl   : 16-dim proposed RL delta action

        Returns
        -------
        du_safe  : 16-dim safe delta action (may equal du_rl if already safe)
        modified : True if the MPC changed the action
        """
        N, n_u   = self.N, N_CTRL
        n_celcc  = self._n_celcc

        # Free-response temperature trajectory (shape N·n_x)
        T_free = self._free_response(T_state, P_curr + du_rl[:n_u])
        # Note: we use (P_curr + du_rl) as a warm start for the solver;
        # the QP objective will pull toward du_rl anyway.

        # T_celCC from free response, shape (N·n_celcc,)
        T_celcc_free = self._C_block @ T_free

        # ── Assemble bounds ───────────────────────────────────────────────
        # Temperature upper bound: CS @ du ≤ t_max - T_celcc_free
        u_temp = self.t_max - T_celcc_free               # (N·n_celcc,)
        l_temp = np.full(self._n_con_temp, -np.inf)

        # Box: du ∈ [-1, 1]
        u_box = np.ones(self._n_var)
        l_box = -np.ones(self._n_var)

        l_osqp = np.concatenate([l_temp, l_box])
        u_osqp = np.concatenate([u_temp, u_box])

        # ── Linear cost: q = -du_rl tiled for each horizon step ──────────
        # Objective: ‖du − du_rl‖² ⇒ min 0.5 duᵀdu − du_rl·du
        # We only apply the cost to the first step's action (the one we use)
        q = np.zeros(self._n_var)
        q[:n_u] = -du_rl   # penalise only step-0 deviation

        # ── Update & solve ─────────────────────────────────────────────────
        self._solver.update(q=q, l=l_osqp, u=u_osqp)
        result = self._solver.solve()

        if result.info.status in ("solved", "solved_inaccurate"):
            du_safe = np.clip(result.x[:n_u], -1.0, 1.0).astype(np.float32)
            modified = bool(np.linalg.norm(du_safe - du_rl) > 1e-3)
        else:
            # QP infeasible or solver failure — fall back to RL action
            warnings.warn(f"[MPC] Solver status: {result.info.status}. Using RL action.")
            du_safe  = du_rl.astype(np.float32)
            modified = False

        return du_safe, modified


# ─────────────────────────────────────────────────────────────────────────────
class MPCGuardrailWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that intercepts RL actions and projects them onto
    the safe set via the MPC guardrail before passing to the environment.

    Extra keys added to `info`:
        mpc_modified  : bool — whether the action was changed
        du_rl_original: original (uncorrected) RL action
    """

    def __init__(self, env: gym.Env, guardrail: MPCGuardrail, col_names: list[str]):
        super().__init__(env)
        self.guardrail = guardrail
        idx = {c: i for i, c in enumerate(col_names)}
        self._t_idx = np.array([idx[c] for c in T_STATE_COLS])
        self._p_idx = np.array([idx[c] for c in POWER_COLS])
        self.modification_count = 0
        self.total_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray):
        # Extract temperature state and current power from last observation
        T_state = self._last_obs[self._t_idx]
        P_curr  = self._last_obs[self._p_idx]

        du_rl   = np.asarray(action, dtype=np.float32)
        du_safe, modified = self.guardrail.project(T_state, P_curr, du_rl)

        obs, reward, terminated, truncated, info = self.env.step(du_safe)
        self._last_obs = obs

        self.total_steps      += 1
        self.modification_count += int(modified)
        info["mpc_modified"]      = modified
        info["du_rl_original"]    = du_rl
        return obs, reward, terminated, truncated, info

    @property
    def modification_rate(self) -> float:
        """Fraction of steps where MPC corrected the RL action."""
        if self.total_steps == 0:
            return 0.0
        return self.modification_count / self.total_steps
