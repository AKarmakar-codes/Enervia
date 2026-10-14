"""
baseline_controllers.py
───────────────────────
Implements non-RL baseline controllers for datacenter cooling:

1. ThermostatController:
   Industrial rule-based feedback controller with deadband hysteresis.
   Increases cooling when temperatures exceed target + deadband,
   decreases cooling when below target - deadband, and maintains state otherwise.

2. AutonomousMPCController:
   Autonomous economic Model Predictive Controller solving an online receding-horizon
   Quadratic Program (QP) via OSQP at each timestep. Minimizes power consumption
   subject to hard thermal safety constraints (T_celCC <= T_max) and actuator rate limits.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import osqp

from MPC.datacenter_env import CLIP_POWER, T_CELCC_MAX, T_CELCC_COLS, POWER_COLS
from MPC.results.mpc_guardrail import LinearDynamicsModel, T_STATE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Industrial Rule-Based Thermostat Controller
# ─────────────────────────────────────────────────────────────────────────────
class ThermostatController:
    """
    Classical rule-based hysteresis deadband controller.

    Parameters
    ----------
    col_names  : list of dataset column names matching observation space.
    t_target   : normalized target setpoint (default: 1.0 sigma).
    deadband   : deadband half-width (default: 0.25 sigma).
    ramp_rate  : power delta increment per step when outside deadband.
    clip_power : maximum power limit.
    """

    def __init__(
        self,
        col_names: list[str],
        t_target: float = 1.0,
        deadband: float = 0.25,
        ramp_rate: float = 0.20,
        clip_power: float = CLIP_POWER,
    ):
        idx = {c: i for i, c in enumerate(col_names)}
        self.celcc_idx = np.array([idx[f"T_celCC-{i}"] for i in range(8)], dtype=int)
        self.power_idx = np.array(
            [idx[f"P_ac-{i}"] for i in range(8)] + [idx[f"P_cu-{i}"] for i in range(8)],
            dtype=int,
        )
        self.t_target   = t_target
        self.deadband   = deadband
        self.ramp_rate  = ramp_rate
        self.clip_power = clip_power

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """
        Computes action delta based on zone temperature deviations.
        Matches the standard agent interface: action, _ = model.predict(obs).
        """
        t_cc = obs[self.celcc_idx]
        p_curr = obs[self.power_idx]
        du = np.zeros(16, dtype=np.float32)

        for i in range(8):
            err = t_cc[i] - self.t_target
            if err > self.deadband:
                # Hot zone: ramp UP cooling power for both AC and CU
                du[i] = self.ramp_rate
                du[8 + i] = self.ramp_rate
            elif err < -self.deadband:
                # Cold zone: ramp DOWN cooling power
                du[i] = -self.ramp_rate
                du[8 + i] = -self.ramp_rate
            else:
                # Within deadband: hold current actuator output
                du[i] = 0.0
                du[8 + i] = 0.0

        # Enforce delta limits [-1, 1] and actuator saturation [-clip_power, clip_power]
        du = np.clip(du, -1.0, 1.0)
        du = np.clip(du, -self.clip_power - p_curr, self.clip_power - p_curr)
        return du.astype(np.float32), None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Autonomous Receding-Horizon Economic MPC Controller
# ─────────────────────────────────────────────────────────────────────────────
class AutonomousMPCController:
    """
    Pure model-based economic receding-horizon controller.
    Solves a multi-step QP online at each time step using OSQP.

    Minimizes:
        J = w_energy * sum(P_k) + w_smooth * sum(||P_k - P_{k-1}||^2)
    Subject to:
        T_celCC[k] <= T_max  (rack safety)
        P_k in [-clip_power, clip_power] (actuator saturation)
        ||P_k - P_{k-1}|| <= 1.0 (slew rate limit)
    """

    def __init__(
        self,
        model: LinearDynamicsModel,
        col_names: list[str],
        horizon: int = 5,
        t_max: float = T_CELCC_MAX,
        w_energy: float = 1.0,
        w_smooth: float = 0.5,
        clip_power: float = CLIP_POWER,
    ):
        self.model      = model
        self.N          = horizon
        self.t_max      = t_max
        self.clip_power = clip_power
        self.n_u        = 16
        self.n_x        = 25
        self.n_celcc    = 8
        self.n_var      = self.N * self.n_u

        idx = {c: i for i, c in enumerate(col_names)}
        self.t_state_idx = [idx[c] for c in T_STATE_COLS]
        self.power_idx   = [idx[c] for c in POWER_COLS]

        A, B = model.A, model.B
        N = self.N

        # Unforced response coefficients: T_drift_k = A^k T0 + sum_{j=0}^{k-1} A^j c
        # Forced response sensitivity matrix S: (N*n_x, N*n_u)
        S = np.zeros((N * self.n_x, N * self.n_u), dtype=np.float32)
        Apow = [np.eye(self.n_x, dtype=np.float32)]
        for _ in range(N):
            Apow.append(Apow[-1] @ A)

        for k in range(N):
            for j in range(k + 1):
                S[k * self.n_x:(k + 1) * self.n_x, j * self.n_u:(j + 1) * self.n_u] = Apow[k - j] @ B

        # Selector matrix for T_celCC (rows 16..23 of 25-dim state)
        C_sel = np.zeros((self.n_celcc, self.n_x), dtype=np.float32)
        for r, c in enumerate(range(16, 24)):
            C_sel[r, c] = 1.0
        C_block = np.kron(np.eye(N, dtype=np.float32), C_sel)
        self.CS = C_block @ S
        self.C_block = C_block

        # Smoothness difference matrix D: D @ P = [P0; P1-P0; P2-P1; ...]
        D = np.zeros((N * self.n_u, N * self.n_u), dtype=np.float32)
        for k in range(N):
            D[k * self.n_u:(k + 1) * self.n_u, k * self.n_u:(k + 1) * self.n_u] = np.eye(self.n_u)
            if k > 0:
                D[k * self.n_u:(k + 1) * self.n_u, (k - 1) * self.n_u:k * self.n_u] = -np.eye(self.n_u)
        self.D = D

        # Objective matrices: 0.5 P^T H P + q^T P
        self.H = 2 * w_smooth * sp.csc_matrix(D.T @ D) + 1e-4 * sp.eye(self.n_var, format="csc")
        self.q = w_energy * np.ones(self.n_var, dtype=np.float32)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        T_state = obs[self.t_state_idx]
        P_curr  = obs[self.power_idx]

        # Natural drift trajectory (unforced, P = 0)
        T_drift = []
        T_curr = T_state.copy()
        for _ in range(self.N):
            T_curr = self.model.A @ T_curr + self.model.c
            T_drift.append(T_curr)
        T_drift_vec = np.concatenate(T_drift)
        T_celcc_drift = self.C_block @ T_drift_vec

        # Constraints:
        # 1. Thermal constraint: CS @ P <= t_max - T_celcc_drift
        u_temp = self.t_max - T_celcc_drift
        l_temp = np.full(len(u_temp), -np.inf)

        # 2. Actuator saturation box: P in [-clip_power, clip_power]
        l_box = np.full(self.n_var, -self.clip_power)
        u_box = np.full(self.n_var, self.clip_power)

        # 3. Slew rate bounds from current power P_curr
        l_slew = np.full(self.n_var, -1.0)
        u_slew = np.full(self.n_var, 1.0)
        l_slew[:self.n_u] = -1.0 + P_curr
        u_slew[:self.n_u] = 1.0 + P_curr

        A_qp = sp.vstack([sp.csc_matrix(self.CS), sp.eye(self.n_var, format="csc"), sp.csc_matrix(self.D)], format="csc")
        l_qp = np.concatenate([l_temp, l_box, l_slew])
        u_qp = np.concatenate([u_temp, u_box, u_slew])

        solver = osqp.OSQP()
        solver.setup(self.H, self.q, A_qp, l_qp, u_qp, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
        res = solver.solve()

        if res.info.status in ("solved", "solved_inaccurate"):
            P_opt0 = res.x[:self.n_u]
            du = P_opt0 - P_curr
        else:
            # Emergency safety fallback: ramp up cooling
            du = np.full(self.n_u, 0.25, dtype=np.float32)

        du = np.clip(du, -1.0, 1.0)
        du = np.clip(du, -self.clip_power - P_curr, self.clip_power - P_curr)
        return du.astype(np.float32), None

