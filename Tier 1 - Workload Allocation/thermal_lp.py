"""
thermal_lp.py — LP models for thermal-aware workload allocation (ME F320 project)

M1 : single-period (steady-state) allocation LP
M2 : multi-period allocation LP with GPU temperature as a decision variable
     (exact zero-order-hold discretisation of a lumped-capacitance thermal model)

All parameters below are SYNTHETIC but chosen to be physically plausible
(order-of-magnitude consistent with consumer / data-centre GPUs). They are NOT
measured values.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

# ----------------------------------------------------------------------------
# 1. Synthetic data-centre instance
# ----------------------------------------------------------------------------
GPU_NAMES = ["G1 4090-AC", "G2 4090-WC", "G3 3090-AC", "G4 3090-WC", "G5 A100-class", "G6 Legacy-AC"]
WL_NAMES = ["W1 Prefill", "W2 Chat decode", "W3 Summarise", "W4 Code gen", "W5 Embedding", "W6 Long-ctx RAG"]

C      = np.array([1.6, 1.6, 1.0, 1.0, 2.0, 0.8])          # capacity, work-units/s
P_IDLE = np.array([30., 32., 25., 27., 50., 30.])         # idle power, W
K      = np.array([7.2, 12.0, 6.0, 10.0, 8.5, 5.6])        # cooling conductance, W/degC
MC     = np.array([1100., 2000., 900., 1600., 1400., 750.])  # thermal capacitance, J/degC
T_THR  = np.array([83., 83., 83., 83., 85., 83.])          # throttling threshold, degC

E_BASE = np.array([230., 235., 300., 305., 190., 380.])    # J per unit, reference workload
A_WL   = np.array([1.10, 0.80, 1.00, 0.95, 0.70, 1.05])    # workload intensity
# affinity: compute-bound (W1,W4,W6) favour Ada (4090); memory-bound (W2,W5) favour A100 (HBM)
AFF = np.ones((6, 6))
for i in (0, 3, 5):
    AFF[i, [0, 1]] = 0.92; AFF[i, 4] = 1.05
for i in (1, 4):
    AFF[i, 4] = 0.85; AFF[i, [2, 3, 5]] = 1.08
GAMMA = np.rint(E_BASE[None, :] * A_WL[:, None] * AFF)     # J/unit, shape (m, n), integers

SHARE = np.array([0.20, 0.25, 0.15, 0.15, 0.10, 0.15])
DELTA = 2.0          # safety margin below throttle threshold, degC
DT    = 10.0         # time step, s
NT    = 180          # 30 minutes


def demand_profile(nt=NT, seed=7, base=5.0):
    """Total demand: slow wave + two short bursts; split by workload share with mild noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(nt)
    tot = base + 0.6 * np.sin(2 * np.pi * t / nt)
    tot[50:62] += 2.0      # 2-minute burst
    tot[120:129] += 2.3    # 1.5-minute burst
    D = SHARE[:, None] * tot[None, :] * (1 + 0.03 * rng.standard_normal((len(SHARE), nt)))
    return np.clip(D, 0, None)


def instance(n_gpu=6, n_wl=6):
    return dict(C=C[:n_gpu], P_IDLE=P_IDLE[:n_gpu], K=K[:n_gpu], MC=MC[:n_gpu],
                T_THR=T_THR[:n_gpu], GAMMA=GAMMA[:n_wl, :n_gpu])


# ----------------------------------------------------------------------------
# 2. Model M1 — steady-state single-period LP (rate form, objective in W)
#    min  sum_ij g_ij x_ij
#    s.t. sum_j x_ij = D_i ; sum_i x_ij <= C_j ;
#         sum_i g_ij x_ij <= Q_j = k_j (T_thr_j - delta - T_amb) - P_idle_j
# ----------------------------------------------------------------------------
def heat_budget(inst, T_amb, delta=DELTA):
    return inst["K"] * (inst["T_THR"] - delta - T_amb) - inst["P_IDLE"]


def build_m1(inst, D, T_amb, thermal=True, delta=DELTA):
    g = inst["GAMMA"]; m, n = g.shape
    c = g.flatten()
    A_eq = np.zeros((m, m * n)); b_eq = np.asarray(D, float)
    for i in range(m):
        A_eq[i, i * n:(i + 1) * n] = 1
    A_cap = np.zeros((n, m * n))
    A_th = np.zeros((n, m * n))
    for j in range(n):
        A_cap[j, j::n] = 1
        A_th[j, j::n] = g[:, j]
    A_ub, b_ub = [A_cap], [inst["C"]]
    if thermal:
        A_ub.append(A_th); b_ub.append(heat_budget(inst, T_amb, delta))
    return c, np.vstack(A_ub), np.concatenate(b_ub), A_eq, b_eq


def solve_m1(inst, D, T_amb, thermal=True, method="highs-ds", delta=DELTA):
    c, A_ub, b_ub, A_eq, b_eq = build_m1(inst, D, T_amb, thermal, delta)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method=method)
    m, n = inst["GAMMA"].shape
    out = dict(status=res.status, res=res)
    if res.status == 0:
        out["x"] = res.x.reshape(m, n)
        out["Z"] = res.fun                             # dynamic power, W
        out["P_total"] = res.fun + inst["P_IDLE"].sum()
        out["y_eq"] = res.eqlin.marginals
        out["y_cap"] = res.ineqlin.marginals[:n]
        out["y_th"] = res.ineqlin.marginals[n:] if thermal else np.zeros(n)
        # dual objective (lower bounds are 0, contribute nothing)
        out["dual_obj"] = b_eq @ res.eqlin.marginals + b_ub @ res.ineqlin.marginals
    return out


# ----------------------------------------------------------------------------
# 3. Model M2 — multi-period LP with thermal state (exact ZOH discretisation)
#    T_{j,t+1} = a_j T_{j,t} + (1-a_j) [T_amb + (P_idle_j + sum_i g_ij x_ijt)/k_j],
#    a_j = exp(-k_j dt / (m c)_j). Affine in (x, T) -> constraint stays linear.
# ----------------------------------------------------------------------------
def zoh_a(inst, dt=DT):
    return np.exp(-inst["K"] * dt / inst["MC"])


def build_m2(inst, D, T_amb, T0, dt=DT, delta=DELTA):
    g = inst["GAMMA"]; m, n = g.shape; nt = D.shape[1]
    nx, nT = m * n * nt, n * nt
    xi = lambda i, j, t: t * m * n + i * n + j
    Ti = lambda j, t: nx + t * n + j                   # temperature AFTER step t (t = 0..nt-1)
    a = zoh_a(inst, dt)

    c = np.zeros(nx + nT)
    for t in range(nt):
        for i in range(m):
            for j in range(n):
                c[xi(i, j, t)] = g[i, j] * dt          # J

    rows, cols, vals, b_eq = [], [], [], []
    r = 0
    for t in range(nt):                                 # demand
        for i in range(m):
            for j in range(n):
                rows.append(r); cols.append(xi(i, j, t)); vals.append(1.0)
            b_eq.append(D[i, t]); r += 1
    for t in range(nt):                                 # thermal dynamics
        for j in range(n):
            rows.append(r); cols.append(Ti(j, t)); vals.append(1.0)
            if t > 0:
                rows.append(r); cols.append(Ti(j, t - 1)); vals.append(-a[j])
            for i in range(m):
                rows.append(r); cols.append(xi(i, j, t)); vals.append(-(1 - a[j]) * g[i, j] / inst["K"][j])
            rhs = (1 - a[j]) * (T_amb + inst["P_IDLE"][j] / inst["K"][j])
            if t == 0:
                rhs += a[j] * T0[j]
            b_eq.append(rhs); r += 1
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(r, nx + nT))

    rows, cols, vals = [], [], []
    r = 0
    for t in range(nt):                                 # capacity
        for j in range(n):
            for i in range(m):
                rows.append(r); cols.append(xi(i, j, t)); vals.append(1.0)
            r += 1
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(r, nx + nT))
    b_ub = np.tile(inst["C"], nt)

    lb = np.concatenate([np.zeros(nx), np.full(nT, -np.inf)])
    ub = np.concatenate([np.full(nx, np.inf), np.tile(inst["T_THR"] - delta, nt)])
    return c, A_ub, b_ub, A_eq, np.array(b_eq), lb, ub, (m, n, nt)


def solve_m2(inst, D, T_amb, T0=None, method="highs-ds", dt=DT, delta=DELTA):
    if T0 is None:
        T0 = np.full(len(inst["C"]), T_amb + 25.0)
    c, A_ub, b_ub, A_eq, b_eq, lb, ub, (m, n, nt) = build_m2(inst, D, T_amb, T0, dt, delta)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(lb, ub)), method=method)
    out = dict(status=res.status, res=res, T0=T0)
    if res.status == 0:
        nx = m * n * nt
        out["x"] = res.x[:nx].reshape(nt, m, n)
        out["T"] = res.x[nx:].reshape(nt, n)
        out["E_dyn"] = res.fun
        out["E_gpu"] = res.fun + inst["P_IDLE"].sum() * dt * nt
        out["mu_T"] = res.upper.marginals[nx:].reshape(nt, n)   # J per degC (<=0)
        finite_ub = np.isfinite(ub)
        out["dual_obj"] = (b_eq @ res.eqlin.marginals + b_ub @ res.ineqlin.marginals
                           + ub[finite_ub] @ res.upper.marginals[finite_ub])
    return out


# ----------------------------------------------------------------------------
# 4. Thermal simulator + thermal-agnostic baselines
# ----------------------------------------------------------------------------
def simulate(inst, x, T_amb, T0, dt=DT):
    """x: (nt, m, n) allocation. Returns temperature after each step (nt, n) and GPU energy (J)."""
    a = zoh_a(inst, dt); g = inst["GAMMA"]
    nt = x.shape[0]; T = np.zeros((nt, len(a))); Tc = np.array(T0, float)
    E = 0.0
    for t in range(nt):
        P = inst["P_IDLE"] + np.einsum("ij,ij->j", g, x[t])
        Tc = a * Tc + (1 - a) * (T_amb + P / inst["K"])
        T[t] = Tc; E += P.sum() * dt
    return T, E


def baseline_capprop(inst, D):
    share = inst["C"] / inst["C"].sum()
    return np.einsum("it,j->tij", D, share)


def baseline_greedy(inst, D):
    g = inst["GAMMA"]; m, n = g.shape; nt = D.shape[1]
    order = np.dstack(np.unravel_index(np.argsort(g, axis=None), g.shape))[0]
    x = np.zeros((nt, m, n))
    for t in range(nt):
        rem_d = D[:, t].copy(); rem_c = inst["C"].copy()
        for i, j in order:
            q = min(rem_d[i], rem_c[j])
            x[t, i, j] = q; rem_d[i] -= q; rem_c[j] -= q
    return x


def baseline_myopic(inst, D, T_amb, T0, dt=DT, delta=DELTA):
    """Thermal-aware but myopic greedy: each step, fill cheapest (i,j) pairs subject to
    capacity AND a one-step thermal budget that keeps T_{t+1} <= T_thr - delta.
    If demand cannot be placed within budgets, the remainder goes to the GPU with the
    most spare capacity (counted as 'forced' work, which may violate the limit)."""
    g = inst["GAMMA"]; m, n = g.shape; nt = D.shape[1]
    a = zoh_a(inst, dt); T_lim = inst["T_THR"] - delta
    order = np.dstack(np.unravel_index(np.argsort(g, axis=None), g.shape))[0]
    x = np.zeros((nt, m, n)); Tc = np.array(T0, float); forced = 0.0
    for t in range(nt):
        P_budget = ((T_lim - a * Tc) / (1 - a) - T_amb) * inst["K"] - inst["P_IDLE"]
        P_budget = np.clip(P_budget, 0, None)
        rem_d = D[:, t].copy(); rem_c = inst["C"].copy()
        for i, j in order:
            q = min(rem_d[i], rem_c[j], P_budget[j] / g[i, j])
            q = max(q, 0.0)
            x[t, i, j] += q; rem_d[i] -= q; rem_c[j] -= q; P_budget[j] -= q * g[i, j]
        for i in range(m):                      # forced placement of leftover demand
            while rem_d[i] > 1e-12:
                j = int(np.argmax(rem_c)); q = min(rem_d[i], rem_c[j])
                if q <= 0: break
                x[t, i, j] += q; rem_d[i] -= q; rem_c[j] -= q; forced += q * dt
        P = inst["P_IDLE"] + np.einsum("ij,ij->j", g, x[t])
        Tc = a * Tc + (1 - a) * (T_amb + P / inst["K"])
    return x, forced


def violation_stats(inst, T, dt=DT):
    over = T - inst["T_THR"][None, :]
    return dict(steps_over=int((over > 0).sum()),
                frac_over=float((over > 0).mean()),
                degree_seconds=float(np.clip(over, 0, None).sum() * dt),
                peak=float(T.max()))
