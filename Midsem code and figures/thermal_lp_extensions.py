"""
thermal_lp_extensions.py
────────────────────────
Advanced extensions for the Linear Programming (LP) workload allocation track:

1. Silicon Leakage Power Model:
   Adds temperature-dependent leakage: P_leak_j = beta_j * (T_{j,t} - T_ref).
   Remains strictly a linear program (LP), but introduces an interior optimal
   temperature setpoint where cooling savings balance leakage power increases.

2. Receding-Horizon Rolling LP (Online MPC for Workloads):
   Re-solves M2 dynamically over a sliding prediction horizon H (e.g. H = 12 steps = 2 mins)
   without assuming 30-minute perfect foresight of incoming LLM inference traffic.

3. Stochastic Demand Spikes & Robustness Evaluation:
   Tests rolling LP performance under unexpected LLM query surges.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

# Add project root to path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_lp import instance, demand_profile, zoh_a, simulate, violation_stats, solve_m2, GPU_NAMES, WL_NAMES, DT, NT, DELTA

# ── Leakage parameters ────────────────────────────────────────────────────────
# beta_j: W per degree C above T_ref (25 C)
# High-power GPUs (4090, A100) have ~0.8 to 1.5 W/degC leakage slope
BETA_LEAK = np.array([1.2, 1.2, 0.9, 0.9, 1.5, 0.6], dtype=float)
T_REF     = 25.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model M2 with Temperature-Dependent Silicon Leakage Power
# ─────────────────────────────────────────────────────────────────────────────
def build_m2_leakage(inst, D, T_amb, T0, dt=DT, delta=DELTA, beta=BETA_LEAK, t_ref=T_REF):
    """
    Formulates M2 with leakage power:
    P_{j,t} = P_idle_j + sum_i gamma_ij x_ijt + beta_j (T_jt - T_ref).
    Objective: min sum_t dt * [ sum_ij gamma_ij x_ijt + sum_j beta_j T_jt ]
    Thermal dynamics update incorporates leakage feedback into effective conductance:
    dT/dt = (P_dyn + P_idle + beta(T - T_ref) - k(T - T_amb)) / mc
          = (P_dyn + P_idle - beta*T_ref + k*T_amb - (k - beta) T) / mc
    Effective conductance k_eff = k - beta.
    Exact ZOH decay: a_eff = exp(-(k - beta) * dt / mc).
    """
    g = inst["GAMMA"]
    m, n = g.shape
    nt = D.shape[1]
    nx, nT = m * n * nt, n * nt

    xi = lambda i, j, t: t * m * n + i * n + j
    Ti = lambda j, t: nx + t * n + j

    k_eff = inst["K"] - beta
    a_eff = np.exp(-k_eff * dt / inst["MC"])

    # Objective: dynamic energy + leakage energy
    c = np.zeros(nx + nT)
    for t in range(nt):
        for i in range(m):
            for j in range(n):
                c[xi(i, j, t)] = g[i, j] * dt
        for j in range(n):
            c[Ti(j, t)] = beta[j] * dt

    rows, cols, vals, b_eq = [], [], [], []
    r = 0

    # Demand equalities: sum_j x_ijt = D_it
    for t in range(nt):
        for i in range(m):
            for j in range(n):
                rows.append(r); cols.append(xi(i, j, t)); vals.append(1.0)
            b_eq.append(D[i, t]); r += 1

    # Thermal dynamics: T_jt - a_eff_j T_{j,t-1} - (1 - a_eff_j) sum_i (gamma_ij / k_eff_j) x_ijt = rhs
    for t in range(nt):
        for j in range(n):
            rows.append(r); cols.append(Ti(j, t)); vals.append(1.0)
            if t > 0:
                rows.append(r); cols.append(Ti(j, t - 1)); vals.append(-a_eff[j])
            for i in range(m):
                rows.append(r); cols.append(xi(i, j, t))
                vals.append(-(1.0 - a_eff[j]) * g[i, j] / k_eff[j])

            rhs = (1.0 - a_eff[j]) * (T_amb * inst["K"][j] + inst["P_IDLE"][j] - beta[j] * t_ref) / k_eff[j]
            if t == 0:
                rhs += a_eff[j] * T0[j]
            b_eq.append(rhs); r += 1

    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(r, nx + nT))

    # Capacity constraints: sum_i x_ijt <= C_j
    rows_u, cols_u, vals_u = [], [], []
    ru = 0
    for t in range(nt):
        for j in range(n):
            for i in range(m):
                rows_u.append(ru); cols_u.append(xi(i, j, t)); vals_u.append(1.0)
            ru += 1

    A_ub = sp.csr_matrix((vals_u, (rows_u, cols_u)), shape=(ru, nx + nT))
    b_ub = np.tile(inst["C"], nt)

    lb = np.concatenate([np.zeros(nx), np.full(nT, -np.inf)])
    ub = np.concatenate([np.full(nx, np.inf), np.tile(inst["T_THR"] - delta, nt)])

    return c, A_ub, b_ub, A_eq, np.array(b_eq), lb, ub, (m, n, nt)


def solve_m2_leakage(inst, D, T_amb, T0=None, dt=DT, delta=DELTA):
    if T0 is None:
        T0 = np.full(len(inst["C"]), T_amb + 25.0)
    c, A_ub, b_ub, A_eq, b_eq, lb, ub, (m, n, nt) = build_m2_leakage(inst, D, T_amb, T0, dt, delta)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(lb, ub)), method="highs-ds")

    out = dict(status=res.status, res=res, T0=T0)
    if res.status == 0:
        nx = m * n * nt
        out["x"] = res.x[:nx].reshape(nt, m, n)
        out["T"] = res.x[nx:].reshape(nt, n)
        out["E_total"] = res.fun + (inst["P_IDLE"].sum() - (BETA_LEAK * T_REF).sum()) * dt * nt
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Receding-Horizon Rolling LP (Workload-Level MPC)
# ─────────────────────────────────────────────────────────────────────────────
def run_rolling_horizon_lp(inst, D_actual, T_amb, horizon_steps=12, dt=DT, delta=DELTA):
    """
    Simulates real-time receding horizon LP execution:
    At each step t:
      - Takes current temperatures T_t
      - Solves M2 over sliding window [t, t + horizon_steps]
      - Applies step 0 allocation x_t
      - Advances true thermal simulator
    """
    nt = D_actual.shape[1]
    m, n = inst["GAMMA"].shape
    a = zoh_a(inst, dt)

    T_curr = np.full(n, T_amb + 25.0)
    x_applied = np.zeros((nt, m, n))
    T_recorded = np.zeros((nt, n))
    solve_times = []

    for t in range(nt):
        # Sliding demand window
        t_end = min(t + horizon_steps, nt)
        D_window = D_actual[:, t:t_end]
        if D_window.shape[1] < horizon_steps:
            # Pad with last known demand
            pad_len = horizon_steps - D_window.shape[1]
            D_window = np.hstack([D_window, np.tile(D_window[:, -1:], pad_len)])

        t_start = time.perf_counter()
        sol = solve_m2(inst, D_window, T_amb, T0=T_curr, dt=dt, delta=delta)
        solve_times.append(time.perf_counter() - t_start)

        if sol["status"] == 0:
            x_step = sol["x"][0]
        else:
            # Infeasible window: fallback to capacity-proportional
            share = inst["C"] / inst["C"].sum()
            x_step = np.outer(D_actual[:, t], share)

        x_applied[t] = x_step

        # Step true physical thermal simulator
        P = inst["P_IDLE"] + np.einsum("ij,ij->j", inst["GAMMA"], x_step)
        T_curr = a * T_curr + (1.0 - a) * (T_amb + P / inst["K"])
        T_recorded[t] = T_curr

    return {
        "x": x_applied,
        "T": T_recorded,
        "solve_time_mean_ms": float(np.mean(solve_times) * 1000.0),
        "solve_time_p95_ms": float(np.percentile(solve_times, 95) * 1000.0),
        "stats": violation_stats(inst, T_recorded, dt=dt),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Evaluation of Extensions
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  RUNNING ADVANCED LP EXTENSIONS (LEAKAGE & ROLLING HORIZON)")
    print("=" * 65)

    inst = instance()
    D = demand_profile()
    T_amb = 41.0

    # 1. Leakage Setpoint Sweep
    print("\n1. Sweeping ambient setpoint with temperature-dependent leakage …")
    T_sweep = np.arange(25.0, 48.0, 1.0)
    facility_energy_no_leak = []
    facility_energy_with_leak = []
    R0 = 0.6  # cooling energy ratio at 25 C
    s = 0.05  # 5% saving per degree

    for Ta in T_sweep:
        r_base = solve_m2(inst, D, Ta)
        r_leak = solve_m2_leakage(inst, D, Ta)

        if r_base["status"] == 0 and r_leak["status"] == 0:
            # Facility = IT * (1 + R0 * (1 - s)^(Ta - 25))
            cool_factor = (1.0 + R0 * (1.0 - s) ** (Ta - 25.0))
            facility_energy_no_leak.append(r_base["E_gpu"] * cool_factor / 1e6)
            facility_energy_with_leak.append(r_leak["E_total"] * cool_factor / 1e6)
        else:
            facility_energy_no_leak.append(np.nan)
            facility_energy_with_leak.append(np.nan)

    # Plot optimal setpoint comparison
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(T_sweep, facility_energy_no_leak, label="Without Leakage (M2 baseline)", color="#2a78d6", linestyle="--", marker="o", ms=3)
    ax.plot(T_sweep, facility_energy_with_leak, label="With Leakage ($P_{\\text{leak}} = \\beta(T - T_{\\text{ref}})$)", color="#eb6834", marker="s", ms=3.5, linewidth=2)

    # Find interior minimum
    valid_leak = [(T, e) for T, e in zip(T_sweep, facility_energy_with_leak) if np.isfinite(e)]
    if valid_leak:
        best_T, best_E = min(valid_leak, key=lambda x: x[1])
        ax.scatter([best_T], [best_E], color="#008300", s=80, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(f"Optimal Setpoint\n$T^* = {best_T:.0f}$ °C", (best_T, best_E),
                    xytext=(best_T - 5, best_E + 0.1), fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c3c2b7", lw=0.6),
                    arrowprops=dict(arrowstyle="->", color="#008300", lw=1.2))

    ax.set_xlabel("Ambient Setpoint $T_{\\text{amb}}$ (°C)", fontsize=10)
    ax.set_ylabel("Total Facility Energy (MJ)", fontsize=10)
    ax.set_title("Interior Optimal Setpoint Under Silicon Leakage", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig7_leakage_setpoint_optimum.png", dpi=200)
    plt.close(fig)
    print("  → Saved figures/fig7_leakage_setpoint_optimum.png")

    # 2. Rolling Horizon LP Test
    print("\n2. Benchmarking Receding-Horizon Rolling LP (Workload MPC) …")
    rolling_results = run_rolling_horizon_lp(inst, D, T_amb, horizon_steps=12)
    print(f"  Mean solve time: {rolling_results['solve_time_mean_ms']:.2f} ms")
    print(f"  95th percentile solve time: {rolling_results['solve_time_p95_ms']:.2f} ms")
    print(f"  Violation fraction: {rolling_results['stats']['frac_over']*100:.2f}%")
    print(f"  Peak temperature: {rolling_results['stats']['peak']:.2f} °C")

    # 3. Stochastic Burst Spikes Test
    print("\n3. Testing robustness against unanticipated stochastic demand spikes …")
    D_spike = D.copy()
    D_spike[:, 80:95] *= 1.40  # 40% surge at minute 13
    rolling_spike = run_rolling_horizon_lp(inst, D_spike, T_amb, horizon_steps=12)
    print(f"  Spike test violation fraction: {rolling_spike['stats']['frac_over']*100:.2f}%")
    print(f"  Spike test peak temperature: {rolling_spike['stats']['peak']:.2f} °C")

    # Export results
    results_out = {
        "best_setpoint_with_leakage": float(best_T) if valid_leak else None,
        "rolling_horizon_lp": {
            "horizon_steps": 12,
            "mean_solve_ms": rolling_results["solve_time_mean_ms"],
            "p95_solve_ms": rolling_results["solve_time_p95_ms"],
            "violations_pct": rolling_results["stats"]["frac_over"] * 100.0,
            "peak_temp": rolling_results["stats"]["peak"],
        },
        "stochastic_spike_test": {
            "violations_pct": rolling_spike["stats"]["frac_over"] * 100.0,
            "peak_temp": rolling_spike["stats"]["peak"],
        }
    }
    (HERE / "extensions_results.json").write_text(json.dumps(results_out, indent=2))
    print(f"  → Saved extensions_results.json")
    print("\nAll advanced LP extensions completed successfully!")


if __name__ == "__main__":
    main()

