"""
run_midsem.py — validation suite + preliminary results + figures for the midsem report.
Writes: results.json and figures/*.png
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from scipy.optimize import linprog
import cvxopt
from cvxopt import matrix, spmatrix, solvers
import scipy.sparse as sp

from thermal_lp import *

os.makedirs("figures", exist_ok=True)
R = {}

# ---------------------------------------------------------------- style
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
})

inst = instance()
D = demand_profile()
Dm = D.mean(1)
Dp = D[:, D.sum(0).argmax()]
R["instance"] = dict(gpus=GPU_NAMES, workloads=WL_NAMES, C=C.tolist(), P_idle=P_IDLE.tolist(),
                     k=K.tolist(), mc=MC.tolist(), T_thr=T_THR.tolist(), tau=(MC / K).round(0).tolist(),
                     gamma=GAMMA.astype(int).tolist(), delta=DELTA, dt=DT, NT=NT,
                     demand_total_min=float(D.sum(0).min()), demand_total_max=float(D.sum(0).max()),
                     demand_total_mean=float(D.sum(0).mean()), capacity_total=float(C.sum()),
                     n_vars_m2=int(D.size * len(C) + len(C) * NT))

# ======================================================================
# VALIDATION
# ======================================================================
V = {}

# V1 — hand-solvable 2x2 instance: graphical method / vertex enumeration vs simplex
g = np.array([[200., 260.], [150., 220.]]); Dh = np.array([1.0, 1.2]); Ch = np.array([1.5, 1.5]); Qh = np.array([220., 400.])
c = g.flatten()
A_eq = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], float)
A_ub = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [g[0, 0], 0, g[1, 0], 0], [0, g[0, 1], 0, g[1, 1]]])
b_ub = np.concatenate([Ch, Qh])
rh = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=Dh, bounds=(0, None), method="highs-ds")
# reduced 2-D problem in (u, v) = (x11, x21); x12 = D1-u, x22 = D2-v
# objective = const + (g11-g12) u + (g21-g22) v
lines = [  # a u + b v <= r
    (1, 1, Ch[0]),                                    # cap G1
    (-1, -1, Ch[1] - Dh.sum()),                       # cap G2
    (g[0, 0], g[1, 0], Qh[0]),                        # thermal G1
    (-g[0, 1], -g[1, 1], Qh[1] - g[0, 1] * Dh[0] - g[1, 1] * Dh[1]),  # thermal G2
    (-1, 0, 0), (0, -1, 0), (1, 0, Dh[0]), (0, 1, Dh[1]),
]
verts = []
for p in range(len(lines)):
    for q in range(p + 1, len(lines)):
        M = np.array([lines[p][:2], lines[q][:2]], float)
        if abs(np.linalg.det(M)) < 1e-12: continue
        uv = np.linalg.solve(M, [lines[p][2], lines[q][2]])
        if all(a * uv[0] + b * uv[1] <= r + 1e-9 for a, b, r in lines):
            verts.append(uv)
verts = np.unique(np.round(verts, 9), axis=0)
const = g[0, 1] * Dh[0] + g[1, 1] * Dh[1]
obj = lambda uv: const + (g[0, 0] - g[0, 1]) * uv[0] + (g[1, 0] - g[1, 1]) * uv[1]
vals = [obj(v) for v in verts]
best = verts[int(np.argmin(vals))]
V["V1_hand"] = dict(vertices=[[float(a), float(b), float(o)] for (a, b), o in zip(verts, vals)],
                    enum_opt=[float(best[0]), float(best[1])], enum_Z=float(min(vals)),
                    simplex_x=rh.x.round(6).tolist(), simplex_Z=float(rh.fun),
                    duals_eq=rh.eqlin.marginals.round(6).tolist(), duals_ub=rh.ineqlin.marginals.round(6).tolist())

# figure: feasible region of hand instance
fig, ax = plt.subplots(figsize=(3.6, 3.0))
uu, vv = np.meshgrid(np.linspace(-0.05, 1.05, 500), np.linspace(-0.05, 1.3, 500))
feas = np.ones_like(uu, bool)
for a, b, r in lines: feas &= a * uu + b * vv <= r + 1e-9
ax.contourf(uu, vv, feas, levels=[0.5, 1.5], colors=["#cde2fb"], alpha=0.9)
uline = np.linspace(-0.05, 1.05, 10)
ax.plot(uline, (Qh[0] - g[0, 0] * uline) / g[1, 0], color=SER[1], lw=1.4, label="Thermal G1 (binding)")
ax.plot(uline, Ch[0] - uline, color=SER[0], lw=1.4, ls="--", label="Capacity G1")
ax.axhline(Dh[1], color=SER[2], lw=1.4, ls=":", label=r"$x_{21}\leq D_2$")
for k_ in [0, 1]:
    zl = rh.fun + 15 * (k_ + 1)
    ax.plot(uline, (zl - const - (g[0, 0] - g[0, 1]) * uline) / (g[1, 0] - g[1, 1]), color=MUTED, lw=0.8, ls="-.")
ax.plot(uline, (rh.fun - const - (g[0, 0] - g[0, 1]) * uline) / (g[1, 0] - g[1, 1]), color=INK, lw=1.0, ls="-.", label="Iso-cost line at optimum")
ax.scatter(verts[:, 0], verts[:, 1], s=14, color=INK2, zorder=3)
ax.scatter([best[0]], [best[1]], s=40, color=SER[1], edgecolor="white", lw=1.5, zorder=4)
ax.annotate(f"optimum ({best[0]:.1f}, {best[1]:.1f})\nZ = {min(vals):.0f} W", best, xytext=(0.22, 0.88),
            fontsize=7.5, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=AXIS, lw=0.6),
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.3)
ax.set_xlabel(r"$x_{11}$ (units/s of W1 on G1)"); ax.set_ylabel(r"$x_{21}$ (units/s of W2 on G1)")
ax.legend(fontsize=6.5, loc="lower left")
fig.savefig("figures/fig1_hand_instance.png"); plt.close(fig)

# V2 / V3 — cross-solver agreement and duality gap (M1 and M2)
def cvx_solve(c, A_ub, b_ub, A_eq, b_eq, lb, ub):
    nvar = len(c)
    A_ub = sp.coo_matrix(A_ub); A_eq = sp.coo_matrix(A_eq)
    Gr, Gc, Gv, h = list(A_ub.row), list(A_ub.col), list(A_ub.data), list(b_ub)
    r0 = A_ub.shape[0]
    for k_ in range(nvar):
        if np.isfinite(lb[k_]): Gr.append(r0); Gc.append(k_); Gv.append(-1.0); h.append(-lb[k_]); r0 += 1
        if np.isfinite(ub[k_]): Gr.append(r0); Gc.append(k_); Gv.append(1.0); h.append(ub[k_]); r0 += 1
    G = spmatrix([float(v) for v in Gv], [int(v) for v in Gr], [int(v) for v in Gc], (int(r0), int(nvar)))
    A = spmatrix([float(v) for v in A_eq.data], [int(v) for v in A_eq.row], [int(v) for v in A_eq.col],
                 (int(A_eq.shape[0]), int(A_eq.shape[1])))
    solvers.options.update(show_progress=False, abstol=1e-7, reltol=1e-8, feastol=1e-8, maxiters=300)
    s = solvers.lp(matrix(np.asarray(c, float)), G, matrix(np.asarray(h, float)), A, matrix(np.asarray(b_eq, float)))
    return s["status"], float(s["primal objective"])

xs = {}
for name, Ta in [("M1_35", 35.0), ("M1_41", 41.0)]:
    cc, Au, bu, Ae, be = build_m1(inst, Dm, Ta)
    ds = linprog(cc, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=(0, None), method="highs-ds")
    ip = linprog(cc, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=(0, None), method="highs-ipm")
    st, cz = cvx_solve(cc, Au, bu, Ae, be, np.zeros(len(cc)), np.full(len(cc), np.inf))
    r = solve_m1(inst, Dm, Ta)
    xs[name] = dict(highs_ds=ds.fun, highs_ipm=ip.fun, cvxopt=cz, cvx_status=st,
                    dual_obj=r["dual_obj"], gap=abs(ds.fun - r["dual_obj"]),
                    ds_iters=int(ds.nit))
for name, Ta in [("M2_35", 35.0), ("M2_41", 41.0)]:
    T0 = np.full(len(C), Ta + 25.0)
    cc, Au, bu, Ae, be, lb, ub, _ = build_m2(inst, D, Ta, T0)
    bnds = list(zip(lb, ub))
    t0 = time.time(); ds = linprog(cc, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=bnds, method="highs-ds"); tds = time.time() - t0
    t0 = time.time(); ip = linprog(cc, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=bnds, method="highs-ipm"); tip = time.time() - t0
    t0 = time.time(); st, cz = cvx_solve(cc, Au, bu, Ae, be, lb, ub); tcv = time.time() - t0
    r = solve_m2(inst, D, Ta)
    xs[name] = dict(highs_ds=ds.fun, highs_ipm=ip.fun, cvxopt=cz, cvx_status=st,
                    dual_obj=r["dual_obj"], gap=abs(ds.fun - r["dual_obj"]),
                    t_ds=tds, t_ipm=tip, t_cvx=tcv, ds_iters=int(ds.nit))
for v in xs.values():
    v["max_rel_dev"] = max(abs(v["highs_ipm"] - v["highs_ds"]), abs(v["cvxopt"] - v["highs_ds"])) / abs(v["highs_ds"])
    v["rel_gap"] = v["gap"] / abs(v["highs_ds"])
V["V2_V3_cross_solver"] = xs

# V4 — shadow price vs finite difference
fd = {}
Ta = 41.0
r0 = solve_m1(inst, Dm, Ta)
jb = int(np.argmin(r0["y_th"]))  # most negative dual = most binding
h = 0.05
inst2 = instance(); inst2["T_THR"] = inst2["T_THR"].copy(); inst2["T_THR"][jb] += h
r1 = solve_m1(inst2, Dm, Ta)
fd["M1"] = dict(gpu=GPU_NAMES[jb], dual_per_W=float(r0["y_th"][jb]),
                dual_per_degC=float(r0["y_th"][jb] * K[jb]), fd_per_degC=float((r1["Z"] - r0["Z"]) / h))
r0 = solve_m2(inst, D, Ta)
musum = r0["mu_T"].sum(0)
jb2 = int(np.argmin(musum))
inst2 = instance(); inst2["T_THR"] = inst2["T_THR"].copy(); inst2["T_THR"][jb2] += h
r1 = solve_m2(inst2, D, Ta, T0=r0["T0"])
fd["M2"] = dict(gpu=GPU_NAMES[jb2], dual_sum_J_per_degC=float(musum[jb2]), fd_J_per_degC=float((r1["E_dyn"] - r0["E_dyn"]) / h))
V["V4_shadow_fd"] = fd

# V5 — transportation reduction vs network simplex (networkx), thermal constraints removed
r_nt = solve_m1(inst, Dm, 35.0, thermal=False)
S = 100000
G_ = nx.DiGraph()
dem_int = np.rint(Dm * S).astype(int)
cap_int = np.rint(C * S).astype(int)
G_.add_node("src", demand=-int(dem_int.sum()))
G_.add_node("sink", demand=int(dem_int.sum()))
for i in range(len(Dm)):
    G_.add_edge("src", f"w{i}", capacity=int(dem_int[i]), weight=0)
    for j in range(len(C)):
        G_.add_edge(f"w{i}", f"g{j}", capacity=int(dem_int.sum()), weight=int(GAMMA[i, j]))
for j in range(len(C)):
    G_.add_edge(f"g{j}", "sink", capacity=int(cap_int[j]), weight=0)
flow_cost, _ = nx.network_simplex(G_)
V["V5_transport"] = dict(lp_Z=float(r_nt["Z"]), network_simplex_Z=flow_cost / S,
                         rel_dev=abs(r_nt["Z"] - flow_cost / S) / r_nt["Z"])

# V6 — thermal model: ZOH vs analytical solution, Euler error, steady state
j = 0; P = 400.0; Ta = 35.0; T0 = 40.0
tau = MC[j] / K[j]; Tss = Ta + (P_IDLE[j] + P) / K[j]
th = {}
for dt in [5.0, 10.0, 30.0]:
    n = int(1500 / dt); t = dt * np.arange(1, n + 1)
    exact = Tss + (T0 - Tss) * np.exp(-t / tau)
    a = np.exp(-K[j] * dt / MC[j]); Tz, Te = T0, T0; ez, ee = 0, 0
    for k_ in range(n):
        Tz = a * Tz + (1 - a) * Tss
        Te = Te + dt / MC[j] * ((P_IDLE[j] + P) - K[j] * (Te - Ta))
        ez = max(ez, abs(Tz - exact[k_])); ee = max(ee, abs(Te - exact[k_]))
    th[f"dt{int(dt)}"] = dict(zoh_max_err=ez, euler_max_err=ee)
th["steady_state_analytic"] = Tss; th["final_zoh"] = float(Tz); th["tau_s"] = float(tau)
V["V6_thermal"] = th

# V7 — identical-GPU limiting case
inst_id = dict(C=np.full(4, 2.0), P_IDLE=np.full(4, 30.), K=np.full(4, 50.), MC=np.full(4, 1000.),
               T_THR=np.full(4, 83.), GAMMA=np.repeat(np.array([[200.], [150.], [250.]]), 4, axis=1))
Did = np.array([1.5, 2.0, 1.0])
rid = solve_m1(inst_id, Did, 30.0)
V["V7_identical"] = dict(lp_Z=float(rid["Z"]), analytic=float((Did * np.array([200, 150, 250])).sum()))

# V8 — infeasibility detection; V9 — LP temperature vs independent re-simulation
r_inf = solve_m2(inst, D, 70.0)
r41 = solve_m2(inst, D, 41.0)
Tsim, Esim = simulate(inst, r41["x"], 41.0, r41["T0"])
V["V8_infeasible_status"] = int(r_inf["status"])
V["V9_resim"] = dict(max_T_diff=float(np.abs(Tsim - r41["T"]).max()), E_diff=float(abs(Esim - r41["E_gpu"])))
R["validation"] = V

# ======================================================================
# PRELIMINARY RESULTS
# ======================================================================
P_ = {}

# R1 — M1 at 35 and 41 degC, mean demand
for Ta in [35.0, 41.0]:
    r = solve_m1(inst, Dm, Ta)
    util = r["x"].sum(0) / C
    heat = (GAMMA * r["x"]).sum(0)
    Q = heat_budget(inst, Ta)
    P_[f"M1_{int(Ta)}"] = dict(P_total=r["P_total"], util=util.tolist(), heat=heat.tolist(), Q=Q.tolist(),
                               y_th=r["y_th"].tolist(), y_th_degC=(r["y_th"] * K).tolist(),
                               y_cap=r["y_cap"].tolist(), y_eq=r["y_eq"].tolist(), x=r["x"].tolist())
r_unc = solve_m1(inst, Dm, 41.0, thermal=False)
P_["M1_41_unconstrained_P"] = r_unc["P_total"]
P_["M1_41_unconstrained_util"] = (r_unc["x"].sum(0) / C).tolist()

# fig 2 — allocation heat maps, thermal-agnostic vs thermal-aware at 41 degC
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
xa = np.array(P_["M1_41"]["x"]); xu = r_unc["x"]
vmax = max(xa.max(), xu.max())
for ax, X, ttl in [(axs[0], xu, "Thermal constraints removed"), (axs[1], xa, "Thermal-aware M1, $T_{amb}$ = 41 °C")]:
    im = ax.imshow(X, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels([s.split()[0] for s in GPU_NAMES])
    ax.set_yticks(range(6)); ax.set_yticklabels(WL_NAMES, fontsize=7.5)
    ax.grid(False); ax.set_title(ttl, fontsize=8.5, color=INK)
    for i in range(6):
        for jj in range(6):
            if X[i, jj] > 0.005:
                ax.text(jj, i, f"{X[i, jj]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if X[i, jj] > 0.55 * vmax else INK)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("units/s", fontsize=7.5); cb.outline.set_visible(False)
fig.savefig("figures/fig2_allocation.png"); plt.close(fig)

# R2 — M2 vs baselines at 35 and 41 degC
cmp = {}
traj = {}
for Ta in [35.0, 41.0]:
    r = solve_m2(inst, D, Ta); T0 = r["T0"]
    rows = {}
    T_lp, E_lp = simulate(inst, r["x"], Ta, T0)
    rows["LP (M2)"] = dict(E=E_lp, **violation_stats(inst, T_lp), forced=0.0)
    for nm, x in [("Greedy lowest-energy", baseline_greedy(inst, D)),
                  ("Capacity-proportional", baseline_capprop(inst, D))]:
        T_, E_ = simulate(inst, x, Ta, T0); rows[nm] = dict(E=E_, **violation_stats(inst, T_), forced=0.0)
        if Ta == 41.0 and nm.startswith("Greedy"): traj["greedy"] = T_
    xm, forced = baseline_myopic(inst, D, Ta, T0)
    T_, E_ = simulate(inst, xm, Ta, T0); rows["Myopic thermal greedy"] = dict(E=E_, **violation_stats(inst, T_), forced=forced)
    for v in rows.values(): v["E_vs_LP_pct"] = 100 * (v["E"] / E_lp - 1)
    cmp[int(Ta)] = rows
    if Ta == 41.0: traj["lp"] = T_lp; traj["mu"] = r["mu_T"]
P_["M2_compare"] = cmp

# fig 3 — demand + temperature trajectories (LP vs greedy) at 41 degC
tmin = DT * np.arange(1, NT + 1) / 60
fig, axs = plt.subplots(3, 1, figsize=(6.6, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[0.7, 1, 1]))
axs[0].plot(tmin, D.sum(0), color=INK2, lw=1.4)
axs[0].axhline(C.sum(), color=MUTED, lw=0.8, ls="--"); axs[0].text(0.3, C.sum() - 0.45, "total capacity", fontsize=7, color=MUTED)
axs[0].set_ylabel("Demand\n(units/s)"); axs[0].set_ylim(3.5, 8.6)
show = [0, 2, 4, 5]
for ax, key, ttl in [(axs[1], "greedy", "Greedy lowest-energy (thermal-agnostic)"), (axs[2], "lp", "LP model M2 (thermal-aware)")]:
    for s_, jj in enumerate(show):
        ax.plot(tmin, traj[key][:, jj], color=SER[s_], lw=1.3, label=GPU_NAMES[jj])
    ax.axhline(83, color="#d03b3b", lw=0.9, ls="--"); ax.text(29.8, 84.0, "throttle 83 °C", fontsize=7, color="#d03b3b", ha="right")
    ax.set_ylabel("GPU temp. (°C)"); ax.set_ylim(40, 97)
    ax.set_title(ttl, fontsize=8.5, loc="left", color=INK)
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.55, 0.955))
axs[2].set_xlabel("Time (min), $T_{amb}$ = 41 °C")
fig.savefig("figures/fig3_trajectories.png"); plt.close(fig)

# R3 — ambient sweep
Tgrid = np.arange(25.0, 50.5, 1.0)
sweep = dict(T=Tgrid.tolist(), m1_mean=[], m1_peak=[], m2=[], viol_greedy=[], viol_cap=[], viol_myopic=[], viol_lp=[],
             forced_myopic=[], E_myopic=[], E_greedy=[], E_cap=[], mu_total=[], mu_by_gpu=[])
for Ta in Tgrid:
    a1 = solve_m1(inst, Dm, Ta); a2 = solve_m1(inst, Dp, Ta); r = solve_m2(inst, D, Ta)
    sweep["m1_mean"].append(a1["P_total"] * DT * NT if a1["status"] == 0 else None)
    sweep["m1_peak"].append(a2["P_total"] if a2["status"] == 0 else None)
    sweep["m2"].append(r["E_gpu"] if r["status"] == 0 else None)
    T0 = r["T0"]
    Tg, Eg = simulate(inst, baseline_greedy(inst, D), Ta, T0)
    Tc, Ec = simulate(inst, baseline_capprop(inst, D), Ta, T0)
    xm, fm = baseline_myopic(inst, D, Ta, T0); Tm, Em = simulate(inst, xm, Ta, T0)
    sweep["viol_greedy"].append(violation_stats(inst, Tg)["frac_over"])
    sweep["viol_cap"].append(violation_stats(inst, Tc)["frac_over"])
    sweep["viol_myopic"].append(violation_stats(inst, Tm)["frac_over"])
    sweep["forced_myopic"].append(fm); sweep["E_myopic"].append(Em); sweep["E_greedy"].append(Eg); sweep["E_cap"].append(Ec)
    if r["status"] == 0:
        Tl, _ = simulate(inst, r["x"], Ta, T0); sweep["viol_lp"].append(violation_stats(inst, Tl)["frac_over"])
        sweep["mu_total"].append(float(r["mu_T"].sum())); sweep["mu_by_gpu"].append(r["mu_T"].sum(0).tolist())
    else:
        sweep["viol_lp"].append(None); sweep["mu_total"].append(None); sweep["mu_by_gpu"].append(None)

def limit(fun, lo, hi, tol=0.05):
    if fun(hi): return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fun(mid) else (lo, mid)
    return lo
lim = dict(
    m1_peak=limit(lambda T: solve_m1(inst, Dp, T)["status"] == 0, 25, 60),
    m1_mean=limit(lambda T: solve_m1(inst, Dm, T)["status"] == 0, 25, 70),
    m2=limit(lambda T: solve_m2(inst, D, T)["status"] == 0, 25, 70),
    m2_fixedT0=limit(lambda T: solve_m2(inst, D, T, T0=np.full(6, 60.0))["status"] == 0, 25, 70),
)
P_["limits"] = lim
P_["sweep"] = sweep

# cooling extension — total facility energy vs set-point (parametric, each point an LP)
R0 = 0.6   # cooling energy / IT energy at 25 degC  (CRAC 30-40 %, IT 50-60 % in [1])
cool = {}
for s_ in [0.05, 0.08]:
    tot = [None if e is None else e * (1 + R0 * (1 - s_) ** (T - 25)) for e, T in zip(sweep["m2"], Tgrid)]
    cool[str(s_)] = tot
P_["cooling"] = dict(R0=R0, total=cool)

def tot_at(T, s_, E):
    return E * (1 + R0 * (1 - s_) ** (T - 25))
E36 = solve_m2(inst, D, np.floor(lim["m1_peak"]))["E_gpu"]; T36 = float(np.floor(lim["m1_peak"]))
Tm2 = float(np.floor(lim["m2"])); Em2 = solve_m2(inst, D, Tm2)["E_gpu"]
E25 = solve_m2(inst, D, 25.0)["E_gpu"]
P_["setpoint_gain"] = {
    str(s_): dict(T_static=T36, T_dynamic=Tm2,
                  total_static=tot_at(T36, s_, E36), total_dynamic=tot_at(Tm2, s_, Em2), total_25=tot_at(25.0, s_, E25),
                  saving_dyn_vs_static_pct=100 * (1 - tot_at(Tm2, s_, Em2) / tot_at(T36, s_, E36)),
                  saving_dyn_vs_25_pct=100 * (1 - tot_at(Tm2, s_, Em2) / tot_at(25.0, s_, E25)),
                  gpu_energy_increase_pct=100 * (Em2 / E25 - 1))
    for s_ in [0.05, 0.08]}
E41 = solve_m2(inst, D, 41.0)["E_gpu"]
P_["setpoint_gain_41"] = {str(s_): dict(saving_41_vs_static_pct=100 * (1 - tot_at(41.0, s_, E41) / tot_at(T36, s_, E36)),
                                         saving_41_vs_25_pct=100 * (1 - tot_at(41.0, s_, E41) / tot_at(25.0, s_, E25)),
                                         gpu_energy_increase_pct=100 * (E41 / E25 - 1))
                          for s_ in [0.05, 0.08]}

# fig 4 — violation fraction vs ambient
fig, ax = plt.subplots(figsize=(4.6, 2.7))
for s_, (key, lab, ls) in enumerate([("viol_greedy", "Greedy lowest-energy", "-"), ("viol_cap", "Capacity-proportional", "--"),
                                    ("viol_myopic", "Myopic thermal greedy", "-."), ("viol_lp", "LP model M2", "-")]):
    y = [np.nan if v is None else 100 * v for v in sweep[key]]
    ax.plot(Tgrid, y, color=SER[s_], ls=ls, marker="o", ms=3, label=lab)
ax.set_xlabel("Ambient (inlet-air) set-point $T_{amb}$ (°C)"); ax.set_ylabel("GPU-time above throttle threshold (%)")
ax.legend(fontsize=7, loc="upper left"); ax.set_xlim(25, 50)
ax.annotate(f"LP infeasible above {lim['m2']:.1f} °C", (47, 0), xytext=(40.5, 7), fontsize=7, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
fig.savefig("figures/fig4_violations.png"); plt.close(fig)

# fig 5 — energy vs ambient: (a) GPU energy, (b) total incl. cooling  [two panels, one axis each]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
e = lambda arr: [np.nan if v is None else v / 1e6 for v in arr]
axs[0].plot(Tgrid, e(sweep["m2"]), color=SER[0], marker="o", ms=3, label="LP model M2")
axs[0].plot(Tgrid, e(sweep["E_myopic"]), color=SER[2], ls="-.", label="Myopic thermal greedy")
axs[0].plot(Tgrid, e(sweep["E_greedy"]), color=SER[1], ls="--", label="Greedy (violates limits)")
axs[0].axvline(lim["m1_peak"], color=MUTED, lw=0.8, ls=":"); axs[0].text(lim["m1_peak"] + 0.4, 2.153, "M1 (peak demand)\ninfeasible →", fontsize=6.5, color=MUTED, ha="left")
axs[0].set_xlim(24, 48.5); axs[0].set_ylim(2.15, 2.24); axs[0].set_xlabel("$T_{amb}$ (°C)"); axs[0].set_ylabel("GPU energy over 30 min (MJ)")
axs[0].legend(fontsize=6.5, loc="upper left"); axs[0].set_title("(a) GPU energy", fontsize=8.5, loc="left", color=INK)
for s_, key, ls in [(0, "0.05", "-"), (1, "0.08", "--")]:
    axs[1].plot(Tgrid, e(cool[key]), color=SER[s_ + 3], ls=ls, marker="o", ms=3, label=f"cooling saving {int(float(key)*100)} %/°C")
axs[1].axvline(lim["m1_peak"], color=MUTED, lw=0.8, ls=":")
axs[1].axvline(lim["m2"], color=MUTED, lw=0.8, ls=":")
axs[1].text(lim["m2"] - 0.3, 3.05, "M2\nlimit", fontsize=6.5, color=MUTED, ha="right")
axs[1].text(lim["m1_peak"] - 0.3, 3.05, "M1 (peak)\nlimit", fontsize=6.5, color=MUTED, ha="right")
axs[1].set_xlabel("$T_{amb}$ (°C)"); axs[1].set_ylabel("GPU + cooling energy (MJ)")
axs[1].legend(fontsize=6.5, loc="upper right"); axs[1].set_title("(b) Facility energy (GPU + CRAC)", fontsize=8.5, loc="left", color=INK)
fig.tight_layout(); fig.savefig("figures/fig5_energy_vs_ambient.png"); plt.close(fig)

# fig 6 — shadow price of thermal headroom by GPU vs ambient
fig, ax = plt.subplots(figsize=(4.6, 2.7))
mb = np.array([[np.nan] * 6 if v is None else v for v in sweep["mu_by_gpu"]])
for s_, jj in enumerate(range(6)):
    ax.plot(Tgrid, -mb[:, jj] / 1e3, color=SER[s_], marker="o", ms=2.5, lw=1.2, label=GPU_NAMES[jj])
ax.set_xlabel("$T_{amb}$ (°C)"); ax.set_ylabel("Value of +1 °C headroom (kJ per 30 min)")
ax.legend(fontsize=6.5, ncol=2, loc="upper left"); ax.set_xlim(25, 48)
fig.savefig("figures/fig6_shadow_prices.png"); plt.close(fig)

# R4 — runtime scaling (M2) vs horizon length
sc = []
for nt in [60, 180, 360, 720]:
    Dn = demand_profile(nt=nt)
    t0 = time.time(); r = solve_m2(inst, Dn, 38.0); el = time.time() - t0
    sc.append(dict(NT=nt, n_vars=int(6 * 6 * nt + 6 * nt), seconds=el, status=int(r["status"])))
P_["scaling"] = sc
R["results"] = P_

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o
json.dump(clean(R), open("results.json", "w"), indent=1)
print("done")
