"""
train_compare.py
────────────────
Trains and benchmarks four datacenter cooling control strategies:

  Strategy 1 — Rule-Based Thermostat : Industrial deadband feedback baseline
  Strategy 2 — Autonomous MPC        : Receding-horizon economic QP via OSQP
  Strategy 3 — Pure RL               : PPO agent acting freely on closed-loop DatacenterEnv
  Strategy 4 — Hybrid Safe RL        : PPO agent projected through OSQP safety guardrail

Outputs (saved in RESULTS_DIR)
──────────────────────────────
  learning_curves.png           Training reward vs timesteps
  eval_metrics.png              4-way comparison of Energy, Violations, Peak Temps
  pareto_energy_safety.png      Tradeoff frontier (Energy vs. Violation Rate)
  mpc_intervention_rate.png     Intervention rate of the MPC safety filter
  metrics_summary.txt           Publication-ready numerical benchmark table
  eval_results_all.json         Machine-readable metrics for LaTeX tables
  model_pure_rl.zip             Trained PPO weights (Pure RL)
  model_rl_mpc.zip              Trained PPO weights (Safe RL)
"""

from __future__ import annotations

import os
import sys
import time
import json
import warnings
from pathlib import Path

# Add project root and local directories to path for robust imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TIER2_DIR = HERE.parent
for p in [ROOT, TIER2_DIR, HERE]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

try:
    from datacenter_env import DatacenterEnv, T_CELCC_MAX, EPISODE_LEN, CLIP_POWER
    from mpc_guardrail import (
        LinearDynamicsModel, MPCGuardrail, MPCGuardrailWrapper,
        T_STATE_COLS, POWER_COLS as MPC_POWER_COLS,
    )
    from baseline_controllers import ThermostatController, AutonomousMPCController
except ImportError:
    from MPC.datacenter_env import DatacenterEnv, T_CELCC_MAX, EPISODE_LEN, CLIP_POWER
    from MPC.results.mpc_guardrail import (
        LinearDynamicsModel, MPCGuardrail, MPCGuardrailWrapper,
        T_STATE_COLS, POWER_COLS as MPC_POWER_COLS,
    )
    from MPC.baseline_controllers import ThermostatController, AutonomousMPCController

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = HERE
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if (HERE / "final_dataset_std.csv").exists():
    DATA_PATH = HERE / "final_dataset_std.csv"
elif (HERE.parent / "final_dataset_std.csv").exists():
    DATA_PATH = HERE.parent / "final_dataset_std.csv"
elif (ROOT / "Tier 2 - Facility Cooling Control" / "final_dataset_std.csv").exists():
    DATA_PATH = ROOT / "Tier 2 - Facility Cooling Control" / "final_dataset_std.csv"
elif (ROOT / "Tier 2" / "final_dataset_std.csv").exists():
    DATA_PATH = ROOT / "Tier 2" / "final_dataset_std.csv"
else:
    DATA_PATH = ROOT / "MPC" / "final_dataset_std.csv"

# ── Training config ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 100_000   # 100k timesteps
EVAL_EPISODES   = 50        # episodes for final evaluation
SEED            = 42

PPO_KWARGS = dict(
    policy       = "MlpPolicy",
    n_steps      = 2048,
    batch_size   = 256,
    n_epochs     = 10,
    gamma        = 0.99,
    learning_rate= 3e-4,
    ent_coef     = 0.01,
    clip_range   = 0.2,
    policy_kwargs= dict(net_arch=[128, 128]),
    verbose      = 0,
    seed         = SEED,
)

PALETTE = {
    "Thermostat": "#898781",
    "Autonomous MPC": "#1BAF7A",
    "Pure RL": "#E07B54",
    "Safe RL (RL+MPC)": "#2A78D6",
}
sns.set_theme(style="darkgrid", context="paper", font_scale=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: record per-episode rewards for learning curves
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_timesteps: list[int] = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_timesteps.append(self.num_timesteps)
            self._ep_reward = 0.0
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper: roll out n episodes with paired seeds
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_strategy(
    controller,
    env: DatacenterEnv | MPCGuardrailWrapper,
    n_episodes: int = EVAL_EPISODES,
    is_mpc: bool = False,
    name: str = "Controller",
) -> dict:
    """
    Evaluates any controller over n_episodes with fixed paired seeds.
    Measures rewards, energy costs, violations, peak temperatures, and compute latency.
    """
    rewards, energies, violations, peaks, latencies = [], [], [], [], []
    mpc_mods = []

    if is_mpc:
        env.modification_count = 0
        env.total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        ep_reward = ep_energy = ep_viol = ep_mod = 0.0
        ep_max_t = -999.0

        while not done:
            t0 = time.perf_counter()
            action, _ = controller.predict(obs, deterministic=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_energy += info["energy_cost"]
            ep_viol   += info["constraint_violation"]
            ep_max_t   = max(ep_max_t, info.get("peak_temp", -999.0))
            if is_mpc:
                ep_mod += int(info.get("mpc_modified", False))

        rewards.append(ep_reward)
        energies.append(ep_energy)
        violations.append(ep_viol / EPISODE_LEN)
        peaks.append(ep_max_t)
        if is_mpc:
            mpc_mods.append(ep_mod / EPISODE_LEN)

    out = {
        "name"        : name,
        "rewards"     : np.array(rewards),
        "energies"    : np.array(energies),
        "violations"  : np.array(violations),
        "peaks"       : np.array(peaks),
        "latency_ms"  : float(np.mean(latencies)),
    }
    if is_mpc:
        out["mpc_modifications"] = np.array(mpc_mods)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting & Artifact Generation
# ─────────────────────────────────────────────────────────────────────────────
def smooth(x: np.ndarray, w: int = 15) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(cb_pure: EpisodeRewardCallback, cb_safe: EpisodeRewardCallback):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, cb, col in [
        ("Pure RL", cb_pure, PALETTE["Pure RL"]),
        ("Safe RL (RL+MPC)", cb_safe, PALETTE["Safe RL (RL+MPC)"]),
    ]:
        t = np.array(cb.episode_timesteps)
        r = np.array(cb.episode_rewards)
        if len(r) == 0:
            continue
        r_sm = smooth(r)
        t_sm = t[len(t) - len(r_sm):]
        ax.plot(t_sm, r_sm, label=label, color=col, linewidth=2.2)
        ax.fill_between(t_sm, r_sm - 100, r_sm + 100, alpha=0.15, color=col)

    ax.set_xlabel("Training Timesteps", fontsize=11)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=11)
    ax.set_title("Training Convergence: Pure RL vs. Safe RL (MPC Guardrails)", fontsize=12)
    ax.legend(framealpha=0.9, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_curves.png", dpi=200)
    plt.close(fig)
    print(f"  → Saved learning_curves.png")


def plot_comparative_metrics(all_results: dict[str, dict]):
    names = list(all_results.keys())
    colors = [PALETTE[k] for k in names]

    metrics = [
        ("energies",   "Total Energy Cost\n(lower = better)", False),
        ("violations", "Violation Rate (fraction)\n(lower = better)", False),
        ("peaks",      "Peak Temperature (σ)\n(safety limit = 2.0σ)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (key, title, _) in zip(axes, metrics):
        means = [all_results[n][key].mean() for n in names]
        stds  = [all_results[n][key].std() for n in names]

        bars = ax.bar(
            range(len(names)), means, yerr=stds,
            color=colors, capsize=5, width=0.55,
            edgecolor="white", linewidth=1.2,
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
        ax.set_title(title, fontsize=11)

        if key == "peaks":
            ax.axhline(T_CELCC_MAX, color="#d03b3b", linestyle="--", linewidth=1.2, label=r"$T_{\max}=2.0$")
            ax.legend(fontsize=8, loc="upper left")

        for bar, val in zip(bars, means):
            offset = max(abs(val) * 0.05, 0.02)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eval_metrics.png", dpi=200)
    plt.close(fig)

    # Pareto Frontier: Energy vs Violations
    fig_p, ax_p = plt.subplots(figsize=(7, 4.5))
    for n in names:
        e_mean = all_results[n]["energies"].mean()
        v_mean = all_results[n]["violations"].mean() * 100.0
        ax_p.scatter(e_mean, v_mean, color=PALETTE[n], s=120, edgecolors="white", linewidth=1.5, zorder=4)
        ax_p.annotate(
            n, (e_mean, v_mean),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=9, fontweight="bold", color=PALETTE[n],
        )

    ax_p.set_xlabel("Mean Total Energy Cost (lower = better)", fontsize=11)
    ax_p.set_ylabel("Mean Thermal Violation Rate (%) (lower = better)", fontsize=11)
    ax_p.set_title("Pareto Frontier: Energy Efficiency vs. Thermal Safety", fontsize=12)
    fig_p.tight_layout()
    fig_p.savefig(RESULTS_DIR / "pareto_energy_safety.png", dpi=200)
    plt.close(fig_p)

    # Intervention rate plot
    if "Safe RL (RL+MPC)" in all_results and "mpc_modifications" in all_results["Safe RL (RL+MPC)"]:
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.2))
        mod_arr = all_results["Safe RL (RL+MPC)"]["mpc_modifications"]
        ax2.bar(
            ["Safe RL\n(RL+MPC)"], [mod_arr.mean() * 100.0],
            yerr=[mod_arr.std() * 100.0],
            color=PALETTE["Safe RL (RL+MPC)"], capsize=6, width=0.4,
            edgecolor="white", linewidth=1.2,
        )
        ax2.set_ylabel("Steps Corrected by Filter (%)", fontsize=10)
        ax2.set_title("MPC Safety Filter Intervention Rate", fontsize=11)
        ax2.set_ylim(0, 100.0)
        fig2.tight_layout()
        fig2.savefig(RESULTS_DIR / "mpc_intervention_rate.png", dpi=200)
        plt.close(fig2)

    print("  → Saved eval_metrics.png, pareto_energy_safety.png, mpc_intervention_rate.png")


def save_summary_table(all_results: dict[str, dict]):
    names = list(all_results.keys())
    lines = [
        "=" * 85,
        "  FACILITY COOLING CONTROL BENCHMARK (mean ± std over 50 test episodes)",
        "=" * 85,
        f"{'Control Paradigm':<22} {'Energy Cost':>15} {'Violation Rate':>16} {'Peak Temp (σ)':>16} {'Latency (ms)':>12}",
        "-" * 85,
    ]

    clean_dict = {}
    for n in names:
        res = all_results[n]
        e_str = f"{res['energies'].mean():.1f}±{res['energies'].std():.1f}"
        v_str = f"{res['violations'].mean()*100:.1f}±{res['violations'].std()*100:.1f}%"
        p_str = f"{res['peaks'].mean():.2f}±{res['peaks'].std():.2f}"
        l_str = f"{res['latency_ms']:.2f} ms"
        lines.append(f"{n:<22} {e_str:>15} {v_str:>16} {p_str:>16} {l_str:>12}")

        clean_dict[n] = {
            "energy_mean": float(res["energies"].mean()),
            "energy_std": float(res["energies"].std()),
            "violation_pct_mean": float(res["violations"].mean() * 100.0),
            "violation_pct_std": float(res["violations"].std() * 100.0),
            "peak_temp_mean": float(res["peaks"].mean()),
            "peak_temp_std": float(res["peaks"].std()),
            "latency_ms": float(res["latency_ms"]),
        }
        if "mpc_modifications" in res:
            clean_dict[n]["intervention_pct"] = float(res["mpc_modifications"].mean() * 100.0)

    lines.append("=" * 85)
    text = "\n".join(lines)
    print("\n" + text)
    (RESULTS_DIR / "metrics_summary.txt").write_text(text)
    (RESULTS_DIR / "eval_results_all.json").write_text(json.dumps(clean_dict, indent=2))
    print(f"  → Saved metrics_summary.txt and eval_results_all.json")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Dataset path: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH, sep=";")
    col_names = list(df.columns)
    data = df.values.astype(np.float32)
    print(f"Loaded dataset: {data.shape[0]:,} rows × {data.shape[1]} features")

    # 1. Fit linear dynamics model on 80% train data
    print("\nFitting physics-consistent linear dynamics model …")
    split_pt = int(0.8 * len(data))
    train_data = data[:split_pt]
    dyn_model = LinearDynamicsModel(alpha=10.0).fit(train_data, col_names)

    # 2. Build environments with closed-loop digital twin
    print("\nBuilding closed-loop environments …")
    guardrail     = MPCGuardrail(dyn_model, horizon=5, t_max=T_CELCC_MAX)

    train_env_rl  = Monitor(DatacenterEnv(data, col_names, dynamics_model=dyn_model, split="train"))
    test_env_raw  = DatacenterEnv(data, col_names, dynamics_model=dyn_model, split="test")

    train_env_mpc = Monitor(MPCGuardrailWrapper(
        DatacenterEnv(data, col_names, dynamics_model=dyn_model, split="train"), guardrail, col_names
    ))
    test_env_mpc  = MPCGuardrailWrapper(
        DatacenterEnv(data, col_names, dynamics_model=dyn_model, split="test"), guardrail, col_names
    )

    # 3. Instantiate classical baselines
    print("\nInitializing classical baseline controllers …")
    thermostat = ThermostatController(col_names, t_target=1.0, deadband=0.25)
    auto_mpc   = AutonomousMPCController(dyn_model, col_names, horizon=5, t_max=T_CELCC_MAX)

    # 4. Train Strategy 3: Pure RL
    print(f"\n{'='*60}")
    print(f"  STRATEGY 3 — Pure RL (PPO) ({TOTAL_TIMESTEPS:,} timesteps)")
    print(f"{'='*60}")
    cb_pure = EpisodeRewardCallback()
    model_pure = PPO(env=train_env_rl, **PPO_KWARGS)
    t0 = time.time()
    model_pure.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb_pure, progress_bar=True)
    print(f"  Training time: {time.time()-t0:.1f}s")
    model_pure.save(RESULTS_DIR / "model_pure_rl")

    # 5. Train Strategy 4: Safe RL (RL + MPC)
    print(f"\n{'='*60}")
    print(f"  STRATEGY 4 — Safe RL (RL + MPC Guardrail) ({TOTAL_TIMESTEPS:,} timesteps)")
    print(f"{'='*60}")
    cb_safe = EpisodeRewardCallback()
    model_safe = PPO(env=train_env_mpc, **PPO_KWARGS)
    t0 = time.time()
    model_safe.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb_safe, progress_bar=True)
    print(f"  Training time: {time.time()-t0:.1f}s")
    model_safe.save(RESULTS_DIR / "model_rl_mpc")

    # 6. Evaluate all 4 strategies on identical test episodes
    print(f"\n{'='*60}")
    print(f"  EVALUATION  ({EVAL_EPISODES} paired test episodes across all 4 paradigms)")
    print(f"{'='*60}")
    all_results = {}

    print("Evaluating Thermostat Controller …")
    all_results["Thermostat"] = evaluate_strategy(
        thermostat, test_env_raw, n_episodes=EVAL_EPISODES, is_mpc=False, name="Thermostat"
    )

    print("Evaluating Autonomous Receding-Horizon MPC …")
    all_results["Autonomous MPC"] = evaluate_strategy(
        auto_mpc, test_env_raw, n_episodes=EVAL_EPISODES, is_mpc=False, name="Autonomous MPC"
    )

    print("Evaluating Pure RL (PPO) …")
    all_results["Pure RL"] = evaluate_strategy(
        model_pure, test_env_raw, n_episodes=EVAL_EPISODES, is_mpc=False, name="Pure RL"
    )

    print("Evaluating Safe RL (PPO + MPC Guardrail) …")
    all_results["Safe RL (RL+MPC)"] = evaluate_strategy(
        model_safe, test_env_mpc, n_episodes=EVAL_EPISODES, is_mpc=True, name="Safe RL (RL+MPC)"
    )

    # 7. Generate plots and summary tables
    print("\nCompiling comparative figures and summary statistics …")
    plot_learning_curves(cb_pure, cb_safe)
    plot_comparative_metrics(all_results)
    save_summary_table(all_results)

    print(f"\nAll benchmark results successfully saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
