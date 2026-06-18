"""
train_compare.py
────────────────
Trains and compares two datacenter cooling control strategies:

  Strategy 1 — Pure RL  : PPO agent acts freely on DatacenterEnv
  Strategy 2 — RL + MPC : same PPO, but actions are projected through the
                           LinearMPC safety filter before reaching the env

Usage
─────
    python train_compare.py

Outputs
───────
    results/learning_curves.png      Episode reward vs. timestep
    results/eval_metrics.png         Energy / violation / MPC intervention
    results/metrics_summary.txt      Numerical summary table
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from datacenter_env import DatacenterEnv, T_CELCC_MAX, EPISODE_LEN
from mpc_guardrail import (
    LinearDynamicsModel, MPCGuardrail, MPCGuardrailWrapper,
    T_STATE_COLS, POWER_COLS as MPC_POWER_COLS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
DATA_PATH   = HERE / "final_dataset_std.csv"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Training config ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES   = 50       # episodes for final evaluation
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
# Evaluation helper: roll out n episodes, collect per-step info
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_strategy(
    model: PPO,
    env: DatacenterEnv,
    n_episodes: int = EVAL_EPISODES,
    is_mpc: bool = False,
) -> dict:
    """
    Returns a dict with arrays of per-episode metrics:
        rewards, energy_costs, constraint_violations, [mpc_modifications]
    """
    rewards, energies, violations = [], [], []
    mpc_mods = []

    # Reset modification counter if MPC wrapper
    if is_mpc:
        env.modification_count = 0
        env.total_steps = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = ep_energy = ep_viol = ep_mod = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_energy += info["energy_cost"]
            ep_viol   += info["constraint_violation"]
            if is_mpc:
                ep_mod += int(info.get("mpc_modified", False))
        rewards.append(ep_reward)
        energies.append(ep_energy)
        violations.append(ep_viol / EPISODE_LEN)   # fraction of steps violated
        if is_mpc:
            mpc_mods.append(ep_mod / EPISODE_LEN)

    out = {
        "rewards"   : np.array(rewards),
        "energies"  : np.array(energies),
        "violations": np.array(violations),
    }
    if is_mpc:
        out["mpc_modifications"] = np.array(mpc_mods)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def smooth(x: np.ndarray, w: int = 20) -> np.ndarray:
    """Simple moving-average smoothing."""
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


PALETTE = {"Pure RL": "#E07B54", "RL + MPC": "#4E8FBF"}
sns.set_theme(style="darkgrid", context="paper", font_scale=1.2)


def plot_learning_curves(cb1: EpisodeRewardCallback, cb2: EpisodeRewardCallback):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, cb, col in [
        ("Pure RL",  cb1, PALETTE["Pure RL"]),
        ("RL + MPC", cb2, PALETTE["RL + MPC"]),
    ]:
        t = np.array(cb.episode_timesteps)
        r = np.array(cb.episode_rewards)
        r_sm = smooth(r)
        t_sm = t[len(t) - len(r_sm):]
        ax.plot(t_sm, r_sm, label=label, color=col, linewidth=2)
        ax.fill_between(t_sm, r_sm - 50, r_sm + 50, alpha=0.15, color=col)

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Episode Reward (smoothed)")
    ax.set_title("Learning Curves: Pure RL vs. RL + MPC Guardrails")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved learning_curves.png")


def plot_eval_metrics(m1: dict, m2: dict):
    metrics = [
        ("rewards",    "Mean Episode Reward\n(higher = better)",  True),
        ("energies",   "Mean Total Energy Cost\n(lower = better)", False),
        ("violations", "Mean Violation Rate\n(lower = better)",    False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (key, ylabel, higher_better) in zip(axes, metrics):
        vals  = [m1[key].mean(), m2[key].mean()]
        errs  = [m1[key].std(),  m2[key].std()]
        bars  = ax.bar(
            ["Pure RL", "RL + MPC"], vals, yerr=errs,
            color=[PALETTE["Pure RL"], PALETTE["RL + MPC"]],
            capsize=6, width=0.5, edgecolor="white", linewidth=1.2,
        )
        ax.set_title(ylabel, fontsize=11)
        ax.set_ylabel("")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + errs[0] * 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10,
            )

    # 4th panel: MPC intervention rate (RL+MPC only)
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.bar(
        ["RL + MPC"], [m2["mpc_modifications"].mean()],
        yerr=[m2["mpc_modifications"].std()],
        color=PALETTE["RL + MPC"], capsize=6, width=0.4,
        edgecolor="white", linewidth=1.2,
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction of Steps Corrected")
    ax2.set_title("MPC Intervention Rate")
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "mpc_intervention_rate.png", dpi=150)
    plt.close(fig2)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eval_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved eval_metrics.png, mpc_intervention_rate.png")


def save_summary(m1: dict, m2: dict):
    lines = [
        "=" * 60,
        "  EVALUATION SUMMARY  (mean ± std over 50 test episodes)",
        "=" * 60,
        f"{'Metric':<35} {'Pure RL':>10} {'RL+MPC':>10}",
        "-" * 60,
    ]
    for key, label in [
        ("rewards",           "Episode Reward"),
        ("energies",          "Total Energy Cost"),
        ("violations",        "Violation Rate (fraction)"),
    ]:
        v1 = f"{m1[key].mean():+.3f}±{m1[key].std():.3f}"
        v2 = f"{m2[key].mean():+.3f}±{m2[key].std():.3f}"
        lines.append(f"{label:<35} {v1:>12} {v2:>12}")
    lines.append(
        f"{'MPC Intervention Rate':<35} {'—':>12}"
        f" {m2['mpc_modifications'].mean():.3f}±{m2['mpc_modifications'].std():.3f}"
    )
    lines.append("=" * 60)

    text = "\n".join(lines)
    print("\n" + text)
    (RESULTS_DIR / "metrics_summary.txt").write_text(text)
    print(f"  → Saved metrics_summary.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("Loading dataset …")
    df = pd.read_csv(DATA_PATH, sep=";")
    col_names = list(df.columns)
    data = df.values.astype(np.float32)
    print(f"  {data.shape[0]:,} rows × {data.shape[1]} features")

    # ── 2. Fit linear dynamics model ─────────────────────────────────────────
    print("\nFitting linear dynamics model (Ridge regression) …")
    split_pt = int(0.8 * len(data))
    train_data = data[:split_pt]
    dyn_model = LinearDynamicsModel(alpha=1.0).fit(train_data, col_names)

    # ── 3. Build environments ─────────────────────────────────────────────────
    print("\nBuilding environments …")
    train_env_rl  = Monitor(DatacenterEnv(data, col_names, split="train"))
    test_env_rl   = DatacenterEnv(data, col_names, split="test")

    guardrail     = MPCGuardrail(dyn_model, horizon=5, t_max=T_CELCC_MAX)
    train_env_mpc = Monitor(MPCGuardrailWrapper(
        DatacenterEnv(data, col_names, split="train"), guardrail, col_names
    ))
    test_env_mpc  = MPCGuardrailWrapper(
        DatacenterEnv(data, col_names, split="test"), guardrail, col_names
    )

    # ── 4. Train Strategy 1: Pure RL ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  STRATEGY 1 — Pure RL  ({TOTAL_TIMESTEPS:,} timesteps)")
    print(f"{'='*55}")
    cb1   = EpisodeRewardCallback()
    model1= PPO(env=train_env_rl, **PPO_KWARGS)
    t0    = time.time()
    model1.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb1, progress_bar=True)
    print(f"  Training time: {time.time()-t0:.1f}s")
    model1.save(RESULTS_DIR / "model_pure_rl")

    # ── 5. Train Strategy 2: RL + MPC ────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  STRATEGY 2 — RL + MPC  ({TOTAL_TIMESTEPS:,} timesteps)")
    print(f"{'='*55}")
    cb2   = EpisodeRewardCallback()
    model2= PPO(env=train_env_mpc, **PPO_KWARGS)
    t0    = time.time()
    model2.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb2, progress_bar=True)
    print(f"  Training time: {time.time()-t0:.1f}s")
    model2.save(RESULTS_DIR / "model_rl_mpc")

    # ── 6. Evaluate on held-out test set ─────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EVALUATION  ({EVAL_EPISODES} episodes on test split)")
    print(f"{'='*55}")
    metrics_rl  = evaluate_strategy(model1, test_env_rl,  n_episodes=EVAL_EPISODES, is_mpc=False)
    metrics_mpc = evaluate_strategy(model2, test_env_mpc, n_episodes=EVAL_EPISODES, is_mpc=True)

    # ── 7. Plot & summarise ───────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_learning_curves(cb1, cb2)
    plot_eval_metrics(metrics_rl, metrics_mpc)
    save_summary(metrics_rl, metrics_mpc)

    print(f"\nAll results saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
