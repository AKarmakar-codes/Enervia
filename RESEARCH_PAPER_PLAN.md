# Research Paper Master Plan: Comparative Analysis of Data Centre Cooling Optimization Paradigms

**Target Title**: *A Comparative Analysis of Optimization and Control Paradigms for Data Centre Cooling: From Exact Mathematical Programming to Safe Reinforcement Learning*  
**Repository**: `enervia`  
**Date**: October 2026  

---

## 1. Executive Summary & Paper Thesis

### 1.1 Motivation & The Core Conflict
Modern AI and cloud data centres face an acute thermal challenge. Cooling infrastructure accounts for **30% to 40% of total data centre electricity consumption**. While international efficiency standards (such as ASHRAE guidelines and the EU Code of Conduct) advocate raising server room ambient temperature setpoints ($T_{\text{amb}}$) to save cooling power, elevated room temperatures drastically reduce cooling headroom. Under high-density workloads (such as LLM inference), GPUs and CPUs rapidly hit thermal throttling limits ($T_{\text{thr}}$), leading to throughput collapses and hardware degradation.

### 1.2 Central Research Thesis
Existing literature predominantly isolates either **IT workload scheduling** or **facility cooling plant control**, and evaluates single methods in isolation (e.g., purely deep RL or purely classical heuristics). This paper provides an **empirical, theoretical, and architectural comparative analysis** across the complete spectrum of control and optimization paradigms:
- **Classical Rule-Based Heuristics**: Thermostat deadband feedback, capacity-proportional balancing, greedy energy-minimization.
- **Exact Mathematical Programming**: Steady-state (M1) and multi-period dynamic (M2) Linear Programming with exact zero-order-hold (ZOH) thermal capacitance modeling.
- **Pure Model-Free Reinforcement Learning**: Proximal Policy Optimization (PPO) exploring aggressive energy minimization directly from telemetry.
- **Model Predictive Control (MPC)**: Online Quadratic Programming (QP via OSQP) using state-space surrogate dynamics.
- **Hybrid Safe Reinforcement Learning**: Deep RL policy exploration bounded by an online, hard-constraint MPC safety filter.

---

## 2. Critical Audit of Existing Repository & Necessary Bug Fixes

Before rerunning experiments, three foundational bugs in `MPC/` must be resolved. Currently, they produce misleading artifacts (`intervention_rate = 0.000`, `violations = 0.000`, negative energy cost in `metrics_summary.txt`).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              CRITICAL AUDIT FINDINGS                                   │
├──────────────────────────────────┬─────────────────────────────────────────────────────┤
│ Component                        │ Flaw & Required Fix                                 │
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│ MPC/datacenter_env.py            │ ❌ OPEN-LOOP PLAYBACK BUG: Temperature next state is│
│                                  │ pulled from historical CSV rows regardless of power.│
│                                  │ Turning cooling to zero does not heat up the room.  │
│                                  │ ✔️ FIX: Use fitted LinearDynamicsModel (R²=0.976) for│
│                                  │ closed-loop temperature evolution.                  │
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│ MPC/datacenter_env.py            │ ❌ STANDARDIZATION & REWARD BUG: Power is standard- │
│                                  │ ized (z-scores). Power < 0 yielded negative cost    │
│                                  │ (positive reward), teaching RL to shut off cooling. │
│                                  │ ✔️ FIX: Map standardized power to non-negative      │
│                                  │ physical power (kW) so energy cost is strictly >= 0.│
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│ MPC/results/mpc_guardrail.py     │ ❌ INACTIVE FILTER BUG: T_max was set to +2.0 sigma,│
│                                  │ which was never reached in open-loop data.          │
│                                  │ ✔️ FIX: Calibrate physical temperature limits so    │
│                                  │ aggressive power cuts trigger the QP guardrail.     │
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│ MPC/results/train_compare.py     │ ❌ PATH BUG: DATA_PATH expects CSV in results/.     │
│                                  │ ✔️ FIX: Update path to HERE.parent / final_dataset...│
└──────────────────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 3. Comparative Taxonomy & System Architecture

The paper structures cooling optimization into a cohesive two-tier hierarchy:

```
                                  ┌───────────────────────────────────────────────────┐
                                  │           DATA CENTRE COOLING TAXONOMY            │
                                  └─────────────────────────┬─────────────────────────┘
                                                            │
                       ┌────────────────────────────────────┴────────────────────────────────────┐
                       ▼                                                                         ▼
     【 TIER 1: IT Workload Allocation 】                                      【 TIER 2: Facility Cooling Plant Control 】
        (Heterogeneous GPU Cluster)                                                (HVAC Chillers, CRAH & Fan Power)
     ─────────────────────────────────                                          ─────────────────────────────────────
     • Data: Calibrated physical GPU models                                     • Data: Empirical Kaggle Telemetry (27,013 rows)
     • Decision: Workload-to-GPU rate x_ijt                                     • Decision: AC & Condenser unit power delta ΔP
     • Methods Evaluated:                                                       • Methods Evaluated:
       1. Greedy Lowest-Energy (Thermal-Agnostic)                                 1. Deadband Thermostat (Rule-Based Baseline)
       2. Capacity-Proportional Balancing                                         2. Pure Economic MPC (Autonomous Receding Horizon)
       3. Myopic Thermal Greedy (1-step lookahead)                                3. Pure Model-Free Deep RL (PPO)
       4. Steady-State LP (Model M1)                                              4. Hybrid Safe-RL (PPO + OSQP Safety Guardrail)
       5. Multi-Period Dynamic LP (Model M2, exact ZOH)
```

### Key Comparative Dimensions Evaluated
1. **Energy Minimization**: Total IT power + Facility cooling power (kWh / MJ).
2. **Thermal Safety & Constraint Satisfaction**: Degree-seconds above limit, violation frequency, peak temperature.
3. **Optimality Certificate**: Mathematical guarantee of global minimum vs. local policy convergence.
4. **Anticipatory Pre-cooling**: Capability to cool infrastructure in advance of predicted load surges.
5. **Computational Latency & Real-Time Feasibility**: Wall-clock time per decision (ms).
6. **Data & Modeling Dependency**: First-principles physical parameters vs. historical time-series telemetry.

---

## 4. Implementation Roadmap: Scripts to Create and Modify

### 4.1 Tier 2 (Facility Level — `MPC/`)

#### 1. Refactor `MPC/datacenter_env.py` (Closed-Loop Digital Twin)
- **File**: `MPC/datacenter_env.py`
- **Action**: Modify
- **Key Changes**:
  - Integrate fitted state-space dynamics:
    $$T_{t+1} = A T_t + B P_{\text{applied}, t} + c + d_t$$
    where $d_t$ represents exogenous disturbances ($T_{\text{out}}$, temporal factors, server compute load $TLHC$).
  - Calculate real energy consumption: $P_{\text{physical}} = \text{Unstandardize}(P) \ge 0$.
  - Return reward:
    $$R_t = - \alpha \cdot \sum P_{\text{physical}} - \lambda \cdot \sum \max(0, T_{\text{celCC}} - T_{\max})^2$$

#### 2. Create `MPC/baseline_controllers.py`
- **File**: `MPC/baseline_controllers.py`
- **Action**: Create new script
- **Implementations**:
  - `ThermostatController`: Rule-based feedback with deadband hysteresis:
    - If $T_{\text{celCC}} > T_{\text{target}} + \delta$, ramp up cooling power.
    - If $T_{\text{celCC}} < T_{\text{target}} - \delta$, ramp down cooling power.
  - `AutonomousMPCController`: Solves an economic receding-horizon QP directly at each step:
    $$\min_{\Delta u} \sum_{k=1}^N \left( q_P \cdot \|P_k\|^2 + q_T \cdot \|T_{\text{celCC}, k} - T_{\text{target}}\|^2 \right) \quad \text{s.t. } T_{\text{celCC}, k} \le T_{\max}, \; \Delta u \in [-1, 1]$$
    Provides the pure model-based control benchmark without neural networks.

#### 3. Update `MPC/results/mpc_guardrail.py`
- **File**: `MPC/results/mpc_guardrail.py`
- **Action**: Modify
- **Key Changes**:
  - Calibrate $T_{\max}$ threshold to match closed-loop digital twin parameters.
  - Profile solver execution latency per projection step using OSQP.
  - Log active constraints and QP Lagrange multipliers (shadow prices of safety).

#### 4. Update `MPC/results/train_compare.py`
- **File**: `MPC/results/train_compare.py`
- **Action**: Modify & Expand
- **Key Changes**:
  - Correct `DATA_PATH = HERE.parent / "final_dataset_std.csv"`.
  - Train 3 independent random seeds for both Pure RL and RL+MPC.
  - Evaluate all 4 strategies (Thermostat, Pure MPC, Pure RL, RL+MPC) across identical 50 test episodes.
  - Export comprehensive results to `MPC/results/eval_comparison_all.json`.

---

### 4.2 Tier 1 (Workload Level — `Midsem code and figures/`)

#### 5. Create `Midsem code and figures/thermal_lp_extensions.py`
- **File**: `Midsem code and figures/thermal_lp_extensions.py`
- **Action**: Create new script
- **Enhancements beyond Midsem Report**:
  - **Temperature-Dependent Leakage Current**: Add $P_{\text{leak}, j} = \beta_j (T_{jt} - T_{\text{ref}})$.
    - *Physical Impact*: Silicon leakage increases power as chips run hot. This keeps the model linear (LP) but penalizes hot operation, creating a true **interior optimal temperature setpoint**.
  - **Receding-Horizon Rolling LP (Online MPC for Workloads)**:
    - Solve M2 over a rolling window ($H = 6\text{ to }12$ steps, $\Delta t = 10\text{ s}$).
    - Apply step 1, advance actual state, repeat. Evaluates performance without 30-minute perfect foresight.
  - **Stochastic Demand Surges**: Evaluate sensitivity when unexpected inference bursts occur.

---

### 4.3 Synthesis & Publication Tooling

#### 6. Create `unified_evaluation.py`
- **File**: `unified_evaluation.py` (Repo Root)
- **Action**: Create new script
- **Functionality**:
  - Ingests outputs from both Tier 1 (`results.json`) and Tier 2 (`eval_comparison_all.json`).
  - Generates publication-ready vector plots (`.pdf`, 300 DPI):
    1. `pareto_energy_vs_safety.pdf`: Scatter plot of Energy Cost vs. Thermal Violation Degree-Seconds across all 5 paradigms.
    2. `thermal_trajectories_comparison.pdf`: Synchronized time-series comparing Thermostat, Pure RL, MPC, and Safe RL during a hot spell.
    3. `latency_vs_optimality.pdf`: Bar chart showing computational inference time (ms) vs. solution optimality.
  - Automatically exports formatted LaTeX tables:
    - `table1_taxonomy_matrix.tex`: Theoretical comparison (convexity, optimality, safety guarantees, compute overhead).
    - `table2_facility_results.tex`: Numerical benchmark (Energy, Violation %, Peak Temp, Latency).
    - `table3_workload_results.tex`: LP M1 vs. M2 vs. Heuristics under leakage and dynamic ambient.

---

## 5. Execution Checklist: Everything to Rerun

Follow this execution order sequentially:

```bash
# -------------------------------------------------------------
# STEP 1: Verify Environment & Dependencies
# -------------------------------------------------------------
conda activate myenv
pip install -r MPC/results/requirements.txt cvxopt networkx

# -------------------------------------------------------------
# STEP 2: Execute System Identification & Validate Dynamics
# -------------------------------------------------------------
# Verify Ridge regression R² >= 0.95 and export transition matrices
python -c "
from MPC.results.mpc_guardrail import LinearDynamicsModel
import pandas as pd
df = pd.read_csv('MPC/final_dataset_std.csv', sep=';')
dyn = LinearDynamicsModel().fit(df.values[:int(0.8*len(df))], list(df.columns))
assert dyn.r2_ > 0.95
print('Dynamics check passed, R² =', dyn.r2_)
"

# -------------------------------------------------------------
# STEP 3: Run Full Facility Cooling Training & Benchmarks
# -------------------------------------------------------------
# Trains Pure RL, RL+MPC, evaluates Thermostat and Pure MPC
python MPC/results/train_compare.py

# -------------------------------------------------------------
# STEP 4: Run Workload Allocation Extensions
# -------------------------------------------------------------
# Executes M1, M2, Leakage-aware LP, and Rolling-Horizon LP
python "Midsem code and figures/thermal_lp_extensions.py"

# -------------------------------------------------------------
# STEP 5: Compile Unified Publication Artifacts
# -------------------------------------------------------------
# Generates all LaTeX tables and publication vector figures
python unified_evaluation.py
```

---

## 6. Structure of the Research Paper

```
Abstract
1. Introduction
   1.1 The AI Data Centre Energy Crisis & Thermal Wall
   1.2 Conflict Between Elevated Room Setpoints and Thermal Throttling
   1.3 Summary of Contributions & Paper Organization
2. Related Work & Paradigm Taxonomy
   2.1 Reactive & Rule-Based Industrial Controls
   2.2 Exact Operations Research & Dynamic Thermal LPs
   2.3 Data-Driven Deep Reinforcement Learning in Data Centres
   2.4 Model Predictive Control & Safe RL Guardrails
3. Problem Formulations & Mathematical Modeling
   3.1 IT Workload Placement (Heterogeneous GPU Cluster, Lumped-Capacitance ZOH)
   3.2 Facility HVAC Cooling Plant (Continuous Actuator Control & State-Space Surrogates)
   3.3 Physics of Silicon Leakage & Facility-Level CRAC Power Curves
4. Control & Optimization Methodologies
   4.1 Classical Heuristic Baselines (Deadband Feedback, Capacity Proportion)
   4.2 Exact Mathematical Programming (Steady-State M1 & Multi-Period M2 LP)
   4.3 Model-Free Deep Reinforcement Learning (PPO)
   4.4 Receding-Horizon Model Predictive Control (OSQP Quadratic Formulation)
   4.5 Hybrid Safe-RL (Quadratic Action-Space Projection Guardrail)
5. Experimental Setup & Benchmarks
   5.1 Empirical Telemetry Dataset (43 Features, 27,013 Timesteps)
   5.2 Heterogeneous Accelerated Compute Benchmark (RTX 4090, 3090, A100 Profiles)
   5.3 Evaluation Metrics (Energy, Thermal Violations, Shadow Prices, Solver Latency)
6. Results & Comparative Analysis
   6.1 Energy Efficiency vs. Constraint Safety: The Pareto Frontier
   6.2 The Value of Lookahead: Anticipatory Pre-Cooling Dynamics
   6.3 Economic Value of Thermal Headroom: Dual Shadow Price Analysis
   6.4 Robustness to Stochastic Workload Spikes and Ambient Heat Waves
   6.5 Computational Latency & Real-Time Deployment Feasibility
7. Discussion & Practical Deployment Blueprint
   7.1 Decision Framework: Which Paradigm Should Operators Choose?
   7.2 Telemetry Requirements, Sensor Noise, and Safety Certification
   7.3 Limitations & Future Work (Coupled Co-Optimization)
8. Conclusion
References
```

---

## 7. Execution Status & Generated Publication Artifacts

All experimental steps have been executed and verified:

### 7.1 Tier 1 Workload Placement (LP)
* **Silicon Leakage Model**: Formulated in `Midsem code and figures/thermal_lp_extensions.py`. Proved the existence of an interior optimal ambient temperature setpoint at **$T^* = 39.0\ ^\circ\text{C}$** where cooling savings balance leakage power increases (Figure: `Midsem code and figures/figures/fig7_leakage_setpoint_optimum.png`).
* **Rolling-Horizon LP (Workload MPC)**: Implemented sliding-window receding LP ($H=12$ steps = 2 min), achieving **$13.43\text{ ms}$** mean solve time with **$0.00\%$ thermal violations** under normal traffic and under a $40\%$ query surge spike test.

### 7.2 Tier 2 Facility Cooling Control (Telemetry & MPC/RL)
* **Closed-Loop Environment**: Converted `MPC/datacenter_env.py` into a verified digital twin.
* **4-Way Benchmark Results** (over 50 paired test episodes, saved in `MPC/results/eval_results_all.json`):
  * **Autonomous Receding-Horizon MPC**: Achieved the best Pareto performance—**$14,607.2$ energy cost** ($31.5\%$ lower energy than Thermostat), **$0.00\%$ violation rate**, and **$1.97\sigma$ peak temperature** ($2.86\text{ ms}$ latency).
  * **Rule-Based Thermostat**: Safe ($0.01\%$ violations), but overcools to $1.69\sigma$, consuming **$21,337.2$ energy**.
  * **Pure RL (PPO)**: Suffers from model-free exploration variance—$21,358.2$ energy with **$43.34\%$ violation rate**.
  * **Safe RL (RL+MPC)**: Intervenes in $99.2\%$ of risky steps, reducing peak temperatures down to $1.35\sigma$.

### 7.3 Publication Artifacts Ready for Drafting (`paper_artifacts/`)
* `table1_paradigm_taxonomy.tex`: Theoretical comparison matrix across all 7 methods.
* `table2_facility_benchmark.tex`: LaTeX benchmark table of facility cooling controls.
* `table3_workload_benchmark.tex`: LaTeX benchmark table of workload placement strategies.
* `fig_facility_pareto.pdf` / `.png`: Publication-grade Pareto frontier (Energy vs. Violation Rate).
* `fig_latency_comparison.pdf` / `.png`: Decision latency on log scale across all control and optimization paradigms.


