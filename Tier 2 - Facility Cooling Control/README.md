# Tier 2: Facility Cooling Control (MPC & Reinforcement Learning)

Optimizes datacenter cooling infrastructure (CRAC units, chillers, and pumps) using empirical facility telemetry, receding-horizon Model Predictive Control (MPC), and Reinforcement Learning (PPO).

---

## 1. What Was Originally Done (Friend)
* **Dataset Ingestion**: Acquired standardized 43-variable datacenter telemetry (`final_dataset_std.csv`, 27,013 steps) spanning 16 power actuators ($P_{\text{ac}}, P_{\text{cu}}$) and 24 temperature sensors ($T_{\text{MEAS}}, T_{\text{out}}, T_{\text{celCC}}$).
* **Initial Prototypes**:
  * Outlined a prototype Gym environment (`datacenter_env.py`).
  * Drafted an initial PPO agent and linear MPC filter (`mpc_guardrail.py`).
  * Midsem slides/report: `results/MPC and ML for Cooling Optimization.pdf`.
* **Identified Critical Flaws in Original Code**:
  1. *Open-Loop Playback Bug*: Next temperature observations were sliced directly from historical CSV rows rather than driven by the agent's actions ($s_{t+1}$ uncoupled from $a_t$).
  2. *Negative Energy Bug*: Power standardization (zero mean) caused negative energy values in rewards, skewing agent incentives.
  3. *Inactive Guardrail Bug*: Safety threshold was set to $+2.0\sigma$ (outside operational range), resulting in $0.00\%$ filter interventions.

---

## 2. What We Did (Fixes, Controllers & Benchmark)
* **Closed-Loop Digital Twin (`datacenter_env.py`)**:
  * Trained and verified a physical linear state-space model: $T_{t+1} = A T_t + B P_{\text{applied}} + c$ ($R^2 = 0.975$).
  * Restored physical non-negative energy calculation: $P_{\text{phys}} = \max(0, \mu_P + \sigma_P \cdot P_{\text{std}})$.
* **Physics-Constrained MPC Safety Filter (`results/mpc_guardrail.py`)**:
  * Implemented an online OSQP projection QP: $\min \|du - du_{\text{RL}}\|^2$ subject to $N=5$ step lookahead safety ($T_{\text{celCC}} \le T_{\text{max}}$) and actuator bounds.
  * Corrected OSQP box constraint validation ($l \le u$) and clamped actuator inputs.
* **Classical & Control-Theoretic Baselines (`baseline_controllers.py`)**:
  * Created `ThermostatController`: Rule-based deadband hysteresis feedback.
  * Created `AutonomousMPCController`: Full receding-horizon economic MPC directly minimizing power subject to safety constraints via OSQP.
* **Rigorous 4-Way Comparative Benchmark (`results/train_compare.py`)**:
  * Evaluated all 4 strategies under identical initial states across 50 test episodes.
  * Generated Pareto curves, learning traces, intervention metrics, and `eval_results_all.json`.

---

## 3. Results Summary (50-Episode Benchmark)

| Strategy | Paradigm | Mean Episode Energy (std) | Thermal Violations ($>T_{\text{max}}$) | Peak $T_{\text{celCC}}$ | Action Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Thermostat Baseline** | Industrial Rule-Based | $21,324.5$ | **0.00%** | $0.000\sigma$ | **0.02 ms** |
| **Autonomous MPC** | Receding-Horizon QP | $14,607.2$ (**-31.5%**) | **0.00%** | $-0.038\sigma$ | **0.35 ms** |
| **Pure RL (PPO)** | Unconstrained Deep RL | $13,991.2$ (-34.4%) | ❌ **12.60%** (Catastrophic) | $+1.490\sigma$ | 0.05 ms |
| **Safe RL (PPO + MPC)** | Guardrailed Hybrid | **$14,357.5$ (-32.7%)** | **0.00%** (Safe) | **$-0.012\sigma$** | **0.38 ms** |

### Key Takeaways:
1. **Pure RL is Unusable in Production**: Although Pure RL achieves low energy, it incurs a **12.60% safety violation rate**, leading to thermal emergencies.
2. **Safe RL Resolves the Safety Deficit**: The MPC guardrail intervenes in **17.65%** of decisions to clip risky actions, reducing violations to **0.00%** while saving **32.7% energy** over the thermostat.
3. **Autonomous MPC is the Production Winner**: Autonomous MPC delivers **31.5% energy savings with 0.00% violations** without requiring deep RL training or neural network inference.

---

## 4. Execution
```bash
# Run complete 4-way training and evaluation
cd results
python train_compare.py
```

