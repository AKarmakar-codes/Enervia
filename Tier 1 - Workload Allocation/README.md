# Tier 1: Workload Allocation (Thermal-Aware LP)

Optimizes GPU workload dispatch across heterogeneous server clusters under thermal and SLA constraints.

---

## 1. What Was Originally Done (Friend)
* **Formulations**: 
  * **Model 1 (M1)**: Steady-state LP assuming instantaneous thermal equilibrium ($R_{\text{th}} P \le T_{\text{max}} - T_{\text{amb}}$).
  * **Model 2 (M2)**: Dynamic LP incorporating thermal capacitance via Zero-Order Hold (ZOH) discretization over a 24-hour horizon ($H=48$, $\Delta t=30\text{ min}$).
* **Validation**: Implemented 9 numerical validation checks ($V_1$–$V_9$) verifying energy conservation, non-negativity, and thermal bounds.
* **Outputs**: Produced baseline `results.json`, Figures 1–6 in `figures/`, and presentation report (`Midsem - ROLLNO1 ROLLNO2.pdf`).
* **Data Input**: Programmatic synthetic LLM workload demand profile (`demand_profile()`) and physical GPU parameters; does not use external CSV.

---

## 2. What We Did (Extensions & Additions)
* **Silicon Leakage Closed-Form Optimum (`thermal_lp_extensions.py`)**:
  * Coupled cooling power with temperature-dependent semiconductor leakage: $P_{\text{leak}} = \beta (T - T_{\text{ref}})$.
  * Solved the multi-objective tradeoff between chiller energy and silicon leakage, identifying the interior optimal facility setpoint $T^*$.
* **Receding-Horizon Rolling LP (`thermal_lp_extensions.py`)**:
  * Implemented an online rolling-horizon controller ($H=12$, $\Delta t=1\text{ min}$, execution time $13.43\text{ ms}$).
  * Subjected the cluster to unexpected $+40\%$ traffic surges and abrupt cooling degradation ($R_{\text{th}} \times 1.4$).
* **Artifacts**: Exported `extensions_results.json` and `figures/fig7_leakage_setpoint_optimum.png`.

---

## 3. Results Summary

### M1 vs. M2 Comparison
| Metric | Model 1 (Steady-State LP) | Model 2 (Dynamic ZOH LP) | Impact |
| :--- | :---: | :---: | :---: |
| **Total Jobs Processed** | 33,333 jobs | **43,800 jobs** | **+31.4% cluster throughput** |
| **Peak GPU Temperature** | $36.3\ ^\circ\text{C}$ (overly conservative) | **$47.7\ ^\circ\text{C}$** (safe exploitation) | Explores thermal inertia |
| **Thermal Violations ($>70\ ^\circ\text{C}$)** | 0.00% | **0.00%** | Zero constraint violations |
| **Solver Time** | 2.1 ms | **18.7 ms** | Real-time capable |

### Extension Findings
* **Optimal Ambient Setpoint ($T^*$)**: **$39.0\ ^\circ\text{C}$**. Below $39.0\ ^\circ\text{C}$, excessive chiller power dominates; above $39.0\ ^\circ\text{C}$, exponential silicon leakage dominates. Running at $T^* = 39.0\ ^\circ\text{C}$ achieves **16.6% total power reduction** vs. traditional $20.0\ ^\circ\text{C}$ over-chilling.
* **Rolling LP Robustness**: Sustained **0.00% thermal violations** during $+40\%$ traffic surges with average QP/LP re-solve time of **$13.43\text{ ms}$**.

---

## 4. Execution
```bash
# Run baseline midsem experiments
python run_midsem.py

# Run leakage optimum and rolling LP extensions
python thermal_lp_extensions.py
```

