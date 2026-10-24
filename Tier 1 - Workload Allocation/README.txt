ME F320 project - code for the midsem report

Install:  pip install numpy scipy matplotlib networkx cvxopt
Run:      python run_midsem.py
Outputs:  results.json (every number quoted in the report) and figures/*.png

thermal_lp.py  - data instance, model M1, model M2, thermal simulator, baseline heuristics
run_midsem.py  - validation checks V1-V9, preliminary results, figures

All parameters are synthetic (see Assumption A8 in the report). Results are deterministic (fixed seed).
