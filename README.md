# Ordered-Median Inverse Optimization (OMIO)

Code for the paper *Ordered-Median Inverse Optimization for Recovery of Decisions from Noisy Records* by K. T. Nguyen, N. N. D. Duy, L. H. Hai and N. T. Hung.
The scripts reproduce every table and figure in Section 5 (computational experiments).

## Requirements

Python 3.10 or later (the paper used Python 3.13).

```bash
pip install -r requirements.txt
```

All linear programs are solved with HiGHS through `scipy.optimize.linprog`, and the mixed-integer programs through `scipy.optimize.milp`. No commercial solver is needed.

## Files

| File | Purpose |
|---|---|
| `section5_experiments.py` | Core implementation (facet decomposition, LP and MILP subproblems, tie-breaking rule of Remark 1, instance generator) and Experiments 1–2 |
| `timing_experiment.py`, `plot_timing.py` | Experiment 3: solve times (Table 3); `plot_timing.py` prints the values of Table 3 and draws an optional figure (not in the paper) |
| `fig_instance2d.py` | Figure 1: 2-D illustration of the random instance generator |
| `app_common.py` | Shared driver for the engineering applications (redundant-constraint removal, estimators, cost regret, distance to the optimal set) |
| `dispatch_model.py`, `exp_dispatch.py` | Section 5.2: electricity dispatch on the ACTIVSg200 synthetic grid (via pandapower) |
| `get_data.py`, `traffic_model.py`, `exp_traffic.py` | Section 5.3: route-flow recovery on the Sioux Falls network |
| `make_tables_data.py` | Data for the descriptive Tables 4, 5, 7, 8 and 9 |
| `diag_apps.py` | Diagnostics quoted in Sections 5.2–5.3 (share of erroneous records discarded by the trimmed member, distinct daily patterns) |
| `check_uniqueness.py` | Uniqueness of the imputed target with and without the tie-breaking rule (Remark 1) |
| `fig_activsg200.py` | Figure 4: the ACTIVSg200 grid, units by marginal cost and the congested line |
| `fig_applications.py` | Figures 5 and 7 |
| `fig_siouxfalls.py` | Figure 6: the Sioux Falls network, studied zones, optimal flows and capacity-constrained links |

## Reproducing the paper

```bash
python section5_experiments.py   # ~2 min: Experiments 1-2, Figures 2 and 3, res_section5.json
python fig_instance2d.py         # Figure 1
python timing_experiment.py      # ~3 min: Table 3 -> timing_v2.json
python plot_timing.py            # values of Table 3
python get_data.py               # Sioux Falls data -> ./data
python exp_dispatch.py           # ~10 min: Table 6 -> res_dispatch.json
python exp_traffic.py            # ~3 min:  Table 10 -> res_traffic.json
python make_tables_data.py       # Tables 4, 5, 7, 8, 9 -> tab_*_data.json
python diag_apps.py dispatch     # ~10 min: diagnostics quoted in Section 5.2
python diag_apps.py traffic      # diagnostics quoted in Section 5.3
python check_uniqueness.py dispatch   # ~10 min: effect of the tie-breaking rule of Remark 1
python check_uniqueness.py traffic
python fig_activsg200.py         # Figure 4 (after make_tables_data.py)
python fig_applications.py       # Figures 5 and 7
python fig_siouxfalls.py         # Figure 6
```

| Paper item | Produced by |
|---|---|
| Figure 1 | `fig_instance2d.py` |
| Experiment 1 (target error, objective, cosine similarity, planted-facet share), Figure 2 | `section5_experiments.py` |
| Experiment 2 (0–6 outliers), Figure 3 | `section5_experiments.py` |
| Table 3 (solve times) | `timing_experiment.py`, `plot_timing.py` |
| Section 5.2: Tables 4–6, Figures 4–5 | `make_tables_data.py`, `exp_dispatch.py`, `diag_apps.py`, `fig_activsg200.py`, `fig_applications.py` |
| Section 5.3: Tables 7–10, Figures 6–7 | `make_tables_data.py`, `exp_traffic.py`, `diag_apps.py`, `fig_siouxfalls.py`, `fig_applications.py` |
| Remark 1 (uniqueness of the target; not needed for the paper's results) | `check_uniqueness.py` |

## Notes

- **Tie-breaking.** With the l∞ distance the OMIO-optimal target is generally not unique. `solve_convex` and `solve_trimmed` therefore select, among the optimal targets on the chosen facet, one with the smallest total l1 distance to the observations (to the retained observations for the trimmed member); see Remark 1 of the paper. Pass `tie_break=False` to obtain the solver's arbitrary optimal target instead.
- Every experiment uses fixed random seeds, so all accuracy results are reproduced exactly. Solve times depend on the machine; the paper's timings were measured on an Intel Core i7-10610U laptop with 16 GB RAM.
- `python section5_experiments.py --diet` additionally runs a diet-planning illustration from an earlier version of the paper (not in the current paper). Its data are the Stigler (1945) nutrient table and U.S. Bureau of Labor Statistics average retail prices for August 2026 (series APU0000701111, 701312, 702111, 709112, 708111, 710212, 711211, 712112), hard-coded in `section5_experiments.py`.
- Solve times (Table 3) depend on the machine, so a rerun reproduces their trend, not their exact values; `plot_timing.py` prints a rerun's values in the layout of Table 3.
- The dispatch data (ACTIVSg200, Birchfield et al. 2017) ship with pandapower. The Sioux Falls data come from https://github.com/bstabler/TransportationNetworks and are downloaded by `get_data.py`.
- Constraints that can never be tight are removed before solving the applications. In the random instances of Section 5.1, some random facets may be redundant; their facet subproblems are infeasible and are skipped.

## Funding

Vietnam National Foundation for Science and Technology Development (NAFOSTED), grant 101.01-2025.15.
