"""Figure 4: scalability of the facet decomposition, from timing_v2.json (written by timing_experiment.py).
    python timing_experiment.py   # writes timing_v2.json
    python plot_timing.py         # writes fig_timing.png
Table 3 should be filled from the same timing_v2.json so that table and figure come from one run.
"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from section5_experiments import COL, MK, LAB

T = json.load(open("timing_v2.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
ms = [row["m"] for row in T["convex_m"]]
for meth in ["average", "cvar50", "max"]:
    ax[0].plot(ms, [row["cell"][meth] for row in T["convex_m"]], MK[meth], color=COL[meth],
               label=LAB[meth], lw=1.8, ms=7)
ax[0].set_xlabel("number of facets $m$  ($n=3$, $k=15$)"); ax[0].set_ylabel("wall-clock time (s)")
ax[0].set_title("(a) Convex members vs. facets"); ax[0].legend(frameon=False, fontsize=9)
ks = [row["k"] for row in T["trim_k"]]
ax[1].semilogy(ks, [row["total_s"] for row in T["trim_k"]], MK["trim"], color=COL["trim"],
               label=LAB["trim"] + r" ($r=\lfloor k/5\rfloor$)", lw=1.8, ms=9)
ax[1].set_xlabel("number of observations $k$  ($n=3$, $m=11$)"); ax[1].set_ylabel("wall-clock time (s, log scale)")
ax[1].set_title("(b) Trimmed member vs. observations"); ax[1].legend(frameon=False, fontsize=9, loc="upper left")
plt.tight_layout(); plt.savefig("fig_timing.png", bbox_inches="tight"); plt.close(); print("fig_timing.png")

# values for Table 3
print("Table 3 (s):")
for row in T["convex_m"]:
    print(f"  m={row['m']:.0f}  " + "  ".join(f"{k}={v:.2f}" for k, v in row["cell"].items()))
for row in T["trim_m"]:
    print(f"  trimmed m={row['m']:.0f} (k=12, r=2): {row['total_s']:.2f}")
for row in T["convex_k"]:
    print(f"  k={row['k']}  " + "  ".join(f"{k}={v:.2f}" for k, v in row["cell"].items()))
for row in T["trim_k"]:
    print(f"  trimmed k={row['k']} (r={row['r']}): {row['total_s']:.2f}")
