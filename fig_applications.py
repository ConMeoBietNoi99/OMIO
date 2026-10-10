"""Figures for the engineering applications (Sections 5.2 and 5.3): cost regret of the imputed
decision for each OMIO member, clean records versus two erroneous records.
    python fig_applications.py   (needs res_dispatch.json, res_traffic.json)
    -> fig_dispatch.png, fig_traffic.png
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import section5_experiments as S   # shared style (serif font, grid)

DISP = {"average": "Average", "cvar50": "CVaR", "max": "Max", "trim": "Trimmed"}
LABS = ["average", "cvar50", "max", "trim"]
COLS = ["#8ecae6", "#f2a65a", "#c0504d"]
FIGS = [
    ("res_dispatch.json", "fig_dispatch.png", "cost increase over optimal dispatch (%)",
     [("clean", "clean records"), ("random_unit", "drop-out, random unit"), ("largest_unit", "drop-out, largest unit")]),
    ("res_traffic.json", "fig_traffic.png", "travel-time increase over optimal flows (%)",
     [("clean", "clean records"), ("random_route", "inflated count, random route"), ("busiest_route", "inflated count, busiest route")]),
]

for src, out, ylab, scen in FIGS:
    R = json.load(open(src))["summary"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(LABS)); w = 0.26
    for j, ((key, name), col) in enumerate(zip(scen, COLS)):
        ax.bar(x + (j - 1) * w, [R[m]["regret_" + key][0] for m in LABS], w,
               yerr=[R[m]["regret_" + key][1] for m in LABS], capsize=2.5, label=name,
               color=col, edgecolor="k", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([DISP[m] for m in LABS]); ax.set_ylabel(ylab)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close(); print(out)
