"""Figure 6: the Sioux Falls network with the studied zones, the optimal route flows of the studied
origin-destination pairs and the capacity-constrained links (Section 5.3).
    python fig_siouxfalls.py   -> fig_siouxfalls.png, fig_siouxfalls.pdf
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import app_common as C
import exp_traffic as ET
import traffic_model as tm

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": False})
BLUE, RED, ORANGE, GREY, INK = "#2563c9", "#c0504d", "#d9731a", "#b8b8b8", "#222222"

# node coordinates (longitude, latitude)
xy = {}
for line in open(os.path.join(tm.DATA, "SiouxFalls_node.tntp")):
    p = line.replace(";", "").split()
    if p and p[0].isdigit():
        xy[int(p[0])] = (float(p[1]), float(p[2]))

M = ET.M
links = tm.read_net()
xt = C.forward(ET.c0, ET.A, ET.b, ET.lo, ET.hi)
load = M["inc"] @ xt                                   # studied flow on each directed link (veh/h)
und = {}                                               # undirected segment -> max load over both directions
for idx, (i, j, cap, fft) in enumerate(links):
    key = tuple(sorted((i, j))); und[key] = max(und.get(key, 0.0), load[idx])

# capacity-constrained links (non-redundant capacity rows) and those binding in the optimal flows
used_idx = np.where(M["used"])[0]; nd, npth = len(M["dem"]), len(ET.c0)
slack = ET.A @ xt - ET.b; binding, constrained = set(), set()
for row, kind in enumerate(M["kinds"][ET.keep]):
    if kind == "capacity":
        i, j = links[used_idx[ET.keep[row] - (nd + 2 * npth)]][:2]
        (binding if abs(slack[row]) < 1e-6 else constrained).add(tuple(sorted((i, j))))
constrained -= binding

zones = sorted({z for od in M["studied"] for z in od})


def draw(ax):
    wmax = max(und.values())
    for (i, j), q in und.items():
        (x1, y1), (x2, y2) = xy[i], xy[j]
        ax.plot([x1, x2], [y1, y2], color=GREY, lw=1.2, zorder=1, solid_capstyle="round")
        if q > 1e-6:
            ax.plot([x1, x2], [y1, y2], color=BLUE, lw=3 + 9 * q / wmax, alpha=0.8, zorder=2, solid_capstyle="round")
        if (i, j) in binding:
            ax.plot([x1, x2], [y1, y2], color=RED, lw=2.4, zorder=3, dashes=(4, 2))
        elif (i, j) in constrained:
            ax.plot([x1, x2], [y1, y2], color=ORANGE, lw=2.2, zorder=3, dashes=(4, 2))
    for n, (x, y) in xy.items():
        z = n in zones
        ax.scatter(x, y, s=260 if z else 190, color="white", edgecolor=INK if z else "#777777",
                   linewidth=1.6 if z else 0.8, zorder=4)
        ax.text(x, y, str(n), ha="center", va="center", fontsize=9 if z else 8,
                fontweight="bold" if z else "normal", color=INK, zorder=5)
    ax.set_aspect(1 / np.cos(np.deg2rad(43.55)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, ax = plt.subplots(figsize=(6.6, 7.6))
draw(ax)
handles = [Line2D([], [], color=GREY, lw=1.2, label="link"),
           Line2D([], [], color=BLUE, lw=5, alpha=0.8, label="studied flows, optimal assignment\n(width $\\propto$ veh/h)"),
           Line2D([], [], color=RED, lw=2.4, dashes=(4, 2), label="link at capacity"),
           Line2D([], [], color=ORANGE, lw=2.2, dashes=(4, 2), label="capacity-constrained link,\nnot binding"),
           Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.6,
                  markersize=10, label="zone of a studied\norigin-destination pair")]
fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=(0, 0.09, 1, 1))
plt.savefig("fig_siouxfalls.png", dpi=200, bbox_inches="tight"); plt.savefig("fig_siouxfalls.pdf", bbox_inches="tight")
print("fig_siouxfalls.png; binding:", sorted(binding), "constrained:", sorted(constrained), "zones:", zones)
