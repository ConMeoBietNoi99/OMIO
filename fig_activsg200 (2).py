"""Figure 4: the ACTIVSg200 grid (Section 5.2) with the generating units grouped by marginal cost
and the congested line. Uses the schematic bus coordinates shipped with pandapower and
tab_dispatch_data.json (written by make_tables_data.py).
    python fig_activsg200.py   -> fig_activsg200.png, fig_activsg200.pdf
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandapower.networks as pn

plt.rcParams.update({"font.family": "serif", "font.size": 9})
GREY, RED, INK = "#c4c4c4", "#c0504d", "#222222"
CLS = [(0.0, "#2a9d8f", "s", "0 \\$/MWh"), (6.71, "#e9a03b", "D", "6.71 \\$/MWh"),
       (19.0, "#2563c9", "o", "19.00 \\$/MWh"), (23.15, "#8e5ba8", "^", "23.15–23.19 \\$/MWh")]

net = pn.case_illinois200()
xy = {i + 1: tuple(json.loads(g)["coordinates"]) for i, g in enumerate(net.bus.geo.values)}  # bus numbers 1..200
edges = [(int(f) + 1, int(t) + 1) for f, t in zip(net.line.from_bus, net.line.to_bus)]
edges += [(int(h) + 1, int(l) + 1) for h, l in zip(net.trafo.hv_bus, net.trafo.lv_bus)]
D = json.load(open("tab_dispatch_data.json"))
cong = (D["congested_line"]["from_bus"], D["congested_line"]["to_bus"])

fig, ax = plt.subplots(figsize=(5.2, 5.0))
for i, j in edges:
    (x1, y1), (x2, y2) = xy[i], xy[j]
    ax.plot([x1, x2], [y1, y2], color=GREY, lw=0.6, zorder=1)
(x1, y1), (x2, y2) = xy[cong[0]], xy[cong[1]]
ax.plot([x1, x2], [y1, y2], color=RED, lw=2.6, zorder=3, solid_capstyle="round")
X = np.array(list(xy.values()))
ax.scatter(X[:, 0], X[:, 1], s=4, color="#9a9a9a", zorder=2)
for c in D["classes"]:
    key = 23.15 if c["cost"] >= 23 else c["cost"]
    _, col, mk, _ = next(k for k in CLS if k[0] == key)
    for bus in c["buses"]:
        ax.scatter(*xy[bus], s=42, marker=mk, color=col, edgecolor="white", linewidth=0.5, zorder=4)
for b in cong:
    ax.annotate(str(b), xy[b], xytext=(4, 4), textcoords="offset points", fontsize=8, color=RED, zorder=5)
ax.annotate("189", xy[189], xytext=(4, -9), textcoords="offset points", fontsize=8, color=INK, zorder=5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
h = [Line2D([], [], marker=mk, ls="", color=col, markersize=6, label=f"units at {lab}") for _, col, mk, lab in CLS]
h += [Line2D([], [], color=RED, lw=2.6, label="congested line 187–121"), Line2D([], [], color=GREY, lw=1, label="line or transformer")]
ax.legend(handles=h, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2)
plt.tight_layout()
plt.savefig("fig_activsg200.png", dpi=300, bbox_inches="tight"); plt.savefig("fig_activsg200.pdf", bbox_inches="tight")
print("fig_activsg200.png")
