"""Figure: the Sioux Falls network with the studied zones, the optimal route flows of the studied
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
dirload = {(i, j): load[idx] for idx, (i, j, cap, fft) in enumerate(links)}
AREA = [4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
# label placement per link: (side of the link, offset, position along the link from the lower node number)
LABEL_SIDE = {(10, 16): (1, 0.0040, 0.5), (10, 17): (-1, 0.0050, 0.72), (16, 17): (1, 0.0045, 0.5),
              (10, 15): (-1, 0.0050, 0.62), (10, 11): (-1, 0.0045, 0.5)}


def draw(ax, zoom):
    wmax = max(und.values())
    for (i, j), q in und.items():
        (x1, y1), (x2, y2) = xy[i], xy[j]
        ax.plot([x1, x2], [y1, y2], color=GREY, lw=1.2, zorder=1, solid_capstyle="round")
        if q > 1e-6:
            ax.plot([x1, x2], [y1, y2], color=BLUE, lw=(2 if not zoom else 3) + (5 if not zoom else 9) * q / wmax,
                    alpha=0.8, zorder=2, solid_capstyle="round")
        if (i, j) in binding:
            ax.plot([x1, x2], [y1, y2], color=RED, lw=2.4, zorder=3, dashes=(4, 2))
        elif (i, j) in constrained:
            ax.plot([x1, x2], [y1, y2], color=ORANGE, lw=2.2, zorder=3, dashes=(4, 2))
        if zoom and q > 1e-6:      # label both directions: "i->j / j->i" in veh/h
            a, b2 = dirload.get((i, j), 0.0), dirload.get((j, i), 0.0)
            side, off, t = LABEL_SIDE.get((i, j), (1, 0.0042, 0.5))
            mx, my = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            dx, dy = x2 - x1, y2 - y1; L = np.hypot(dx, dy); nx, ny = -dy / L, dx / L
            ax.text(mx + side * off * nx, my + side * off * ny, f"{i}$\\to${j}: {a:.0f}\n{j}$\\to${i}: {b2:.0f}",
                    fontsize=7.2, ha="center", va="center", color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for n, (x, y) in xy.items():
        if zoom and n not in AREA + [7]:
            continue
        z = n in zones
        ax.scatter(x, y, s=(260 if zoom else 170) if z else (190 if zoom else 120), color="white",
                   edgecolor=INK if z else "#777777", linewidth=1.6 if z else 0.8, zorder=4)
        ax.text(x, y, str(n), ha="center", va="center", fontsize=(9 if zoom else 7.5) if z else (8 if zoom else 6.8),
                fontweight="bold" if z else "normal", color=INK, zorder=5)
    ax.set_aspect(1 / np.cos(np.deg2rad(43.55)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(zoom); sp.set_color("#555555")


fig, axs = plt.subplots(1, 2, figsize=(11.5, 7.2), gridspec_kw={"width_ratios": [1, 1.25]})
draw(axs[0], False); draw(axs[1], True)
ax_x = [xy[n][0] for n in AREA]; ax_y = [xy[n][1] for n in AREA]
pad = 0.008
x0, x1, y0, y1 = min(ax_x) - pad, max(ax_x) + pad, min(ax_y) - pad, max(ax_y) + pad
axs[1].set_xlim(x0, x1); axs[1].set_ylim(y0, y1)
axs[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#555555", lw=0.9, zorder=6))
axs[0].set_title("(a) Sioux Falls network", fontsize=11); axs[1].set_title("(b) Study area, studied flows in veh/h", fontsize=11)
handles = [Line2D([], [], color=GREY, lw=1.2, label="link"),
           Line2D([], [], color=BLUE, lw=5, alpha=0.8, label="studied flows in the optimal assignment (width $\\propto$ veh/h)"),
           Line2D([], [], color=RED, lw=2.4, dashes=(4, 2), label="link at capacity"),
           Line2D([], [], color=ORANGE, lw=2.2, dashes=(4, 2), label="capacity-constrained link, not binding"),
           Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.6,
                  markersize=10, label="zone of a studied origin-destination pair")]
fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig("fig_siouxfalls.png", dpi=200, bbox_inches="tight"); plt.savefig("fig_siouxfalls.pdf", bbox_inches="tight")
print("fig_siouxfalls.png; binding:", sorted(binding), "constrained:", sorted(constrained), "zones:", zones)
