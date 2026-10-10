"""Figure 1: 2-D illustration of the random instance generator (make_instance with n=2).
    python fig_instance2d.py   -> fig_instance2d.png and fig_instance2d.pdf
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import HalfspaceIntersection
import section5_experiments as S

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3   # seed used for Figure 1
n, m_extra, k, sigma = 2, 4, 15, 0.25
rng = np.random.default_rng(seed)
I = S.make_instance(rng, n, m_extra)
A, b, x0, c0 = I["A"], I["b"], I["x0"], I["cstar"]
Xhat = np.array([x0 + rng.normal(0, sigma, size=n) for _ in range(k)])
v = m_extra                                   # index of planted facet

# polygon X = {Ax >= b}
hs = np.hstack([-A, b[:, None]])              # -a.x + b <= 0
P = HalfspaceIntersection(hs, x0 + 0.5 * c0 / np.linalg.norm(c0))
V = P.intersections; cen = V.mean(0)
V = V[np.argsort(np.arctan2(V[:, 1] - cen[1], V[:, 0] - cen[0]))]

BLUE, ORANGE, INK, MUTE = "#2563c9", "#d9731a", "#222222", "#8a8a8a"
plt.rcParams.update({"font.family": "serif", "font.size": 11, "mathtext.fontset": "cm", "axes.grid": False})
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})

def line_seg(a, bb, lo=-1, hi=11):
    # points of a.x = bb inside the plotting window
    if abs(a[1]) > 1e-9:
        xs = np.array([lo, hi]); ys = (bb - a[0] * xs) / a[1]
    else:
        ys = np.array([lo, hi]); xs = (bb - a[1] * ys) / a[0]
    return xs, ys

for j, a_ in enumerate(ax):
    a_.add_patch(Polygon(V, closed=True, facecolor="#e9edf3", edgecolor=INK, lw=1.4, zorder=1))
    for i in range(m_extra):               # random valid facets
        xs, ys = line_seg(A[i], b[i]); a_.plot(xs, ys, color=MUTE, lw=0.9, ls="--", zorder=2)
    xs, ys = line_seg(A[v], b[v]); a_.plot(xs, ys, color=BLUE, lw=1.0, ls=":", zorder=2)
    # planted facet: segment that belongs to X
    on = [p for p in V if abs(A[v] @ p - b[v]) < 1e-7]
    if len(on) == 2:
        on = np.array(on); a_.plot(on[:, 0], on[:, 1], color=BLUE, lw=3, zorder=3, solid_capstyle="round")
    a_.set_aspect("equal")

# ---- panel (a): whole instance ----
a0 = ax[0]
a0.add_patch(Rectangle((0, 0), 10, 10, fill=False, ec=INK, lw=0.8, ls="-", zorder=0))
a0.set_xlim(-0.6, 10.8); a0.set_ylim(-0.6, 10.8)
a0.scatter(*x0, marker="*", s=170, color=INK, zorder=6)
u = c0 / np.linalg.norm(c0)
a0.annotate("", xy=x0 + 1.9 * u, xytext=x0, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8), zorder=7)
a0.text(*(x0 + 2.05 * u + np.array([0.05, 0.05])), r"$\boldsymbol{c}^0$", color=BLUE, fontsize=13)
a0.text(*(x0 + np.array([-0.35, -0.75])), r"$\boldsymbol{x}^0$", color=INK, fontsize=13, ha="right")
a0.scatter(Xhat[:, 0], Xhat[:, 1], s=9, color=ORANGE, zorder=5)
a0.text(7.6, 7.6, r"$X$", fontsize=16, color=INK)
a0.text(10.0, -0.45, r"box $[0,10]^2$", fontsize=9.5, color=INK, ha="right")
# zoom window
h = 1.25
a0.add_patch(Rectangle(x0 - h, 2 * h, 2 * h, fill=False, ec=INK, lw=0.8, zorder=8))
a0.set_title(r"(a) Feasible region and planted facet", fontsize=11.5)
a0.set_xlabel(r"$x_1$"); a0.set_ylabel(r"$x_2$")

# ---- panel (b): zoom with observations ----
a1 = ax[1]
a1.set_xlim(x0[0] - h, x0[0] + h); a1.set_ylim(x0[1] - h, x0[1] + h)
inside = (A @ Xhat.T - b[:, None] >= -1e-12).all(0)
a1.scatter(Xhat[inside, 0], Xhat[inside, 1], s=46, color=ORANGE, ec="white", lw=0.8, zorder=5,
           label=r"observation $\hat{\boldsymbol{x}}_i\in X$")
a1.scatter(Xhat[~inside, 0], Xhat[~inside, 1], s=46, facecolor="white", ec=ORANGE, lw=1.6, zorder=5,
           label=r"observation $\hat{\boldsymbol{x}}_i\notin X$")
a1.scatter(*x0, marker="*", s=230, color=INK, zorder=6, label=r"planted optimum $\boldsymbol{x}^0$")
for r in (sigma, 2 * sigma):
    a1.add_patch(plt.Circle(x0, r, fill=False, ec=MUTE, lw=0.8, ls=(0, (2, 2)), zorder=4))
a1.text(x0[0] + 2 * sigma * 0.72, x0[1] + 2 * sigma * 0.72 + 0.04, r"$2\sigma$", color=MUTE, fontsize=10)
a1.annotate("", xy=x0 + 0.85 * u, xytext=x0, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6), zorder=7)
a1.text(*(x0 + 0.92 * u), r"$\boldsymbol{c}^0$", color=BLUE, fontsize=13)
a1.plot([], [], color=BLUE, lw=3, label=r"planted facet ($\boldsymbol{c}^0$ = normalized normal)")
a1.plot([], [], color=MUTE, lw=0.9, ls="--", label=r"random valid facets")
a1.set_title(r"(b) Zoom: $\hat{\boldsymbol{x}}_i=\boldsymbol{x}^0+\varepsilon_i$, $\varepsilon_i\sim\mathcal{N}(\mathbf{0},\sigma^2\boldsymbol{I})$", fontsize=11.5)
a1.set_xlabel(r"$x_1$")
fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9.5, columnspacing=1.6, handlelength=2.2,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.08, 1, 1))
out = sys.argv[2] if len(sys.argv) > 2 else "fig_instance2d"
plt.savefig(out + ".png", dpi=200, bbox_inches="tight"); plt.savefig(out + ".pdf", bbox_inches="tight")
print("seed", seed, "x0", x0.round(2), "c0", c0.round(3), "outside X:", int((~inside).sum()), "verts", len(V))
