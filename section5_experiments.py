import json, time
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREE = (None, None)
NONNEG = (0, None)
plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
COL = {"average": "#1f77b4", "cvar50": "#ff7f0e", "max": "#2ca02c", "trim": "#9467bd"}
LAB = {"average": "Average", "cvar50": "CVaR ($t=k/2$)", "max": "Max (discrete robust)",
       "trim": "Trimmed"}
MK = {"average": "o-", "cvar50": "s-", "max": "^-", "trim": "*-"}


# =====================================================================
#  Small dense LP/MILP builder addressed by integer variable ids
# =====================================================================
class Prog:
    def __init__(self):
        self.nv = 0; self.bounds = []; self.integ = []; self.obj = {}
        self.Aub, self.bub, self.Aeq, self.beq = [], [], [], []

    def add(self, n=1, bounds=NONNEG, integer=False):
        ids = list(range(self.nv, self.nv + n)); self.nv += n
        self.bounds += [bounds] * n; self.integ += [1 if integer else 0] * n
        return ids[0] if n == 1 else ids

    def cost(self, i, v): self.obj[i] = self.obj.get(i, 0.0) + v
    def le(self, row, rhs): self.Aub.append(row); self.bub.append(rhs)
    def eq(self, row, rhs): self.Aeq.append(row); self.beq.append(rhs)

    def _mat(self, rows):
        data, ri, ci = [], [], []
        for r, row in enumerate(rows):
            for j, v in row.items():
                if v != 0.0:
                    data.append(v); ri.append(r); ci.append(j)
        return sparse.csr_matrix((data, (ri, ci)), shape=(len(rows), self.nv))

    def _cvec(self):
        c = np.zeros(self.nv)
        for j, v in self.obj.items():
            c[j] = v
        return c

    def solve(self):
        c = self._cvec()
        if any(self.integ):
            cons = []
            if self.Aub:
                cons.append(LinearConstraint(self._mat(self.Aub), -np.inf, np.array(self.bub)))
            if self.Aeq:
                cons.append(LinearConstraint(self._mat(self.Aeq), np.array(self.beq), np.array(self.beq)))
            lb = np.array([b[0] if b[0] is not None else -np.inf for b in self.bounds])
            ub = np.array([b[1] if b[1] is not None else np.inf for b in self.bounds])
            return milp(c, constraints=cons, integrality=np.array(self.integ),
                        bounds=Bounds(lb, ub), options={"disp": False})
        A_ub = self._mat(self.Aub) if self.Aub else None
        b_ub = np.array(self.bub) if self.Aub else None
        A_eq = self._mat(self.Aeq) if self.Aeq else None
        b_eq = np.array(self.beq) if self.Aeq else None
        return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=self.bounds, method="highs")


# =====================================================================
#  Ordered-median weights and named schemes
# =====================================================================
def betas_from_lambda(lam):
    lam = np.asarray(lam, float); k = len(lam)
    return [(t, float(lam[k - t] - lam[k - t - 1]) if t < k else float(lam[0]))
            for t in range(1, k + 1)]


def scheme_lambda(name, k):
    if name == "average":
        return np.ones(k) / k
    if name == "max":
        w = np.zeros(k); w[-1] = 1.0; return w
    if name.startswith("cvar"):
        t = max(1, int(round(float(name[4:]) / 100.0 * k)))
        w = np.zeros(k); w[k - t:] = 1.0 / t; return w
    raise ValueError(name)


# =====================================================================
#  Facet subproblems (L-infinity), enumeration, and metrics
# =====================================================================
def _x_and_facet(P, A, b, v):
    m, n = A.shape
    x = P.add(n, bounds=FREE); x = [x] if n == 1 else x
    P.eq({x[j]: A[v, j] for j in range(n)}, b[v])
    for r in range(m):
        P.le({x[j]: -A[r, j] for j in range(n)}, -b[r])
    return x


def _linf_distances(P, x, Xhat):
    k = len(Xhat); n = len(x)
    d = P.add(k, bounds=NONNEG); d = [d] if k == 1 else d
    for i in range(k):
        for j in range(n):
            P.le({x[j]: 1.0, d[i]: -1.0}, Xhat[i][j])
            P.le({x[j]: -1.0, d[i]: -1.0}, -Xhat[i][j])
    return d


def _convex_facet(A, b, Xhat, betas, v):
    P = Prog(); x = _x_and_facet(P, A, b, v); d = _linf_distances(P, x, Xhat)
    k = len(Xhat)
    for (t, bt) in betas:
        eta = P.add(1, bounds=FREE); P.cost(eta, bt * t)
        u = P.add(k, bounds=NONNEG); u = [u] if k == 1 else u
        for i in range(k):
            P.cost(u[i], bt); P.le({d[i]: 1.0, eta: -1.0, u[i]: -1.0}, 0.0)
    res = P.solve()
    if not res.success:
        return None
    return np.array([res.x[x[j]] for j in range(len(x))]), float(res.fun)


def _trim_facet(A, b, Xhat, r, Mb, v):
    P = Prog(); x = _x_and_facet(P, A, b, v); d = _linf_distances(P, x, Xhat)
    k = len(Xhat)
    z = P.add(k, bounds=(0, 1), integer=True); z = [z] if k == 1 else z
    u = P.add(k, bounds=NONNEG); u = [u] if k == 1 else u
    P.eq({z[i]: 1.0 for i in range(k)}, k - r)
    for i in range(k):
        P.cost(u[i], 1.0)
        P.le({d[i]: 1.0, z[i]: Mb[i], u[i]: -1.0}, Mb[i])
    res = P.solve()
    if not res.success:
        return None
    return np.array([res.x[x[j]] for j in range(len(x))]), float(res.fun)


def solve_convex(A, b, Xhat, lam):
    betas = [(t, bt) for (t, bt) in betas_from_lambda(lam) if bt > 1e-12]
    best = None
    for v in range(A.shape[0]):
        out = _convex_facet(A, b, Xhat, betas, v)
        if out and (best is None or out[1] < best[0] - 1e-12):
            best = (out[1], v, out[0])
    obj, v, x = best
    return {"c": A[v] / np.abs(A[v]).sum(), "x": x, "obj": obj, "facet": v}


def solve_trimmed(A, b, Xhat, r, box):
    lo, hi = box; Xhat = np.asarray(Xhat)
    Mb = [float(max(np.max(Xhat[i] - lo), np.max(hi - Xhat[i]), 0.0)) + 1e-6
          for i in range(len(Xhat))]
    best = None
    for v in range(A.shape[0]):
        out = _trim_facet(A, b, Xhat, r, Mb, v)
        if out and (best is None or out[1] < best[0] - 1e-9):
            best = (out[1], v, out[0])
    obj, v, x = best
    return {"c": A[v] / np.abs(A[v]).sum(), "x": x, "obj": obj, "facet": v}


def forward_solve(c, A, b):
    res = linprog(c, A_ub=-A, b_ub=-b, bounds=[FREE] * A.shape[1], method="highs")
    return (res.x, res.fun) if res.success else (None, None)


def cheb_center(A, b, v):
    m, n = A.shape
    P = Prog(); x = P.add(n, bounds=FREE); x = [x] if n == 1 else x
    t = P.add(1, bounds=NONNEG); P.cost(t, -1.0)
    for i in range(m):
        if i == v:
            continue
        P.le({**{x[j]: -A[i, j] for j in range(n)}, t: 1.0}, -b[i])
    P.eq({x[j]: A[v, j] for j in range(n)}, b[v])
    res = P.solve()
    return np.array([res.x[x[j]] for j in range(n)])


def _linf(x, xh): return float(np.max(np.abs(np.asarray(x) - np.asarray(xh))))
def omf_value(lam, x, Xhat):
    return float(np.dot(np.asarray(lam, float), np.sort([_linf(x, xh) for xh in Xhat])))
def cosine(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# =====================================================================
#  Synthetic instance generator (box [0,10]^n + random facets, planted x0)
# =====================================================================
def make_instance(rng, n, m_extra, box=10.0):
    xc = rng.uniform(3.0, 7.0, size=n)
    av = rng.uniform(0.2, 1.0, size=n)
    rows, rhs = [], []
    for _ in range(m_extra):
        a = rng.uniform(0.2, 1.0, size=n); s = rng.uniform(0.5, 2.0)
        rows.append(a); rhs.append(a @ xc - s)
    v_idx = len(rows)
    rows.append(av.copy()); rhs.append(av @ xc)
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0; rows.append(e); rhs.append(0.0)
    for j in range(n):
        e = np.zeros(n); e[j] = -1.0; rows.append(e); rhs.append(-box)
    A = np.array(rows); b = np.array(rhs)
    return {"A": A, "b": b, "cstar": av / np.abs(av).sum(),
            "x0": cheb_center(A, b, v_idx), "box": (np.zeros(n), box * np.ones(n))}


CONVEX = ["average", "cvar50", "max"]


def eval_syn(A, b, Xhat, cstar, x0, box, r, k):
    res = {}
    for m in CONVEX:
        o = solve_convex(A, b, Xhat, scheme_lambda(m, k))
        res[m] = (cosine(o["c"], cstar), float(np.max(np.abs(o["x"] - x0))),
                  omf_value(scheme_lambda(m, k), o["x"], Xhat))
    t = solve_trimmed(A, b, Xhat, r, box)
    lam_tr = np.concatenate([np.ones(k - r), np.zeros(r)])
    res["trim"] = (cosine(t["c"], cstar), float(np.max(np.abs(t["x"] - x0))),
                   omf_value(lam_tr, t["x"], Xhat))
    return res


# =====================================================================
#  Experiment 1: single outlier   ->  fig_influence
# =====================================================================
def experiment1(n=3, m_extra=4, k=15, noise=0.25, rhos=(0, 2, 4, 8, 16),
                ninst=20, r=3, seed=200):
    labs = CONVEX + ["trim"]
    terr = {m: {rho: [] for rho in rhos} for m in labs}
    obj = {m: {rho: [] for rho in rhos} for m in labs}
    for s in range(ninst):
        rng = np.random.default_rng(seed + s)
        I = make_instance(rng, n, m_extra)
        base = [I["x0"] + rng.normal(0, noise, size=n) for _ in range(k)]
        direction = rng.normal(size=n); direction /= np.linalg.norm(direction)
        for rho in rhos:
            X = [x.copy() for x in base]
            X[0] = I["x0"] + rho * direction
            res = eval_syn(I["A"], I["b"], X, I["cstar"], I["x0"], I["box"], r, k)
            for m in labs:
                terr[m][rho].append(res[m][1]); obj[m][rho].append(res[m][2])
    summ = {"rhos": list(rhos),
            "terr": {m: [float(np.mean(terr[m][rho])) for rho in rhos] for m in labs},
            "obj":  {m: [float(np.mean(obj[m][rho]))  for rho in rhos] for m in labs}}
    return summ


def fig_influence(E, path="fig_influence.png"):
    rhos = E["rhos"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in CONVEX + ["trim"]:
        ax[0].plot(rhos, E["terr"][m], MK[m], color=COL[m], label=LAB[m], lw=1.8, ms=7)
    ax[0].set_xlabel("outlier magnitude $\\rho$"); ax[0].set_ylabel("$\\|x^*-x^0\\|_\\infty$")
    ax[0].set_title("(a) Influence on the estimate"); ax[0].legend(frameon=False, fontsize=9)
    for m in CONVEX + ["trim"]:
        ax[1].plot(rhos, E["obj"][m], MK[m], color=COL[m], label=LAB[m], lw=1.8, ms=7)
    ax[1].set_xlabel("outlier magnitude $\\rho$"); ax[1].set_ylabel("optimal objective value")
    ax[1].set_title("(b) Influence on the objective"); ax[1].legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(); print(path)


# =====================================================================
#  Experiment 2: breakdown   ->  fig_breakdown
# =====================================================================
def experiment2(n=3, m_extra=4, k=15, noise=0.25, mag=8.0, counts=(0, 1, 2, 3, 4),
                ninst=20, r=3, seed=300):
    labs = CONVEX + ["trim"]
    terr = {m: {c: [] for c in counts} for m in labs}
    for s in range(ninst):
        rng = np.random.default_rng(seed + s)
        I = make_instance(rng, n, m_extra)
        base = [I["x0"] + rng.normal(0, noise, size=n) for _ in range(k)]
        direction = rng.normal(size=n); direction /= np.linalg.norm(direction)
        for cnt in counts:
            X = [x.copy() for x in base]
            for j in range(cnt):
                X[j] = I["x0"] + mag * direction
            res = eval_syn(I["A"], I["b"], X, I["cstar"], I["x0"], I["box"], r, k)
            for m in labs:
                terr[m][cnt].append(res[m][1])
    return {"counts": list(counts),
            "terr": {m: [float(np.mean(terr[m][c])) for c in counts] for m in labs}}


def fig_breakdown(E, r=3, path="fig_breakdown.png"):
    cnt = E["counts"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for m in ["average", "max", "trim"]:
        ax.plot(cnt, E["terr"][m], MK[m], color=COL[m], label=LAB[m], lw=1.8, ms=7)
    ax.axvline(r, color="k", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("number of corrupted observations"); ax.set_ylabel("$\\|x^*-x^0\\|_\\infty$")
    ax.set_title("Breakdown of the trimmed estimator ($r=%d$)" % r)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(); print(path)


# =====================================================================
#  Experiment 3: Stigler diet (real 1939 USDA nutrition, 2026 BLS prices)
#  Nutrition per unit is fixed; the cost vector is the modern retail price.
#  Sources: Stigler (1945); BLS Average Price Data (U.S. city avg, Aug 2026,
#  series APU0000701111/701312/702111/709112/708111/710212/711211/712112).
# =====================================================================
S_FOODS = ["Wheat Flour", "Rice", "White Bread", "Milk", "Eggs", "Cheddar",
           "Bananas", "Potatoes"]
S_REQ = np.array([3.0, 70.0, 0.8, 12.0, 5.0, 1.8, 2.7, 18.0, 75.0])
S_PRICE = np.array([5.48, 1.111, 1.823, 1.0572, 2.272, 5.983, 0.652, 14.64])  # 2026 $/pkg
S_N = np.array([                                        # 9 nutrients x 8 foods, per pkg
    [16.092, 1.59, 1.185, 0.671, 0.9454, 1.7908, 0.2989, 4.862],     # Calories
    [507.96, 34.5, 38.552, 34.1, 77.588, 108.416, 3.66, 114.24],     # Protein
    [0.72, 0.045, 0.1975, 1.155, 0.326, 3.9688, 0.0244, 0.612],      # Calcium
    [131.4, 3.075, 9.085, 1.98, 16.952, 4.598, 1.83, 40.12],         # Iron
    [0.0, 0.0, 0.0, 1.848, 6.0636, 6.8002, 1.0614, 2.278],           # Vit A
    [19.944, 0.15, 1.0902, 0.44, 0.9128, 0.1936, 0.1525, 9.996],     # Vit B1
    [11.988, 0.36, 0.6715, 1.76, 2.119, 2.4926, 0.2135, 2.414],      # Vit B2
    [158.76, 4.5, 9.954, 0.77, 0.326, 0.968, 1.708, 67.32],          # Niacin
    [0.0, 0.0, 0.0, 19.47, 0.0, 0.0, 30.378, 857.48],                # Vit C
])


def stigler_instance(floor_frac=0.02):
    # floor_frac: minimum serving of every food, as a fraction of its cap
    # (a dietary-variety constraint).  floor_frac=0.0 -> original all-zero-prone
    # least-cost diet;  floor_frac~0.02 -> every food enters, no zero entries.
    n = len(S_FOODS); cap = 2 * S_REQ[0] / S_N[0]
    lb = floor_frac * cap
    A = np.vstack([S_N, np.eye(n), -np.eye(n)])
    b = np.concatenate([S_REQ, lb, -cap])
    return A, b, cap, (np.zeros(n), cap), S_PRICE / S_PRICE.sum()


def _gen_diet(rng, A, b, c0, var=0.25):
    c = c0 * (1 + rng.uniform(-var, var, size=len(c0))); c = np.clip(c, 1e-4, None); c /= c.sum()
    x, _ = forward_solve(c, A, b); return x


def _recover_pref(A, b, xstar, fallback_c, n_nutr, n, tol=1e-4):
    active = [i for i in range(n_nutr + n) if abs(A[i] @ xstar - b[i]) <= tol]
    if active:
        P = Prog()
        mu = {i: P.add(1, bounds=NONNEG) for i in active}
        c = P.add(n, bounds=NONNEG); c = [c] if n == 1 else c
        tau = P.add(1, bounds=NONNEG); P.cost(tau, 1.0)
        for j in range(n):
            row = {c[j]: 1.0}
            for i in active:
                row[mu[i]] = row.get(mu[i], 0.0) - A[i, j]
            P.eq(row, 0.0); P.le({c[j]: 1.0, tau: -1.0}, 0.0)
        P.eq({c[j]: 1.0 for j in range(n)}, 1.0)
        res = P.solve()
        if res.success:
            cv = np.array([res.x[c[j]] for j in range(n)])
            if cv.sum() > 1e-6:
                return cv / cv.sum()
    cv = np.clip(fallback_c, 0, None)
    return cv / cv.sum() if cv.sum() > 1e-6 else np.ones(n) / n


def _impute_diet(method, A, b, X, box, c0, n_nutr, n, r=2):
    if method == "trim":
        o = solve_trimmed(A, b, X, r, box)
    else:
        o = solve_convex(A, b, X, scheme_lambda(method, len(X)))
    return _recover_pref(A, b, o["x"], o["c"], n_nutr, n)


def experiment3(k=12, n_cheat=2, r=2, S=15, seed=20, milk_error=4.70):
    # milk_error: recorded milk quantity on a mis-recorded day.  4.70 is about
    # three times a heavy normal day (~1.57 qt) and well below the 8.94 cap --
    # a plausible over-record that still fools the worst-case member.
    A, b, cap, box, c0 = stigler_instance()
    n = len(S_FOODS); n_nutr = S_N.shape[0]; milk = S_FOODS.index("Milk")
    labs = ["average", "cvar50", "max", "trim"]

    def make_days(rng):
        X = [_gen_diet(rng, A, b, c0) for _ in range(k)]
        clean = [x.copy() for x in X]
        idx = rng.choice(k, size=n_cheat, replace=False)
        for ci in idx:
            X[ci] = X[ci].copy(); X[ci][milk] = milk_error
        return X, clean, idx

    def evaluate(X):
        return {m: {"chat": _impute_diet(m, A, b, X, box, c0, n_nutr, n, r).tolist(),
                    "spearman": float(spearmanr(_impute_diet(m, A, b, X, box, c0, n_nutr, n, r), c0).statistic)}
                for m in labs}

    xt, _ = forward_solve(c0, A, b)
    diet = {S_FOODS[j]: round(xt[j], 3) for j in range(n) if xt[j] > 1e-3}
    rng = np.random.default_rng(1)
    X, clean, idx = make_days(rng)
    illus = {"clean": evaluate(clean), "contaminated": evaluate(X),
             "cheat_idx": sorted(idx.tolist()), "days_cont": [x.tolist() for x in X],
             "cap_milk": float(cap[milk])}
    agg = {m: {"c": [], "x": []} for m in labs}
    for s in range(S):
        rng = np.random.default_rng(seed + s)
        Xs, cls, _ = make_days(rng)
        rc = evaluate(cls); rx = evaluate(Xs)
        for m in labs:
            agg[m]["c"].append(rc[m]["spearman"]); agg[m]["x"].append(rx[m]["spearman"])
    agg = {m: {"clean": [float(np.mean(agg[m]["c"])), float(np.std(agg[m]["c"]))],
               "cont":  [float(np.mean(agg[m]["x"])), float(np.std(agg[m]["x"]))]} for m in labs}
    return {"diet_true": diet, "c0": c0.tolist(), "illustrative": illus,
            "aggregate": agg, "milk": milk}


def fig_real_diet(E, path="fig_real_diet.png"):
    labs = ["average", "cvar50", "max", "trim"]
    disp = {"average": "Average", "cvar50": "CVaR", "max": "Max", "trim": "Trimmed"}
    agg = E["aggregate"]; cf = E["milk"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labs)); w = 0.38
    rc = [agg[m]["clean"][0] for m in labs]; rcs = [agg[m]["clean"][1] for m in labs]
    rx = [agg[m]["cont"][0] for m in labs]; rxs = [agg[m]["cont"][1] for m in labs]
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].bar(x - w/2, rc, w, yerr=rcs, capsize=3, label="clean",
              color="#8ecae6", edgecolor="k", linewidth=0.6)
    ax[0].bar(x + w/2, rx, w, yerr=rxs, capsize=3, label="contaminated (2 error days)",
              color="#e07a5f", edgecolor="k", linewidth=0.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels([disp[m] for m in labs], fontsize=9)
    ax[0].set_ylabel("Spearman $\\rho_S(\\hat c,\\, \\mathrm{price})$")
    ax[0].set_title("(a) Recovery of the real price ranking")
    ax[0].set_ylim(0, 1.0); ax[0].legend(frameon=False, fontsize=9, loc="lower left")
    rice_clean = [E["illustrative"]["clean"][m]["chat"][cf] for m in labs]
    rice_cont = [E["illustrative"]["contaminated"][m]["chat"][cf] for m in labs]
    ax[1].bar(x - w/2, rice_clean, w, label="clean", color="#8ecae6", edgecolor="k", linewidth=0.6)
    ax[1].bar(x + w/2, rice_cont, w, label="contaminated", color="#e07a5f", edgecolor="k", linewidth=0.6)
    ax[1].axhline(E["c0"][cf], color="k", ls="--", lw=1.3, label="true (real price)")
    ax[1].set_xticks(x); ax[1].set_xticklabels([disp[m] for m in labs], fontsize=9)
    ax[1].set_ylabel("imputed cost of Milk (mis-recorded)")
    ax[1].set_title("(b) Distortion of the mis-recorded food")
    ax[1].legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close(); print(path)


# =====================================================================
#  Main: run all three experiments, print numbers, save figures
# =====================================================================
def main():
    t0 = time.time()
    print("== Experiment 1: single outlier ==", flush=True)
    E1 = experiment1()
    print("  rhos", E1["rhos"])
    for m in CONVEX + ["trim"]:
        print(f"  {m:8} terr", [round(v, 2) for v in E1["terr"][m]],
              " obj", [round(v, 2) for v in E1["obj"][m]])
    fig_influence(E1)

    print(f"== Experiment 2: breakdown ==  ({time.time()-t0:.0f}s)", flush=True)
    E2 = experiment2()
    for m in CONVEX + ["trim"]:
        print(f"  {m:8} terr", [round(v, 2) for v in E2["terr"][m]])
    fig_breakdown(E2)

    print(f"== Experiment 3: Stigler diet ==  ({time.time()-t0:.0f}s)", flush=True)
    E3 = experiment3()
    print("  modern least-cost diet:", E3["diet_true"])
    print(f"  {'method':8}{'clean':>16}{'contaminated':>16}")
    for m in ["average", "cvar50", "max", "trim"]:
        a = E3["aggregate"][m]
        print(f"  {m:8}{a['clean'][0]:>8.3f}+/-{a['clean'][1]:<5.3f}"
              f"{a['cont'][0]:>8.3f}+/-{a['cont'][1]:<5.3f}")
    fig_real_diet(E3)

    with open("res_section5.json", "w") as f:
        json.dump({"exp1": E1, "exp2": E2, "exp3": E3}, f, indent=2)
    print(f"saved res_section5.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
