"""Shared driver for the engineering applications (electricity dispatch, traffic assignment).

Forward problem: min c'x  s.t.  A x >= b,  lo <= x <= hi  (bounds are also rows of A).
Each application supplies A, b, lo, hi, the true cost c0, an observation generator and a
contamination rule. Every OMIO member is solved with the facet decomposition of
section5_experiments.py (Theorem 1 for convex members, Theorem 2 for the trimmed member).
"""
import numpy as np
from scipy.optimize import linprog
import section5_experiments as S

METHODS = ["average", "cvar50", "max", "trim"]


def forward(c, A, b, lo, hi):
    r = linprog(c, A_ub=-A, b_ub=-b, bounds=list(zip(lo, hi)), method="highs")
    return r.x if r.success else None


def prune(A, b, lo, hi, tol=1e-7):
    """Indices of rows that can be tight on X; the others are redundant and are removed
    (the paper assumes X is described without redundant constraints)."""
    keep = []
    for v in range(A.shape[0]):
        r = linprog(A[v], A_ub=-A, b_ub=-b, bounds=list(zip(lo, hi)), method="highs")
        if r.success and r.fun <= b[v] + tol * max(1.0, abs(b[v])):
            keep.append(v)
    return np.array(keep)


def dist_to_optimal_set(x, c0, A, b, lo, hi, opt):
    """min ||x - z||_inf over optimal decisions z (c0'z = opt, z in X). Handles ties in c0."""
    n = len(x)
    # variables (z, t): minimise t
    cost = np.r_[np.zeros(n), 1.0]
    Aub = [np.c_[-A, np.zeros((A.shape[0], 1))],                 # A z >= b
           np.c_[np.eye(n), -np.ones((n, 1))],                     # z - x <= t
           np.c_[-np.eye(n), -np.ones((n, 1))]]                    # x - z <= t
    bub = [-b, x, -x]
    Aeq = np.r_[c0, 0.0][None, :]
    r = linprog(cost, A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), A_eq=Aeq,
                b_eq=[opt * (1 + 1e-9)], bounds=list(zip(lo, hi)) + [(0, None)], method="highs")
    if not r.success:   # tiny numerical slack on the optimal-value equality
        r = linprog(cost, A_ub=np.vstack(Aub + [np.r_[c0, 0.0][None, :]]),
                    b_ub=np.concatenate(bub + [[opt * (1 + 1e-7) + 1e-7]]),
                    bounds=list(zip(lo, hi)) + [(0, None)], method="highs")
    return float(r.fun)


def estimate(method, A, b, X, lo, hi, r):
    if method == "trim":
        return S.solve_trimmed(A, b, X, r, (lo, hi))
    return S.solve_convex(A, b, X, S.scheme_lambda(method, len(X)))


def run(A, b, lo, hi, c0, make_obs, contaminations, r, ninst=15, seed=0, verbose=True):
    """For each instance: clean observations and each contamination scenario -> every OMIO member
    -> (cost regret %, distance of the imputed target to the optimal set).
    `contaminations` maps a scenario name to a function (list of observations, rng) -> list."""
    xt = forward(c0, A, b, lo, hi); opt = float(c0 @ xt)
    scen = ["clean"] + list(contaminations)
    res = {m: {f"{q}_{s}": [] for q in ("regret", "dist") for s in scen} for m in METHODS}
    for s in range(ninst):
        rng = np.random.default_rng(seed + s)
        clean = make_obs(rng)
        data = {"clean": clean}
        for name, f in contaminations.items():
            data[name] = f([x.copy() for x in clean], rng)
        for name, X in data.items():
            for m in METHODS:
                x = estimate(m, A, b, X, lo, hi, r)["x"]
                res[m][f"regret_{name}"].append(100.0 * (c0 @ x - opt) / opt)
                res[m][f"dist_{name}"].append(dist_to_optimal_set(x, c0, A, b, lo, hi, opt))
        if verbose:
            print(f"  instance {s + 1}/{ninst} done", flush=True)
    ms = lambda v: [float(np.mean(v)), float(np.std(v))]
    summary = {m: {k: ms(v) for k, v in d.items()} for m, d in res.items()}
    return {"opt": opt, "scenarios": scen, "raw": res, "summary": summary}


def print_summary(out, unit):
    for q, u in (("regret", "%"), ("dist", unit)):
        print(f"  {q} ({u}):  " + "".join(f"{sc:>22}" for sc in out["scenarios"]))
        for m in METHODS:
            a = out["summary"][m]
            print(f"  {m:14}" + "".join(f"{a[q + '_' + sc][0]:>13.3f}+/-{a[q + '_' + sc][1]:<6.3f}"
                                        for sc in out["scenarios"]))
