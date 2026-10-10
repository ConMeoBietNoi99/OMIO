"""Uniqueness check for the imputed target (reported in the paper).

For each estimator, on the facet it selects, we compute the range of the true cost c0'x over
(a) all targets attaining the OMIO value (no tie-break), and
(b) all targets attaining the OMIO value and the minimal l1 distance of the tie-break.
    python check_uniqueness.py dispatch|traffic
"""
import sys
import numpy as np
from scipy.optimize import linprog
import app_common as C
import section5_experiments as S


def ranges(A, b, lo, hi, Xs, v, c0, opt):
    Xs = np.asarray(Xs); n = A.shape[1]; k = len(Xs)
    # variables: x (n), d (k) l_inf distances, e (k*n) l1 deviations
    nv = n + k + k * n
    Aub, bub = [np.c_[-A, np.zeros((A.shape[0], k + k * n))]], [-b]
    for i in range(k):
        for sgn in (1, -1):
            E = np.zeros((n, nv)); E[:, :n] = sgn * np.eye(n); E[:, n + i] = -1; Aub.append(E); bub.append(sgn * Xs[i])
            F = np.zeros((n, nv)); F[:, :n] = sgn * np.eye(n); F[:, n + k + i * n: n + k + (i + 1) * n] = -np.eye(n)
            Aub.append(F); bub.append(sgn * Xs[i])
    Aub = np.vstack(Aub); bub = np.concatenate(bub)
    Aeq = np.r_[A[v], np.zeros(k + k * n)][None, :]
    bounds = list(zip(lo, hi)) + [(0, None)] * (k + k * n)
    g = np.r_[np.zeros(n), np.ones(k), np.zeros(k * n)]
    h = np.r_[np.zeros(n + k), np.ones(k * n)]
    cc = np.r_[c0, np.zeros(k + k * n)]
    lp = lambda obj, A2, b2: linprog(obj, A_ub=A2, b_ub=b2, A_eq=Aeq, b_eq=[b[v]], bounds=bounds, method="highs")
    Z = lp(g, Aub, bub).fun
    A1 = np.vstack([Aub, g]); b1 = np.r_[bub, Z + S._tol(Z)]
    W = lp(h, A1, b1).fun
    A2 = np.vstack([A1, h]); b2 = np.r_[b1, W + S._tol(W)]
    pct = lambda val: 100 * (val - opt) / opt
    return ((pct(lp(cc, A1, b1).fun), pct(-lp(-cc, A1, b1).fun)),
            (pct(lp(cc, A2, b2).fun), pct(-lp(-cc, A2, b2).fun)))


def check(mod, seed, ninst=None):
    """Average and Trimmed members (their OMIO value is a plain sum of l_inf distances)."""
    A, b, lo, hi, c0 = mod.A, mod.b, mod.lo, mod.hi, mod.c0
    opt = c0 @ C.forward(c0, A, b, lo, hi); k = mod.K; ninst = ninst or mod.NINST
    out = {m: {"width_no_tb": [], "width_tb": []} for m in ("average", "trim")}
    for s in range(ninst):
        rng = np.random.default_rng(seed + s); X = mod.make_obs(rng)
        for m in ("average", "trim"):
            o = C.estimate(m, A, b, X, lo, hi, mod.R)
            if m == "trim":
                d = np.array([np.max(np.abs(o["x"] - xi)) for xi in X]); keep = np.argsort(d, kind="stable")[:k - mod.R]
            else:
                keep = range(k)
            (a0, a1), (b0, b1) = ranges(A, b, lo, hi, [X[i] for i in keep], o["facet"], c0, opt)
            out[m]["width_no_tb"].append(a1 - a0); out[m]["width_tb"].append(b1 - b0)
        print(f"  instance {s + 1}/{ninst} done", flush=True)
    for m, r in out.items():
        w0, w1 = np.array(r["width_no_tb"]), np.array(r["width_tb"])
        print(f"  {m:8} regret range over optimal targets: without tie-break mean width {w0.mean():6.2f} pp "
              f"(max {w0.max():6.2f}); with tie-break mean width {w1.mean():6.2f} pp (max {w1.max():6.2f}), "
              f"unique in {(w1 < 1e-3).sum()}/{len(w1)} instances")
    return out


if __name__ == "__main__":
    import json
    if sys.argv[1] == "dispatch":
        import exp_dispatch as mod; res = check(mod, 100)
    else:
        import exp_traffic as mod; res = check(mod, 200)
    json.dump(res, open(f"uniqueness_{sys.argv[1]}.json", "w"), indent=1)
