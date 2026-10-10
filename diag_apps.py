"""Diagnostics quoted in the text: (i) share of erroneous records among the r records discarded by
the trimmed estimator; (ii) how often the trimmed target coincides with an observed decision and how
often the optimal decision x*(c0) is among the observations."""
import sys, json
import numpy as np
import app_common as C

def linf(a, b): return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

def diag(mod, seed, scen_names):
    A, b, lo, hi, c0 = mod.A, mod.b, mod.lo, mod.hi, mod.c0
    xt = C.forward(c0, A, b, lo, hi); opt = float(c0 @ xt)
    out = {"caught": {s: 0 for s in scen_names}, "trim_at_obs": 0, "xstar_in_obs": 0, "distinct_obs": [], "n": 0}
    for s in range(mod.NINST):
        rng = np.random.default_rng(seed + s)
        clean = mod.make_obs(rng)
        conts = {name: mod.CONTAMINATIONS[name]([x.copy() for x in clean], rng) for name in scen_names}
        out["distinct_obs"].append(len({tuple(np.round(x, 3)) for x in clean}))
        out["xstar_in_obs"] += int(any(C.dist_to_optimal_set(x, c0, A, b, lo, hi, opt) < 1e-3 for x in clean))
        xtr = C.estimate("trim", A, b, clean, lo, hi, mod.R)["x"]
        out["trim_at_obs"] += int(min(linf(xtr, x) for x in clean) < 1e-3)
        for name, X in conts.items():
            bad = [i for i in range(len(X)) if linf(X[i], clean[i]) > 1e-9]
            x = C.estimate("trim", A, b, X, lo, hi, mod.R)["x"]
            d = np.array([linf(x, xi) for xi in X]); dropped = set(np.argsort(d)[-mod.R:])
            out["caught"][name] += len(dropped & set(bad))
        out["n"] += 1
    out["caught_share"] = {k: v / (out["n"] * mod.N_ERR) for k, v in out["caught"].items()}
    return out

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "dispatch":
        import exp_dispatch as mod; res = diag(mod, 100, ["random_unit", "largest_unit"])
    else:
        import exp_traffic as mod; res = diag(mod, 200, ["random_route", "busiest_route"])
    print(which, json.dumps(res))
    json.dump(res, open(f"diag_{which}.json", "w"), indent=1)
