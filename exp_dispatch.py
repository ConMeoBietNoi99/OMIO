"""Application A: recovering the nominal economic dispatch of the ACTIVSg200 (Illinois 200-bus)
synthetic grid from metered dispatch records, two of which contain telemetry drop-outs.

    python exp_dispatch.py      -> console summary, res_dispatch.json
"""
import json
import numpy as np
import dispatch_model as dm
import app_common as C

LOAD_SCALE, K, GAMMA, N_ERR, R, NINST = 1.10, 12, 0.25, 2, 2, 15

M = dm.build(LOAD_SCALE)
keep = C.prune(M["A"], M["b"], M["pmin"], M["pmax"])
A, b, lo, hi, c0 = M["A"][keep], M["b"][keep], M["pmin"], M["pmax"], M["c0"]


def make_obs(rng):
    """K daily dispatches at the same peak-hour load, under +/-25% fuel-cost shocks."""
    out = []
    for _ in range(K):
        c = np.maximum(c0 * (1 + rng.uniform(-GAMMA, GAMMA, size=len(c0))), 0.0)
        out.append(C.forward(c, A, b, lo, hi))
    return out


def dropout_random(X, rng):
    """Telemetry drop-out: on N_ERR random days, one randomly chosen unit producing at least
    50 MW is recorded at 0 MW."""
    for i in rng.choice(K, size=N_ERR, replace=False):
        j = int(rng.choice(np.where(X[i] >= 50.0)[0])); X[i][j] = 0.0
    return X


def dropout_largest(X, rng):
    """Gross drop-out: on N_ERR random days, the largest unit is recorded at 0 MW."""
    for i in rng.choice(K, size=N_ERR, replace=False):
        j = int(np.argmax(X[i])); X[i][j] = 0.0
    return X


CONTAMINATIONS = {"random_unit": dropout_random, "largest_unit": dropout_largest}


if __name__ == "__main__":
    kinds = dict(zip(*np.unique(M["kinds"][keep], return_counts=True)))
    print(f"ACTIVSg200 at {LOAD_SCALE:.0%} load: D = {M['D']:.0f} MW, {M['ng']} units, "
          f"{len(keep)} non-redundant constraints {kinds}")
    out = C.run(A, b, lo, hi, c0, make_obs, CONTAMINATIONS, R, NINST, seed=100)
    print(f"  optimal cost {out['opt']:.1f} $/h")
    C.print_summary(out, "MW")
    json.dump({"summary": out["summary"], "raw": out["raw"], "opt": out["opt"], "scenarios": out["scenarios"],
               "D": M["D"], "n_units": M["ng"], "m": int(len(keep))},
              open("res_dispatch.json", "w"), indent=1)
