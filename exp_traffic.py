"""Application B: recovering the nominal peak-hour route flows of the Sioux Falls network from
daily route counts, two of which are inflated by a detector error.

    python exp_traffic.py      -> console summary, res_traffic.json
"""
import json
import numpy as np
import traffic_model as tm
import app_common as C

PEAK, K, GAMMA, N_ERR, R, NINST = 0.20, 12, 0.50, 2, 2, 15

M = tm.build(peak_factor=PEAK)
keep = C.prune(M["A"], M["b"], M["lo"], M["hi"])
A, b, lo, hi, c0 = M["A"][keep], M["b"][keep], M["lo"], M["hi"], M["c0"]


def make_obs(rng):
    """K days of route flows; each day link travel times vary by +/-50% (weather, incidents)."""
    out = []
    for _ in range(K):
        t = M["fft"] * (1 + rng.uniform(-GAMMA, GAMMA, size=len(M["fft"])))
        out.append(C.forward(M["inc"].T @ t, A, b, lo, hi))
    return out


def inflate_random(X, rng):
    """Detector error: on N_ERR random days, the count of one randomly chosen route that carries
    traffic is recorded three times too high."""
    for i in rng.choice(K, size=N_ERR, replace=False):
        j = int(rng.choice(np.where(X[i] > 1.0)[0])); X[i][j] *= 3.0
    return X


def inflate_busiest(X, rng):
    """Gross detector error: on N_ERR random days, the busiest route's count is recorded three times too high."""
    for i in rng.choice(K, size=N_ERR, replace=False):
        j = int(np.argmax(X[i])); X[i][j] *= 3.0
    return X


CONTAMINATIONS = {"random_route": inflate_random, "busiest_route": inflate_busiest}


if __name__ == "__main__":
    kinds = dict(zip(*np.unique(M["kinds"][keep], return_counts=True)))
    print(f"Sioux Falls: {len(M['studied'])} OD pairs, {len(c0)} routes, peak demand {M['dem'].sum():.0f} veh/h, "
          f"{len(keep)} non-redundant constraints {kinds}")
    out = C.run(A, b, lo, hi, c0, make_obs, CONTAMINATIONS, R, NINST, seed=200)
    print(f"  optimal total travel time {out['opt']:.0f} veh-min")
    C.print_summary(out, "veh/h")
    json.dump({"summary": out["summary"], "raw": out["raw"], "opt": out["opt"], "scenarios": out["scenarios"],
               "n_od": len(M["studied"]), "n_routes": len(c0), "m": int(len(keep)),
               "demand": float(M["dem"].sum())}, open("res_traffic.json", "w"), indent=1)
