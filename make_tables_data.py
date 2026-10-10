"""Data for the descriptive tables of Sections 5.2 and 5.3 (unit classes, congested line, recorded
dispatch of one instance; studied routes, capacity-constrained links, recorded route flows of one
instance). Writes tab_dispatch_data.json and tab_traffic_data.json.
    python make_tables_data.py
"""
import json
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.idx_brch import F_BUS, T_BUS, RATE_A
import app_common as C
import exp_dispatch as ED
import exp_traffic as ET
import traffic_model as tm


def dispatch_data():
    M, c0, lo, hi = ED.M, ED.c0, ED.lo, ED.hi
    ng = M["ng"]
    net = pn.case_illinois200()
    bus = np.array(list(net.gen.bus.values) + list(net.ext_grid.bus.values)) + 1
    xt = C.forward(c0, ED.A, ED.b, lo, hi)
    # cost classes: totals are identical in every optimal dispatch (checked by LP)
    classes = []
    for cost in sorted(set(np.round(c0, 2))):
        J = np.where(np.round(c0, 2) == cost)[0]
        classes.append({"cost": float(cost), "units": int(len(J)), "buses": sorted(bus[J].tolist()),
                        "pmin": float(lo[J].sum()), "pmax": float(hi[J].sum()), "optimal_output": float(xt[J].sum())})
    # congested line
    li = ED.keep[np.where(M["kinds"][ED.keep] == "flow+")[0][0]] - (1 + 2 * ng)
    pp.rundcpp(net); br = net._ppc["branch"]; bi = np.where(br[:, RATE_A] > 0)[0][li]
    lk = net._pd2ppc_lookups["bus"]; back = {int(lk[i]): i for i in range(len(lk))}
    line = {"from_bus": back[int(br[bi, F_BUS])] + 1, "to_bus": back[int(br[bi, T_BUS])] + 1, "rating": float(br[bi, RATE_A])}
    # recorded dispatch of the first instance (seed 100)
    rng = np.random.default_rng(100); clean = ED.make_obs(rng)
    cr = ED.dropout_random([x.copy() for x in clean], rng); cl = ED.dropout_largest([x.copy() for x in clean], rng)
    X = np.array(clean); top = sorted(np.argsort(-X.std(0))[:8], key=lambda j: bus[j])
    diff = lambda Y: [(i, int(j), int(bus[j]), float(clean[i][j])) for i in range(len(X)) for j in range(ng)
                      if abs(Y[i][j] - clean[i][j]) > 1e-9]
    return {"classes": classes, "congested_line": line, "top_units": [int(j) for j in top],
            "top_bus": [int(bus[j]) for j in top], "nominal": [float(xt[j]) for j in top],
            "days": X[:, top].round(1).tolist(), "daycost": [float(c0 @ x) for x in X],
            "err_random": diff(cr), "err_largest": diff(cl), "opt": float(c0 @ xt)}


def traffic_data():
    M, c0, lo, hi = ET.M, ET.c0, ET.lo, ET.hi
    xt = C.forward(c0, ET.A, ET.b, lo, hi); od = tm.read_trips(); links = tm.read_net()
    routes = []
    for p, (path, w) in enumerate(zip(M["paths"], M["path_od"])):
        o, d = M["studied"][w]
        routes.append({"p": p, "od": f"{o}-{d}", "daily": od[(o, d)], "peak": float(M["dem"][w]),
                       "route": "-".join(map(str, path)), "tau": float(c0[p]), "nominal": float(xt[p])})
    used_idx = np.where(M["used"])[0]; nd, npth = len(M["dem"]), len(c0)
    kinds = M["kinds"][ET.keep]; slack = ET.A @ xt - ET.b
    lk = []
    for row, kind in enumerate(kinds):
        if kind != "capacity":
            continue
        li = used_idx[ET.keep[row] - (nd + 2 * npth)]; i, j, cap, _ = links[li]
        lk.append({"link": f"{i}-{j}", "capacity": cap, "background": float(M["bg"][li]),
                   "residual": float(M["residual"][li]), "nominal_load": float(M["inc"][li] @ xt),
                   "binding": bool(abs(slack[row]) < 1e-6)})
    rng = np.random.default_rng(200); clean = ET.make_obs(rng)
    cr = ET.inflate_random([x.copy() for x in clean], rng); cb = ET.inflate_busiest([x.copy() for x in clean], rng)
    X = np.array(clean); var = [p for p in range(npth) if X[:, p].std() > 1e-6]
    diff = lambda Y: [(i, int(j), float(clean[i][j]), float(Y[i][j])) for i in range(len(X)) for j in range(npth)
                      if abs(Y[i][j] - clean[i][j]) > 1e-9]
    return {"routes": routes, "links": lk, "var": var, "days": X[:, var].round(1).tolist(),
            "daycost": [float(c0 @ x) for x in X], "err_random": diff(cr), "err_busiest": diff(cb), "opt": float(c0 @ xt)}


if __name__ == "__main__":
    json.dump(dispatch_data(), open("tab_dispatch_data.json", "w"), indent=1)
    json.dump(traffic_data(), open("tab_traffic_data.json", "w"), indent=1)
    print("tab_dispatch_data.json, tab_traffic_data.json")
