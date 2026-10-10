"""Path-based capacitated traffic assignment on the Sioux Falls network, in the form A f >= b.

Data: Transportation Networks for Research (github.com/bstabler/TransportationNetworks),
files SiouxFalls_net.tntp and SiouxFalls_trips.tntp (placed in ./data).
Daily trip-table volumes are converted to the peak hour with a peak-hour factor.
The study covers the N_OD largest origin-destination flows, each with its K_PATHS shortest
free-flow routes. All other demand is loaded all-or-nothing on free-flow shortest paths and
treated as background traffic; the studied flows use the residual capacity of each link.
"""
import os, re
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def read_net(path=os.path.join(DATA, "SiouxFalls_net.tntp")):
    links = []
    for line in open(path):
        t = line.strip()
        if not t or t.startswith("<") or t.startswith("~"):
            continue
        p = t.replace(";", "").split()
        links.append((int(p[0]), int(p[1]), float(p[2]), float(p[4])))   # i, j, capacity, free-flow time
    return links


def read_trips(path=os.path.join(DATA, "SiouxFalls_trips.tntp")):
    od, orig = {}, None
    for line in open(path):
        t = line.strip()
        if t.startswith("Origin"):
            orig = int(t.split()[1]); continue
        if orig is None or not t:
            continue
        for dest, val in re.findall(r"(\d+)\s*:\s*([\d.]+)", t):
            if int(dest) != orig and float(val) > 0:
                od[(orig, int(dest))] = float(val)
    return od


def build(n_od=8, k_paths=3, peak_factor=0.10, min_residual=0.10):
    links = read_net(); od = read_trips()
    G = nx.DiGraph()
    for idx, (i, j, cap, fft) in enumerate(links):
        G.add_edge(i, j, idx=idx, cap=cap, fft=fft)
    nl = len(links)
    cap = np.array([l[2] for l in links]); fft = np.array([l[3] for l in links])
    studied = sorted(od, key=od.get, reverse=True)[:n_od]
    # background traffic: all other OD pairs, all-or-nothing on free-flow shortest paths
    bg = np.zeros(nl)
    for (o, d), q in od.items():
        if (o, d) in studied:
            continue
        path = nx.shortest_path(G, o, d, weight="fft")
        for u, v in zip(path[:-1], path[1:]):
            bg[G[u][v]["idx"]] += peak_factor * q
    residual = np.maximum(cap - bg, min_residual * cap)
    # route sets
    paths, path_od, inc = [], [], []
    for w, (o, d) in enumerate(studied):
        for p in list(nx.shortest_simple_paths(G, o, d, weight="fft"))[:k_paths]:
            e = np.zeros(nl)
            for u, v in zip(p[:-1], p[1:]):
                e[G[u][v]["idx"]] = 1.0
            paths.append(p); path_od.append(w); inc.append(e)
    inc = np.array(inc).T                       # links x paths
    npth = len(paths); path_od = np.array(path_od)
    dem = np.array([peak_factor * od[w] for w in studied])
    c0 = inc.T @ fft                            # path free-flow times (min)
    used = inc.sum(1) > 0
    Dm = np.zeros((len(studied), npth)); Dm[path_od, np.arange(npth)] = 1.0
    lo = np.zeros(npth); hi = dem[path_od].astype(float)
    # A f >= b : OD demand, f >= 0, f <= demand, link residual capacity
    A = np.vstack([Dm, np.eye(npth), -np.eye(npth), -inc[used]])
    b = np.concatenate([dem, lo, -hi, -residual[used]])
    kinds = np.array(["demand"] * len(studied) + ["lower"] * npth + ["upper"] * npth + ["capacity"] * int(used.sum()))
    return dict(A=A, b=b, c0=c0, lo=lo, hi=hi, kinds=kinds, studied=studied, dem=dem, paths=paths,
                path_od=path_od, inc=inc, fft=fft, residual=residual, used=used, bg=bg, cap=cap)


if __name__ == "__main__":
    from scipy.optimize import linprog
    M = build()
    print("studied OD pairs and demand:", [(w, M["dem"][i]) for i, w in enumerate(M["studied"])])
    print("paths", len(M["paths"]), "rows", len(M["b"]), "links used", int(M["used"].sum()))
    r = linprog(M["c0"], A_ub=-M["A"], b_ub=-M["b"], bounds=list(zip(M["lo"], M["hi"])), method="highs")
    print("feasible", r.success)
    if r.success:
        sl = M["A"] @ r.x - M["b"]
        u, n = np.unique(M["kinds"][np.abs(sl) < 1e-6], return_counts=True); print("tight", dict(zip(u, n)))
        print("path costs", np.round(M["c0"], 1))
