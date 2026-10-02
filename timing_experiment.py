import time, json
import numpy as np
from section5_experiments import (make_instance, solve_convex, solve_trimmed,
                                   scheme_lambda)

def tcall(fn, reps=2):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def convex_cell(seed, n, m_extra, k, ninst=3, noise=0.25):
    tt = {"average": [], "cvar50": [], "max": []}; mv = []
    for s in range(ninst):
        rng = np.random.default_rng(seed + s)
        I = make_instance(rng, n, m_extra); A, b = I["A"], I["b"]; mv.append(A.shape[0])
        X = [I["x0"] + rng.normal(0, noise, size=n) for _ in range(k)]
        for mth in tt:
            lam = scheme_lambda(mth, k)
            tt[mth].append(tcall(lambda: solve_convex(A, b, X, lam)))
    m = float(np.mean(mv))
    return {mth: float(np.mean(v)) for mth, v in tt.items()}, m

def trim_cell(seed, n, m_extra, k, r, ninst=3, noise=0.25):
    ts = []; mv = []
    for s in range(ninst):
        rng = np.random.default_rng(seed + s)
        I = make_instance(rng, n, m_extra); A, b, box = I["A"], I["b"], I["box"]; mv.append(A.shape[0])
        X = [I["x0"] + rng.normal(0, noise, size=n) for _ in range(k)]
        ts.append(tcall(lambda: solve_trimmed(A, b, X, r, box), reps=1))
    return float(np.mean(ts)), float(np.mean(mv))

def main():
    t0 = time.time(); out = {"convex_k": [], "convex_m": [], "convex_n": [], "trim_k": [], "trim_m": []}
    save = lambda: json.dump(out, open("timing_v2.json", "w"), indent=2)

    print("== convex: vary k (n=3,m_extra=4) ==", flush=True)
    for k in [10, 20, 40, 80, 160]:
        c, m = convex_cell(1000, 3, 4, k)
        out["convex_k"].append({"n":3,"m":m,"k":k,"cell":c}); save()
        print(f"  k={k:3d} m={m:.0f}  "+"  ".join(f"{x}={c[x]*1e3:.1f}ms" for x in c), flush=True)

    print("== convex: vary m (n=3,k=15) ==", flush=True)
    for me in [4, 12, 26, 46, 96]:
        c, m = convex_cell(2000, 3, me, 15)
        out["convex_m"].append({"n":3,"m":m,"k":15,"cell":c}); save()
        print(f"  m={m:.0f}  "+"  ".join(f"{x}={c[x]*1e3:.1f}ms" for x in c), flush=True)

    print("== convex: vary n (m_extra=6,k=15) ==", flush=True)
    for n in [2, 3, 5, 8, 12]:
        c, m = convex_cell(3000, n, 6, 15)
        out["convex_n"].append({"n":n,"m":m,"k":15,"cell":c}); save()
        print(f"  n={n:2d} m={m:.0f}  "+"  ".join(f"{x}={c[x]*1e3:.1f}ms" for x in c), flush=True)

    print("== trim: vary k (n=3,m_extra=4,r=k//5) ==", flush=True)
    for k in [10, 15, 20, 25, 30]:
        r = max(1, k // 5)
        s, m = trim_cell(4000, 3, 4, k, r)
        out["trim_k"].append({"n":3,"m":m,"k":k,"r":r,"total_s":s}); save()
        print(f"  k={k:2d} m={m:.0f} r={r}  trim={s*1e3:.0f}ms  ({time.time()-t0:.0f}s)", flush=True)

    print("== trim: vary m (n=3,k=12,r=2) ==", flush=True)
    for me in [4, 12, 26, 46]:
        s, m = trim_cell(5000, 3, me, 12, 2)
        out["trim_m"].append({"n":3,"m":m,"k":12,"r":2,"total_s":s}); save()
        print(f"  m={m:.0f}  trim={s*1e3:.0f}ms  per_facet={s/m*1e3:.1f}ms  ({time.time()-t0:.0f}s)", flush=True)

    print(f"done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
