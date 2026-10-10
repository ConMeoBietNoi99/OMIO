"""DC economic dispatch on the ACTIVSg200 (Illinois 200-bus) synthetic grid, in the form A p >= b."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pypower.idx_bus import PD, BUS_TYPE, REF
from pandapower.pypower.idx_brch import RATE_A
from pandapower.pypower.idx_gen import GEN_BUS, PMAX, PMIN

def build(load_scale=1.0):
    net = pn.case_illinois200()
    pp.rundcpp(net)                                   # initialise internal ppc
    ppc = net._ppc
    bus, branch, gen, base = ppc["bus"], ppc["branch"], ppc["gen"], ppc["baseMVA"]
    nb = bus.shape[0]
    slack = int(np.where(bus[:, BUS_TYPE] == REF)[0][0])
    H = makePTDF(base, bus, branch, slack)            # branch x bus
    # generator data in ppc order, costs from poly_cost (linear coefficient = marginal cost)
    pc = net.poly_cost.set_index(["et", "element"])
    order = [("gen", i) for i in net.gen.index] + [("ext_grid", i) for i in net.ext_grid.index]
    gbus_pp = list(net.gen.bus.values) + list(net.ext_grid.bus.values)
    lookup = net._pd2ppc_lookups["bus"]
    gbus = np.array([lookup[b] for b in gbus_pp])
    pmax = np.array(list(net.gen.max_p_mw.values) + list(net.ext_grid.max_p_mw.values), float)
    pmin = np.array(list(net.gen.min_p_mw.values) + list(net.ext_grid.min_p_mw.values), float)
    c0 = np.array([pc.loc[o, "cp1_eur_per_mw"] for o in order], float)
    ng = len(c0)
    Pd = bus[:, PD] * load_scale
    D = Pd.sum()
    Cg = np.zeros((nb, ng)); Cg[gbus, np.arange(ng)] = 1.0
    G = H @ Cg; f0 = H @ Pd                           # flow = G p - f0
    rate = branch[:, RATE_A]
    lim = rate > 0
    G, f0, rate = G[lim], f0[lim], rate[lim]
    # A p >= b : demand, lower bounds, upper bounds, flow <= rate, flow >= -rate
    A = np.vstack([np.ones((1, ng)), np.eye(ng), -np.eye(ng), -G, G])
    b = np.concatenate([[D], pmin, -pmax, -rate - f0, -rate + f0])
    kinds = (["demand"] + ["pmin"] * ng + ["pmax"] * ng + ["flow+"] * len(rate) + ["flow-"] * len(rate))
    return dict(A=A, b=b, c0=c0, pmin=pmin, pmax=pmax, D=D, kinds=np.array(kinds), ng=ng)

if __name__ == "__main__":
    import numpy as np
    from app_common import forward
    for scale in (1.0, 1.05, 1.1):
        M = build(scale)
        x = forward(M["c0"], M["A"], M["b"], M["pmin"], M["pmax"])
        tight = M["kinds"][np.abs(M["A"] @ x - M["b"]) < 1e-6]
        print(f"load x{scale}: D = {M['D']:.0f} MW, binding constraints:", dict(zip(*np.unique(tight, return_counts=True))))
