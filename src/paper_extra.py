"""Second device (WORK/results_v3_torino) and 20-qubit chain (<results dir>/n20_*), per circuit family: BlockGeo-QEM
against the best baseline, paired Wilcoxon over the 50 test circuits at the 5% level, as in tables/tab_wtl.tex.
For the 20-qubit chain also the errors of the model transferred from n = 10 on the two halves of the chain. GNN trunk
(<results dir>/gnnb_*, run_gnnb.py): error ratios to BlockGeo-QEM and to GNN-QEM.

usage: paper_extra.py [torino] [n20] [gnnb]      (default: all)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v3 as A
import paths
import qemlib as L

MET = ["local_bures", "local_trace", "mae_off", "mae_diag", "err_purity", "err_concurrence"]
FAMILY = {"tfim": "uniform TFIM", "dtfim": "disordered TFIM"}


def compare(df, ours, baselines, mets=MET, alpha=0.05):
    rows = []
    for key, g in df.groupby(["family", "n", "budget", "depth"]):
        if ours not in set(g.method):
            continue
        bl = [b for b in baselines if b in set(g.method)]
        for met in mets:
            P = g.pivot(index="sample", columns="method", values=met)
            best = min(bl, key=lambda b: P[b].mean())
            r = P[ours].mean() / P[best].mean()
            p = wilcoxon(P[ours], P[best]).pvalue if np.any(P[ours] != P[best]) else 1.0
            rows.append(dict(zip(["family", "n", "budget", "depth"], key), metric=met, ours=ours, best=best,
                             ours_mean=P[ours].mean(), best_mean=P[best].mean(), ratio=r, p=p,
                             outcome="win" if r < 1 and p < alpha else "loss" if r > 1 and p < alpha else "tie"))
    return pd.DataFrame(rows)


def summary(c):
    wtl = lambda s: f"{(s == 'win').sum()}/{(s == 'tie').sum()}/{(s == 'loss').sum()}"
    gm = lambda x: float(np.exp(np.log(x).mean()))
    return pd.concat([c.groupby(["ours", "budget", "metric"]).outcome.apply(wtl).rename("W/T/L"),
                      c.groupby(["ours", "budget", "metric"]).ratio.apply(gm).round(2).rename("ratio")], axis=1)


def torino():
    R = paths.results_dir("_v3_torino") + os.sep
    fs = [R + f for f in sorted(os.listdir(R)) if f.endswith("_samples.csv")]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    out = []
    for fam in [f for f in FAMILY if f in set(df.family)]:
        dff = df[df.family == fam]
        c = compare(dff, "BlockGeo-QEM", A.BASELINES)
        groups = [(f"{FAMILY[fam]}, all chain lengths pooled", c, dff)]
        if c.n.nunique() > 1:                           # the same tables for every chain length
            groups += [(f"{FAMILY[fam]} n={n}", cn, dff[dff.n == n]) for n, cn in c.groupby("n")]
        for title, cg, dg in groups:
            print(f"== ibm_torino (CZ), {title}\n" + summary(cg).to_string())
            print(cg[cg.metric == "local_bures"].pivot_table(index="budget", columns="depth", values="ratio").round(2))
            print(cg[cg.metric == "local_bures"].pivot_table(index="budget", columns="depth", values="best",
                                                             aggfunc="first"))
            abl = dg[dg.method.isin(["BlockGeo-QEM", "abl: HS state loss"])]
            print(abl.groupby(["budget", "method", "depth"]).local_bures.mean().unstack().round(4))
        out.append(c)
    return pd.concat(out)


def half_chain(fam, R, method="BlockGeo-QEM (transfer 10->20)"):
    """Errors of the model transferred from n = 10 on the two halves of the 20-qubit chain: blocks 0-8 lie on the
    default 10-qubit chain of the training circuits, blocks 10-18 on the added qubits 10-19 (block 9 straddles both).
    Mean over test circuits of the RMS block Bures distance and of the mean block trace distance of each half."""
    key = method.replace(" ", "_").replace(">", "")
    rows = []
    for f in sorted(os.listdir(R + "pred")):
        if not (f.startswith(f"n20_{fam}_B") and f.endswith(".npz")):
            continue
        budget, depth = f[len(f"n20_{fam}_B"):-4].split("_d")
        z = np.load(R + "pred/" + f)
        if key not in z.files:
            continue
        Y, S = z["__ideal__"], z[key]
        rho = lambda X, i, k: L.obs_to_rho(X[i, k], 2)
        bs = np.array([[max(L.bures_sq(rho(Y, i, k), rho(S, i, k)), 0.0) for k in range(Y.shape[1])]
                       for i in range(len(Y))])
        td = np.array([[L.trace_dist(rho(Y, i, k), rho(S, i, k)) for k in range(Y.shape[1])] for i in range(len(Y))])
        for met, old, new in (("local_bures", np.sqrt(bs[:, :9].mean(1)).mean(), np.sqrt(bs[:, 10:].mean(1)).mean()),
                              ("local_trace", td[:, :9].mean(), td[:, 10:].mean())):
            rows.append(dict(budget=int(budget), depth=int(depth), metric=met, qubits_0_9=old, qubits_10_19=new,
                             ratio=new / old))
    if rows:
        t = pd.DataFrame(rows).sort_values(["budget", "metric", "depth"])
        print(f"== {FAMILY[fam]} n=20, {method} on the two halves of the chain (largest deviation of the ratio from 1: "
              f"{np.abs(t.ratio - 1).max():.3f})\n" + t.round(4).to_string(index=False))


def n20():
    R = paths.RES_DIR + os.sep
    fs = [R + f for f in sorted(os.listdir(R)) if f.startswith("n20_") and f.endswith("_samples.csv")]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    out = []
    for fam in [f for f in FAMILY if f in set(df.family)]:
        dff = df[df.family == fam]
        base = [m for m in dff.method.unique() if "BlockGeo" not in m and not m.startswith("abl")]
        c = pd.concat([compare(dff, ours, base) for ours in dff.method.unique() if ours.startswith("BlockGeo")])
        print(f"== {FAMILY[fam]} n=20 (tensor-network data)\n" + summary(c).to_string())
        co = c[c.ours == "BlockGeo-QEM"]
        print("BlockGeo-QEM trained at n=20 over the best baseline, ratio and outcome per depth")
        print(co.pivot_table(index=["budget", "metric"], columns="depth", values="ratio").round(2).to_string())
        print(co.pivot_table(index=["budget", "metric"], columns="depth", values="outcome", aggfunc="first").to_string())
        print(dff.groupby(["budget", "method", "depth"]).local_bures.mean().unstack().round(4).to_string())
        half_chain(fam, R)
        out.append(c)
    return pd.concat(out)


def gnnb():
    """The GNN trunk with the heads, loss and training of BlockGeo-QEM (GNN-QEM (Bures)) against BlockGeo-QEM and
    against GNN-QEM, the same network trained with the squared error: ratios of the mean local Bures and trace
    distances per family, budget and depth (n = 8), with their range over the depths."""
    R = paths.RES_DIR + os.sep
    rows = []
    for fam in ("tfim", "dtfim"):
        for B in (100000, 10000):
            if not os.path.exists(R + f"gnnb_{fam}_n8_B{B}_samples.csv"):
                continue
            g = pd.read_csv(R + f"gnnb_{fam}_n8_B{B}_samples.csv")
            o, b = pd.read_csv(R + f"ours_{fam}_n8_B{B}_samples.csv"), pd.read_csv(R + f"base_{fam}_n8_B{B}_samples.csv")
            for met in ("local_bures", "local_trace"):
                trunk = g.groupby("depth")[met].mean()
                ours = o[o.method == "BlockGeo-QEM"].groupby("depth")[met].mean()
                mse = b[b.method == "GNN-QEM"].groupby("depth")[met].mean()
                rows += [dict(family=fam, budget=B, metric=met, depth=d, trunk_over_ours=trunk[d] / ours[d],
                              gnn_over_trunk=mse[d] / trunk[d]) for d in trunk.index]
    if not rows:
        print("no gnnb_*_samples.csv files; run src/run_gnnb.py first")
        return None
    c = pd.DataFrame(rows)
    for col, lab in (("trunk_over_ours", "GNN trunk / BlockGeo-QEM"), ("gnn_over_trunk", "GNN-QEM / GNN trunk")):
        t = c.pivot_table(index=["family", "budget", "metric"], columns="depth", values=col)
        t["min"], t["max"] = t.min(1), t.max(1)
        print(f"== {lab}, n=8, ratio of mean errors\n" + t.round(2).to_string())
        for met in ("local_bures", "local_trace"):
            s = c[c.metric == met].groupby("family")[col]
            print(f"   {met} over both budgets: " + ", ".join(f"{fam} {v.min():.2f}-{v.max():.2f}" for fam, v in s))
    return c


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    which = sys.argv[1:] or ["torino", "n20", "gnnb"]
    if "torino" in which:
        torino()
    if "n20" in which:
        n20()
    if "gnnb" in which:
        gnnb()
