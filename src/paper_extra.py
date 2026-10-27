"""Second device (WORK/results_v3_torino) and 20-qubit chain (<results dir>/n20_*): BlockGeo-QEM against the best
baseline, paired Wilcoxon over the 50 test circuits at the 5% level, as in tables/tab_wtl.tex. GNN trunk
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

MET = ["local_bures", "local_trace", "mae_off", "mae_diag", "err_purity", "err_concurrence"]


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
    c = compare(df, "BlockGeo-QEM", A.BASELINES)
    print("== ibm_torino (CZ), uniform TFIM, all chain lengths pooled\n" + summary(c).to_string())
    print(c[c.metric == "local_bures"].pivot_table(index="budget", columns="depth", values="ratio").round(2))
    print(c[c.metric == "local_bures"].pivot_table(index="budget", columns="depth", values="best", aggfunc="first"))
    abl = df[df.method.isin(["BlockGeo-QEM", "abl: HS state loss"])].groupby(["budget", "method", "depth"])
    print(abl.local_bures.mean().unstack().round(4))
    if c.n.nunique() > 1:                               # the same tables for every chain length
        for n, cn in c.groupby("n"):
            print(f"== ibm_torino (CZ), uniform TFIM n={n}\n" + summary(cn).to_string())
            print(cn[cn.metric == "local_bures"].pivot_table(index="budget", columns="depth", values="ratio").round(2))
            print(cn[cn.metric == "local_bures"].pivot_table(index="budget", columns="depth", values="best",
                                                             aggfunc="first"))
            dn = df[(df.n == n) & df.method.isin(["BlockGeo-QEM", "abl: HS state loss"])]
            print(dn.groupby(["budget", "method", "depth"]).local_bures.mean().unstack().round(4))
    return c


def n20():
    R = paths.RES_DIR + os.sep
    fs = [R + f for f in sorted(os.listdir(R)) if f.startswith("n20_") and f.endswith("_samples.csv")]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    base = [m for m in df.method.unique() if "BlockGeo" not in m and not m.startswith("abl")]
    out = []
    for ours in [m for m in df.method.unique() if m.startswith("BlockGeo")]:
        out.append(compare(df, ours, base))
    c = pd.concat(out)
    print("== uniform TFIM n=20 (tensor-network data)\n" + summary(c).to_string())
    print(df.groupby(["budget", "method", "depth"]).local_bures.mean().unstack().round(4).to_string())
    return c



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
