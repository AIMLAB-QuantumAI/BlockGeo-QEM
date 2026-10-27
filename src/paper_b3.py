"""Tables and printed statistics of the three-qubit-block study (n = 8, uniform and disordered TFIM).

For each family it compares BlockGeo-QEM with the best baseline for two- and three-qubit blocks and resolves the gain
of the Bures geometry by the ideal block purity and by the smallest eigenvalue of the ideal block. Two-qubit results
are read from results_v3, three-qubit results from results_v3b3 (runs with BLOCKGEO_TAG=_v3b3 BLOCKGEO_B=3, including
run_gnn_b3.py for the three-qubit graph network).

Writes, per family F (suffix "" for tfim, "_dtfim" for dtfim),
  results_v3b3/compare_b2_b3_F_n8.csv                ratio to the best baseline and W/T/L per block size, budget, depth
  results_v3b3/purity_resolved_b2_b3_F_n8_B100000.csv  block Bures distances with ideal purity and smallest eigenvalue
  tables/tab_b3<suffix>.tex                           BlockGeo-QEM / best baseline (geometric mean and per depth),
                                                      local Bures and trace distances at 1e4 and 1e5 shots, with W/T/L
  tables/tab_b3_gain<suffix>.tex                      Bures distance of the Hilbert-Schmidt model over that of
                                                      BlockGeo-QEM by purity and by smallest eigenvalue (1e5 shots)
and, when both families are done, tables/tab_b3_gain_both.tex with the gain columns of both families side by side.

usage: python src/paper_b3.py [tfim] [dtfim]        (default: both)
"""
import os
import sys

import numpy as np
import pandas as pd

from paths import TAB_DIR, results_dir

R2, R3 = results_dir("_v3"), results_dir("_v3b3")
DEPTHS = [1, 2, 3, 4, 6, 8]
BASELINES = ["Noisy", "ZNE-lin", "ZNE-Rich", "ZNE-exp", "PEC", "CDR", "vnCDR", "CDR-family", "OLS", "vnOLS", "RF",
             "MLP-QEM", "GNN-QEM", "BlockGeo-v1"]
# two-qubit predictions with BlockGeo-QEM and the HS model at 1e5 shots (uniform TFIM: the --tag nl run)
OURS2 = {"tfim": "oursnl_tfim_n8_B100000", "dtfim": "ours_dtfim_n8_B100000"}


def suffix(fam):
    return "" if fam == "tfim" else "_" + fam


def load_results(fam, b, B):
    R = R3 if b == 3 else R2
    parts = [pd.read_csv(os.path.join(R, f"base_{fam}_n8_B{B}_samples.csv")),
             pd.read_csv(os.path.join(R, f"ours_{fam}_n8_B{B}_samples.csv"))]
    if b == 3:
        parts.append(pd.read_csv(os.path.join(R, f"gnn_{fam}_n8_B{B}_samples.csv")))
    return pd.concat(parts, ignore_index=True)


def build_inputs(fam):
    from scipy.stats import wilcoxon

    import qemlib as L

    rows = []
    for b in (2, 3):
        for B in (100000, 10000):
            df = load_results(fam, b, B)
            for d in sorted(df.depth.unique()):
                s = df[df.depth == d]
                for met in ("local_bures", "local_trace"):
                    P = s.pivot_table(index="sample", columns="method", values=met)
                    best = min([m for m in BASELINES if m in P], key=lambda m: P[m].mean())
                    o = P["BlockGeo-QEM"]
                    r, pv = o.mean() / P[best].mean(), wilcoxon(o, P[best]).pvalue
                    rows.append(dict(b=b, B=B, depth=d, metric=met, ratio=r, best=best,
                                     out="W" if (r < 1 and pv < .05) else "L" if (r > 1 and pv < .05) else "T",
                                     hs_gain=P["abl: HS state loss"].mean() / o.mean() if "abl: HS state loss" in P
                                     else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(R3, f"compare_b2_b3_{fam}_n8.csv"), index=False)

    def block_bures(S, Y, b):
        return np.array([[np.sqrt(max(L.bures_sq(L.obs_to_rho(Y[i, k], b), L.obs_to_rho(S[i, k], b)), 0.0))
                          for k in range(S.shape[1])] for i in range(len(S))])

    recs = []
    for b, P, pre in ((2, os.path.join(R2, "pred"), OURS2[fam]), (3, os.path.join(R3, "pred"), f"ours_{fam}_n8_B100000")):
        for d in DEPTHS:
            z = dict(np.load(os.path.join(P, f"{pre}_d{d}.npz")))
            Y = z["__ideal__"]
            pur = (1 + (Y ** 2).sum(-1)) / 2 ** b
            lmin = np.array([[np.linalg.eigvalsh(L.obs_to_rho(Y[i, k], b)).min() for k in range(Y.shape[1])]
                             for i in range(len(Y))])
            recs.append(pd.DataFrame({"b": b, "depth": d, "purity": pur.ravel(), "lmin": lmin.ravel(),
                                      "ours": block_bures(z["BlockGeo-QEM"], Y, b).ravel(),
                                      "hs": block_bures(z["abl_HS_state_loss"], Y, b).ravel()}))
    pd.concat(recs, ignore_index=True).to_csv(os.path.join(R3, f"purity_resolved_b2_b3_{fam}_n8_B100000.csv"),
                                              index=False)


def write_table(name, rows):
    body = "\n".join(r for r in rows if r)
    body = body[:-3] if body.endswith(" \\\\") else body          # no trailing \\: it would add an empty row
    with open(os.path.join(TAB_DIR, name), "w") as f:
        f.write(body + "\n")


def ratio_table(fam):
    t = pd.read_csv(os.path.join(R3, f"compare_b2_b3_{fam}_n8.csv"))
    rows = []
    for met, mname in (("local_bures", "$D_B$"), ("local_trace", "$T$")):
        for B, bname in ((10000, "$10^4$"), (100000, "$10^5$")):
            for b in (2, 3):
                x = t[(t.metric == met) & (t.B == B) & (t.b == b)].set_index("depth").loc[DEPTHS]
                wtl = f"{(x.out == 'W').sum()}/{(x.out == 'T').sum()}/{(x.out == 'L').sum()}"
                gm = float(np.exp(np.log(x.ratio).mean()))
                cells = " & ".join(f"{r:.2f}" for r in x.ratio)
                head = f"\\multirow{{2}}{{*}}{{{mname}}} & \\multirow{{2}}{{*}}{{{bname}}}" if b == 2 else " & "
                rows.append(f"{head} & {b} & {wtl} & {gm:.2f} & {cells} \\\\")
            rows.append("\\hline" if not (met == "local_trace" and B == 100000) else "")
    write_table(f"tab_b3{suffix(fam)}.tex", rows)
    return t


def gain_table(fam):
    df = pd.read_csv(os.path.join(R3, f"purity_resolved_b2_b3_{fam}_n8_B100000.csv"))
    pb = [0.0, 0.4, 0.55, 0.7, 0.85, 0.95, 1.0001]
    lb = [1e-13, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 1.0]
    df["pbin"] = pd.cut(df.purity, pb, right=False)
    df["lbin"] = pd.cut(df.lmin.clip(lower=1e-12), lb, right=False)
    plabels = ["$<0.4$", "0.4--0.55", "0.55--0.7", "0.7--0.85", "0.85--0.95", "$>0.95$"]
    llabels = ["$<10^{-6}$", "$10^{-6}$--$10^{-4}$", "$10^{-4}$--$10^{-3}$", "$10^{-3}$--$10^{-2}$",
               "$10^{-2}$--0.05", "$>0.05$"]
    rows = []
    for key, labels in (("pbin", plabels), ("lbin", llabels)):
        g = df.groupby([key, "b"], observed=True).agg(blocks=("ours", "size"), ours=("ours", "mean"),
                                                      hs=("hs", "mean"), lmin=("lmin", "median"))
        for bin_, lab in zip(sorted(df[key].cat.categories), labels):
            cells = []
            for b in (2, 3):
                if (bin_, b) in g.index:
                    r = g.loc[(bin_, b)]
                    cells.append(f"{int(r.blocks)} & {r.hs / r.ours:.2f}")
                else:
                    cells.append("-- & --")
            rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        rows.append("\\hline")
    write_table(f"tab_b3_gain{suffix(fam)}.tex", rows[:-1])
    return df


def merge_gain_tables():
    """tab_b3_gain_both.tex: the label column and the gain columns of uniform and of disordered TFIM."""
    def rows(name):
        return [line.rstrip("\n") for line in open(os.path.join(TAB_DIR, name)) if line.strip()]

    a, b = rows("tab_b3_gain.tex"), rows("tab_b3_gain_dtfim.tex")
    assert len(a) == len(b), (len(a), len(b))
    out = []
    for la, lb in zip(a, b):
        if la.strip() == "\\hline":
            assert lb.strip() == "\\hline"
            out.append("\\hline")
            continue
        fa = [x.strip() for x in la.replace("\\\\", "").split("&")]
        fb = [x.strip() for x in lb.replace("\\\\", "").split("&")]
        assert fa[0] == fb[0], (fa[0], fb[0])
        out.append(" & ".join(fa + fb[1:]) + " \\\\")
    write_table("tab_b3_gain_both.tex", out)


def numbers(fam, t, df):
    pd.set_option("display.width", 200)
    print(f"== {fam}: BlockGeo-QEM / best baseline, and best baseline")
    print(t.pivot_table(index=["metric", "B", "b"], columns="depth", values="ratio").round(2).to_string())
    print(t.pivot_table(index=["metric", "B", "b"], columns="depth", values="best", aggfunc="first").to_string())
    print(f"== {fam}: HS / BlockGeo-QEM per depth (1e5)")
    print(t[(t.B == 100000)].pivot_table(index=["metric", "b"], columns="depth", values="hs_gain").round(2).to_string())
    print(f"== {fam}: median smallest eigenvalue by purity bin")
    print(df.groupby(["pbin", "b"], observed=True).lmin.median().unstack().to_string())
    print(f"== {fam}: mean Bures distance of BlockGeo-QEM by purity bin")
    print(df.groupby(["pbin", "b"], observed=True).ours.mean().unstack().round(4).to_string())
    for b in (2, 3):
        g = load_results(fam, b, 100000).groupby(["method", "depth"]).local_bures.mean().unstack()
        print(f"== {fam}: b={b}, 1e5 shots, Bures distance of Noisy, GNN-QEM and BlockGeo-QEM")
        print(g.loc[[m for m in ("Noisy", "GNN-QEM", "BlockGeo-QEM") if m in g.index]].round(4).to_string())


if __name__ == "__main__":
    families = [a for a in sys.argv[1:]] or ["tfim", "dtfim"]
    bad = [f for f in families if f not in OURS2]
    if bad:
        raise SystemExit(f"unknown family {bad}; choose from {sorted(OURS2)}")
    os.makedirs(TAB_DIR, exist_ok=True)
    for fam in families:
        build_inputs(fam)
        t = ratio_table(fam)
        df = gain_table(fam)
        numbers(fam, t, df)
        for name in (f"tab_b3{suffix(fam)}.tex", f"tab_b3_gain{suffix(fam)}.tex"):
            print(f"== {name}\n" + open(os.path.join(TAB_DIR, name)).read())
    if all(os.path.exists(os.path.join(TAB_DIR, f"tab_b3_gain{suffix(f)}.tex")) for f in OURS2):
        merge_gain_tables()
        print("== tab_b3_gain_both.tex\n" + open(os.path.join(TAB_DIR, "tab_b3_gain_both.tex")).read())
