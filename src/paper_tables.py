"""Win/tie/loss table, written to WORK/tables/tab_wtl.tex (paths.TAB_DIR), and summary numbers of the studies.

usage: paper_tables.py              tab_wtl.tex
       paper_tables.py numbers      summary values of the main grid and of the follow-up studies
       paper_tables.py cdr          the CDR-input variant (run_cdrbb.py) against OLS+CDR
       paper_tables.py marginals    median purity and smallest eigenvalue of the ideal block marginals (gen_v3.py data)

tab_wtl.tex  BlockGeo-QEM against the best baseline (analyze_v3.BASELINES): wins/ties/losses over the six depths
             (paired Wilcoxon, 5% level) and the geometric mean over depths of the error ratio, for the local
             Bures and trace distances at 1e4, 1e5 and 1e6 shots (run_v3 results, no CDR input); the last rows
             are the second device (WORK/results_v3_torino), one block per circuit family.
marginals    needs only the _test files of gen_v3.py in the feats directory, no results.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v3 as A
import paths

R = paths.RES_DIR + os.sep
OUT = paths.TAB_DIR + os.sep
BUDGETS = (10000, 100000, 1000000)
METS = ("local_bures", "local_trace")
FAM = {"tfim": "Uniform TFIM", "dtfim": "Disordered TFIM", "random": "Random circuits"}


def gm(x):
    return float(np.exp(np.log(x).mean()))


def cell(sub):
    if sub is None or sub.empty:
        return "\\multicolumn{2}{c}{--}"
    o = sub.outcome
    return f"{(o == 'win').sum()}/{(o == 'tie').sum()}/{(o == 'loss').sum()} & {gm(sub.ratio):.2f}"


def plain_table():
    c = A.compare(A.load(), "BlockGeo-QEM")
    c["budget"] = c.budget.astype(int)
    return c


def main():
    c = plain_table()
    rows = []
    for fam in ("tfim", "dtfim", "random"):         # the CDR-input variant is reported separately (cdr())
        label = FAM[fam]
        for k, n in enumerate((6, 8, 10)):
            cells = []
            for B in BUDGETS:
                for met in METS:
                    sub = c[(c.family == fam) & (c.n == n) & (c.budget == B) & (c.metric == met)]
                    cells.append(cell(sub))
            head = (f"\\multirow{{3}}{{*}}{{\\parbox{{2.3cm}}{{{label}}}}}" if k == 0 else "")
            rows.append(f"{head} & {n} & " + " & ".join(cells) + " \\\\")
        rows.append("\\hline" if fam != "random" else "")
    # second device (WORK/results_v3_torino): one block per circuit family at 1e4, 1e5 and 1e6 shots, one row per n
    tor_dir = paths.results_dir("_v3_torino") + os.sep
    tor = [tor_dir + f for f in sorted(os.listdir(tor_dir)) if f.endswith("_samples.csv")] if os.path.isdir(tor_dir) else []
    if tor:
        ct = A.compare(pd.concat([pd.read_csv(f) for f in tor], ignore_index=True), "BlockGeo-QEM")
        ct["budget"] = ct.budget.astype(int)
        for fam in [f for f in ("tfim", "dtfim", "random") if f in set(ct.family)]:
            cf = ct[ct.family == fam]
            ns = sorted(cf.n.unique())
            if rows and rows[-1] == "":
                rows[-1] = "\\hline"
            for k, n in enumerate(ns):
                cells = []
                for B in BUDGETS:
                    for met in METS:
                        sub = cf[(cf.n == n) & (cf.budget == B) & (cf.metric == met)]
                        cells.append(cell(sub if len(sub) == 6 else None))
                two = "\\begin{tabular}[c]{@{}l@{}}%s,\\\\ ibm\\_torino\\end{tabular}" % FAM[fam]   # two lines
                label = "\\multirow{%d}{*}{%s}" % (len(ns), two) if len(ns) > 1 else two
                rows.append((label if k == 0 else "") + f" & {n} & " + " & ".join(cells) + " \\\\")
            rows.append("")
    body = "\n".join(r for r in rows if r)
    body = body.rsplit("\\hline", 1)[0] if body.endswith("\\hline") else body
    body = body.rstrip()
    body = body[:-3] if body.endswith(" \\\\") else body          # no trailing \\: it would add an empty row
    os.makedirs(OUT, exist_ok=True)
    with open(OUT + "tab_wtl.tex", "w") as f:
        f.write(body + "\n")
    print(body)


def numbers():
    """Summary values of the main grid and of the follow-up studies."""
    pd.set_option("display.width", 250)
    df = A.load()
    m = lambda fam, n, B, meth, met: df[(df.family == fam) & (df.n == n) & (df.budget == B) & (df.method == meth)
                                        ].groupby("depth")[met].mean()
    print("== TFIM one and eight steps, 1e5 and 1e4: BlockGeo-QEM vs GNN (Bures, trace)")
    for B in (100000, 10000):
        for n in (6, 8, 10):
            o, g = m("tfim", n, B, "BlockGeo-QEM", "local_bures"), m("tfim", n, B, "GNN-QEM", "local_bures")
            ot, gt = m("tfim", n, B, "BlockGeo-QEM", "local_trace"), m("tfim", n, B, "GNN-QEM", "local_trace")
            print(f"B={B} n={n} bures 1: {o[1]:.4f} vs {g[1]:.4f} (x{g[1] / o[1]:.1f}); 8: {o[8]:.4f} vs {g[8]:.4f} "
                  f"(x{g[8] / o[8]:.2f}) | trace 1: {ot[1]:.4f} vs {gt[1]:.4f}; 8: {ot[8]:.4f} vs {gt[8]:.4f}")
    print("== PEC sampling overhead gamma^2 (median over test circuits), TFIM")
    F = paths.FEAT_DIR + os.sep
    for n in (6, 8, 10):
        print(n, {d: round(float(np.median(np.load(F + f"tfim_n{n}_d{d}_test.npz")["gamma"] ** 2)), 1)
                  for d in (1, 4, 8)})
    print("== exact limit (B=0): BlockGeo-QEM, -ms and best non-PEC baseline, mean over depths (Bures)")
    for fam in ("tfim", "dtfim", "random"):
        for n in (6, 8, 10):
            s = df[(df.family == fam) & (df.n == n) & (df.budget == 0)]
            if s.empty:
                continue
            g = s.groupby("method").local_bures.mean()
            bl = g[[b for b in A.BASELINES if b in g and b != "PEC"]]
            print(fam, n, f"ours {g['BlockGeo-QEM']:.4f} ms {g['BlockGeo-QEM-ms']:.4f} best {bl.idxmin()} {bl.min():.4f}")
    print("== spectral quantities (n=6,8; 1e4, 1e5): win/tie/loss vs best baseline, ratio at step 1")
    import analyze_nl as NL
    t = NL.table(NL.load())                             # p-values come from NL.table
    t["out"] = np.where((t.ours < t.best_val) & (t.p < .05), "win", np.where((t.ours > t.best_val) & (t.p < .05),
                                                                            "loss", "tie"))
    print(t.groupby("metric").out.value_counts().unstack(fill_value=0))
    s1 = t[t.depth == 1].assign(r=lambda x: x.ours / x.best_val)
    print(s1.groupby("metric").r.agg(["min", "max"]).round(2))
    for a in NL.ABL:
        print(a, (s1.ours / s1[a]).groupby(s1.metric).agg(["min", "max"]).round(2).to_dict())
    print("== ablations at 1e5 (TFIM): ratio of ablation to full model, local Bures and trace")
    for n in (6, 8, 10):
        s = df[(df.family == "tfim") & (df.n == n) & (df.budget == 100000)]
        g = s.groupby(["method", "depth"])[["local_bures", "local_trace"]].mean()
        full = g.loc["BlockGeo-QEM"]
        for a in ("abl: HS state loss", "abl: coordinate head only", "abl: no circuit context"):
            r = (g.loc[a] / full).round(2)
            print(n, a, "bures", r.local_bures.to_dict(), "trace", r.local_trace.to_dict())
    print("== transfer to n=10 (1e5): Bures by step")
    t8 = pd.read_csv(R + "transfer_tfim_n8to10_B100000_samples.csv")
    t6 = pd.read_csv(R + "transfer_tfim_n6to10_B100000_samples.csv")
    tab = pd.concat([t8.groupby(["method", "depth"]).local_bures.mean().unstack().rename(lambda x: x + " 8->10"),
                     t6.groupby(["method", "depth"]).local_bures.mean().unstack().rename(lambda x: x + " 6->10"),
                     pd.DataFrame([m("tfim", 10, 100000, "GNN-QEM", "local_bures").rename("GNN in-size"),
                                   m("tfim", 10, 100000, "BlockGeo-QEM", "local_bures").rename("ours in-size")])])
    print(tab.round(4).to_string())
    t10 = pd.read_csv(R + "transfer_tfim_n8to10_B10000_samples.csv")
    print(pd.concat([t10.groupby(["method", "depth"]).local_bures.mean().unstack(),
                     pd.DataFrame([m("tfim", 10, 10000, "GNN-QEM", "local_bures").rename("GNN in-size 1e4")])]
                    ).round(4).to_string())
    print("== network size of BlockGeo-QEM (trainable parameters, single-scale input)")
    import torch
    import v3
    model = v3.BlockGeoQEM2(beta=1.0, device="cpu")
    nets = {n: model._net({"x": torch.zeros(1, n - 1, v3.NS)}, 0) for n in (6, 8, 10, 20)}   # n - 1 blocks
    print({n: sum(p.numel() for p in net.parameters() if p.requires_grad) for n, net in nets.items()})


def marginals():
    """Median purity Tr(rho^2) = (1 + |s|^2)/4 and median smallest eigenvalue of the ideal block states rho(s), over
    the test circuits and blocks of every family, n and depth in the feats directory (gen_v3.py _test files).
    Returns one row per (family, n, depth)."""
    import glob
    import re

    import qemlib as L
    pm = L.pauli_ops(2)
    rows = []
    for f in glob.glob(os.path.join(paths.FEAT_DIR, "*_test.npz")):
        key = re.fullmatch(r"(tfim|dtfim|random)_n(\d+)_d(\d+)_test\.npz", os.path.basename(f))
        if key is None:
            continue
        S = np.load(f)["S_ideal"]                                   # [circuits, blocks, 15]
        rho = (np.eye(4) + np.einsum("...k,kij->...ij", S.astype(np.complex128), pm)) / 4
        rows.append(dict(family=key.group(1), n=int(key.group(2)), depth=int(key.group(3)),
                         purity=float(np.median((1 + (S ** 2).sum(-1)) / 4)),
                         lambda_min=float(np.median(np.linalg.eigvalsh(rho)[..., 0]))))
    if not rows:
        sys.exit(f"no gen_v3.py test files in {paths.FEAT_DIR}")
    return pd.DataFrame(rows).sort_values(["family", "n", "depth"], ignore_index=True)


def print_marginals(t):
    pd.set_option("display.width", 250)
    print("== ideal block marginals of the test circuits: median over circuits and blocks of the purity and of the "
          "smallest eigenvalue")
    for fam, label in FAM.items():
        s = t[t.family == fam]
        if s.empty:
            continue
        print(f"-- {label}")
        print((s.pivot(index="depth", columns="n", values=["purity", "lambda_min"]).round(4) + 0.0).to_string())
        print(f"median purity between {s.purity.min():.4f} and {s.purity.max():.4f} over n and depths")


def cdr():
    """BlockGeo-QEM with the per-circuit CDR estimate as a second input (run_cdrbb.py) against OLS+CDR, the linear
    blend with the same information, and BlockGeo-QEM without the CDR input against OLS+CDR."""
    import glob
    from scipy.stats import wilcoxon
    rows = []
    for f in sorted(glob.glob(R + "cdrbb_*_samples.csv")):
        cb = pd.read_csv(f)
        fam, n, B = cb.family.iloc[0], int(cb.n.iloc[0]), int(cb.budget.iloc[0])
        o = pd.read_csv(R + f"ours_{fam}_n{n}_B{B}_samples.csv")
        df = pd.concat([cb, o[o.method == "BlockGeo-QEM"]])
        for met in METS:
            for d, g in df.groupby("depth"):
                P = g.pivot(index="sample", columns="method", values=met)
                for a, b in (("BlockGeo-QEM+CDR", "OLS+CDR"), ("BlockGeo-QEM", "OLS+CDR"), ("BlockGeo-QEM+CDR", "BlockGeo-QEM")):
                    r, p = P[a].mean() / P[b].mean(), wilcoxon(P[a], P[b]).pvalue
                    rows.append(dict(fam=fam, n=n, B=B, met=met, d=d, pair=f"{a} / {b}", r=r,
                                     out="win" if r < 1 and p < .05 else "loss" if r > 1 and p < .05 else "tie"))
    t = pd.DataFrame(rows)
    wtl = lambda x: f"{(x == 'win').sum()}/{(x == 'tie').sum()}/{(x == 'loss').sum()}"
    pd.set_option("display.width", 250)
    for pair, s in t.groupby("pair"):
        print("==", pair)
        print(pd.concat([s.groupby(["fam", "B", "met"]).out.apply(wtl).unstack("met"),
                         s.groupby(["fam", "B", "met"]).r.apply(gm).unstack("met").round(2)], axis=1).to_string())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "numbers":
        numbers()
    elif sys.argv[1] == "cdr":
        cdr()
    elif sys.argv[1] == "marginals":
        print_marginals(marginals())
    else:
        sys.exit("usage: paper_tables.py [numbers | cdr | marginals]")
