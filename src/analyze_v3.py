"""v3 results: BlockGeo-QEM vs the best baseline (BASELINES) per (family, n, budget, depth, metric), paired Wilcoxon.
In the exact regime the idealised PEC is excluded from the 'best baseline' (its error is zero by construction)."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import paths

RES = paths.RES_DIR
FAMILIES = ("tfim", "dtfim", "random")
METRICS = ["local_bures", "local_trace", "mae_off", "mae_diag"]
BASELINES = ["Noisy (raw)", "Noisy", "ZNE-lin", "ZNE-Rich", "ZNE-exp", "PEC", "CDR", "vnCDR", "CDR-family", "OLS",
             "vnOLS", "RF", "MLP-QEM", "GNN-QEM", "BlockGeo-v1"]


def load():
    """base_ and ours_ results of the circuit families in FAMILIES."""
    fs = [f for part in ("base", "ours") for f in sorted(glob.glob(os.path.join(RES, f"{part}_*_samples.csv")))
          if os.path.basename(f).split("_")[1] in FAMILIES]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def compare(df, ours="BlockGeo-QEM", alpha=0.05):
    rows = []
    for key, g in df.groupby(["family", "n", "budget", "depth"]):
        if ours not in set(g.method):
            continue
        bl = [b for b in BASELINES if b in set(g.method) and not (key[2] == 0 and b == "PEC")]
        if not bl:
            continue
        for met in METRICS:
            P = g.pivot(index="sample", columns="method", values=met)
            best = min(bl, key=lambda b: P[b].mean())
            diff = P[ours] - P[best]
            p = wilcoxon(P[ours], P[best]).pvalue if np.any(diff != 0) else 1.0
            r = P[ours].mean() / P[best].mean()
            rows.append(dict(zip(["family", "n", "budget", "depth"], key), metric=met, ratio=r, best=best, p=p,
                             outcome="win" if r < 1 and p < alpha else "loss" if r > 1 and p < alpha else "tie"))
    return pd.DataFrame(rows)


def wtl(c):
    t = c.groupby(["metric", "outcome"]).size().unstack(fill_value=0)
    for col in ("win", "tie", "loss"):
        if col not in t:
            t[col] = 0
    return t.reindex(METRICS)[["win", "tie", "loss"]].apply(lambda r: f"{r.win}/{r.tie}/{r.loss}", axis=1)


if __name__ == "__main__":
    import sys

    df = load()
    ours = sys.argv[1] if len(sys.argv) > 1 else "BlockGeo-QEM"
    c = compare(df, ours)
    c["budget"] = c["budget"].astype(int)
    c.to_csv(os.path.join(RES, f"compare_{ours.replace(' ', '_')}.csv"), index=False)
    pd.set_option("display.width", 250)
    for fam in ("tfim", "random"):
        rows = {}
        for n in sorted(c.n.unique()):
            for b in (10000, 100000, 1000000, 0):
                sub = c[(c.family == fam) & (c.n == n) & (c.budget == b)]
                if len(sub):
                    rows[(n, b or "exact")] = wtl(sub)
        if rows:
            print(f"\n=== {fam}: {ours} vs best baseline (win/tie/loss over depths)")
            print(pd.DataFrame(rows).T.to_string())
            sub = c[(c.family == fam) & (c.outcome == "loss")]
            if len(sub):
                print("baselines beating it:", sub.groupby(["budget", "best"]).size().to_dict())
