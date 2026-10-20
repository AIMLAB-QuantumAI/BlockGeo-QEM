"""Nonlinear block quantities (purity, Renyi-2 entropy, concurrence): BlockGeo-QEM vs best baseline and
vs its Euclidean ablations, from the 'nl' re-runs (<results dir>/basenl_*, oursnl_*)."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import paths

RES = paths.RES_DIR
NL = ["err_purity", "err_renyi2", "err_concurrence"]
BASELINES = ["Noisy", "ZNE-lin", "ZNE-Rich", "ZNE-exp", "PEC", "CDR", "vnCDR", "CDR-family", "OLS", "vnOLS", "RF",
             "MLP-QEM", "GNN-QEM", "BlockGeo-v1"]
ABL = ["abl: HS state loss", "abl: coordinate head only"]


def load():
    fs = sorted(glob.glob(os.path.join(RES, "basenl_*_samples.csv")) + glob.glob(os.path.join(RES, "oursnl_*_samples.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def table(df, ours="BlockGeo-QEM"):
    rows = []
    for key, g in df.groupby(["family", "n", "budget", "depth"]):
        if ours not in set(g.method):
            continue
        for met in NL:
            P = g.pivot(index="sample", columns="method", values=met)
            bl = [b for b in BASELINES if b in P]
            best = min(bl, key=lambda b: P[b].mean())
            row = dict(zip(["family", "n", "budget", "depth"], key), metric=met, ours=P[ours].mean(), best=best,
                       best_val=P[best].mean(), p=wilcoxon(P[ours], P[best]).pvalue)
            for a in ABL:
                if a in P:
                    row[a] = P[a].mean()
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = table(load())
    pd.set_option("display.width", 250)
    t["ratio_vs_best"] = (t.ours / t.best_val).round(2)
    for a in ABL:
        if a in t:
            t[f"ratio_vs_{a.split(': ')[1].split()[0]}"] = (t.ours / t[a]).round(2)
    print(t.drop(columns=[c for c in ABL if c in t]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
