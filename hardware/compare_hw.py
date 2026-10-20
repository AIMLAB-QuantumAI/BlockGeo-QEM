"""Mitigation on hardware data: the learned methods are trained on the hardware measurements of the training circuits
and their classically computed ideal marginals, and evaluated on the hardware measurements of the test circuits.
Baselines that need extra circuits (ZNE, CDR, PEC) are not available. The ideal marginals are read from
<hardware dir>/<tag>/labels.npz (make_labels.py). Results: <hardware dir>/<tag>/results.csv and pred.npz.

usage: compare_hw.py <tag>
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import baselines as B
import baselines_v3 as B3
from analyze_hw import ROOT, load_labels, load_run
from bench import metrics
from v3 import BlockGeoQEM2

MET = ["local_bures", "local_trace", "mae_off", "mae_diag", "err_purity", "err_concurrence"]


def run(tag, seeds=(0, 1, 2)):
    S_raw, S_rem, A, meta = load_run(tag)
    n, d = len(meta["chain"]), meta["depth"]
    lab = load_labels(tag)
    Y, Yte = lab["S_train"][: meta["n_train"]], lab["S_test"][: meta["n_test"]]
    X, Xte = S_rem["train"], S_rem["test"]
    cq, ce = np.array(meta["calibration"]["cq"]), np.array(meta["calibration"]["ce"])
    methods = {
        "Noisy (raw)": lambda: S_raw["test"],
        "Noisy": lambda: Xte,
        "OLS": lambda: B.ols_per_coord({1: X}, Y, {1: Xte}, (1,)),
        "RF": lambda: B.rf_block({1: X}, Y, {1: Xte}, (1,)),
        "MLP-QEM": lambda: B.mlp_global({1: X}, Y, {1: Xte}, (1,)),
        "GNN-QEM": lambda: B3.gnn_qem(X, Y, Xte, cq, ce),
        "abl: HS state loss": lambda: BlockGeoQEM2(beta=1.0, loss="hs", seeds=seeds).fit({1: X}, Y).predict({1: Xte}),
        "BlockGeo-QEM": lambda: BlockGeoQEM2(beta=1.0, seeds=seeds).fit({1: X}, Y).predict({1: Xte}),
    }
    rows, preds = [], {"__ideal__": Yte}
    for name, fn in methods.items():
        t0 = time.time()
        p = fn()
        preds[name] = p
        df = metrics(name, p, Yte, Xte)
        df["fit_time_s"] = time.time() - t0
        rows.append(df)
        print(f"{name:20s} bures {df.local_bures.mean():.4f} trace {df.local_trace.mean():.4f} "
              f"purity {df.err_purity.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["backend"], df["depth"], df["n"] = meta["backend"], d, n
    df.to_csv(os.path.join(ROOT, tag, "results.csv"), index=False)
    np.savez_compressed(os.path.join(ROOT, tag, "pred.npz"), **{k.replace(" ", "_").replace(":", ""): v
                                                                 for k, v in preds.items()})
    print("\nBlockGeo-QEM vs best baseline (paired Wilcoxon over test circuits):")
    bl = [m for m in methods if "BlockGeo" not in m and not m.startswith("abl")]
    for met in MET:
        P = df.pivot(index="sample", columns="method", values=met)
        best = min(bl, key=lambda m: P[m].mean())
        r, p = P["BlockGeo-QEM"].mean() / P[best].mean(), wilcoxon(P["BlockGeo-QEM"], P[best]).pvalue
        hs = P["abl: HS state loss"].mean() / P["BlockGeo-QEM"].mean()
        print(f"  {met:16s} ratio {r:.2f} vs {best:12s} p={p:.3g}   HS/ours {hs:.2f}")
    return df


if __name__ == "__main__":
    run(sys.argv[1])
