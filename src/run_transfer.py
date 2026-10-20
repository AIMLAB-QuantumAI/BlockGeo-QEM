"""Size transfer: train on an n_src-qubit chain, mitigate n_tgt-qubit test circuits (same family, depth, budget).

Methods that can be applied to another chain length:
  BlockGeo-QEM (transfer)  - boundary-class positions, pooled backbone, mean-pooled circuit context
  GNN-QEM (transfer)       - message passing on the chain; the test chain's calibration is used at test time
  OLS-pooled (transfer)    - per-coordinate linear map pooled over blocks of each boundary class
Reference: the same boundary-class BlockGeo-QEM trained on the target size ("in-size").
Results: <results dir>/transfer_{family}_n{src}to{tgt}_B{budget}_samples.csv (+ predictions in <results dir>/pred).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import baselines_v3 as B3
from bench import metrics
from paths import RES_DIR
from run_v3 import draws
from v3 import Backbone, BlockGeoQEM2


def run(fam, n_src, n_tgt, d, budget):
    S, T = draws(fam, n_src, d, budget), draws(fam, n_tgt, d, budget)
    Ys, Yt = S["tr"]["S_ideal"], T["te"]["S_ideal"]
    cal_s, cal_t = B3.chain_calibration(n_src), B3.chain_calibration(n_tgt)
    methods = {
        "BlockGeo-QEM (transfer)": lambda: BlockGeoQEM2(beta=1.0, pos_mode="boundary").fit(
            {1: S["tr1"]}, Ys).predict({1: T["te1"]}),
        "GNN-QEM (transfer)": lambda: B3.gnn_qem(S["tr1"], Ys, T["te1"], *cal_s, test_calib=cal_t),
        "OLS-pooled (transfer)": lambda: np.clip(Backbone("boundary").fit(S["tr1"], Ys, 1).predict(T["te1"]), -1, 1),
        "BlockGeo-QEM (in-size, boundary)": lambda: BlockGeoQEM2(beta=1.0, pos_mode="boundary").fit(
            {1: T["tr1"]}, T["tr"]["S_ideal"]).predict({1: T["te1"]}),
    }
    rows, preds = [], {"__ideal__": Yt}
    for name, fn in methods.items():
        t0 = time.time()
        p = fn()
        preds[name] = p
        df = metrics(name, p, Yt, T["te1"])
        df["fit_time_s"] = time.time() - t0
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["family"], df["n_src"], df["n"], df["depth"], df["budget"] = fam, n_src, n_tgt, d, budget
    return df, preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="tfim")
    ap.add_argument("--src", type=int, required=True)
    ap.add_argument("--tgt", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--depths", default="1,2,3,4,6,8")
    args = ap.parse_args()
    os.makedirs(os.path.join(RES_DIR, "pred"), exist_ok=True)
    stem = f"transfer_{args.family}_n{args.src}to{args.tgt}_B{args.budget}"
    out = os.path.join(RES_DIR, f"{stem}_samples.csv")
    done = pd.read_csv(out) if os.path.exists(out) else None
    dfs = [done] if done is not None else []
    for d in [int(x) for x in args.depths.split(",")]:
        if done is not None and d in set(done.depth):
            continue
        t0 = time.time()
        df, preds = run(args.family, args.src, args.tgt, d, args.budget)
        np.savez_compressed(os.path.join(RES_DIR, "pred", f"{stem}_d{d}.npz"),
                            **{k.replace(" ", "_"): v for k, v in preds.items()})
        dfs.append(df)
        pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
        cur = df.groupby("method", sort=False)[["local_bures", "local_trace", "mae_off", "err_purity"]].mean()
        print(f"== {stem} depth={d} ({time.time() - t0:.0f}s)\n" + cur.to_string(float_format=lambda x: f"{x:.4g}"),
              flush=True)
