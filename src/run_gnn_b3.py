"""GNN-QEM baseline for three-qubit blocks on the run_v3 draws (the same measured training and test data as the other
methods). The three-qubit runs use BLOCKGEO_TAG=_v3b3 and BLOCKGEO_B=3, which this script sets if they are unset.
Results: <results dir>/gnn_{family}_n{n}_B{budget}_samples.csv (method GNN-QEM), resumed per depth, and predictions in
<results dir>/pred/gnn_{family}_n{n}_B{budget}_d{depth}.npz.

usage: python src/run_gnn_b3.py --family tfim --n 8 --budget 100000 --depths 1,2,3,4,6,8
"""
import os

os.environ.setdefault("BLOCKGEO_TAG", "_v3b3")
os.environ.setdefault("BLOCKGEO_B", "3")

import argparse
import time

import numpy as np
import pandas as pd

import baselines_v3 as B3
from bench import metrics
from gnn_b3 import gnn_qem_b3
from paths import RES_DIR
from run_v3 import BLOCK, draws

if __name__ == "__main__":
    if BLOCK != 3:
        raise SystemExit("run_gnn_b3.py needs BLOCKGEO_B=3 (three-qubit blocks)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--depths", default="1,2,3,4,6,8")
    args = ap.parse_args()
    os.makedirs(os.path.join(RES_DIR, "pred"), exist_ok=True)
    out = os.path.join(RES_DIR, f"gnn_{args.family}_n{args.n}_B{args.budget}_samples.csv")
    done = pd.read_csv(out) if os.path.exists(out) else None
    dfs = [done] if done is not None else []
    cq, ce = B3.chain_calibration(args.n)
    for d in [int(x) for x in args.depths.split(",")]:
        if done is not None and d in set(done.depth):
            continue
        t0 = time.time()
        D = draws(args.family, args.n, d, args.budget)
        p = gnn_qem_b3(D["tr1"], D["tr"]["S_ideal"], D["te1"], cq, ce)
        df = metrics("GNN-QEM", p, D["te"]["S_ideal"], D["te1"], b=BLOCK)
        df["fit_time_s"] = time.time() - t0
        df["family"], df["n"], df["depth"], df["budget"] = args.family, args.n, d, args.budget
        dfs.append(df)
        pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
        np.savez_compressed(os.path.join(RES_DIR, "pred", f"gnn_{args.family}_n{args.n}_B{args.budget}_d{d}.npz"),
                            GNN_QEM=p)
        print(f"== gnn b=3 {args.family} n={args.n} B={args.budget} d={d} ({time.time() - t0:.0f}s): "
              f"bures {df.local_bures.mean():.4f} trace {df.local_trace.mean():.4f}", flush=True)
