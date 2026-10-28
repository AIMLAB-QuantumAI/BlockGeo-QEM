"""GNN-QEM (Bures) on the run_v3 draws: the message-passing network of GNN-QEM as the trunk of BlockGeo-QEM, with
the backbone, heads, loss and training of BlockGeo-QEM (gnn_bures.py). Two-qubit blocks only.
Results: <results dir>/gnnb_{family}_n{n}_B{budget}_samples.csv (method GNN-QEM (Bures), selected epochs per seed
in column epochs), resumed per depth, and predictions in <results dir>/pred/gnnb_{family}_n{n}_B{budget}_d{depth}.npz.

usage: python src/run_gnnb.py --family tfim --n 8 --budget 100000 --depths 1,2,3,4,6,8
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import baselines_v3 as B3
from bench import metrics
from gnn_bures import GNNBures
from paths import RES_DIR
from run_v3 import BLOCK, draws

if __name__ == "__main__":
    if BLOCK != 2:
        raise SystemExit("run_gnnb.py reads two-qubit blocks; unset BLOCKGEO_B")
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--depths", default="1,2,3,4,6,8")
    ap.add_argument("--max-epochs", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(os.path.join(RES_DIR, "pred"), exist_ok=True)
    stem = f"gnnb_{args.family}_n{args.n}_B{args.budget}"
    out = os.path.join(RES_DIR, f"{stem}_samples.csv")
    done = pd.read_csv(out) if os.path.exists(out) else None
    dfs = [done] if done is not None else []
    cq, ce = B3.chain_calibration(args.n)
    for d in [int(x) for x in args.depths.split(",")]:
        if done is not None and d in set(done.depth):
            continue
        t0 = time.time()
        D = draws(args.family, args.n, d, args.budget)
        m = GNNBures(cq, ce, beta=1.0, max_epochs=args.max_epochs).fit({1: D["tr1"]}, D["tr"]["S_ideal"])
        p = m.predict({1: D["te1"]})
        df = metrics("GNN-QEM (Bures)", p, D["te"]["S_ideal"], D["te1"])
        df["fit_time_s"], df["epochs"] = time.time() - t0, str(m.best_eps)
        df["family"], df["n"], df["depth"], df["budget"] = args.family, args.n, d, args.budget
        dfs.append(df)
        pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
        np.savez_compressed(os.path.join(RES_DIR, "pred", f"{stem}_d{d}.npz"), __ideal__=D["te"]["S_ideal"],
                            GNN_QEM_Bures=p)
        print(f"== {stem} depth={d} ({time.time() - t0:.0f}s, epochs {m.best_eps}): bures {df.local_bures.mean():.4f} "
              f"trace {df.local_trace.mean():.4f} purity {df.err_purity.mean():.4f}", flush=True)
