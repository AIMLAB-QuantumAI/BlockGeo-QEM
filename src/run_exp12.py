"""exp 1 (data size) and exp 2 (calibration drift) for TFIM n=8 at one shot budget.

exp 1: learned methods trained on the first N in {25, 50, 100, 200, 400} training circuits, same test draws.
exp 2: models trained once on the original calibration; test circuits (and, for CDR, its near-Clifford circuits)
       executed with every gate error probability scaled by f. ZNE, CDR and PEC use test-time data only, so they
       follow the drift; PEC knows the drifted noise exactly.
Results: <results dir>/datasize_{fam}_n{n}_B{B}_samples.csv and <results dir>/drift_{fam}_n{n}_B{B}_samples.csv
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import baselines as B
import baselines_v3 as B3
import measure_v3 as M
from bench import metrics
from paths import FEAT_DIR, RES_DIR
from run_v3 import draws
from v3 import BlockGeoQEM2

SIZES = (25, 50, 100, 200, 400)
DRIFT = (0.7, 0.85, 1.15, 1.3)


def learners(n):
    cq, ce = B3.chain_calibration(n)
    return {
        "OLS": lambda X, Y, T: B.ols_per_coord({1: X}, Y, {1: T}, (1,)),
        "RF": lambda X, Y, T: B.rf_block({1: X}, Y, {1: T}, (1,)),
        "MLP-QEM": lambda X, Y, T: B.mlp_global({1: X}, Y, {1: T}, (1,)),
        "GNN-QEM": lambda X, Y, T: B3.gnn_qem(X, Y, T, cq, ce),
        "BlockGeo-QEM": lambda X, Y, T: BlockGeoQEM2(beta=1.0).fit({1: X}, Y).predict({1: T}),
    }


def exp_datasize(fam, n, d, budget):
    D = draws(fam, n, d, budget)
    x = dict(np.load(os.path.join(FEAT_DIR, f"{fam}_n{n}_d{d}_trainx.npz")))
    s1 = None if budget == 0 else budget // 9
    Xx, _ = M.measure(x["Q_s1"], n, s1, np.random.default_rng([0, n, d, budget, 6]))
    X = np.concatenate([D["tr1"], Xx])
    Y = np.concatenate([D["tr"]["S_ideal"], x["S_ideal"]])
    rows = []
    for N in SIZES:
        for name, fn in learners(n).items():
            t0 = time.time()
            df = metrics(name, fn(X[:N], Y[:N], D["te1"]), D["te"]["S_ideal"], D["te1"])
            df["n_train"], df["fit_time_s"] = N, time.time() - t0
            rows.append(df)
        print(f"  datasize d={d} N={N} done", flush=True)
    return pd.concat(rows, ignore_index=True)


def exp_drift(fam, n, d, budget):
    D = draws(fam, n, d, budget)
    s1 = None if budget == 0 else budget // 9
    s3 = None if budget == 0 else budget // 27
    rng = lambda k: np.random.default_rng([0, n, d, budget, k])       # same streams as run_v3.draws
    Yte, Ytr = D["te"]["S_ideal"], D["tr"]["S_ideal"]
    models = {}
    cq, ce = B3.chain_calibration(n)
    blockgeo = BlockGeoQEM2(beta=1.0).fit({1: D["tr1"]}, Ytr)
    rows = []
    for f in (1.0,) + DRIFT:
        if f == 1.0:
            te1, te3, cdr1, gamma = D["te1"], D["te3"], D["cdr1"], D["te"]["gamma"]
        else:
            z = dict(np.load(os.path.join(FEAT_DIR, "drift", f"{fam}_n{n}_d{d}_f{f}.npz")))
            te1, _ = M.measure(z["Q_s1"], n, s1, rng(1))
            te3 = {s: M.measure(z[f"Q_s{s}"], n, s3, rng(20 + s))[0] for s in (1, 3, 5)}
            cdr1, _ = M.measure(z["cdr_Q_s1"], n, s1, rng(2))
            gamma = z["gamma"]
        stack = np.stack([te3[s] for s in (1, 3, 5)])
        preds = {
            "Noisy": te1,
            "ZNE-lin": B.zne_poly(stack, order=1),
            "ZNE-Rich": B.zne_poly(stack, order=2),
            "PEC": B3.pec(Yte, gamma, n, budget, rng(4), M.readout_flip_scale(n)),
            "CDR": B3.cdr(cdr1, D["cdr"]["S_ideal"], te1),
            "OLS": B.ols_per_coord({1: D["tr1"]}, Ytr, {1: te1}, (1,)),
            "RF": B.rf_block({1: D["tr1"]}, Ytr, {1: te1}, (1,)),
            "MLP-QEM": B.mlp_global({1: D["tr1"]}, Ytr, {1: te1}, (1,)),
            "GNN-QEM": B3.gnn_qem(D["tr1"], Ytr, te1, cq, ce),
            "BlockGeo-QEM": blockgeo.predict({1: te1}),
        }
        for name, p in preds.items():
            df = metrics(name, p, Yte, te1)
            df["drift"] = f
            rows.append(df)
        print(f"  drift d={d} f={f} done", flush=True)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["datasize", "drift"], required=True)
    ap.add_argument("--family", default="tfim")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget", type=int, default=100000)
    ap.add_argument("--depths", default="1,3,6")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f"{args.exp}_{args.family}_n{args.n}_B{args.budget}_samples.csv")
    done = pd.read_csv(out) if os.path.exists(out) else None
    dfs = [done] if done is not None else []
    fn = exp_datasize if args.exp == "datasize" else exp_drift
    for d in [int(x) for x in args.depths.split(",")]:
        if done is not None and d in set(done.depth):
            continue
        t0 = time.time()
        df = fn(args.family, args.n, d, args.budget)
        df["family"], df["n"], df["depth"], df["budget"] = args.family, args.n, d, args.budget
        dfs.append(df)
        pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
        print(f"== {args.exp} d={d} ({time.time() - t0:.0f}s)", flush=True)
