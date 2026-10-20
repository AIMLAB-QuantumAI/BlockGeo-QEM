"""BlockGeo-QEM with a per-circuit CDR estimate as a second input channel.

Each circuit (training and test) gets a CDR estimate from its own near-Clifford training set. The model sees
[measured coordinates, CDR estimate]; its ridge backbone learns a per-coordinate blend of the two (CDR itself is
one such blend), and the residual network and Bures-trained state head act on top.
Results: <results dir>/cdrbb_{family}_n{n}_B{budget}_samples.csv (test draws identical to run_v3.py, so paired).
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
from paths import FEAT_DIR, RES_DIR, load_generated
from run_v3 import draws
from v3 import BlockGeoQEM2


def run(fam, n, d, budget):
    D = draws(fam, n, d, budget)
    tc = load_generated(os.path.join(FEAT_DIR, f"{fam}_n{n}_d{d}_trcdr.npz"),
                        f"python src/gen_v3_trcdr.py --family {fam} --n {n} --depths {d}")
    s1 = None if budget == 0 else budget // 9
    tr_nc, _ = M.measure(tc["Q_s1"], n, s1, np.random.default_rng([0, n, d, budget, 5]))
    cdr_tr = B3.cdr(tr_nc, tc["S_ideal"], D["tr1"])
    cdr_te = B3.cdr(D["cdr1"], D["cdr"]["S_ideal"], D["te1"])
    Y, Yte = D["tr"]["S_ideal"], D["te"]["S_ideal"]
    S_tr, S_te = {1: D["tr1"], 2: cdr_tr}, {1: D["te1"], 2: cdr_te}
    methods = {
        "CDR": lambda: cdr_te,
        "OLS+CDR": lambda: B.ols_per_coord(S_tr, Y, S_te, (1, 2)),
        "BlockGeo-QEM+CDR": lambda: BlockGeoQEM2(beta=1.0, scales=(1, 2)).fit(S_tr, Y).predict(S_te),
    }
    rows, preds = [], {"__ideal__": Yte}
    for name, fn in methods.items():
        t0 = time.time()
        p = fn()
        preds[name] = p
        df = metrics(name, p, Yte, D["te1"])
        df["fit_time_s"] = time.time() - t0
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["family"], df["n"], df["depth"], df["budget"] = fam, n, d, budget
    return df, preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--depths", default="1,2,3,4,6,8")
    args = ap.parse_args()
    os.makedirs(os.path.join(RES_DIR, "pred"), exist_ok=True)
    stem = f"cdrbb_{args.family}_n{args.n}_B{args.budget}"
    out = os.path.join(RES_DIR, f"{stem}_samples.csv")
    done = pd.read_csv(out) if os.path.exists(out) else None
    dfs = [done] if done is not None else []
    for d in [int(x) for x in args.depths.split(",")]:
        if done is not None and d in set(done.depth):
            continue
        t0 = time.time()
        df, preds = run(args.family, args.n, d, args.budget)
        np.savez_compressed(os.path.join(RES_DIR, "pred", f"{stem}_d{d}.npz"),
                            **{k.replace(" ", "_").replace("+", "p"): v for k, v in preds.items()})
        dfs.append(df)
        pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
        cur = df.groupby("method", sort=False)[["local_bures", "local_trace", "mae_off", "mae_diag"]].mean()
        print(f"== {stem} depth={d} ({time.time() - t0:.0f}s)\n" + cur.to_string(float_format=lambda x: f"{x:.4g}"),
              flush=True)
