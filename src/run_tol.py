"""Tolerance of the validation check: BlockGeo-QEM with a stricter check that also requires the held-out coordinate
error to stay within 2% of the backbone's ("tol 0.02") and with the default rule, the held-out loss alone ("no check").
Records the selected epochs of the three seeds (-1 = no epoch qualified, the projected backbone is returned).
Results: <results dir>/tol_{family}_n{n}_B{budget}_samples.csv (test draws identical to run_v3.py, so paired).
"""
import argparse
import os
import time

import pandas as pd

from bench import metrics
from paths import RES_DIR
from run_v3 import draws
from v3 import BlockGeoQEM2

VARIANTS = {"BlockGeo-QEM (tol 0.02)": {"tol": 0.02}, "BlockGeo-QEM (no check)": {"tol": float("inf")}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--budget", type=int, default=100000)
    ap.add_argument("--depths", default="1,2,3,4,6,8")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f"tol_{args.family}_n{args.n}_B{args.budget}_samples.csv")
    done = pd.read_csv(out) if os.path.exists(out) else None
    dfs = [done] if done is not None else []
    for d in [int(x) for x in args.depths.split(",")]:
        if done is not None and d in set(done.depth):
            continue
        D = draws(args.family, args.n, d, args.budget)
        Y, Yte = D["tr"]["S_ideal"], D["te"]["S_ideal"]
        for name, kw in VARIANTS.items():
            t0 = time.time()
            m = BlockGeoQEM2(beta=1.0, **kw).fit({1: D["tr1"]}, Y)
            df = metrics(name, m.predict({1: D["te1"]}), Yte, D["te1"])
            df["fit_time_s"], df["epochs"] = time.time() - t0, " ".join(map(str, m.best_eps))
            df["family"], df["n"], df["depth"], df["budget"] = args.family, args.n, d, args.budget
            dfs.append(df)
            print(f"{args.family} n={args.n} d={d} {name}: eps {m.best_eps} bures {df.local_bures.mean():.4f} "
                  f"trace {df.local_trace.mean():.4f}", flush=True)
        pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
