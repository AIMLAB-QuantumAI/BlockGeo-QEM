"""beta sweep of BlockGeo-QEM (state-fidelity vs observable-error front), TFIM at 1e5 shots.

Results: <results dir>/beta_{family}_n{n}_B{budget}_samples.csv; methods are named "beta=<value>" (beta = 1 is the
BlockGeo-QEM row of run_v3.py). Test draws are identical to run_v3.py.
"""
import argparse
import os
import time

import pandas as pd

from bench import metrics
from paths import RES_DIR
from run_v3 import draws
from v3 import BlockGeoQEM2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="tfim")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--budget", type=int, default=100000)
    ap.add_argument("--depths", default="1,3,6")
    ap.add_argument("--betas", default="0.1,0.3,3,10")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f"beta_{args.family}_n{args.n}_B{args.budget}_samples.csv")
    dfs = [pd.read_csv(out)] if os.path.exists(out) else []
    for d in [int(x) for x in args.depths.split(",")]:
        V = draws(args.family, args.n, d, args.budget)
        tr, te = V["tr"], V["te"]
        for beta in [float(b) for b in args.betas.split(",")]:
            name = f"beta={beta:g}"
            if dfs and ((dfs[0].depth == d) & (dfs[0].method == name)).any():
                continue
            t0 = time.time()
            p = BlockGeoQEM2(beta=beta).fit({1: V["tr1"]}, tr["S_ideal"]).predict({1: V["te1"]})
            df = metrics(name, p, te["S_ideal"], V["te1"])
            df["family"], df["n"], df["depth"], df["budget"] = args.family, args.n, d, args.budget
            dfs.append(df)
            pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
            print(f"{args.family} n={args.n} d={d} {name}: D {df.local_bures.mean():.4f} T {df.local_trace.mean():.4f}"
                  f" ({time.time() - t0:.0f}s)", flush=True)
