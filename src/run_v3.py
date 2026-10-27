"""v3 benchmark (chain layout, twirled gate noise, readout errors + REM) for one (family, n, budget).

--part base : Noisy (raw), Noisy, ZNE x3, PEC, CDR, vnCDR, CDR-family, OLS, vnOLS, RF, MLP-QEM, GNN-QEM, BlockGeo-v1
--part ours : BlockGeo-QEM, -ms, (Clifford-trained); ablations at 1e5 shots
BLOCKGEO_B=3 (with BLOCKGEO_TAG=_v3b3) uses three-qubit blocks; --part base then leaves out GNN-QEM and BlockGeo-v1,
and run_gnn_b3.py runs GNN-QEM for them.
budget = 0 means exact expectation values (readout errors applied exactly, then REM).
Results: <results dir>/{part}{tag}_{family}_n{n}_B{budget}_samples.csv, predictions in <results dir>/pred/.
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
from v3 import BlockGeoQEM, BlockGeoQEM2

SCALES = (1, 3, 5)
MAIN_BUDGET = 100000
BLOCK = int(os.environ.get("BLOCKGEO_B", "2"))       # block size (BLOCKGEO_TAG selects the matching data, e.g. _v3b3)


def load(fam, n, d):
    cmd = f"python src/gen_v3.py --family {fam} --n {n} --depths {d}"
    get = lambda part: load_generated(os.path.join(FEAT_DIR, f"{fam}_n{n}_d{d}_{part}.npz"), cmd)
    return get("train"), get("test"), get("cdr"), get("cliff")


def draws(fam, n, d, budget, seed=0):
    tr, te, cdr, cl = load(fam, n, d)
    ns = 3 ** BLOCK                                     # covering settings: 9 for b=2, 27 for b=3
    s1 = None if budget == 0 else budget // ns         # single-scale protocol
    s3 = None if budget == 0 else budget // (3 * ns)   # multi-scale protocol: settings x 3 fold scales
    rng = lambda k: np.random.default_rng([seed, n, d, budget, k])
    D = {"tr": tr, "te": te, "cdr": cdr, "cl": cl}
    D["tr1"], D["tr_p"] = M.measure(tr["Q_s1"], n, s1, rng(0), b=BLOCK)
    D["te1"], D["te_p"] = M.measure(te["Q_s1"], n, s1, rng(1), b=BLOCK)
    D["te_raw"], _ = M.measure(te["Q_s1"], n, s1, rng(1), rem=False, b=BLOCK)      # same shots, no REM
    D["tr3"] = {s: M.measure(tr[f"Q_s{s}"], n, s3, rng(10 + s), b=BLOCK)[0] for s in SCALES}
    D["te3"] = {s: M.measure(te[f"Q_s{s}"], n, s3, rng(20 + s), b=BLOCK)[0] for s in SCALES}
    D["cdr1"], _ = M.measure(cdr["Q_s1"], n, s1, rng(2), b=BLOCK)
    D["cdr3"] = {s: M.measure(cdr[f"Q_s{s}"], n, s3, rng(30 + s), b=BLOCK)[0] for s in SCALES}
    D["cl1"], _ = M.measure(cl["Q_s1"], n, s1, rng(3), b=BLOCK)
    D["pec_rng"] = rng(4)
    return D


def base_methods(D, n, budget):
    Y, te = D["tr"]["S_ideal"], D["te"]
    tr1, te1 = {1: D["tr1"]}, {1: D["te1"]}
    stack = np.stack([D["te3"][s] for s in SCALES])
    cq, ce = B3.chain_calibration(n)
    yield "Noisy (raw)", lambda: D["te_raw"]
    yield "Noisy", lambda: D["te1"]
    yield "ZNE-lin", lambda: B.zne_poly(stack, order=1)
    yield "ZNE-Rich", lambda: B.zne_poly(stack, order=2)
    yield "ZNE-exp", lambda: B.zne_exp(stack)
    yield "PEC", lambda: B3.pec(te["S_ideal"], te["gamma"], n, budget, D["pec_rng"], M.readout_flip_scale(n), BLOCK)
    yield "CDR", lambda: B3.cdr(D["cdr1"], D["cdr"]["S_ideal"], D["te1"])
    yield "vnCDR", lambda: B3.vncdr(D["cdr3"], D["cdr"]["S_ideal"], D["te3"])
    yield "CDR-family", lambda: B.ols_per_coord({1: D["cl1"]}, D["cl"]["S_ideal"], te1, (1,))
    yield "OLS", lambda: B.ols_per_coord(tr1, Y, te1, (1,))
    yield "vnOLS", lambda: B.ols_per_coord(D["tr3"], Y, D["te3"], SCALES)
    yield "RF", lambda: B.rf_block(tr1, Y, te1, (1,))
    yield "MLP-QEM", lambda: B.mlp_global(tr1, Y, te1, (1,))
    if BLOCK != 2:                                      # the graph network and BlockGeo-v1 read two-qubit blocks;
        return                                          # run_gnn_b3.py runs the three-qubit graph network
    yield "GNN-QEM", lambda: B3.gnn_qem(D["tr1"], Y, D["te1"], cq, ce)
    from bench_shots import paper_nn_shots
    yield "BlockGeo-v1", lambda: paper_nn_shots(D["tr1"], D["tr_p"], D["tr"], D["te1"], D["te_p"], te, n, "blockgeo")


def our_methods(D, n, budget):
    Y = D["tr"]["S_ideal"]
    tr1, te1 = {1: D["tr1"]}, {1: D["te1"]}
    yield "BlockGeo-QEM", lambda: BlockGeoQEM2(beta=1.0, b=BLOCK).fit(tr1, Y).predict(te1)
    yield "BlockGeo-QEM-ms", lambda: BlockGeoQEM2(beta=1.0, scales=SCALES, b=BLOCK).fit(D["tr3"], Y).predict(D["te3"])
    if budget == MAIN_BUDGET:
        yield "BlockGeo-QEM (Clifford-trained)", lambda: BlockGeoQEM2(beta=1.0, b=BLOCK).fit(
            {1: D["cl1"]}, D["cl"]["S_ideal"]).predict(te1)
        yield "abl: HS state loss", lambda: BlockGeoQEM2(beta=1.0, loss="hs", b=BLOCK).fit(tr1, Y).predict(te1)
        yield "abl: coordinate head only", lambda: BlockGeoQEM(output="coords", loss="hs", ctx=16, b=BLOCK).fit(
            tr1, Y).predict(te1)
        yield "abl: no circuit context", lambda: BlockGeoQEM2(beta=1.0, ctx=0, b=BLOCK).fit(tr1, Y).predict(te1)


def run(part, fam, n, d, budget, methods=None, pred_path=None):
    D = draws(fam, n, d, budget)
    Yte = D["te"]["S_ideal"]
    gen = base_methods if part == "base" else our_methods
    rows, preds = [], {"__ideal__": Yte}
    for name, fn in gen(D, n, budget):
        if methods and name not in methods:
            continue
        t0 = time.time()
        p = fn()
        preds[name] = p
        df = metrics(name, p, Yte, D["te1"], b=BLOCK)
        df["fit_time_s"] = time.time() - t0
        rows.append(df)
    if pred_path:
        np.savez_compressed(pred_path, **{k.replace(" ", "_").replace(":", ""): v for k, v in preds.items()})
    df = pd.concat(rows, ignore_index=True)
    df["family"], df["n"], df["depth"], df["budget"] = fam, n, d, budget
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["base", "ours"], required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--depths", required=True)
    ap.add_argument("--tag", default="", help="suffix for the output files (e.g. 'nl' for metric re-runs)")
    ap.add_argument("--methods", default="", help="comma-separated subset of methods")
    args = ap.parse_args()
    os.makedirs(os.path.join(RES_DIR, "pred"), exist_ok=True)
    stem = f"{args.part}{args.tag}_{args.family}_n{args.n}_B{args.budget}"
    out = os.path.join(RES_DIR, f"{stem}_samples.csv")
    methods = [m for m in args.methods.split(",") if m] or None
    done = pd.read_csv(out) if os.path.exists(out) else None
    dfs = [done] if done is not None else []
    for d in [int(x) for x in args.depths.split(",")]:
        if done is not None and d in set(done.depth):
            continue
        t0 = time.time()
        dfs.append(run(args.part, args.family, args.n, d, args.budget, methods,
                       os.path.join(RES_DIR, "pred", f"{stem}_d{d}.npz")))
        pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
        cur = dfs[-1].groupby("method", sort=False)[["local_bures", "local_trace", "mae_off", "mae_diag"]].mean()
        print(f"== {args.part} {args.family} n={args.n} B={args.budget} depth={d} ({time.time() - t0:.0f}s)\n"
              + cur.to_string(float_format=lambda x: f"{x:.4g}"), flush=True)
