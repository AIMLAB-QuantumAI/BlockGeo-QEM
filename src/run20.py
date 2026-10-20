"""Uniform TFIM on a 20-qubit chain (Trotter steps 1-3), from the tensor-network data of gen20.py.

In-size learned methods are trained on 200 twenty-qubit training circuits whose ideal marginals come from the
tensor-network simulation; BlockGeo-QEM (boundary positions), the graph network and the pooled OLS map are also
transferred from n = 8 and n = 10 without retraining. Per-circuit baselines that need no folded circuits: unmitigated,
idealized PEC and CDR. ZNE and the multi-noise-level methods are left out because the folded 20-qubit circuits cannot
be simulated exactly. All measurements, including those of the n = 8 and n = 10 training circuits, are drawn with
sample_blocks.measure_blocks.
Requires BLOCKGEO_CHAIN with the 20-qubit chain (first 10 qubits = default chain).
Results: <results dir>/n20_tfim_B{budget}_samples.csv (+ predictions in <results dir>/pred).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import baselines as B
import baselines_v3 as B3
import measure_v3 as M
import noise_layout as NL
import sample_blocks as SB
from bench import metrics
from paths import FEAT_DIR, RES_DIR, TAG, feats_dir
from v3 import Backbone, BlockGeoQEM2

F20 = feats_dir(TAG + "_n20")                       # data of gen20.py (WORK/feats_v3_n20 by default)


def draws20(d, budget, n=20, seed=0):
    rng = lambda k: np.random.default_rng([seed, n, d, budget, k])
    shots = None if budget == 0 else budget
    get = lambda part: dict(np.load(os.path.join(F20, f"tfim_n{n}_d{d}_{part}.npz")))
    tr, te, cdr = get("train"), get("test"), get("cdr")
    return dict(tr=tr, te=te, cdr=cdr,
                tr1=SB.measure_blocks(tr["S_s1"], n, shots, rng(0)),
                te1=SB.measure_blocks(te["S_s1"], n, shots, rng(1)),
                te_raw=SB.measure_blocks(te["S_s1"], n, shots, rng(1), rem=False),
                cdr1=SB.measure_blocks(cdr["S_s1"], n, shots, rng(2)), pec_rng=rng(4))


def source(n_src, d, budget, seed=0):
    """Training circuits of the n_src-qubit chain (gen_v3.py data), measured with the same block sampler."""
    tr = dict(np.load(os.path.join(FEAT_DIR, f"tfim_n{n_src}_d{d}_train.npz")))
    shots = None if budget == 0 else budget
    X = SB.measure_blocks(tr["S_s1"], n_src, shots, np.random.default_rng([seed, n_src, d, budget, 0]))
    return X, tr["S_ideal"]


def methods(D, d, budget, n=20):
    Y, te = D["tr"]["S_ideal"], D["te"]
    tr1, te1 = {1: D["tr1"]}, {1: D["te1"]}
    cal_t = B3.chain_calibration(n)
    yield "Noisy (raw)", lambda: D["te_raw"]
    yield "Noisy", lambda: D["te1"]
    yield "PEC", lambda: B3.pec(te["S_ideal"], te["gamma"], n, budget, D["pec_rng"], M.readout_flip_scale(n))
    yield "CDR", lambda: B3.cdr(D["cdr1"], D["cdr"]["S_ideal"], D["te1"])
    yield "OLS", lambda: B.ols_per_coord(tr1, Y, te1, (1,))
    yield "RF", lambda: B.rf_block(tr1, Y, te1, (1,))
    yield "MLP-QEM", lambda: B.mlp_global(tr1, Y, te1, (1,))
    yield "GNN-QEM", lambda: B3.gnn_qem(D["tr1"], Y, D["te1"], *cal_t)
    yield "BlockGeo-QEM", lambda: BlockGeoQEM2(beta=1.0).fit(tr1, Y).predict(te1)
    yield "abl: HS state loss", lambda: BlockGeoQEM2(beta=1.0, loss="hs").fit(tr1, Y).predict(te1)
    for n_src in (8, 10):
        Xs, Ys = source(n_src, d, budget)
        cal_s = B3.chain_calibration(n_src)
        yield f"BlockGeo-QEM (transfer {n_src}->20)", lambda Xs=Xs, Ys=Ys: BlockGeoQEM2(
            beta=1.0, pos_mode="boundary").fit({1: Xs}, Ys).predict(te1)
        yield f"GNN-QEM (transfer {n_src}->20)", lambda Xs=Xs, Ys=Ys, cal_s=cal_s: B3.gnn_qem(
            Xs, Ys, D["te1"], *cal_s, test_calib=cal_t)
        yield f"OLS-pooled (transfer {n_src}->20)", lambda Xs=Xs, Ys=Ys: np.clip(
            Backbone("boundary").fit(Xs, Ys, 1).predict(D["te1"]), -1, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--depths", default="1,2,3")
    args = ap.parse_args()
    assert len(NL.CHAIN) >= 20, "set BLOCKGEO_CHAIN to the 20-qubit chain"
    os.makedirs(os.path.join(RES_DIR, "pred"), exist_ok=True)
    stem = f"n20_tfim_B{args.budget}"
    out = os.path.join(RES_DIR, f"{stem}_samples.csv")
    done = pd.read_csv(out) if os.path.exists(out) else None
    dfs = [done] if done is not None else []
    for d in [int(x) for x in args.depths.split(",")]:
        if done is not None and d in set(done.depth):
            continue
        t0 = time.time()
        D = draws20(d, args.budget)
        Yte = D["te"]["S_ideal"]
        rows, preds = [], {"__ideal__": Yte}
        for name, fn in methods(D, d, args.budget):
            t1 = time.time()
            p = fn()
            preds[name] = p
            df = metrics(name, p, Yte, D["te1"])
            df["fit_time_s"] = time.time() - t1
            rows.append(df)
        df = pd.concat(rows, ignore_index=True)
        df["family"], df["n"], df["depth"], df["budget"] = "tfim", 20, d, args.budget
        np.savez_compressed(os.path.join(RES_DIR, "pred", f"{stem}_d{d}.npz"),
                            **{k.replace(" ", "_").replace(">", ""): v for k, v in preds.items()})
        dfs.append(df)
        pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
        cur = df.groupby("method", sort=False)[["local_bures", "local_trace", "mae_off", "err_purity"]].mean()
        print(f"== {stem} depth={d} ({time.time() - t0:.0f}s)\n" + cur.to_string(float_format=lambda x: f"{x:.4g}"),
              flush=True)
