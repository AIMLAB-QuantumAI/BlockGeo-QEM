"""Quench dynamics on hardware: the models trained on the hardware measurements of the random-angle family
(<hardware dir>/<backend>_d<d>_...) mitigate the fixed-parameter quench circuit measured at the same step count
(<hardware dir>/quench_<backend>). Statistical uncertainty: R multinomial resamples of the measured frequencies.
The ideal marginals of the training circuits are read from quench_<backend>/labels.npz (make_labels.py).
Results: <hardware dir>/quench_<backend>/results.csv (rows as the quench_*.csv of run_quench.py, plus method "ideal")
and pred_d<d>.npz.

usage: quench_hw.py ibm_fez "1:ibm_fez_d1_tr100_te30_s1000,2:ibm_fez_d2_tr100_te30_s500"
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from qiskit.quantum_info import DensityMatrix

import baselines as B
import baselines_v3 as B3
import qemlib as L
from analyze_hw import ROOT, confusion_from_cal, estimate, load_run
from run_quench import TH_H, TH_J, quantities
from v3 import BlockGeoQEM2

R = 30


def main(backend, runs):
    qd = os.path.join(ROOT, f"quench_{backend}")
    meta = json.load(open(os.path.join(qd, "meta.json")))
    z = np.load(os.path.join(qd, "job0.npz"))
    n, ns = len(meta["chain"]), len(meta["settings"])
    A = confusion_from_cal(z["cal"], n)
    rows = []
    for d, tag in runs.items():
        freq = np.zeros((ns, 2 ** n))
        for j, (dd, s) in enumerate(meta["pairs"]):
            if dd == d:
                freq[s] = z["probs"][j]
        rng = np.random.default_rng([13, d])
        freq = freq.astype(np.float64) / freq.sum(-1, keepdims=True)
        res = np.stack([[rng.multinomial(meta["shots"], f) / meta["shots"] for f in freq] for _ in range(R)])
        te = estimate(res, n, A)                                             # [R, n-1, 15] with REM
        S_ideal = L.block_paulis(DensityMatrix(L.tfim_circuit(n, d, TH_H, TH_J)).data, n, 2)
        _, S_rem, _, m = load_run(tag)
        Y = np.load(os.path.join(qd, "labels.npz"))[f"S_train_d{d}"][: m["n_train"]]
        X = S_rem["train"]
        cq, ce = np.array(m["calibration"]["cq"]), np.array(m["calibration"]["ce"])
        preds = {
            "Noisy": te,
            "OLS": B.ols_per_coord({1: X}, Y, {1: te}, (1,)),
            "MLP-QEM": B.mlp_global({1: X}, Y, {1: te}, (1,)),
            "GNN-QEM": B3.gnn_qem(X, Y, te, cq, ce),
            "abl: HS state loss": BlockGeoQEM2(beta=1.0, loss="hs").fit({1: X}, Y).predict({1: te}),
            "BlockGeo-QEM": BlockGeoQEM2(beta=1.0).fit({1: X}, Y).predict({1: te}),
        }
        np.savez_compressed(os.path.join(qd, f"pred_d{d}.npz"), __ideal__=S_ideal,
                            **{k.replace(' ', '_').replace(':', ''): v for k, v in preds.items()})
        ideal = quantities(S_ideal[None])
        rows.append(dict(depth=d, method="ideal", draw=-1, **{k: float(v[0]) for k, v in ideal.items()}))
        for name, P in preds.items():
            q = quantities(P)
            for r in range(R):
                rows.append(dict(depth=d, method=name, draw=r, **{k: float(v[r]) for k, v in q.items()}))
        df = pd.DataFrame(rows)
        print(f"== hardware quench d={d}\n" + df[df.depth == d].groupby("method")[
            ["X", "ZZ", "purity", "concurrence"]].mean().round(4).to_string(), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(qd, "results.csv"), index=False)


if __name__ == "__main__":
    runs = {int(k): v for k, v in (item.split(":") for item in sys.argv[2].split(","))}
    main(sys.argv[1], runs)
