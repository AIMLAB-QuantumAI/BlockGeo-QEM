"""Downstream task: quench dynamics of a fixed-parameter TFIM circuit (theta_h = pi/4, theta_J = -pi/4) on the
8-qubit chain, Trotter steps 1-8, mitigated by models trained on the random-angle family of the same step count.

For every step count the fixed circuit is simulated with the twirled device noise model, measured R times with the
single-scale protocol at the given budget, and mitigated by the learned methods (trained once per step count on the
200 training circuits of the gen_v3.py data). Reported per draw and method: mean <X_i> over sites, mean
<Z_i Z_{i+1}> over bonds, mean block purity and mean concurrence, together with the ideal values.
Results: <results dir>/quench_tfim_n{n}_B{budget}.csv and the predictions quench_pred_tfim_n{n}_B{budget}_d{d}.npz
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

import baselines as B
import baselines_v3 as B3
import measure_v3 as M
import noise_layout as NL
import qemlib as L
from paths import RES_DIR
from run_v3 import draws
from v3 import BlockGeoQEM2

TH_H, TH_J = np.pi / 4, -np.pi / 4
STR = L.local_pauli_strings(2, include_identity=False)
IX, XI, ZZ = STR.index(("I", "X")), STR.index(("X", "I")), STR.index(("Z", "Z"))


def quantities(S):
    """S [..., n-1, 15] -> dict of mean <X_i>, mean <Z_i Z_i+1>, mean purity, mean concurrence over the chain."""
    R = L.block_states(S, 2)
    x = np.concatenate([S[..., :, IX], S[..., -1:, XI]], axis=-1).mean(-1)   # qubit k from block k, last from block n-2
    return dict(X=x, ZZ=S[..., ZZ].mean(-1), purity=L.purity(R).mean(-1), concurrence=L.concurrence(R).mean(-1))


def fixed_circuit_data(n, d, sim_ideal, sim_noisy):
    qc = L.tfim_circuit(n, d, TH_H, TH_J)
    rho_i = L.simulate_density_matrix(qc, sim_ideal, n)
    rho_n = L.simulate_density_matrix(NL.to_device(qc, n), sim_noisy, n)
    return L.block_paulis(rho_i, n, 2), L.setting_probs(rho_n, n, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--depths", default="1,2,3,4,6,8")
    ap.add_argument("--draws", type=int, default=30)
    args = ap.parse_args()
    n, budget, R = args.n, args.budget, args.draws
    sim_ideal = AerSimulator(method="density_matrix")
    sim_noisy = AerSimulator(method="density_matrix", noise_model=NL.twirled_noise_model(n), basis_gates=NL.BASIS)
    cq, ce = B3.chain_calibration(n)
    os.makedirs(RES_DIR, exist_ok=True)
    rows = []
    for d in [int(x) for x in args.depths.split(",")]:
        t0 = time.time()
        S_ideal, Q = fixed_circuit_data(n, d, sim_ideal, sim_noisy)
        s1 = None if budget == 0 else budget // 9
        te = np.stack([M.measure(Q[None], n, s1, np.random.default_rng([11, n, d, budget, r]))[0][0] for r in range(R)])
        D = draws("tfim", n, d, budget)
        X, Y = D["tr1"], D["tr"]["S_ideal"]
        preds = {
            "Noisy": te,
            "OLS": B.ols_per_coord({1: X}, Y, {1: te}, (1,)),
            "MLP-QEM": B.mlp_global({1: X}, Y, {1: te}, (1,)),
            "GNN-QEM": B3.gnn_qem(X, Y, te, cq, ce),
            "abl: HS state loss": BlockGeoQEM2(beta=1.0, loss="hs").fit({1: X}, Y).predict({1: te}),
            "BlockGeo-QEM": BlockGeoQEM2(beta=1.0).fit({1: X}, Y).predict({1: te}),
        }
        np.savez_compressed(os.path.join(RES_DIR, f"quench_pred_tfim_n{n}_B{budget}_d{d}.npz"), __ideal__=S_ideal,
                            **{k.replace(' ', '_').replace(':', ''): v for k, v in preds.items()})
        ideal = quantities(S_ideal[None])
        rows.append(dict(depth=d, method="ideal", draw=-1, **{k: float(v[0]) for k, v in ideal.items()}))
        for name, P in preds.items():
            q = quantities(P)
            for r in range(R):
                rows.append(dict(depth=d, method=name, draw=r, **{k: float(v[r]) for k, v in q.items()}))
        out = os.path.join(RES_DIR, f"quench_tfim_n{n}_B{budget}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        df = pd.DataFrame(rows)
        g = df[df.depth == d].groupby("method")[["X", "ZZ", "purity", "concurrence"]].mean()
        print(f"== quench d={d} ({time.time() - t0:.0f}s)\n" + g.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
