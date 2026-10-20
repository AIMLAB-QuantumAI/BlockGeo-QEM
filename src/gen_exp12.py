"""Data for the data-size (exp 1) and calibration-drift (exp 2) studies, TFIM n=8.

exp 1: training circuits 200..399 (same RNG stream as the first 200)
       -> <feats dir>/{fam}_n{n}_d{d}_trainx.npz: S_ideal, P_ideal, Q_s1
exp 2: the 50 test circuits and their 24 CDR circuits each, run with every gate error probability scaled by f
       -> <feats dir>/drift/{fam}_n{n}_d{d}_f{f}.npz: Q_s{1,3,5}, gamma (test) and cdr_Q_s1 [50, 24, 9, 2^n]
"""
import argparse
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import noise_layout as NL
import qemlib as L
from gen_v3 import near_clifford
from paths import FEAT_DIR

SCALES = (1, 3, 5)
_S = {}


def _init(n, fs):
    from qiskit_aer import AerSimulator

    _S["ideal"] = AerSimulator(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    for f in fs:
        _S[f] = AerSimulator(method="density_matrix", noise_model=NL.twirled_noise_model(n, scale=f),
                             basis_gates=NL.BASIS, max_parallel_threads=1, max_parallel_experiments=1)


def _work(task):
    key, kind, qc, n, f = task
    if kind == "ideal":
        rho = L.simulate_density_matrix(qc, _S["ideal"], n)
        P = np.clip(np.real(np.diag(rho)), 0, None)
        return key, kind, L.block_paulis(rho, n, 2), P / P.sum(), None
    if kind == "gamma":
        return key, kind, NL.circuit_gamma(NL.to_device(qc, n), n, scale=f), None, None
    rho = L.simulate_density_matrix(NL.to_device(L.fold_global(qc, kind), n), _S[f], n)
    return key, kind, None, None, L.setting_probs(rho, n, 2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="tfim")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--depths", default="1,3,6")
    ap.add_argument("--drift", default="0.7,0.85,1.15,1.3")
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    fam, n, K = args.family, args.n, args.K
    fs = [float(x) for x in args.drift.split(",")]
    D, nb = 2 ** n, n - 1
    os.makedirs(os.path.join(FEAT_DIR, "drift"), exist_ok=True)
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(n, [1.0] + fs)) as pool:
        for d in [int(x) for x in args.depths.split(",")]:
            # ---- exp 1: extra training circuits ----
            out = os.path.join(FEAT_DIR, f"{fam}_n{n}_d{d}_trainx.npz")
            if not os.path.exists(out):
                t0 = time.time()
                extra = L.make_circuits(fam, n, d, 400, "train")[200:]
                tasks = [((i,), k, qc, n, 1.0) for i, qc in enumerate(extra) for k in ("ideal", 1)]
                S, P, Q = np.empty((200, nb, 15)), np.empty((200, D)), np.empty((200, 9, D), dtype=np.float32)
                for key, kind, a, b, c in pool.imap_unordered(_work, tasks, chunksize=2):
                    if kind == "ideal":
                        S[key[0]], P[key[0]] = a, b
                    else:
                        Q[key[0]] = c
                np.savez(out, S_ideal=S, P_ideal=P, Q_s1=Q)
                print(f"trainx {fam} n={n} d={d} in {time.time() - t0:.0f}s", flush=True)
            # ---- exp 2: drifted test and CDR circuits ----
            test = L.make_circuits(fam, n, d, 50, "test")
            rng = np.random.default_rng([7, n, d, 0 if fam == "tfim" else 1])      # same stream as gen_v3
            cdr = [[near_clifford(qc, rng) for _ in range(K)] for qc in test]
            for f in fs:
                out = os.path.join(FEAT_DIR, "drift", f"{fam}_n{n}_d{d}_f{f}.npz")
                if os.path.exists(out):
                    continue
                t0 = time.time()
                tasks = [((i,), s, qc, n, f) for i, qc in enumerate(test) for s in SCALES]
                tasks += [((i,), "gamma", qc, n, f) for i, qc in enumerate(test)]
                tasks += [((i, j), 1, c, n, f) for i, row in enumerate(cdr) for j, c in enumerate(row)]
                A = {f"Q_s{s}": np.empty((50, 9, D), dtype=np.float32) for s in SCALES}
                A["gamma"] = np.empty(50)
                A["cdr_Q_s1"] = np.empty((50, K, 9, D), dtype=np.float32)
                for key, kind, a, b, c in pool.imap_unordered(_work, tasks, chunksize=2):
                    if kind == "gamma":
                        A["gamma"][key[0]] = a
                    elif len(key) == 2:
                        A["cdr_Q_s1"][key] = c
                    else:
                        A[f"Q_s{kind}"][key[0]] = c
                np.savez(out, **A)
                print(f"drift {fam} n={n} d={d} f={f} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
