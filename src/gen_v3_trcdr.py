"""Near-Clifford (CDR) training sets for the *training* circuits, so a per-circuit CDR estimate can be used as a
model input during training as well as at test time.

Writes <feats dir>/{family}_n{n}_d{depth}_trcdr.npz: S_ideal [N_train, K, nb, 15], Q_s1 [N_train, K, 9, 2^n].
"""
import argparse
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import qemlib as L
from gen_v3 import _init, _work, near_clifford, path
from paths import FEAT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--depths", required=True)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    n, K = args.n, args.K
    D, nb = 2 ** n, n - 1
    os.makedirs(FEAT_DIR, exist_ok=True)
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(n,)) as pool:
        for d in [int(x) for x in args.depths.split(",")]:
            out = path(args.family, n, d, "trcdr")
            if os.path.exists(out):
                print(f"skip {out}", flush=True)
                continue
            t0 = time.time()
            train = L.make_circuits(args.family, n, d, args.n_train, "train")
            rng = np.random.default_rng([11, n, d, {"tfim": 0, "random": 1, "dtfim": 2}[args.family]])
            tasks = []
            for i, qc in enumerate(train):
                for j in range(K):
                    nc = near_clifford(qc, rng)
                    tasks += [(("trcdr", i, j), "ideal", nc, n), (("trcdr", i, j), 1, nc, n)]
            S = np.empty((args.n_train, K, nb, 15))
            Q = np.empty((args.n_train, K, 9, D), dtype=np.float32)
            for key, kind, a, b, c in pool.imap_unordered(_work, tasks, chunksize=4):
                i, j = key[1], key[2]
                if kind == "ideal":
                    S[i, j] = a
                else:
                    Q[i, j] = c
            np.savez(out, S_ideal=S, Q_s1=Q)
            print(f"{args.family} n={n} d={d} trcdr in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
