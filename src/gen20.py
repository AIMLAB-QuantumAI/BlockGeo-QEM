"""Uniform TFIM on a 20-qubit chain with the MPDO simulator (mpdo.py): exact block coordinates of the ideal state and
of the noisy state at fold scales 1, 3, 5, for the same circuit ensembles and CDR construction as gen_v3.py.

Requires BLOCKGEO_CHAIN to hold a 20-qubit chain whose first 10 qubits are noise_layout's default chain.
Per depth writes to WORK/feats_v3_n20/ (paths.feats_dir(BLOCKGEO_TAG + "_n20")):
  tfim_n20_d{d}_test.npz  : S_ideal, S_s{scale} [50, 19, 15] for the requested fold scales, gamma [50], chi (largest
                            bond dimension). Folded circuits are exact only at small depth (chi grows quickly),
                            so the n = 20 study uses scale 1 and leaves out ZNE.
  tfim_n20_d{d}_train.npz : S_ideal, S_s1 [200, 19, 15]
  tfim_n20_d{d}_cdr.npz   : S_ideal, S_s1 [50, K, 19, 15] for the K near-Clifford circuits of every test circuit
Measurements with finite shots are drawn later from these coordinates (sample_blocks.py).
"""
import argparse
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import mpdo
import noise_layout as NL
import paths
import qemlib as L
from gen_v3 import near_clifford

OUT = paths.feats_dir(paths.TAG + "_n20")
_D = {}


def _init(n):
    _D["noisy"], _D["ideal"] = mpdo.Device(NL.twirled_noise_model(n)), mpdo.Device(None)
    _D["n"] = n


def _work(task):
    key, kind, qc = task
    n = _D["n"]
    if kind == "ideal":
        A, chi, _ = mpdo.simulate(NL.to_device(qc, n), n, _D["ideal"])
        return key, kind, mpdo.block_coords(A)[0], chi
    if kind == "gamma":
        return key, kind, NL.circuit_gamma(NL.to_device(qc, n), n), 0
    A, chi, disc = mpdo.simulate(NL.to_device(L.fold_global(qc, kind), n), n, _D["noisy"])
    if disc > 1e-10:
        raise RuntimeError(f"truncation {disc:.1e} at {key} scale {kind}")
    return key, kind, mpdo.block_coords(A)[0], chi


def run(pool, tasks, shape_of):
    out, chi = {}, 0
    for key, kind, val, c in pool.imap_unordered(_work, tasks, chunksize=1):
        arr = out.setdefault(kind, np.empty(shape_of(kind)))
        arr[key] = val
        chi = max(chi, c)
    return out, chi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="worker processes (default: 8 or the number of CPUs, whichever is smaller; "
                         f"{min(8, os.cpu_count() or 1)} here)")
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--scales", default="1")
    args = ap.parse_args()
    n, K, nb = args.n, args.K, args.n - 1
    scales = [int(x) for x in args.scales.split(",")]
    assert len(NL.CHAIN) >= n, "set BLOCKGEO_CHAIN to a chain of at least n qubits"
    os.makedirs(OUT, exist_ok=True)
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(n,)) as pool:
        for d in [int(x) for x in args.depths.split(",")]:
            t0 = time.time()
            f_test = os.path.join(OUT, f"tfim_n{n}_d{d}_test.npz")
            if not os.path.exists(f_test):
                test = L.make_circuits("tfim", n, d, 50, "test")
                tasks = [((i,), k, qc) for i, qc in enumerate(test) for k in ["ideal", "gamma"] + scales]
                r, chi = run(pool, tasks, lambda k: (50,) if k == "gamma" else (50, nb, 15))
                np.savez(f_test, S_ideal=r["ideal"], gamma=r["gamma"], chi=chi, **{f"S_s{s}": r[s] for s in scales})
                print(f"test d={d} chi {chi} ({time.time() - t0:.0f}s)", flush=True)
            f_train = os.path.join(OUT, f"tfim_n{n}_d{d}_train.npz")
            if not os.path.exists(f_train):
                train = L.make_circuits("tfim", n, d, 200, "train")
                tasks = [((i,), k, qc) for i, qc in enumerate(train) for k in ("ideal", 1)]
                r, chi = run(pool, tasks, lambda k: (200, nb, 15))
                np.savez(f_train, S_ideal=r["ideal"], S_s1=r[1], chi=chi)
                print(f"train d={d} chi {chi} ({time.time() - t0:.0f}s)", flush=True)
            f_cdr = os.path.join(OUT, f"tfim_n{n}_d{d}_cdr.npz")
            if not os.path.exists(f_cdr):
                test = L.make_circuits("tfim", n, d, 50, "test")
                rng = np.random.default_rng([7, n, d, 0])                    # same stream as gen_v3
                cdr = [[near_clifford(qc, rng) for _ in range(K)] for qc in test]
                tasks = [((i, j), k, c) for i, row in enumerate(cdr) for j, c in enumerate(row) for k in ("ideal", 1)]
                r, chi = run(pool, tasks, lambda k: (50, K, nb, 15))
                np.savez(f_cdr, S_ideal=r["ideal"], S_s1=r[1], chi=chi)
                print(f"cdr d={d} chi {chi} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
