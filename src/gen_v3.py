"""v3 data: corrected chain layout, Pauli-twirled gate noise; readout errors are applied at measurement time.

Per (family, n, depth) writes to the feats directory (paths.FEAT_DIR, WORK/feats_v3 by default):
  {fam}_n{n}_d{d}_train.npz / _test.npz : S_ideal, P_ideal, Q_s{1,3,5} [N, 9, 2^n] (true outcome distributions
                                           in the 9 covering settings), S_s{1,3,5}; test also has gamma (PEC overhead)
  {fam}_n{n}_d{d}_cdr.npz               : K=24 near-Clifford training circuits of every test circuit (CDR):
                                           S_ideal [50, K, nb, 15], Q_s{1,3,5} [50, K, 9, 2^n]
  {fam}_n{n}_d{d}_cliff.npz             : near-Clifford version of every training circuit: S_ideal, Q_s1
"""
import argparse
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from qiskit import QuantumCircuit

import noise_layout as NL
import qemlib as L
from paths import FEAT_DIR

SCALES = (1, 3, 5)
_S = {}


def near_clifford(qc, rng, keep_frac=0.3, sigma=1.0):
    """CDR training circuit (Czarnik et al., Quantum 5, 592 (2021)). keep_frac=0.3, sigma=1.0 and K=24 training
    circuits were chosen in a probe on 10 circuits: the defaults (0.1, 0.5) leave most observables constant over the
    training set for uniform-angle Trotter circuits and CDR then ignores the measured value.
    A random max(1, round(keep_frac * N)) of the
    N non-Clifford rotations keep their angles; every other rotation R(theta) is replaced by R(k pi/2), k drawn with
    probability proportional to exp(-||R(theta) - R(k pi/2)||_F^2 / sigma^2) = exp(-8 sin^2((theta - k pi/2)/2) / sigma^2)."""
    rot = [i for i, inst in enumerate(qc.data) if inst.operation.name in ("rx", "ry", "rz")
           and abs(inst.operation.params[0] / (np.pi / 2) - round(inst.operation.params[0] / (np.pi / 2))) > 1e-9]
    keep = set(rng.choice(rot, size=min(len(rot), max(1, int(round(keep_frac * len(rot))))), replace=False)) \
        if rot else set()
    ks = np.arange(4) * (np.pi / 2)
    out = QuantumCircuit(qc.num_qubits)
    for i, inst in enumerate(qc.data):
        op = inst.operation
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if i in rot and i not in keep:
            w = np.exp(-8 * np.sin((float(op.params[0]) - ks) / 2) ** 2 / sigma ** 2)
            getattr(out, op.name)(float(rng.choice(ks, p=w / w.sum())), *qs)
        else:
            out.append(op, qs)
    return out


def _init(n):
    from qiskit_aer import AerSimulator

    _S["ideal"] = AerSimulator(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    _S["noisy"] = AerSimulator(method="density_matrix", noise_model=NL.twirled_noise_model(n), basis_gates=NL.BASIS,
                               max_parallel_threads=1, max_parallel_experiments=1)


def _work(task):
    key, kind, qc, n = task
    if kind == "ideal":
        rho = L.simulate_density_matrix(qc, _S["ideal"], n)
        P = np.clip(np.real(np.diag(rho)), 0, None)
        return key, kind, L.block_paulis(rho, n, 2), P / P.sum(), None
    if kind == "gamma":
        return key, kind, NL.circuit_gamma(NL.to_device(qc, n), n), None, None
    rho = L.simulate_density_matrix(NL.to_device(L.fold_global(qc, kind), n), _S["noisy"], n)
    return key, kind, L.block_paulis(rho, n, 2), None, L.setting_probs(rho, n, 2).astype(np.float32)


def path(fam, n, d, part):
    return os.path.join(FEAT_DIR, f"{fam}_n{n}_d{d}_{part}.npz")


def run_setting(pool, fam, n, d, n_train, n_test, K):
    D, nb, nc, nset = 2 ** n, n - 1, 15, 9               # dimension, blocks, coordinates per block, settings
    train = L.make_circuits(fam, n, d, n_train, "train")
    test = L.make_circuits(fam, n, d, n_test, "test")
    rng = np.random.default_rng([7, n, d, 0 if fam == "tfim" else 1])
    cdr = [[near_clifford(qc, rng) for _ in range(K)] for qc in test]
    cliff = [near_clifford(qc, rng) for qc in train]
    tasks = []
    for split, circs in (("train", train), ("test", test), ("cliff", cliff)):
        for i, qc in enumerate(circs):
            tasks.append(((split, i), "ideal", qc, n))
            for s in (SCALES if split != "cliff" else (1,)):
                tasks.append(((split, i), s, qc, n))
            if split == "test":
                tasks.append(((split, i), "gamma", qc, n))
    for i, row in enumerate(cdr):
        for j, qc in enumerate(row):
            tasks.append((("cdr", i, j), "ideal", qc, n))
            for s in SCALES:
                tasks.append((("cdr", i, j), s, qc, n))
    arr = {"train": {}, "test": {}, "cliff": {}, "cdr": {}}
    shape = {"train": (n_train,), "test": (n_test,), "cliff": (n_train,), "cdr": (n_test, K)}
    for split in arr:
        arr[split]["S_ideal"] = np.empty(shape[split] + (nb, nc))
        if split in ("train", "test"):
            arr[split]["P_ideal"] = np.empty(shape[split] + (D,))
        for s in (SCALES if split != "cliff" else (1,)):
            arr[split][f"Q_s{s}"] = np.empty(shape[split] + (nset, D), dtype=np.float32)
            if split in ("train", "test"):
                arr[split][f"S_s{s}"] = np.empty(shape[split] + (nb, nc))
    arr["test"]["gamma"] = np.empty(n_test)
    for key, kind, a, b, c in pool.imap_unordered(_work, tasks, chunksize=2):
        split, idx = key[0], key[1:]
        A = arr[split]
        if kind == "ideal":
            A["S_ideal"][idx] = a
            if "P_ideal" in A:
                A["P_ideal"][idx] = b
        elif kind == "gamma":
            A["gamma"][idx] = a
        else:
            A[f"Q_s{kind}"][idx] = c
            if f"S_s{kind}" in A:
                A[f"S_s{kind}"][idx] = a
    for split in ("train", "test", "cliff", "cdr"):                     # cdr last: marks the setting complete
        np.savez(path(fam, n, d, split), **arr[split])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--depths", required=True)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1),
                    help="worker processes (default: 14 or the number of CPUs, whichever is smaller; "
                         f"{min(14, os.cpu_count() or 1)} here)")
    args = ap.parse_args()
    os.makedirs(FEAT_DIR, exist_ok=True)
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.n,)) as pool:
        for d in [int(x) for x in args.depths.split(",")]:
            if os.path.exists(path(args.family, args.n, d, "cdr")):        # written last
                print(f"skip {args.family} n={args.n} d={d}", flush=True)
                continue
            t0 = time.time()
            run_setting(pool, args.family, args.n, d, args.n_train, args.n_test, args.K)
            print(f"{args.family} n={args.n} d={d} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
