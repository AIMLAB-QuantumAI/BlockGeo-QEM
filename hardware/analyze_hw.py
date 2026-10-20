"""Block coordinates from a hardware run (<hardware dir>/<tag>): tensored readout-error mitigation with the run's own
calibration circuits, then the covering-setting estimator of measure_v3.py.

    S_raw, S_rem, A, meta = load_run(tag)    # dicts split -> [N, n-1, 15]; A [n, 2, 2] confusion matrices
    lab = load_labels(tag)                   # ideal block coordinates of the measured circuits (make_labels.py)

usage: analyze_hw.py <tag>    readout errors, and the errors of the unmitigated test-circuit estimates next to those
                              of the simulated device model for the same circuits
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import numpy as np

import shots as SH
from paths import HW as ROOT
from readout import apply_per_qubit


def confusion_from_cal(cal, n):
    """cal[0]: outcome distribution for |0..0>, cal[1]: for |1..1> -> A[q, m, t] = P(measured m | prepared t)."""
    A = np.zeros((n, 2, 2))
    for t in (0, 1):
        p = cal[t].reshape([2] * n)
        for q in range(n):
            ax = n - 1 - q
            marg = p.sum(axis=tuple(a for a in range(n) if a != ax))
            A[q, :, t] = marg
    return A


def estimate(freq, n, A=None, b=2):
    """freq [N, 9, 2^n] -> block coordinates [N, n-1, 15] (REM with A if given)."""
    if A is not None:
        freq = apply_per_qubit(freq, np.linalg.inv(A), n)
    signs, wcover, zidx, nb, ns = SH.tables(n, b)
    est = np.einsum("nsd,dm,sm->nm", freq, signs, wcover)
    return np.clip(est, -1, 1).reshape(len(freq), nb, ns)


def load_run(tag):
    d = os.path.join(ROOT, tag)
    meta = json.load(open(os.path.join(d, "meta.json")))
    n, ns = len(meta["chain"]), len(meta["settings"])
    probs, cal = {}, None
    for k in range(len(meta["jobs"])):
        z = np.load(os.path.join(d, f"job{k}.npz"))
        for j, p in zip(z["idx"], z["probs"]):
            probs[int(j)] = p
        if k == 0:
            cal = z["cal"]
    A = confusion_from_cal(cal, n)
    freq = {"train": np.zeros((meta["n_train"], ns, 2 ** n)), "test": np.zeros((meta["n_test"], ns, 2 ** n))}
    for j, (split, i, s) in enumerate(meta["pairs"]):
        freq[split][i, s] = probs[j]
    S_raw = {k: estimate(v, n) for k, v in freq.items() if len(v)}
    S_rem = {k: estimate(v, n, A) for k, v in freq.items() if len(v)}
    return S_raw, S_rem, A, meta


def load_labels(tag):
    """labels.npz of a run: ideal block coordinates of its circuits (make_labels.py)."""
    return dict(np.load(os.path.join(ROOT, tag, "labels.npz")))


if __name__ == "__main__":
    import qemlib as L

    tag = sys.argv[1]
    S_raw, S_rem, A, meta = load_run(tag)
    print("readout: P(1|0) per qubit", np.round(A[:, 1, 0], 3), " P(0|1)", np.round(A[:, 0, 1], 3))
    lab = load_labels(tag)
    Y = lab["S_test"][: meta["n_test"]]
    for name, S in (("raw", S_raw["test"]), ("REM", S_rem["test"])):
        print(f"test circuits, {name}: local Bures {L.local_bures(S, Y, 2).mean():.4f}, "
              f"trace {L.local_trace(S, Y, 2).mean():.4f}, MAE {np.abs(S - Y).mean():.4f}")
    Ssim = lab["S_test_s1"][: meta["n_test"]]
    print(f"FakeSherbrooke simulation of the same circuits (exact, REM): local Bures "
          f"{L.local_bures(Ssim, Y, 2).mean():.4f}, trace {L.local_trace(Ssim, Y, 2).mean():.4f}")
