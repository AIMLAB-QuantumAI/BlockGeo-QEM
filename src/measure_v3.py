"""Measurement model for v3: readout errors on every outcome, optional tensored readout-error mitigation.

Q holds the true outcome distributions of the 9 covering settings. Readout flips are applied with the
chain qubits' calibrated confusion matrices; with shots=None the readout-affected distributions are used
exactly. REM applies the inverse confusion matrices to the observed frequencies before Pauli estimates.
"""
import numpy as np

import noise_layout as NL
import shots as SH
from readout import apply_per_qubit

_A = {}


def confusion(n):
    if n not in _A:
        _A[n] = NL.readout_confusion(n)
    return _A[n]


def measure(Q, n, shots, rng, rem=True):
    """Q: [..., 9, 2^n] -> (S_hat [..., n-1, 15], p_hat [..., 2^n])."""
    A = confusion(n)
    lead, D = Q.shape[:-2], Q.shape[-1]
    Qf = np.asarray(Q, dtype=np.float64).reshape(-1, Q.shape[-2], D)
    Qm = np.clip(apply_per_qubit(Qf, A, n), 0, None)
    Qm /= Qm.sum(-1, keepdims=True)
    freq = Qm if shots is None else rng.multinomial(shots, Qm) / float(shots)
    if rem:
        freq = apply_per_qubit(freq, np.linalg.inv(A), n)
    signs, wcover, zidx, nb, ns = SH.tables(n)
    est = np.einsum("nsd,dm,sm->nm", freq, signs, wcover)
    return np.clip(est, -1, 1).reshape(lead + (nb, ns)), freq[:, zidx].reshape(lead + (D,))


def readout_flip_scale(n):
    """1 / (1 - P(1|0) - P(0|1)) per qubit: the factor by which (twirled) readout mitigation divides."""
    A = confusion(n)
    return 1.0 / (1.0 - A[:, 1, 0] - A[:, 0, 1])
