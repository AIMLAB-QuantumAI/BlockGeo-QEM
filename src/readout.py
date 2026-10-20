"""Tensored readout errors and readout-error mitigation (REM): a 2x2 matrix applied qubit by qubit.

measure_v3.py applies the calibrated confusion matrices (readout flips) and their inverses (REM) to the outcome
distributions of the covering settings; hardware/analyze_hw.py applies the inverses estimated from the calibration
circuits of a hardware run.
"""
import numpy as np


def apply_per_qubit(Q, mats, n):
    """Apply a 2x2 matrix to each qubit axis of distributions Q [..., 2^n] (Qiskit little-endian)."""
    lead = Q.shape[:-1]
    t = Q.reshape(lead + (2,) * n)
    nl = len(lead)
    for q in range(n):
        ax = nl + (n - 1 - q)
        t = np.moveaxis(np.tensordot(mats[q], t, axes=([1], [ax])), 0, ax)
    return t.reshape(Q.shape)
