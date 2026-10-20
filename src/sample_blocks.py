"""Finite-shot measurement of block coordinates from exact block Pauli coordinates, for chains too long for the
2^n outcome distributions of measure_v3.py (n = 20).

For every covering setting the outcomes are drawn as a Markov chain along the chain: qubit 0 from its marginal, then
qubit k+1 from the conditional of the exact pair marginal of (k, k+1). Every pair (k, k+1) therefore has its exact
joint outcome distribution and the shared qubits of overlapping blocks use the same shots, as in the device protocol;
only correlations between non-neighbouring qubits are approximated, and no block estimate depends on them.
Readout flips, tensored readout-error mitigation and the estimator follow measure_v3.py.
"""
import numpy as np

import noise_layout as NL
import qemlib as L

_STR = L.local_pauli_strings(2, include_identity=False)
_COL = {(t1, t0): j for j, (t0, t1) in enumerate(_STR)}      # (basis on site k, basis on site k+1) -> column


def _pair_coeffs(S):
    """S [nb, 15] -> c [nb, 4, 4] with c[k, a, b] = Tr(rho P_a(k) P_b(k+1)), indices 0 I, 1 X, 2 Y, 3 Z."""
    nb = S.shape[0]
    c = np.zeros((nb, 4, 4))
    c[:, 0, 0] = 1.0
    letters = "IXYZ"
    for a in range(4):
        for b in range(4):
            if a or b:
                c[:, a, b] = S[:, _COL[(letters[a], letters[b])]]
    return c


def _pair_probs(c, Bk, Bk1):
    """True outcome distributions p[k, m_k, m_k1] of every pair in bases Bk[k], Bk1[k] (letters X, Y, Z)."""
    idx = {"X": 1, "Y": 2, "Z": 3}
    nb = c.shape[0]
    p = np.zeros((nb, 2, 2))
    for k in range(nb):
        a, b = idx[Bk[k]], idx[Bk1[k]]
        for mk in (0, 1):
            for mk1 in (0, 1):
                sa, sb = (-1) ** mk, (-1) ** mk1
                p[k, mk, mk1] = 0.25 * (c[k, 0, 0] + sa * c[k, a, 0] + sb * c[k, 0, b] + sa * sb * c[k, a, b])
    p = np.clip(p, 0.0, None)
    return p / p.sum((1, 2), keepdims=True)


def measure_blocks(S, n, shots, rng, rem=True, A=None):
    """S: [..., n-1, 15] exact block coordinates of the (noisy) state -> estimated coordinates of the same shape.
    shots: total budget per circuit (split over the 9 settings) or None for exact expectation values."""
    A = NL.readout_confusion(n) if A is None else A                 # A[q, m, t] = P(m | t)
    Ainv = np.linalg.inv(A)
    settings = L.measurement_settings(n, 2)
    lead = S.shape[:-2]
    Sf = S.reshape((-1, n - 1, 15))
    out = np.zeros_like(Sf)
    for i in range(len(Sf)):
        c = _pair_coeffs(Sf[i])
        acc, cnt = np.zeros((n - 1, 15)), np.zeros((n - 1, 15))
        for s in settings:
            p = _pair_probs(c, [s[k] for k in range(n - 1)], [s[k + 1] for k in range(n - 1)])
            if shots is None:
                # exact: readout then REM is the identity on the pair distribution
                f = p
            else:
                m = shots // len(settings)
                bits = np.empty((n, m), dtype=np.int8)
                p0 = p[0].sum(1)
                bits[0] = rng.random(m) < p0[1]
                for k in range(n - 1):
                    cond1 = p[k][:, 1] / np.maximum(p[k].sum(1), 1e-300)          # P(m_k1 = 1 | m_k)
                    bits[k + 1] = rng.random(m) < cond1[bits[k]]
                flip1 = A[:, 1, :]                                                    # P(measure 1 | true t)
                u = rng.random((n, m))
                meas = (u < flip1[np.arange(n)[:, None], bits]).astype(np.int8)
                f = np.zeros((n - 1, 2, 2))
                for k in range(n - 1):
                    code = 2 * meas[k] + meas[k + 1]
                    f[k] = np.bincount(code, minlength=4).reshape(2, 2) / float(m)
                if rem:
                    f = np.einsum("kab,kxa,kyb->kxy", f, Ainv[:-1], Ainv[1:])
            for k in range(n - 1):
                Bk, Bk1 = s[k], s[k + 1]
                for a_l, b_l in ((Bk, "I"), ("I", Bk1), (Bk, Bk1)):
                    col = _COL[(a_l, b_l)]
                    sa = np.array([1, -1]) if a_l != "I" else np.array([1, 1])
                    sb = np.array([1, -1]) if b_l != "I" else np.array([1, 1])
                    acc[k, col] += np.sum(f[k] * np.outer(sa, sb))
                    cnt[k, col] += 1
        out[i] = np.clip(acc / cnt, -1, 1)
    return out.reshape(S.shape)
