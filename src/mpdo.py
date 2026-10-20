"""Noisy circuit simulation on a qubit chain as a matrix-product state of Pauli coefficients (MPDO).

rho = 2^-n sum_P c_P P with c_P = Tr(rho P) is stored as an MPS over local Pauli indices (0 I, 1 X, 2 Y, 3 Z), real
tensors A[i] of shape (chi_l, 4, chi_r). Gates act through their Pauli transfer matrices (PTMs) and the twirled gate
errors of noise_layout.twirled_noise_model are Pauli channels, i.e. diagonal PTMs applied after each gate, exactly as
Aer applies them. Two-qubit gates act on neighbouring sites of the chain (device circuits from noise_layout.to_device).
The block Pauli coordinates of every pair (k, k+1) are the coefficients with identities on all other sites.
Used for the n = 20 study, where density matrices no longer fit in memory; validated against the density-matrix data
of gen_v3.py at n = 8 (test_mpdo.py).
"""
import numpy as np

import qemlib as L

_P = [np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], dtype=complex),
      np.array([[0, -1j], [1j, 0]], dtype=complex), np.array([[1, 0], [0, -1]], dtype=complex)]


def _basis2(order):
    """Operator basis for a two-qubit instruction. order=(0, 1): qargs = (site, site+1), so qargs[0] = site is the
    least significant qubit; order=(1, 0): qargs = (site+1, site)."""
    if order == (0, 1):
        return [np.kron(_P[b], _P[a]) for a in range(4) for b in range(4)]
    return [np.kron(_P[a], _P[b]) for a in range(4) for b in range(4)]


def ptm_unitary(U, basis):
    d = U.shape[0]
    return np.array([[np.real(np.trace(P @ U @ Q @ U.conj().T)) / d for Q in basis] for P in basis])


def ptm_channel(err, basis):
    from qiskit.quantum_info import SuperOp

    S = SuperOp(err).data
    d = basis[0].shape[0]
    out = np.empty((len(basis), len(basis)))
    for j, Q in enumerate(basis):
        EQ = (S @ Q.flatten(order="F")).reshape(d, d, order="F")
        for i, P in enumerate(basis):
            out[i, j] = np.real(np.trace(P @ EQ)) / d
    return out


class Device:
    """PTMs of every (gate, qubits) of a device circuit, gate followed by its twirled error, cached."""

    def __init__(self, noise_model=None):
        self.err = {}
        if noise_model is not None:
            for g, errs in noise_model._local_quantum_errors.items():
                for q, e in errs.items():
                    self.err[(g, tuple(q))] = e
        self.cache = {}

    def ptm(self, inst, qargs):
        op = inst.operation
        name = op.name
        if len(qargs) == 1:
            key = (name, qargs, tuple(float(p) for p in op.params))
            if key not in self.cache:
                from qiskit.quantum_info import Operator

                R = ptm_unitary(Operator(op).data, _P)
                e = self.err.get((name, qargs))
                if e is not None:
                    R = ptm_channel(e, _P) @ R
                self.cache[key] = R
            return self.cache[key]
        order = (0, 1) if qargs[1] == qargs[0] + 1 else (1, 0)
        key = (name, qargs, tuple(float(p) for p in op.params))
        if key not in self.cache:
            from qiskit.quantum_info import Operator

            B = _basis2(order)
            R = ptm_unitary(Operator(op).data, B)
            e = self.err.get((name, qargs))
            if e is not None:
                R = ptm_channel(e, B) @ R
            self.cache[key] = R
        return self.cache[key]


def simulate(device_qc, n, dev, chi_max=2048, cutoff=1e-13):
    """Evolve |0..0><0..0| through a device circuit. Returns the MPS tensors and the largest bond dimension and
    discarded weight encountered."""
    A = [np.array([1.0, 0.0, 0.0, 1.0]).reshape(1, 4, 1) for _ in range(n)]
    chi_peak, discarded = 1, 0.0
    for inst in device_qc.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        qargs = tuple(device_qc.find_bit(q).index for q in inst.qubits)
        R = dev.ptm(inst, qargs)
        if len(qargs) == 1:
            A[qargs[0]] = np.einsum("pq,lqr->lpr", R, A[qargs[0]])
            continue
        s = min(qargs)
        if max(qargs) != s + 1:
            raise ValueError("two-qubit gate on non-neighbouring sites")
        th = np.einsum("lam,mbr->labr", A[s], A[s + 1])
        cl, cr = th.shape[0], th.shape[3]
        th = np.einsum("xy,lyr->lxr", R, th.reshape(cl, 16, cr)).reshape(cl * 4, 4 * cr)
        U, S, Vt = np.linalg.svd(th, full_matrices=False)
        keep = max(1, min(chi_max, int(np.sum(S > cutoff * S[0]))))
        discarded = max(discarded, float(np.sum(S[keep:] ** 2) / np.sum(S ** 2)))
        A[s] = U[:, :keep].reshape(cl, 4, keep)
        A[s + 1] = (S[:keep, None] * Vt[:keep]).reshape(keep, 4, cr)
        chi_peak = max(chi_peak, keep)
    return A, chi_peak, discarded


_STR = L.local_pauli_strings(2, include_identity=False)
_IDX = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def block_coords(A):
    """[n-1, 15] block Pauli coordinates in the order of qemlib.local_pauli_strings(2): string (t0, t1) has t1 on
    site k and t0 on site k+1 (qubit order of baselines_v3.coord_support)."""
    n = len(A)
    left = [np.ones(1)]
    for i in range(n):
        left.append(left[-1] @ A[i][:, 0, :])
    right = [np.ones(1)]
    for i in range(n - 1, -1, -1):
        right.append(A[i][:, 0, :] @ right[-1])
    right = right[::-1]                     # right[i] = product of identity slices of sites i..n-1
    norm = float(left[n][0])
    out = np.empty((n - 1, 15))
    for k in range(n - 1):
        M = np.einsum("l,lam,mbr,r->ab", left[k], A[k], A[k + 1], right[k + 2])
        for j, (t0, t1) in enumerate(_STR):
            out[k, j] = M[_IDX[t1], _IDX[t0]] / norm
    return out, norm
