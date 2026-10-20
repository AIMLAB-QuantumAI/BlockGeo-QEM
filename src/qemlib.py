"""Shared helpers of BlockGeo-QEM: circuit ensembles, reduced block states, Pauli coordinates and state metrics.

Simulation primitives are imported from the workshop implementation (blockgeo_qem_fixed.py, the BlockGeo-v1 code)
so that circuits, noise model and conventions stay identical to it.
"""
import numpy as np

from blockgeo_qem_fixed import (
    BASIS_LABELS,
    basis_change_unitaries,
    fold_global,
    load_fake_backend,
    local_pauli_ops_np,
    local_pauli_strings,
    measurement_settings,
    project_density_np,
    random_circuit,
    rotate_density,
    simulate_density_matrix,
)


# ------------------------------------------------------------------
# Circuit sets (same RNG streams as the workshop paper)
# ------------------------------------------------------------------

def paper_seed(n, depth, split, seed=0):
    return seed + 1000 * n + depth if split == "train" else seed + 9999 + 1000 * n + depth


def tfim_circuit(n, steps, th_h, th_J):
    """Trotterized transverse-field Ising evolution from |0...0>: each step applies
    RX(th_h) on every qubit, then RZZ(th_J) on even bonds and on odd bonds. th_h and th_J may be
    scalars (uniform chain) or per-qubit / per-bond arrays (disordered chain)."""
    from qiskit import QuantumCircuit

    th_h = np.broadcast_to(th_h, (n,))
    th_J = np.broadcast_to(th_J, (n - 1,))
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for q in range(n):
            qc.rx(float(th_h[q]), q)
        for start in (0, 1):
            for a in range(start, n - 1, 2):
                qc.cx(a, a + 1)
                qc.rz(float(th_J[a]), a + 1)
                qc.cx(a, a + 1)
    return qc


def make_circuits(family, n, depth, num, split, seed=0):
    rng = np.random.default_rng(paper_seed(n, depth, split, seed))
    if family == "random":
        return [random_circuit(n, depth, rng) for _ in range(num)]
    if family == "tfim":
        # depth = number of Trotter steps; couplings drawn per circuit, uniform in space and time
        return [tfim_circuit(n, depth, rng.uniform(0, np.pi / 2), rng.uniform(-np.pi / 2, 0))
                for _ in range(num)]
    if family == "dtfim":
        # disordered TFIM: every qubit has its own field and every bond its own coupling (2n-1 parameters)
        return [tfim_circuit(n, depth, rng.uniform(0, np.pi / 2, n), rng.uniform(-np.pi / 2, 0, n - 1))
                for _ in range(num)]
    raise ValueError(f"unknown family {family}")


# ------------------------------------------------------------------
# Reduced states and Pauli coordinates (conventions of the workshop code)
# ------------------------------------------------------------------
# Qiskit is little-endian: qubit q is bit q of the basis index, i.e. tensor axis
# n-1-q. In a block's Pauli string, position j acts on block[b-1-j] (np.kron puts
# the first factor in the most significant slot).

_LET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def reduced_dm(rho, n, qubits):
    """Partial trace onto `qubits`, returned in Qiskit order (qubits[0] least significant)."""
    keep = sorted(qubits)
    t = rho.reshape([2] * (2 * n))
    row = list(_LET[:n])
    col = list(_LET[n:2 * n])
    for q in range(n):
        if q not in keep:
            col[n - 1 - q] = row[n - 1 - q]
    out_r = [row[n - 1 - q] for q in reversed(keep)]
    out_c = [col[n - 1 - q] for q in reversed(keep)]
    expr = "".join(row) + "".join(col) + "->" + "".join(out_r) + "".join(out_c)
    d = 2 ** len(keep)
    return np.einsum(expr, t).reshape(d, d)


_OPS_CACHE = {}


def pauli_ops(b):
    if b not in _OPS_CACHE:
        _, ops = local_pauli_ops_np(b, include_identity=False)
        _OPS_CACHE[b] = np.stack(ops)
    return _OPS_CACHE[b]


def contiguous_blocks(n, b):
    return [tuple(range(s, s + b)) for s in range(n - b + 1)]


def block_paulis(rho, n, b):
    """Pauli coordinates of every contiguous b-block: array [n-b+1, 4^b-1]."""
    ops = pauli_ops(b)
    out = []
    for blk in contiguous_blocks(n, b):
        r = reduced_dm(rho, n, blk)
        out.append(np.real(np.einsum("kij,ji->k", ops, r)))
    return np.asarray(out)


def setting_probs(rho, n, b):
    """Outcome distributions in each of the 3^b covering measurement settings: [3^b, 2^n]."""
    bcu = basis_change_unitaries()
    out = []
    for s in measurement_settings(n, b):
        if all(c == "Z" for c in s):
            p = np.real(np.diag(rho))
        else:
            p = np.real(np.diag(rotate_density(rho, n, [bcu[c] for c in s])))
        p = np.clip(p, 0.0, None)
        out.append(p / p.sum())
    return np.asarray(out)


def diag_offdiag_masks(b):
    strings = local_pauli_strings(b, include_identity=False)
    diag = np.array([set(s) <= {"I", "Z"} for s in strings])
    return diag, ~diag


# ------------------------------------------------------------------
# State-space metrics
# ------------------------------------------------------------------

def project_to_states(rho):
    """Frobenius-nearest density matrix (Smolin, Gambetta & Smith, PRL 108, 070502 (2012)):
    shift all eigenvalues by a common constant, clip at zero, keep unit trace."""
    rho = 0.5 * (rho + rho.conj().T)
    lam, V = np.linalg.eigh(rho)
    lam = np.real(lam)
    srt = np.sort(lam)[::-1]
    css = np.cumsum(srt)
    k = np.arange(1, len(srt) + 1)
    r = np.nonzero(srt - (css - 1.0) / k > 0)[0][-1]
    mu = (css[r] - 1.0) / (r + 1)
    lam = np.clip(lam - mu, 0.0, None)
    return (V * lam) @ V.conj().T


def obs_to_rho(s, b):
    """Pauli coordinates -> nearest density matrix (numpy, single block)."""
    d = 2 ** b
    rho = np.eye(d, dtype=np.complex128) + np.einsum("k,kij->ij", s.astype(np.complex128), pauli_ops(b))
    return project_to_states(rho / d)


def _sqrtm_psd(m):
    m = 0.5 * (m + m.conj().T)
    w, v = np.linalg.eigh(m)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.conj().T


def bures_sq(r, q):
    sr = _sqrtm_psd(r)
    f = np.real(np.trace(_sqrtm_psd(sr @ q @ sr))) ** 2
    return 2.0 * (1.0 - np.sqrt(np.clip(f, 0.0, 1.0)))


def local_bures(pred, ideal, b):
    """RMS block Bures distance per sample. pred/ideal: [N, nblocks, 4^b-1]."""
    N, nb, _ = pred.shape
    out = np.empty(N)
    for i in range(N):
        acc = 0.0
        for k in range(nb):
            acc += bures_sq(obs_to_rho(ideal[i, k], b), obs_to_rho(pred[i, k], b))
        out[i] = np.sqrt(acc / nb)
    return out


def trace_dist(r, q):
    """(1/2)||r - q||_1; 2x this is the largest error of any block observable with ||O|| <= 1."""
    return 0.5 * np.abs(np.linalg.eigvalsh(0.5 * ((r - q) + (r - q).conj().T))).sum()


def local_trace(pred, ideal, b):
    """Mean block trace distance per sample (states projected as in local_bures)."""
    N, nb, _ = pred.shape
    out = np.empty(N)
    for i in range(N):
        out[i] = np.mean([trace_dist(obs_to_rho(ideal[i, k], b), obs_to_rho(pred[i, k], b)) for k in range(nb)])
    return out


# ------------------------------------------------------------------
# Spectral and entanglement quantities of 2-qubit block states
# ------------------------------------------------------------------

_YY = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]]))


def block_states(S, b=2):
    """[..., 4^b-1] Pauli coordinates -> nearest density matrices [..., 2^b, 2^b]."""
    flat = S.reshape(-1, S.shape[-1])
    return np.stack([obs_to_rho(s, b) for s in flat]).reshape(*S.shape[:-1], 2 ** b, 2 ** b)


def purity(R):
    return np.real(np.einsum("...ij,...ji->...", R, R))


def concurrence(R):
    """Wootters concurrence of 2-qubit states [..., 4, 4]."""
    Rt = _YY @ R.conj() @ _YY
    ev = np.linalg.eigvals(R @ Rt)
    lam = np.sort(np.sqrt(np.clip(np.real(ev), 0, None)), axis=-1)[..., ::-1]
    return np.clip(lam[..., 0] - lam[..., 1] - lam[..., 2] - lam[..., 3], 0, None)
