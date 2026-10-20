"""Bures (SLD quantum Fisher) metric in Pauli coordinates.

For rho = (I + sum_P s_P P)/d and a small displacement ds, the squared Bures distance
D_B^2 = 2(1 - sqrt F) satisfies D_B^2 ~= ds^T G(rho) ds with

    G_PQ = 1/(2 d^2) * sum_jk Re(P_jk conj(Q_jk)) / (lambda_j + lambda_k),

where (lambda_j, |j>) is the eigendecomposition of rho. At rho = I/d, G = I/(2d)
(= I/4 for two qubits), so the Bures loss reduces to a rescaled Euclidean loss.
"""
import numpy as np

import qemlib as L


def metric_tensor(s, b=2, eps=1e-3):
    """G for one block state given by Pauli coordinates s (the state is PSD-projected first)."""
    d = 2 ** b
    rho = L.obs_to_rho(s, b)
    lam, V = np.linalg.eigh(rho)
    lam = np.clip(lam, 0.0, None)
    ops = L.pauli_ops(b)                                   # [K, d, d]
    Pe = np.einsum("ij,kjl,lm->kim", V.conj().T, ops, V)   # Paulis in the eigenbasis
    W = 1.0 / (lam[:, None] + lam[None, :] + eps)
    G = np.einsum("pjk,qjk,jk->pq", Pe, Pe.conj(), W).real / (2 * d * d)
    return 0.5 * (G + G.T)


def metric_tensors(S, b=2, eps=1e-3):
    """S: [..., 4^b-1] -> G: [..., 4^b-1, 4^b-1]."""
    flat = S.reshape(-1, S.shape[-1])
    G = np.stack([metric_tensor(s, b, eps) for s in flat])
    return G.reshape(*S.shape, S.shape[-1])
