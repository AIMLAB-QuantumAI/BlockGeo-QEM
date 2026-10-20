"""Bures metric in Pauli coordinates (src/geometry.py) for two-qubit block states.

For rho = (I + sum_P s_P P)/d with eigenvalues lam_j > 0 and a displacement delta rho = (1/d) sum_P ds_P P,
D_B^2(rho, rho + delta rho) = ds^T G ds + O(|ds|^3) with G_PQ = 1/(2 d^2) sum_jk Re(P_jk conj(Q_jk)) / (lam_j + lam_k)
in the eigenbasis of rho. The eigenvalues of G lie in [1/(4 d lam_max), 1/(4 d lam_min)], and G = I/4 at the
maximally mixed state.
The metric is evaluated without the regularisation of its denominators (eps=0) unless stated otherwise.
"""
import numpy as np
import pytest

import geometry as GEO
import qemlib as L

D = 4                        # dimension of a two-qubit block
PAULIS = L.pauli_ops(2)      # the 15 non-identity Pauli strings


def haar_unitary(rng):
    z = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_spectrum(rng, floor):
    """Random eigenvalues on the simplex, each at least `floor`; small Dirichlet parameters give nearly pure states."""
    lam = rng.dirichlet(np.full(D, rng.choice([0.3, 1.0, 3.0])))
    return floor + (1 - D * floor) * lam


def random_state(rng, lam):
    """Density matrix with eigenvalues lam and Haar-random eigenvectors."""
    u = haar_unitary(rng)
    return (u * lam) @ u.conj().T


def coords(rho):
    return np.real(np.einsum("kij,ji->k", PAULIS, rho))


def test_pauli_coordinate_convention():
    """The Pauli strings satisfy Tr(PQ) = 4 delta_PQ, so rho = (I + sum_P s_P P)/4 with s_P = Tr(rho P)."""
    np.testing.assert_allclose(np.einsum("pij,qji->pq", PAULIS, PAULIS), D * np.eye(15), atol=1e-14)
    rng = np.random.default_rng(0)
    rho = random_state(rng, random_spectrum(rng, floor=0.01))
    rebuilt = (np.eye(D) + np.einsum("k,kij->ij", coords(rho), PAULIS)) / D
    np.testing.assert_allclose(rebuilt, rho, atol=1e-14)


def test_metric_eigenvalues_within_bounds():
    """For random full-rank states, including nearly pure ones, G is symmetric and its eigenvalues lie in
    [1/(4 d lam_max), 1/(4 d lam_min)]."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        lam = random_spectrum(rng, floor=1e-4)
        G = GEO.metric_tensor(coords(random_state(rng, lam)), eps=0.0)
        assert G.shape == (15, 15)
        np.testing.assert_array_equal(G, G.T)
        ev = np.linalg.eigvalsh(G)
        assert ev.min() >= (1 - 1e-9) / (4 * D * lam.max())
        assert ev.max() <= (1 + 1e-9) / (4 * D * lam.min())


def test_metric_bounds_are_attained():
    """For the spectrum (a, a, b, b) the largest eigenvalue of G is 1/(4 d a) and the smallest 1/(4 d b), so the bounds
    cannot be tightened."""
    a, b = 0.05, 0.45
    rho = random_state(np.random.default_rng(2), np.array([a, a, b, b]))
    ev = np.linalg.eigvalsh(GEO.metric_tensor(coords(rho), eps=0.0))
    assert ev.max() == pytest.approx(1 / (4 * D * a), rel=1e-9)
    assert ev.min() == pytest.approx(1 / (4 * D * b), rel=1e-9)


def test_metric_spectrum_closed_form():
    """In the eigenbasis of rho the quadratic form is diagonal in the matrix elements, so G has the eigenvalues
    1/(2 d (lam_j + lam_k)) for j < k, each twice, and those of diag(1/(4 d lam)) on traceless diagonal matrices."""
    rng = np.random.default_rng(3)
    proj = np.eye(D) - np.ones((D, D)) / D          # projector onto traceless diagonals
    for _ in range(50):
        lam = random_spectrum(rng, floor=1e-3)
        ev = np.linalg.eigvalsh(GEO.metric_tensor(coords(random_state(rng, lam)), eps=0.0))
        off = [1 / (2 * D * (lam[j] + lam[k])) for j in range(D) for k in range(j + 1, D) for _ in range(2)]
        diag = np.linalg.eigvalsh(proj @ np.diag(1 / (4 * D * lam)) @ proj)[1:]     # drop the zero mode (identity)
        np.testing.assert_allclose(ev, np.sort(np.concatenate([off, diag])), rtol=1e-9)


def test_metric_at_maximally_mixed_state():
    """At rho = I/4 the metric is I/4; with a regularisation eps of the denominators lam_j + lam_k it is
    I/(4 + 2 d eps)."""
    np.testing.assert_allclose(GEO.metric_tensor(np.zeros(15), eps=0.0), np.eye(15) / 4, rtol=0, atol=1e-15)
    eps = 1e-3
    np.testing.assert_allclose(GEO.metric_tensor(np.zeros(15), eps=eps), np.eye(15) / (4 + 2 * D * eps), rtol=0,
                               atol=1e-15)


@pytest.mark.parametrize("state", ["random full-rank", "maximally mixed"])
def test_bures_distance_matches_quadratic_form(state):
    """D_B^2(rho, rho + delta rho) = ds^T G ds to second order: the relative deviation is below 1% at |ds| = 1e-3 and
    shrinks in proportion to |ds|."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        lam = random_spectrum(rng, floor=0.05) if state == "random full-rank" else np.full(D, 1 / D)
        rho = random_state(rng, lam)
        G = GEO.metric_tensor(coords(rho), eps=0.0)
        u = rng.standard_normal(15)
        u /= np.linalg.norm(u)
        dev = []
        for h in (1e-3, 1e-4):
            drho = h * np.einsum("k,kij->ij", u, PAULIS) / D
            dev.append(abs(L.bures_sq(rho, rho + drho) / (h * h * (u @ G @ u)) - 1))
        assert dev[0] < 1e-2
        assert dev[1] < 0.2 * dev[0] + 1e-6


def test_metric_tensors_batched():
    """metric_tensors maps coordinates [..., 15] to metrics [..., 15, 15], block by block."""
    S = np.random.default_rng(5).uniform(-0.2, 0.2, (2, 3, 15))
    G = GEO.metric_tensors(S)
    assert G.shape == (2, 3, 15, 15)
    for i in range(2):
        for k in range(3):
            np.testing.assert_array_equal(G[i, k], GEO.metric_tensor(S[i, k]))
