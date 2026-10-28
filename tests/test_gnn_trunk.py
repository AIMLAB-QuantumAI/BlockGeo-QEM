"""GNN-QEM (Bures), the message-passing network of GNN-QEM as the trunk of BlockGeo-QEM, on random states of a
5-qubit chain (four overlapping two-qubit blocks): physical outputs and the guard against three-qubit blocks."""
import numpy as np
import pytest

import baselines_v3 as B3
import qemlib as L
import v3
from gnn_bures import GNNBures

N = 5
NB, NS = N - 1, 15


def random_states(count, rng):
    """Block coordinates of random mixed states of the chain, ideal and noisy (depolarized, with additive noise)."""
    ideal = []
    for _ in range(count):
        g = rng.normal(size=(2 ** N, 2)) + 1j * rng.normal(size=(2 ** N, 2))
        rho = g @ g.conj().T
        ideal.append(L.block_paulis(rho / np.trace(rho).real, N, 2))
    ideal = np.asarray(ideal)
    noisy = np.clip(0.8 * ideal + 0.01 * rng.normal(size=ideal.shape), -1, 1)
    return ideal, noisy


def test_gnn_trunk_returns_states():
    rng = np.random.default_rng(0)
    (tr_ideal, tr_noisy), (te_ideal, te_noisy) = random_states(30, rng), random_states(6, rng)
    cq, ce = B3.chain_calibration(N)
    model = GNNBures(cq, ce, beta=1.0, max_epochs=20, patience=10, seeds=(0,), device="cpu")
    pred = model.fit({1: tr_noisy}, tr_ideal).predict({1: te_noisy})
    assert pred.shape == (len(te_ideal), NB, NS)
    eig = np.linalg.eigvalsh(v3.coords_to_rho(pred, 2))     # every output block is a density matrix
    assert eig.min() > -1e-8


def test_gnn_trunk_needs_two_qubit_blocks():
    cq, ce = B3.chain_calibration(N)
    with pytest.raises(ValueError):
        GNNBures(cq, ce, b=3, device="cpu")
