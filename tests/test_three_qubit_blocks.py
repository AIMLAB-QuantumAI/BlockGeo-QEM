"""Three-qubit blocks: the metrics, the PEC support of the covering settings and BlockGeo-QEM with b = 3 on random
states of a 5-qubit chain (three overlapping three-qubit blocks with 63 Pauli coordinates each)."""
import numpy as np
import pytest

import baselines_v3 as B3
import qemlib as L
import v3
from bench import metrics

N, B = 5, 3
NB, NS = N - B + 1, 4 ** B - 1


def random_states(count, rng):
    """Block coordinates of random mixed states of the chain, ideal and noisy (depolarized, with additive noise)."""
    ideal = []
    for _ in range(count):
        g = rng.normal(size=(2 ** N, 2)) + 1j * rng.normal(size=(2 ** N, 2))
        rho = g @ g.conj().T
        ideal.append(L.block_paulis(rho / np.trace(rho).real, N, B))
    ideal = np.asarray(ideal)
    noisy = np.clip(0.8 * ideal + 0.01 * rng.normal(size=ideal.shape), -1, 1)
    return ideal, noisy


@pytest.fixture(scope="module")
def states():
    rng = np.random.default_rng(0)
    return random_states(30, rng), random_states(6, rng)


def test_metrics_three_qubit_blocks(states):
    (_, _), (ideal, noisy) = states
    df = metrics("x", noisy, ideal, noisy, b=B)
    assert len(df) == len(ideal)
    assert np.isfinite(df[["local_bures", "local_trace", "mae_diag", "mae_off", "err_purity"]].to_numpy()).all()
    assert df.err_concurrence.isna().all()                  # the concurrence is defined for two qubits only


def test_coordinate_support_three_qubit_blocks():
    sup = B3.coord_support(N, B)
    assert len(sup) == NB * NS
    assert {len(s) for s in sup} == {1, 2, 3}


def test_blockgeo_qem_three_qubit_blocks(states):
    (tr_ideal, tr_noisy), (te_ideal, te_noisy) = states
    model = v3.BlockGeoQEM2(beta=1.0, b=B, max_epochs=20, patience=10, seeds=(0,), device="cpu")
    pred = model.fit({1: tr_noisy}, tr_ideal).predict({1: te_noisy})
    assert pred.shape == (len(te_ideal), NB, NS)
    eig = np.linalg.eigvalsh(v3.coords_to_rho(pred, B))   # every output block is a density matrix
    assert eig.min() > -1e-8
