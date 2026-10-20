"""Block-wise finite-shot sampler (src/sample_blocks.py), which serves chains too long for 2^n outcome distributions,
against the global multinomial sampler of measure_v3.py on the four-qubit chain: the same exact limit, the same means
and spreads over repeated draws, and shared shots on overlapping blocks."""
import numpy as np
import pytest

import gen_v3
import measure_v3 as M
import qemlib as L
import sample_blocks as SB

N, DEPTH = 4, 2
BUDGET = 1800        # shots per circuit, 200 in each of the 9 covering settings
REPS = 400           # independent draws per sampler
Z_MAX = 4.5          # largest accepted standardised difference over the 45 block coordinates


@pytest.fixture(scope="module")
def state():
    """Exact noisy block coordinates S [3, 15] and true covering-setting distributions Q [9, 16] of one uniform-TFIM
    circuit, simulated as in gen_v3.py, and the readout confusion matrices of the chain."""
    gen_v3._init(N)
    qc = L.make_circuits("tfim", N, DEPTH, 1, "test")[0]
    _, _, S, _, Q = gen_v3._work((("test", 0), 1, qc, N))
    return S, Q, M.confusion(N)


@pytest.fixture(scope="module")
def draws(state):
    """REPS draws of each sampler at BUDGET shots per circuit, with readout errors and readout-error mitigation."""
    S, Q, A = state
    glob = M.measure(np.broadcast_to(Q, (REPS,) + Q.shape), N, BUDGET // 9, np.random.default_rng(1))[0]
    blocks = SB.measure_blocks(np.broadcast_to(S, (REPS,) + S.shape), N, BUDGET, np.random.default_rng(2), A=A)
    assert glob.shape == blocks.shape == (REPS, N - 1, 15)
    return glob, blocks


def test_exact_limit(state):
    """Without shot noise both samplers return the exact coordinates, i.e. readout-error mitigation undoes the
    readout errors."""
    S, Q, A = state
    np.testing.assert_allclose(SB.measure_blocks(S[None], N, None, None, A=A)[0], S, rtol=0, atol=1e-12)
    np.testing.assert_allclose(M.measure(Q[None], N, None, None)[0][0], S, rtol=0, atol=1e-6)    # Q is float32


def test_means_agree(state, draws):
    """For every coordinate the mean estimates of the two samplers agree within Z_MAX standard errors, and each
    agrees with the exact value."""
    S = state[0]
    glob, blocks = draws
    se_g, se_b = glob.std(0, ddof=1) / np.sqrt(REPS), blocks.std(0, ddof=1) / np.sqrt(REPS)
    assert se_g.min() > 0 and se_b.min() > 0
    assert np.max(np.abs(glob.mean(0) - blocks.mean(0)) / np.hypot(se_g, se_b)) < Z_MAX
    assert np.max(np.abs(glob.mean(0) - S) / se_g) < Z_MAX
    assert np.max(np.abs(blocks.mean(0) - S) / se_b) < Z_MAX


def test_spreads_agree(draws):
    """The shot noise of the two samplers has the same size: the standard deviations of every coordinate agree within
    30% and their median ratio within 10%."""
    glob, blocks = draws
    ratio = glob.std(0, ddof=1) / blocks.std(0, ddof=1)
    assert 0.9 < np.median(ratio) < 1.1
    assert np.all((ratio > 0.7) & (ratio < 1.3))


def test_overlapping_blocks_share_shots(draws):
    """Blocks (k, k+1) and (k+1, k+2) give the same single-qubit estimates for their shared qubit k+1 in every draw,
    as in the device protocol."""
    strings = L.local_pauli_strings(2, include_identity=False)
    on_second = [strings.index((p, "I")) for p in "XYZ"]     # Pauli on qubit k+1 of block (k, k+1)
    on_first = [strings.index(("I", p)) for p in "XYZ"]      # Pauli on qubit k of block (k, k+1)
    for S in draws:
        np.testing.assert_allclose(S[:, :-1][..., on_second], S[:, 1:][..., on_first], rtol=0, atol=1e-12)
