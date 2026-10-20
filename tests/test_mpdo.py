"""MPDO simulator (src/mpdo.py) against the density-matrix simulation of the data generator (gen_v3.py) on the
four-qubit chain of the default device: the block Pauli coordinates of uniform-TFIM circuits, with and without the
twirled gate noise and at fold scales 1, 3 and 5, agree to 1e-10."""
import numpy as np
import pytest

import gen_v3
import mpdo
import noise_layout as NL
import qemlib as L

N = 4
NUM = 2          # first two test circuits of every ensemble
TOL = 1e-10


@pytest.fixture(scope="module")
def devices():
    """Set up the Aer simulators of gen_v3.py (its pool initialiser) and the MPDO gate tables with and without the
    noise model of the noisy simulator."""
    gen_v3._init(N)
    return {"noisy": mpdo.Device(gen_v3._S["noisy"].options.noise_model), "ideal": mpdo.Device(None)}


def circuits(depth):
    return L.make_circuits("tfim", N, depth, NUM, "test")


def test_device_circuits_use_both_gate_orientations():
    """The device circuits contain two-qubit gates in both orientations along the chain, so both operator orderings
    of the MPDO two-qubit gates are exercised."""
    qc = NL.to_device(circuits(1)[0], N)
    pairs = {tuple(qc.find_bit(q).index for q in inst.qubits) for inst in qc.data if len(inst.qubits) == 2}
    assert any(a < b for a, b in pairs) and any(a > b for a, b in pairs)


@pytest.mark.parametrize("scale", [1, 3, 5])
@pytest.mark.parametrize("depth", [1, 3])
def test_noisy_block_coordinates(devices, depth, scale):
    """With the twirled device noise the MPDO coordinates of the folded circuit agree with the density-matrix
    simulation to 1e-10, the MPDO keeps unit trace and no weight is truncated."""
    for i, qc in enumerate(circuits(depth)):
        _, _, S_dm, _, _ = gen_v3._work((("test", i), scale, qc, N))
        A, _, discarded = mpdo.simulate(NL.to_device(L.fold_global(qc, scale), N), N, devices["noisy"])
        S_mp, norm = mpdo.block_coords(A)
        assert S_mp.shape == S_dm.shape == (N - 1, 15)
        np.testing.assert_allclose(S_mp, S_dm, rtol=0, atol=TOL)
        assert abs(norm - 1) < TOL
        assert discarded < 1e-20


@pytest.mark.parametrize("depth", [1, 3])
def test_ideal_block_coordinates(devices, depth):
    """Without noise the MPDO coordinates of the device circuit agree with the noiseless density-matrix simulation of
    the logical circuit to 1e-10."""
    for i, qc in enumerate(circuits(depth)):
        _, _, S_dm, _, _ = gen_v3._work((("test", i), "ideal", qc, N))
        A, _, _ = mpdo.simulate(NL.to_device(qc, N), N, devices["ideal"])
        S_mp, norm = mpdo.block_coords(A)
        np.testing.assert_allclose(S_mp, S_dm, rtol=0, atol=TOL)
        assert abs(norm - 1) < TOL
