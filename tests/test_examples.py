"""examples/mitigate_own_data.py on a user file: reads X_train, Y_train and X_test and writes the mitigated
coordinates of the test circuits, one density matrix per block."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import qemlib as L

REPO = Path(__file__).resolve().parents[1]


def test_own_data_file(tmp_path):
    """Synthetic coordinates of random two-qubit states with added noise go in; physical block states come out."""
    rng = np.random.default_rng(0)
    P = L.pauli_ops(2)

    def coords(n_circ, n_blocks):
        G = rng.normal(size=(n_circ, n_blocks, 4, 4)) + 1j * rng.normal(size=(n_circ, n_blocks, 4, 4))
        R = G @ np.conj(np.swapaxes(G, -1, -2))
        R /= np.trace(R, axis1=-2, axis2=-1)[..., None, None]
        return np.real(np.einsum("...ij,kji->...k", R, P))

    Y_train, Y_test = coords(40, 2), coords(6, 2)
    noise = lambda Y: np.clip(0.9 * Y + 0.02 * rng.normal(size=Y.shape), -1, 1)
    f = tmp_path / "data.npz"
    np.savez(f, X_train=noise(Y_train), Y_train=Y_train, X_test=noise(Y_test))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, str(REPO / "examples" / "mitigate_own_data.py"), str(f)], env=env, check=True,
                   capture_output=True, text=True)
    out = np.load(tmp_path / "data_mitigated.npz")["Y_test_hat"]
    assert out.shape == Y_test.shape
    rho = (np.eye(4) + np.einsum("...k,kij->...ij", out.astype(complex), P)) / 4
    assert np.linalg.eigvalsh(rho).min() > -1e-6
