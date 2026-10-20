"""Apply BlockGeo-QEM to your own block coordinates.

BlockGeo-QEM learns from training circuits for which you have the measured (noisy) Pauli coordinates of every
two-qubit block and the ideal ones, and then mitigates the measured coordinates of new circuits. All arrays have the
shape (circuits, blocks, 15), where block k holds qubits k and k+1 of the chain and the 15 coordinates are the
expectation values of the Pauli strings in the order of qemlib.local_pauli_strings(2) (IX, IY, IZ, XI, XX, ...; the
right letter acts on qubit k).

    python examples/mitigate_own_data.py              # builds a small example data set with the simulator
    python examples/mitigate_own_data.py data.npz     # your arrays X_train, Y_train, X_test in one .npz file

With your own file, the mitigated coordinates of the test circuits are written to <file>_mitigated.npz (key
Y_test_hat). The example data set takes about a minute on a CPU.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # the CPU is enough for small problems

import numpy as np

import qemlib as L
from v3 import BlockGeoQEM2


def example_data(n=4, steps=1, n_train=100, n_test=20, shots=10000, seed=0):
    """Uniform-TFIM circuits simulated with the device noise model of this repository and measured with `shots`
    shots per circuit in the nine covering settings, followed by readout-error mitigation."""
    from qiskit_aer import AerSimulator

    import measure_v3 as M
    import noise_layout as NL

    ideal = AerSimulator(method="density_matrix")
    noisy = AerSimulator(method="density_matrix", noise_model=NL.twirled_noise_model(n), basis_gates=NL.BASIS)
    rng = np.random.default_rng(seed)

    def simulate(circuits):
        S = np.stack([L.block_paulis(L.simulate_density_matrix(qc, ideal, n), n, 2) for qc in circuits])
        Q = np.stack([L.setting_probs(L.simulate_density_matrix(NL.to_device(qc, n), noisy, n), n, 2)
                      for qc in circuits])
        X, _ = M.measure(Q, n, shots // 9, rng)
        return X, S

    X_train, Y_train = simulate(L.make_circuits("tfim", n, steps, n_train, "train"))
    X_test, Y_test = simulate(L.make_circuits("tfim", n, steps, n_test, "test"))
    return X_train, Y_train, X_test, Y_test


def main():
    if len(sys.argv) > 1:
        z = np.load(sys.argv[1])
        X_train, Y_train, X_test, Y_test = z["X_train"], z["Y_train"], z["X_test"], None
    else:
        X_train, Y_train, X_test, Y_test = example_data()
    print(f"training circuits {X_train.shape[0]}, test circuits {X_test.shape[0]}, blocks {X_train.shape[1]}")

    # The model takes dictionaries {noise scale: coordinates}; with measurements at the native noise level only,
    # the key is 1. Training uses 20% of the training circuits for validation and falls back to the linear backbone
    # when the network does not improve on it.
    model = BlockGeoQEM2(beta=1.0).fit({1: X_train}, Y_train)
    Y_test_hat = model.predict({1: X_test})        # (circuits, blocks, 15), every block a density matrix

    if Y_test is None:
        out = os.path.splitext(sys.argv[1])[0] + "_mitigated.npz"
        np.savez(out, Y_test_hat=Y_test_hat)
        print("written", out)
        return
    for name, S in (("unmitigated", X_test), ("BlockGeo-QEM", Y_test_hat)):
        print(f"{name:13s} mean local Bures distance {L.local_bures(S, Y_test, 2).mean():.4f}, "
              f"trace distance {L.local_trace(S, Y_test, 2).mean():.4f}")


if __name__ == "__main__":
    main()
