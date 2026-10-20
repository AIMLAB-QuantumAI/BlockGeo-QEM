"""Device layout for the corrected noise simulation.

The workshop pipeline transpiled CX(i, i+1) against a simulator without a coupling map, so every
ECR was emitted as ECR(i, i+1). FakeSherbrooke defines most ECR errors only for the reverse
direction, so those gates ran without two-qubit noise, and the chain 0..9 also contains couplers
calibrated as broken (error 1.0). Here the circuits run on a chain of good qubits chosen on the
heavy-hex coupling graph, the noise model is re-indexed to logical qubits 0..n-1, and circuits are
transpiled with the directed coupling map so each ECR carries its calibrated error.
"""
import os

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap

import qemlib as L

# Device: FakeSherbrooke by default. BLOCKGEO_BACKEND, BLOCKGEO_TWOQ (native two-qubit gate) and BLOCKGEO_CHAIN
# (comma-separated physical qubits) select another device, e.g. torino / cz for the second-device study (find_chain.py
# gives its chain).
BACKEND = os.environ.get("BLOCKGEO_BACKEND", "sherbrooke")
TWOQ = os.environ.get("BLOCKGEO_TWOQ", "ecr")
# Default 10-qubit path of FakeSherbrooke (ECR errors 0.41-0.80%); n < 10 uses its prefix.
CHAIN = ([int(q) for q in os.environ["BLOCKGEO_CHAIN"].split(",")] if os.environ.get("BLOCKGEO_CHAIN")
         else [33, 39, 40, 41, 53, 60, 59, 58, 71, 77])
BASIS = [TWOQ, "id", "rz", "sx", "x"]


def chain_noise_model(n, backend=BACKEND):
    from qiskit_aer.noise import NoiseModel

    full = NoiseModel.from_backend(L.load_fake_backend(backend))
    phys = CHAIN[:n]
    idx = {p: i for i, p in enumerate(phys)}
    nm = NoiseModel(basis_gates=full.basis_gates)
    for gate, errs in full._local_quantum_errors.items():
        for qubits, err in errs.items():
            if all(q in idx for q in qubits):
                nm.add_quantum_error(err, gate, [idx[q] for q in qubits])
    for qubits, err in full._local_readout_errors.items():
        if all(q in idx for q in qubits):
            nm.add_readout_error(err, [idx[q] for q in qubits])
    return nm


def chain_coupling(n, backend=BACKEND):
    """Directed logical edges (i, j) for which the device has a calibrated two-qubit gate on (phys_i, phys_j)."""
    tgt = L.load_fake_backend(backend).target
    phys = CHAIN[:n]
    edges = []
    for i in range(n - 1):
        a, b = phys[i], phys[i + 1]
        if (a, b) in tgt[TWOQ]:
            edges.append((i, i + 1))
        elif (b, a) in tgt[TWOQ]:
            edges.append((i + 1, i))
        else:
            raise ValueError(f"no {TWOQ} between physical qubits {a} and {b}")
    return edges


_CMAP = {}


def to_device(qc, n):
    """Transpile a logical nearest-neighbour circuit to the native gates with calibrated two-qubit directions.
    optimization_level=0 keeps folded U U^dag U intact for ZNE."""
    if n not in _CMAP:
        _CMAP[n] = CouplingMap(chain_coupling(n))
    return transpile(qc, basis_gates=BASIS, coupling_map=_CMAP[n], initial_layout=list(range(n)),
                     optimization_level=0)


# ----------------------------- Pauli twirling, readout, PEC -----------------------------

def _paulis(k):
    import itertools

    from qiskit.quantum_info import Pauli

    labels = ["".join(t) for t in itertools.product("IXYZ", repeat=k)]
    return labels, [Pauli(l) for l in labels]


def _commute_signs(paulis):
    return np.array([[1.0 if p.commutes(q) else -1.0 for q in paulis] for p in paulis])


def pauli_fidelities(err):
    """lambda_P = Tr(P E(P)) / 2^k for every k-qubit Pauli P (the diagonal of the Pauli transfer matrix)."""
    from qiskit.quantum_info import SuperOp

    S = SuperOp(err).data
    k = err.num_qubits
    d = 2 ** k
    labels, paulis = _paulis(k)
    lam = []
    for p in paulis:
        M = p.to_matrix()
        EM = (S @ M.flatten(order="F")).reshape(d, d, order="F")
        lam.append(np.real(np.trace(M @ EM)) / d)
    return labels, paulis, np.array(lam)


def twirl_error(err):
    """Pauli twirl of a quantum error: the Pauli channel with the same PTM diagonal (non-unital and
    off-diagonal parts removed). p_Q = 4^-k sum_P chi(P,Q) lambda_P with chi = +-1 for (anti)commuting."""
    from qiskit_aer.noise import pauli_error

    labels, paulis, lam = pauli_fidelities(err)
    p = _commute_signs(paulis) @ lam / len(labels)
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    return pauli_error([(l, float(x)) for l, x in zip(labels, p) if x > 1e-14])


def pec_gamma(err):
    """Sampling overhead of the exact inverse of the twirled channel: sum_Q |q_Q| with
    q_Q = 4^-k sum_P chi(P,Q) / lambda_P."""
    labels, paulis, lam = pauli_fidelities(err)
    q = _commute_signs(paulis) @ (1.0 / lam) / len(labels)
    return float(np.abs(q).sum())


def scale_pauli_error(err, f):
    """Multiply every non-identity probability of a Pauli channel by f (drifted calibration)."""
    from qiskit_aer.noise import pauli_error

    labels, paulis, lam = pauli_fidelities(err)
    p = np.clip(_commute_signs(paulis) @ lam / len(labels), 0.0, None)
    p /= p.sum()
    ident = "I" * err.num_qubits
    p = np.array([x if l == ident else f * x for l, x in zip(labels, p)])
    p[labels.index(ident)] = 1.0 - (p.sum() - p[labels.index(ident)])
    return pauli_error([(l, float(x)) for l, x in zip(labels, p) if x > 1e-14])


def twirled_noise_model(n, backend=BACKEND, scale=1.0):
    """Chain noise model with every gate error replaced by its Pauli twirl; readout errors kept.
    scale != 1 multiplies all gate error probabilities (calibration drift)."""
    from qiskit_aer.noise import NoiseModel

    base = chain_noise_model(n, backend)
    nm = NoiseModel(basis_gates=base.basis_gates)
    for gate, errs in base._local_quantum_errors.items():
        for qubits, err in errs.items():
            tw = twirl_error(err)
            nm.add_quantum_error(tw if scale == 1.0 else scale_pauli_error(tw, scale), gate, list(qubits))
    for qubits, err in base._local_readout_errors.items():
        nm.add_readout_error(err, list(qubits))
    return nm


def readout_confusion(n, backend=BACKEND):
    """Per-qubit column-stochastic confusion matrices A_q[m, t] = P(measured m | true t), logical order."""
    nm = chain_noise_model(n, backend)
    return np.stack([np.asarray(nm._local_readout_errors[(q,)].probabilities).T for q in range(n)])


_GAMMA = {}


def gate_gammas(n, backend=BACKEND, scale=1.0):
    if (n, backend, scale) not in _GAMMA:
        nm = twirled_noise_model(n, backend, scale)
        _GAMMA[(n, backend, scale)] = {(g, tuple(q)): pec_gamma(e) for g, errs in nm._local_quantum_errors.items()
                                       for q, e in errs.items()}
    return _GAMMA[(n, backend, scale)]


def circuit_gamma(device_qc, n, backend=BACKEND, scale=1.0):
    """Product of per-gate PEC overheads over a device-level circuit (unfolded)."""
    G = gate_gammas(n, backend, scale)
    g = 1.0
    for inst in device_qc.data:
        key = (inst.operation.name, tuple(device_qc.find_bit(q).index for q in inst.qubits))
        g *= G.get(key, 1.0)
    return g
