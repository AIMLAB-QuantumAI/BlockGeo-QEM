import os

# Default thread counts of the linear-algebra libraries; values set by the user take precedence.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import importlib
import itertools
import math
import warnings
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import DensityMatrix, partial_trace


# ============================================================
# Utility
# ============================================================

EPS = 1e-10


def project_density_np(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(np.real(vals), 0.0, None)
    if vals.sum() < EPS:
        vals = np.ones_like(vals) / len(vals)
    else:
        vals = vals / vals.sum()
    return (vecs * vals) @ vecs.conj().T


# ============================================================
# Fake backend loader
# ============================================================

def load_fake_backend(name: str):
    """
    Tries to load IBM fake backend from available Qiskit providers.

    Examples:
        --backend manila
        --backend lima
        --backend melbourne
        --backend athens
        --backend yorktown
    """
    name = name.lower().replace("fake_", "")
    class_base = name.capitalize()

    candidate_modules = [
        "qiskit_ibm_runtime.fake_provider",
        "qiskit.providers.fake_provider",
    ]

    candidate_class_names = [
        f"Fake{class_base}V2",
        f"Fake{class_base}",
        f"Fake{name.upper()}",
    ]

    # Some common special cases
    special = {
        "melbourne": ["FakeMelbourneV2", "FakeMelbourne"],
        "lima": ["FakeLimaV2", "FakeLima"],
        "manila": ["FakeManilaV2", "FakeManila"],
        "athens": ["FakeAthensV2", "FakeAthens"],
        "yorktown": ["FakeYorktownV2", "FakeYorktown"],
        "belem": ["FakeBelemV2", "FakeBelem"],
        "quito": ["FakeQuitoV2", "FakeQuito"],
        "jakarta": ["FakeJakartaV2", "FakeJakarta"],
    }

    if name in special:
        candidate_class_names = special[name] + candidate_class_names

    errors = []
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            errors.append(f"{module_name}: {repr(e)}")
            continue

        for cls_name in candidate_class_names:
            if hasattr(module, cls_name):
                return getattr(module, cls_name)()

    raise ImportError(
        f"Could not load fake backend '{name}'. "
        f"Tried classes {candidate_class_names}. "
        f"Errors: {errors}"
    )


# ============================================================
# Circuit generation and simulation
# ============================================================

def random_circuit(n_qubits: int, depth: int, rng: np.random.Generator) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.h(q)

    for layer in range(depth):
        for q in range(n_qubits):
            theta = rng.uniform(-np.pi, np.pi)
            phi = rng.uniform(-np.pi, np.pi)
            qc.ry(theta, q)
            qc.rz(phi, q)

        # Alternating entangling pattern
        if layer % 2 == 0:
            pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
        else:
            pairs = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]

        for a, b in pairs:
            qc.cx(a, b)

    return qc


def fold_global(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Global unitary folding:
        scale 1: U
        scale 3: U U† U
        scale 5: U U† U U† U
    """
    if scale == 1:
        return circuit.copy()

    if scale % 2 == 0 or scale < 1:
        raise ValueError("Scale must be odd: 1, 3, 5, ...")

    folded = circuit.copy()
    inv = circuit.inverse()
    for _ in range((scale - 1) // 2):
        folded = folded.compose(inv)
        folded = folded.compose(circuit)
    return folded


def simulate_density_matrix(
    circuit: QuantumCircuit,
    simulator: AerSimulator,
    n_qubits: int,
    shots: int | None = None,
) -> np.ndarray:
    qc = circuit.copy()

    tqc = transpile(
        qc,
        simulator,
        optimization_level=0,
        num_processes=1,
    )

    tqc.save_density_matrix()

    result = simulator.run(tqc, shots=shots).result()
    rho = np.asarray(result.data(0)["density_matrix"], dtype=np.complex128)
    rho = project_density_np(rho)

    dim = rho.shape[0]
    total_qubits = int(round(np.log2(dim)))

    if total_qubits > n_qubits:
        traced_out = list(range(n_qubits, total_qubits))
        rho = partial_trace(DensityMatrix(rho), traced_out).data
        rho = np.asarray(rho, dtype=np.complex128)
        rho = project_density_np(rho)

    elif total_qubits < n_qubits:
        raise ValueError(
            f"Returned density matrix has only {total_qubits} qubits, "
            f"but expected at least {n_qubits}."
        )

    return rho


def pauli_matrices_np():
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def local_pauli_strings(block_size: int, include_identity: bool = False):
    labels = ["I", "X", "Y", "Z"]
    strings = list(itertools.product(labels, repeat=block_size))
    if not include_identity:
        strings = [s for s in strings if not all(x == "I" for x in s)]
    return strings


def kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def local_pauli_ops_np(block_size: int, include_identity: bool = False):
    pm = pauli_matrices_np()
    strings = local_pauli_strings(block_size, include_identity=include_identity)
    ops = []
    for s in strings:
        ops.append(kron_all([pm[x] for x in s]))
    return strings, ops


# ============================================================
# ZNE
# ============================================================

# ============================================================
# Finite-shot measurement
# ============================================================

BASIS_LABELS = ("X", "Y", "Z")


def basis_change_unitaries():
    """
    Single-qubit rotations that map an X/Y/Z measurement onto a
    computational-basis measurement:  <P> is read off the Z-parity after
    applying u(P).  u(Z)=I, u(X)=H, u(Y)=H S^dagger.
    """
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    return {"Z": np.eye(2, dtype=np.complex128), "X": H, "Y": H @ Sdg}


def measurement_settings(n_qubits: int, block_size: int) -> list[tuple[str, ...]]:
    """
    3^b global basis assignments covering every window of b consecutive qubits,
    via  f(i, a) = sum_j a_j i^j  (mod 3)  for a in {0,1,2}^b.
    Valid for b <= 3; the count is independent of n, and the same settings serve
    overlapping and non-overlapping blockings alike.
    """
    if block_size > 3:
        raise ValueError("The polynomial covering construction requires b <= 3.")
    settings = []
    for a in itertools.product(range(3), repeat=block_size):
        settings.append(tuple(
            BASIS_LABELS[sum(a[j] * (i ** j) for j in range(block_size)) % 3]
            for i in range(n_qubits)
        ))
    return settings


def rotate_density(rho: np.ndarray, n_qubits: int, us: list[np.ndarray]) -> np.ndarray:
    """
    U rho U^dagger for U = tensor of single-qubit unitaries, applied by tensor
    contraction rather than by forming the 2^n x 2^n kronecker product.
    Qiskit is little-endian: qubit q occupies tensor axis n-1-q.
    """
    t = rho.reshape([2] * n_qubits + [2] * n_qubits)
    for q in range(n_qubits):
        u = us[q]
        ax = n_qubits - 1 - q                      # row index of qubit q
        t = np.moveaxis(t, ax, 0)
        t = np.tensordot(u, t, axes=([1], [0]))
        t = np.moveaxis(t, 0, ax)
        axc = 2 * n_qubits - 1 - q                 # column index of qubit q
        t = np.moveaxis(t, axc, 0)
        t = np.tensordot(u.conj(), t, axes=([1], [0]))
        t = np.moveaxis(t, 0, axc)
    return t.reshape(2 ** n_qubits, 2 ** n_qubits)


def _parity_signs(n_qubits: int, qubits: tuple[int, ...]) -> np.ndarray:
    xs = np.arange(2 ** n_qubits)
    bits = ((xs[:, None] >> np.array(qubits)[None, :]) & 1).sum(axis=1) % 2
    return np.where(bits == 0, 1.0, -1.0)


@dataclass
class DatasetPack:
    noisy_p: np.ndarray
    ideal_p: np.ndarray
    noisy_obs: np.ndarray
    ideal_obs: np.ndarray
    zne_p: np.ndarray | None = None
    zne_obs: np.ndarray | None = None

    # Timing fields. These are accumulated over all circuits in this pack.
    # For fair QEM comparison, ideal_sim_time_sec is reported separately
    # because it is only used to obtain labels/ground truth in simulation.
    ideal_sim_time_sec: float = 0.0
    noisy_sim_time_sec: float = 0.0
    zne_sim_time_sec: float = 0.0
    wall_time_sec: float = 0.0


# ============================================================
# Torch Bures loss
# ============================================================

def torch_pauli_ops(block_size: int, device):
    I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    pm = {"I": I, "X": X, "Y": Y, "Z": Z}

    strings = local_pauli_strings(block_size, include_identity=False)
    ops = []
    for s in strings:
        out = pm[s[0]]
        for label in s[1:]:
            out = torch.kron(out, pm[label])
        ops.append(out)
    return torch.stack(ops, dim=0)


def project_density_torch(rho: torch.Tensor) -> torch.Tensor:
    # rho: [B, dim, dim], complex
    rho = 0.5 * (rho + rho.conj().transpose(-1, -2))
    vals, vecs = torch.linalg.eigh(rho)
    vals = torch.clamp(vals.real, min=1e-6)
    vals = vals / vals.sum(dim=-1, keepdim=True)
    rho_proj = vecs @ torch.diag_embed(vals).to(torch.complex64) @ vecs.conj().transpose(-1, -2)
    return rho_proj


def sqrtm_psd_torch(mat: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Batched matrix square root for PSD matrices.

    Do not normalize intermediate fidelity matrices.
    """
    mat = 0.5 * (mat + mat.conj().transpose(-1, -2))

    if normalize:
        mat = project_density_torch(mat)

    vals, vecs = torch.linalg.eigh(mat)
    vals = torch.clamp(vals.real, min=1e-8)
    sqrt_vals = torch.sqrt(vals)

    return vecs @ torch.diag_embed(sqrt_vals).to(torch.complex64) @ vecs.conj().transpose(-1, -2)


def obs_to_density_torch(obs: torch.Tensor, block_size: int, device) -> torch.Tensor:
    """
    obs: [batch, 4^b - 1]
    returns rho: [batch, 2^b, 2^b]
    """
    batch = obs.shape[0]
    dim = 2 ** block_size
    ops = torch_pauli_ops(block_size, device=device)  # [num_obs, dim, dim]

    eye = torch.eye(dim, dtype=torch.complex64, device=device).unsqueeze(0).repeat(batch, 1, 1)
    rho = eye.clone()

    obs_c = obs.to(torch.complex64)
    rho = rho + torch.einsum("bo,oij->bij", obs_c, ops)
    rho = rho / dim
    rho = project_density_torch(rho)
    return rho


def bures_loss_torch_from_obs(
    pred_obs_flat: torch.Tensor,
    target_obs_flat: torch.Tensor,
    blocks: list[tuple[int, ...]],
    device,
) -> torch.Tensor:
    idx = 0
    losses = []

    for block in blocks:
        b = len(block)
        num_obs = 4 ** b - 1

        pred_obs = pred_obs_flat[:, idx: idx + num_obs]
        target_obs = target_obs_flat[:, idx: idx + num_obs]
        idx += num_obs

        rho_pred = obs_to_density_torch(pred_obs, b, device)
        rho_target = obs_to_density_torch(target_obs, b, device)

        sqrt_target = sqrtm_psd_torch(rho_target, normalize=False)
        middle = sqrt_target @ rho_pred @ sqrt_target
        middle = 0.5 * (middle + middle.conj().transpose(-1, -2))
        sqrt_middle = sqrtm_psd_torch(middle, normalize=False)
        fid = torch.real(torch.diagonal(sqrt_middle, dim1=-2, dim2=-1).sum(dim=-1)) ** 2
        fid = torch.clamp(fid, min=0.0, max=1.0)

        d_bures_sq = 2.0 * (1.0 - torch.sqrt(torch.clamp(fid, min=1e-8)))
        losses.append(d_bures_sq)

    return torch.stack(losses, dim=0).mean()


def prob_fid_loss_torch(pred_p: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    pred_p = torch.clamp(pred_p, min=1e-8)
    target_p = torch.clamp(target_p, min=1e-8)
    bc = torch.sum(torch.sqrt(pred_p * target_p), dim=-1)
    fid = bc ** 2
    return torch.mean(1.0 - fid)


# ============================================================
# Neural QEM model
# ============================================================

# Internal variant name -> name used in the result CSVs.
#
# The ablation ladder is:
#   mse       : L_mse                                   (probability only)
#   mse_obs   : L_mse + L_obs                            (+ Euclidean observables)
#   probfid   : L_mse + L_fid + L_obs                    (+ probability fidelity)
#   blockgeo  : L_mse + L_fid + L_obs + L_block          (+ Bures geometry)
#
# blockgeo minus probfid isolates the block-wise Bures term exactly.
VARIANTS = {
    "mse": "MSE-NN-QEM",
    "mse_obs": "MSE+Obs-QEM",
    "probfid": "BlockGeo-QEM w/o L_block",
    "blockgeo": "BlockGeo-QEM",
}


class QEMNet(nn.Module):
    def __init__(self, input_dim: int, prob_dim: int, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.prob_dim = prob_dim
        self.obs_dim = obs_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.prob_delta_head = nn.Linear(hidden, prob_dim)
        self.obs_delta_head = nn.Linear(hidden, obs_dim)

        # Start from the Noisy baseline.
        nn.init.zeros_(self.prob_delta_head.weight)
        nn.init.zeros_(self.prob_delta_head.bias)
        nn.init.zeros_(self.obs_delta_head.weight)
        nn.init.zeros_(self.obs_delta_head.bias)

    def forward(self, x):
        noisy_p = x[:, :self.prob_dim]
        noisy_obs = x[:, self.prob_dim:self.prob_dim + self.obs_dim]

        h = self.net(x)

        delta_p = 0.1 * torch.tanh(self.prob_delta_head(h))
        delta_obs = 0.1 * torch.tanh(self.obs_delta_head(h))

        corrected_p = noisy_p + delta_p
        corrected_p = torch.clamp(corrected_p, min=1e-8)
        pred_p = corrected_p / corrected_p.sum(dim=-1, keepdim=True)

        pred_obs = noisy_obs + delta_obs
        pred_obs = torch.clamp(pred_obs, min=-1.0, max=1.0)

        return pred_p, pred_obs


def train_model(
    train_pack: DatasetPack,
    blocks: list[tuple[int, ...]],
    variant: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    lambda_mse: float,
    lambda_fid: float,
    lambda_block: float,
    lambda_obs: float,
    device: str,
):
    noisy_p = train_pack.noisy_p
    ideal_p = train_pack.ideal_p
    noisy_obs = train_pack.noisy_obs
    ideal_obs = train_pack.ideal_obs

    x = np.concatenate([noisy_p, noisy_obs], axis=1).astype(np.float32)
    y_p = ideal_p.astype(np.float32)
    y_obs = ideal_obs.astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(x),
        torch.tensor(y_p),
        torch.tensor(y_obs),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = x.shape[1]
    prob_dim = y_p.shape[1]
    obs_dim = y_obs.shape[1]

    model = QEMNet(input_dim, prob_dim, obs_dim, hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    variant = variant.lower()

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {sorted(VARIANTS)}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        # Per-term accumulators: lets us see whether each loss term is actually
        # decreasing, and how the terms compare in magnitude. Note that the
        # probability MSE is averaged over 2^n bins and is therefore ~1e-6,
        # several orders of magnitude below the observable/fidelity terms.
        comp = {"mse": 0.0, "fid": 0.0, "obs": 0.0, "block": 0.0}

        for xb, pb, ob in loader:
            xb = xb.to(device)
            pb = pb.to(device)
            ob = ob.to(device)

            pred_p, pred_obs = model(xb)

            loss_mse = F.mse_loss(pred_p, pb)
            loss_obs = F.mse_loss(pred_obs, ob)
            loss_fid = None
            loss_block = None

            if variant == "mse":
                # MSE-NN-QEM: probability MSE ONLY, as in earlier learned
                # mitigators that regress output probabilities.
                # The observable head receives no gradient and is zero-init,
                # so pred_obs == noisy_obs by construction. Its Local Bures is
                # therefore identical to the Noisy baseline -- which is exactly
                # the point: probability-space QEM cannot correct local geometry.
                loss = lambda_mse * loss_mse

            elif variant == "mse_obs":
                # Ablation: probability MSE + Euclidean observable supervision.
                # No probability-fidelity term, no Bures term.
                loss = lambda_mse * loss_mse + lambda_obs * loss_obs

            elif variant == "probfid":
                # BlockGeo-QEM w/o L_block. This is the correct control for
                # isolating the block-wise Bures term: it differs from the
                # blockgeo variant by exactly lambda_block * L_block.
                loss_fid = prob_fid_loss_torch(pred_p, pb)
                loss = (
                    lambda_mse * loss_mse
                    + lambda_fid * loss_fid
                    + lambda_obs * loss_obs
                )

            elif variant == "blockgeo":
                loss_fid = prob_fid_loss_torch(pred_p, pb)
                loss_block = bures_loss_torch_from_obs(pred_obs, ob, blocks, device)
                loss = (
                    lambda_mse * loss_mse
                    + lambda_fid * loss_fid
                    + lambda_obs * loss_obs
                    + lambda_block * loss_block
                )

            else:
                raise ValueError(f"Unknown variant: {variant}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = xb.shape[0]
            total_loss += float(loss.item()) * bs
            comp["mse"] += float(loss_mse.item()) * bs
            comp["obs"] += float(loss_obs.item()) * bs
            if loss_fid is not None:
                comp["fid"] += float(loss_fid.item()) * bs
            if loss_block is not None:
                comp["block"] += float(loss_block.item()) * bs

        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            nd = len(dataset)
            print(
                f"[{variant}] ep{epoch:03d}/{epochs} "
                f"total={total_loss / nd:.3e} | "
                f"mse={comp['mse'] / nd:.3e} "
                f"fid={comp['fid'] / nd:.3e} "
                f"obs={comp['obs'] / nd:.3e} "
                f"block={comp['block'] / nd:.3e}"
            )

    return model


def predict_model(model: QEMNet, pack: DatasetPack, device: str):
    x = np.concatenate([pack.noisy_p, pack.noisy_obs], axis=1).astype(np.float32)
    xb = torch.tensor(x).to(device)

    model.eval()
    preds_p = []
    preds_obs = []

    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(x), batch_size):
            pred_p, pred_obs = model(xb[start:start + batch_size])
            preds_p.append(pred_p.cpu().numpy())
            preds_obs.append(pred_obs.cpu().numpy())

    return np.concatenate(preds_p, axis=0), np.concatenate(preds_obs, axis=0)
