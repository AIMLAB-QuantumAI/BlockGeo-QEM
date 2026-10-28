"""BlockGeo-QEM: learned mitigation of 2-qubit block states, trained in the Bures geometry.

For each overlapping 2-qubit block k of an n-qubit chain, the measured Pauli coordinates x_k
(9 covering settings; optionally at several fold scales) are mapped to a block density matrix.

  1. Backbone      s_lin,k = per-coordinate ridge of the noisy coordinate(s)   (CDR-like)
  2. Correction    f_theta(x_k, x_{k-1}, x_{k+1}, s_lin,k, pos_k, c)  with a circuit context
                   c = mean_k phi(x_k, pos_k)  (DeepSets pooling over all blocks)
  3. Physical output   rho_k = A A^dag / Tr(A A^dag),  A = chol(Pi(s_lin,k)) + f_theta(...)
  4. Loss          D_B^2(rho_k, rho_ideal,k) + beta * (1/4) ||s(rho_k) - s_ideal,k||^2

Pi is the nearest-state projection. The number of training epochs is chosen by the loss on a held-out
split, and the method returns the projected backbone when no epoch lowers that loss. A finite tolerance
tol (argument or BLOCKGEO_TOL) adds a stricter check: epochs whose validation coordinate error exceeds
(1 + tol) times that of the backbone are then not eligible. The default tol = inf disables it.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV

import geometry as GEO
import qemlib as L

# Optional cap on the fraction of GPU memory one process may allocate (e.g. GPU_FRAC=0.2 on a shared GPU).
if torch.cuda.is_available() and os.environ.get("GPU_FRAC"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_FRAC"]))

BLOCK = 2                                    # qubits per block
NS, DIM = 4 ** BLOCK - 1, 2 ** BLOCK         # 15 Pauli coordinates per block, block dimension 4


# ----------------------------- numpy helpers -----------------------------

def stack_scales(S, scales):
    return np.concatenate([S[k] for k in scales], axis=-1)  # [N, nb, 15*K]


def coords_to_rho(S):
    flat = S.reshape(-1, S.shape[-1])
    return np.stack([L.obs_to_rho(s, BLOCK) for s in flat]).reshape(*S.shape[:-1], DIM, DIM)


def sqrtm_psd(R):
    w, V = np.linalg.eigh(R)
    return (V * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.swapaxes(V.conj(), -1, -2)


def positions(nb, mode):
    """Block position ids: 'index' (0..nb-1) or 'boundary' (0 left edge, 1 bulk, 2 right edge)."""
    if mode == "index":
        return np.arange(nb)
    p = np.ones(nb, dtype=int)
    p[0], p[-1] = 0, 2
    return p


# ----------------------------- backbone -----------------------------

class Backbone:
    """Per-coordinate ridge regression of the ideal coordinate on its noisy value(s); one model per
    position id, pooling the blocks that share an id."""

    def __init__(self, pos_mode="index", alphas=np.logspace(-8, 0, 17)):
        self.pos_mode, self.alphas = pos_mode, alphas

    def fit(self, X, Y, K):
        self.K = K
        pos = positions(X.shape[1], self.pos_mode)
        self.models = {}
        for p in np.unique(pos):
            ks = np.where(pos == p)[0]
            Xp, Yp = X[:, ks].reshape(-1, X.shape[2]), Y[:, ks].reshape(-1, NS)
            self.models[p] = [RidgeCV(alphas=self.alphas).fit(Xp[:, j::NS][:, :K], Yp[:, j]) for j in range(NS)]
        return self

    def predict(self, X):
        pos = positions(X.shape[1], self.pos_mode)
        out = np.empty(X.shape[:2] + (NS,))
        for k in range(X.shape[1]):
            ms = self.models[pos[k]]
            out[:, k] = np.stack([m.predict(X[:, k, j::NS][:, :self.K]) for j, m in enumerate(ms)], -1)
        return out


# ----------------------------- network -----------------------------

def bures_sq_t(sigma, sqrt_rho):
    M = sqrt_rho @ sigma @ sqrt_rho
    M = 0.5 * (M + M.conj().transpose(-1, -2))
    try:
        mu = torch.linalg.eigvalsh(M)
    except (torch.linalg.LinAlgError, torch.OutOfMemoryError):
        # cuSOLVER can fail on exactly degenerate spectra (e.g. pure stabilizer targets) or reserve too much
        # workspace for large batches; LAPACK on the CPU handles both
        mu = torch.linalg.eigvalsh(M.cpu()).to(M.device)
    return 2.0 * (1.0 - mu.clamp_min(1e-10).sqrt().sum(-1))


def mlp(d_in, hidden, d_out, depth):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class BlockNet(nn.Module):
    def __init__(self, n_pos, f_in, out_dim, hidden=128, depth=2, emb=8, ctx=16, neighbors=True):
        super().__init__()
        self.neighbors, self.ctx = neighbors, ctx
        self.emb = nn.Embedding(n_pos, emb)
        if ctx:
            self.phi = mlp(f_in + emb, hidden, ctx, 1)
        d_in = f_in * (3 if neighbors else 1) + NS + emb + ctx
        self.f = mlp(d_in, hidden, out_dim, depth)
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, x, base, pos):
        N, nb, _ = x.shape
        e = self.emb(pos)[None].expand(N, -1, -1)
        parts = [x]
        if self.neighbors:
            z = torch.zeros_like(x[:, :1])
            parts += [torch.cat([z, x[:, :-1]], 1), torch.cat([x[:, 1:], z], 1)]
        parts += [base, e]
        if self.ctx:
            c = self.phi(torch.cat([x, e], -1)).mean(1, keepdim=True).expand(-1, nb, -1)
            parts.append(c)
        return self.f(torch.cat(parts, -1))


# ----------------------------- model -----------------------------

class BlockGeoQEM:
    def __init__(self, scales=(1,), output="chol", loss="bures", beta=0.3, hidden=128, depth=2, ctx=16,
                 neighbors=True, pos_mode="index", lr=1e-3, wd=1e-4, max_epochs=4000, patience=400,
                 val_frac=0.2, seeds=(0, 1, 2), tol=None, device=None):
        self.__dict__.update(locals())
        self.__dict__.pop("self")
        if tol is None:                      # coordinate-error tolerance of the validation check; inf = loss alone
            self.tol = float(os.environ.get("BLOCKGEO_TOL", "inf"))
        self.ns, self.D = NS, DIM                                   # coordinates per block, block dimension
        self._tril = torch.tril_indices(self.D, self.D, offset=-1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.P = torch.tensor(L.pauli_ops(BLOCK), dtype=torch.complex128, device=self.device)

    # tensors
    def _T(self, X, B, Y=None):
        dev = self.device
        f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        c128 = lambda a: torch.tensor(a, dtype=torch.complex128, device=dev)
        T = {"x": f32((X - self.mu) / self.sd), "base": f32(B),
             "pos": torch.as_tensor(positions(X.shape[1], self.pos_mode), device=dev)}
        if self.output == "chol":
            T["A0"] = c128(np.linalg.cholesky(coords_to_rho(B) + 1e-6 * np.eye(self.D)))
        if Y is not None:
            T["Y"] = f32(Y)
            if self.output == "chol":
                R = coords_to_rho(Y)
                T["R"], T["sqrtR"] = c128(R), c128(sqrtm_psd(R))
            else:
                G = GEO.metric_tensors(Y) if self.loss == "bures" else np.broadcast_to(
                    np.eye(self.ns) / 4, Y.shape + (self.ns,)).copy()
                T["G"] = f32(G)
        return T

    def _assemble(self, delta, T):
        if self.output == "coords":
            return T["base"] + delta
        d = delta.to(torch.float64)
        A = T["A0"].clone()
        D, M = self.D, self.D * (self.D - 1) // 2
        i = torch.arange(D, device=A.device)
        A[..., i, i] = A[..., i, i] + d[..., :D].to(A.dtype)
        tr = self._tril.to(A.device)
        A[..., tr[0], tr[1]] = A[..., tr[0], tr[1]] + torch.complex(d[..., D:D + M], d[..., D + M:D + 2 * M])
        R = A @ A.conj().transpose(-1, -2)
        return R / torch.diagonal(R, dim1=-2, dim2=-1).real.sum(-1)[..., None, None]

    def _coords(self, out):
        if self.output == "coords":
            return out
        return torch.einsum("nbij,kji->nbk", out, self.P).real.float()

    def _loss(self, out, T):
        if self.output == "coords":
            r = out - T["Y"]
            main = torch.einsum("nbi,nbij,nbj->nb", r, T["G"], r).mean()
        elif self.loss == "hs":
            main = ((out - T["R"]).abs() ** 2).sum((-1, -2)).mean()
        else:
            main = bures_sq_t(out, T["sqrtR"]).mean()
        mse = ((self._coords(out) - T["Y"]) ** 2).sum(-1).mean()
        return main + self.beta * 0.25 * mse, mse

    def _net(self, T, seed):
        torch.manual_seed(seed)
        n_pos = 3 if self.pos_mode == "boundary" else T["x"].shape[1]
        return BlockNet(n_pos, T["x"].shape[2], self.ns if self.output == "coords" else self.D ** 2, self.hidden,
                        self.depth, ctx=self.ctx, neighbors=self.neighbors).to(self.device)

    def _forward(self, net, T):
        return self._assemble(net(T["x"], T["base"], T["pos"]), T)

    def _train(self, T, V=None, epochs=None, seed=0):
        net = self._net(T, seed)
        if epochs == 0:
            return net, -1
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        n_ep = epochs or self.max_epochs
        if V is not None:
            with torch.no_grad():
                best, mse0 = [float(v) for v in self._loss(self._forward(net, V), V)]
            best_ep, best_state, last = -1, None, 0
        for ep in range(n_ep):
            loss, _ = self._loss(self._forward(net, T), T)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if V is not None and (ep % 10 == 0 or ep == n_ep - 1):
                with torch.no_grad():
                    v, mse = [float(u) for u in self._loss(self._forward(net, V), V)]
                if v < best - 1e-12:
                    last = ep
                    if self.tol == float("inf") or mse <= mse0 * (1 + self.tol):
                        best, best_ep = v, ep
                        best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
                elif ep - last > self.patience:
                    break
        if V is None:
            return net, n_ep - 1
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_ep

    def fit(self, S_train, Y_train):
        X = stack_scales(S_train, self.scales)
        K = len(self.scales)
        self.backbone = Backbone(self.pos_mode).fit(X, Y_train, K)
        self.nets, self.best_eps = [], []
        for seed in self.seeds:
            perm = np.random.default_rng(seed).permutation(len(X))
            nva = int(round(self.val_frac * len(X)))
            va, tr = perm[:nva], perm[nva:]
            bb = Backbone(self.pos_mode).fit(X[tr], Y_train[tr], K)
            self.mu, self.sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-8
            _, ep = self._train(self._T(X[tr], bb.predict(X[tr]), Y_train[tr]),
                                self._T(X[va], bb.predict(X[va]), Y_train[va]), seed=seed)
            self.best_eps.append(ep)
        self.mu, self.sd = X.mean((0, 1)), X.std((0, 1)) + 1e-8
        T = self._T(X, self.backbone.predict(X), Y_train)
        for seed, ep in zip(self.seeds, self.best_eps):
            self.nets.append(self._train(T, epochs=ep + 1, seed=seed)[0])
        return self

    def predict(self, S):
        X = stack_scales(S, self.scales)
        T = self._T(X, self.backbone.predict(X))
        with torch.no_grad():
            delta = torch.stack([net(T["x"], T["base"], T["pos"]) for net in self.nets]).mean(0)
            out = self._assemble(delta, T)
        return np.clip(self._coords(out).double().cpu().numpy(), -1, 1)


# ----------------------------- two-head model -----------------------------

def project_states_t(R):
    """Batched nearest-state projection (Smolin-Gambetta-Smith) for Hermitian [..., d, d]; no gradient.
    Computed on the CPU: batched cuSOLVER eigh reserves workspace far beyond the size of these 4x4 problems."""
    dev = R.device
    R = R.detach().cpu()
    lam, V = torch.linalg.eigh(R)
    srt = torch.flip(lam, dims=[-1])
    k = torch.arange(1, lam.shape[-1] + 1, device=R.device, dtype=lam.dtype)
    t = (torch.cumsum(srt, -1) - 1.0) / k
    r = ((srt - t) > 0).to(lam.dtype) * k
    idx = (r.argmax(-1, keepdim=True))
    mu = torch.gather(t, -1, idx)
    lam = (lam - mu).clamp_min(0.0)
    return ((V * lam.to(V.dtype)[..., None, :]) @ V.conj().transpose(-1, -2)).to(dev)


class TwoHeadNet(BlockNet):
    def __init__(self, n_pos, f_in, hidden=128, depth=2, emb=8, ctx=16, neighbors=True):
        super().__init__(n_pos, f_in, NS + DIM ** 2, hidden, depth, emb, ctx, neighbors)


class BlockGeoQEM2(BlockGeoQEM):
    """Two heads on a shared trunk. The coordinate head s_c = s_lin + g(h) is trained with the coordinate
    error; the state head corrects the Cholesky factor of Pi(s_c) (gradient stopped through the
    projection) and is trained with D_B^2 + beta (1/4)||s(rho) - s||^2. The output is the state head."""

    def __init__(self, gamma=1.0, **kw):
        kw.setdefault("ctx", 16)
        super().__init__(output="chol", **kw)
        self.gamma = gamma

    def _T(self, X, B, Y=None):
        T = super()._T(X, B, Y)
        T.pop("A0", None)
        return T

    def _net(self, T, seed):
        torch.manual_seed(seed)
        n_pos = 3 if self.pos_mode == "boundary" else T["x"].shape[1]
        return TwoHeadNet(n_pos, T["x"].shape[2], self.hidden, self.depth, ctx=self.ctx,
                          neighbors=self.neighbors).to(self.device)

    def _heads(self, out, T):
        s_c = T["base"] + out[..., :self.ns]
        with torch.no_grad():
            R0 = (torch.eye(self.D, dtype=torch.complex128, device=self.device)
                  + torch.einsum("nbk,kij->nbij", s_c.detach().to(torch.complex128), self.P)) / self.D
            R0 = project_states_t(0.5 * (R0 + R0.conj().transpose(-1, -2)))
            A0 = torch.linalg.cholesky(R0 + 1e-6 * torch.eye(self.D, dtype=R0.dtype, device=R0.device))
        rho = self._assemble(out[..., self.ns:], {"A0": A0})
        return s_c, rho

    def _forward(self, net, T):
        return net(T["x"], T["base"], T["pos"])

    def _loss(self, out, T):
        s_c, rho = self._heads(out, T)
        main, mse = super()._loss(rho, T)
        coord = 0.25 * ((s_c - T["Y"]) ** 2).sum(-1).mean()
        return main + self.gamma * coord, mse

    def predict(self, S):
        X = stack_scales(S, self.scales)
        T = self._T(X, self.backbone.predict(X))
        with torch.no_grad():
            out = torch.stack([net(T["x"], T["base"], T["pos"]) for net in self.nets]).mean(0)
            _, rho = self._heads(out, T)
        return np.clip(self._coords(rho).double().cpu().numpy(), -1, 1)

    def predict_coords(self, S):
        X = stack_scales(S, self.scales)
        T = self._T(X, self.backbone.predict(X))
        with torch.no_grad():
            out = torch.stack([net(T["x"], T["base"], T["pos"]) for net in self.nets]).mean(0)
            s_c, _ = self._heads(out, T)
        return np.clip(s_c.double().cpu().numpy(), -1, 1)
