"""Additional v3 baselines: Clifford data regression (CDR, vnCDR), idealised PEC, GNN-QEM.

Shapes: block coordinates are [N, nb, 4^b-1] (15 for two-qubit blocks); CDR training sets are [N_test, K, nb, 4^b-1].
GNN-QEM reads two-qubit blocks; gnn_b3.py is its three-qubit version.
"""
import itertools

import numpy as np

import qemlib as L


# ----------------------------- CDR / vnCDR (Czarnik et al. 2021; Lowe et al. 2021) -----------------------------

def _lstsq_batched(X, y, ridge=1e-8):
    """X [..., K, F], y [..., K] -> coefficients [..., F] (tiny ridge for degenerate training sets)."""
    XtX = np.einsum("...kf,...kg->...fg", X, X) + ridge * np.eye(X.shape[-1])
    Xty = np.einsum("...kf,...k->...f", X, y)
    return np.linalg.solve(XtX, Xty[..., None])[..., 0]


def cdr(train_noisy, train_ideal, test_noisy):
    """Per test circuit and observable: fit ideal = a * noisy + b on that circuit's near-Clifford training set."""
    X = np.stack([train_noisy, np.ones_like(train_noisy)], -1)          # [Nt, K, nb, 15, 2]
    X = np.moveaxis(X, 1, -2)                                             # [Nt, nb, 15, K, 2]
    y = np.moveaxis(train_ideal, 1, -1)                                   # [Nt, nb, 15, K]
    w = _lstsq_batched(X, y)
    return np.clip(w[..., 0] * test_noisy + w[..., 1], -1, 1)


def vncdr(train_noisy, train_ideal, test_noisy, scales=(1, 3, 5)):
    """Variable-noise CDR: fit ideal = sum_s c_s noisy_s + b on the near-Clifford training set."""
    X = np.stack([train_noisy[s] for s in scales] + [np.ones_like(train_noisy[scales[0]])], -1)
    X = np.moveaxis(X, 1, -2)
    y = np.moveaxis(train_ideal, 1, -1)
    w = _lstsq_batched(X, y)
    Xt = np.stack([test_noisy[s] for s in scales] + [np.ones_like(test_noisy[scales[0]])], -1)
    return np.clip((w * Xt).sum(-1), -1, 1)


# ----------------------------- idealised PEC -----------------------------

def coord_support(n, b=2):
    """Support qubits of every block coordinate, in the block-major order used everywhere."""
    out = []
    for blk in L.contiguous_blocks(n, b):
        for st in L.local_pauli_strings(b, include_identity=False):
            out.append(tuple(blk[b - 1 - j] for j, c in enumerate(st) if c != "I"))
    return out


def pec(test_ideal, gamma, n, budget, rng, readout_scale, b=2):
    """PEC with exact knowledge of the twirled noise: an unbiased estimator whose single-shot values are
    +-gamma_P, gamma_P = gamma_circuit * prod_{q in supp(P)} readout_scale[q]. A weight-w string of a b-qubit block
    gets the shots of 3^(b-w) settings (budget/3^b shots per setting). budget=0 returns the ideal values."""
    if budget == 0:
        return test_ideal.copy()
    N, nb, ns = test_ideal.shape
    sup = coord_support(n, b)
    g_ro = np.array([np.prod([readout_scale[q] for q in s]) for s in sup]).reshape(nb, ns)
    per = budget // 3 ** b                       # shots per covering setting; a weight-w string is in 3^(b-w) of them
    shots = np.array([per * 3 ** (b - len(s)) for s in sup]).reshape(nb, ns)
    g = gamma[:, None, None] * g_ro[None]
    prob = np.clip((1 + test_ideal / g) / 2, 0, 1)
    k = rng.binomial(np.broadcast_to(shots, prob.shape), prob)
    return g * (2 * k / shots - 1)


# ----------------------------- GNN-QEM -----------------------------

_X1 = {"k": [0, 1, 2], "k+1": [3, 7, 11]}           # X, Y, Z on the first / second qubit of a block
_X2 = [4, 5, 6, 8, 9, 10, 12, 13, 14]               # two-body correlators


def _graph_feats(S, calib_q, calib_e):
    """Node features [N, n, 3 + c_q] and edge features [N, nb, 9 + c_e] from block coordinates."""
    N, nb, _ = S.shape
    n = nb + 1
    node = np.empty((N, n, 3))
    node[:, :nb] = S[:, :, _X1["k"]]
    node[:, nb] = S[:, nb - 1, _X1["k+1"]]
    node = np.concatenate([node, np.broadcast_to(calib_q, (N,) + calib_q.shape)], -1)
    edge = np.concatenate([S[:, :, _X2], np.broadcast_to(calib_e, (N,) + calib_e.shape)], -1)
    return node, edge


def gnn_qem(train_S, train_Y, test_S, calib_q, calib_e, hidden=64, layers=3, lr=1e-3, wd=1e-4, max_epochs=3000,
            patience=300, seed=0, test_calib=None):
    """Message passing on the qubit chain (nodes: qubits with single-qubit estimates and calibration; edges:
    couplers with two-body correlators and ECR error), predicting each block's coordinate residual (MSE)."""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tq, tce = test_calib if test_calib is not None else (calib_q, calib_e)   # test chain may differ (transfer)
    ntr, nte = _graph_feats(train_S, calib_q, calib_e), _graph_feats(test_S, tq, tce)
    mu_n, sd_n = ntr[0].mean((0, 1)), ntr[0].std((0, 1)) + 1e-8
    mu_e, sd_e = ntr[1].mean((0, 1)), ntr[1].std((0, 1)) + 1e-8
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)

    def tens(g, S):
        return t((g[0] - mu_n) / sd_n), t((g[1] - mu_e) / sd_e), t(S)

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, hidden), nn.SiLU(), nn.Linear(hidden, o))

    class Net(nn.Module):
        def __init__(self, dn, de):
            super().__init__()
            self.enc_n, self.enc_e = mlp(dn, hidden), mlp(de, hidden)
            self.upd_e = nn.ModuleList([mlp(3 * hidden, hidden) for _ in range(layers)])
            self.upd_n = nn.ModuleList([mlp(2 * hidden, hidden) for _ in range(layers)])
            self.out = mlp(3 * hidden, 15)
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

        def forward(self, xn, xe, S):
            h, e = self.enc_n(xn), self.enc_e(xe)
            for ue, un in zip(self.upd_e, self.upd_n):
                e = e + ue(torch.cat([e, h[:, :-1], h[:, 1:]], -1))
                z = torch.zeros_like(e[:, :1])
                agg = 0.5 * (torch.cat([z, e], 1) + torch.cat([e, z], 1))   # incident edges of every node
                h = h + un(torch.cat([h, agg], -1))
            return S + self.out(torch.cat([e, h[:, :-1], h[:, 1:]], -1))

    Ytr = t(train_Y)

    def train(idx, val=None, epochs=max_epochs):
        torch.manual_seed(seed)
        xn, xe, S = tens((ntr[0][idx], ntr[1][idx]), train_S[idx])
        net = Net(xn.shape[-1], xe.shape[-1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        best, best_ep, last = np.inf, -1, 0
        for ep in range(epochs):
            loss = ((net(xn, xe, S) - Ytr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if val is not None and ep % 10 == 0:
                with torch.no_grad():
                    vn, ve, vS = tens((ntr[0][val], ntr[1][val]), train_S[val])
                    v = float(((net(vn, ve, vS) - Ytr[val]) ** 2).mean())
                if v < best:
                    best, best_ep, last = v, ep, ep
                elif ep - last > patience:
                    break
        return net, best_ep

    perm = np.random.default_rng(seed).permutation(len(train_S))
    nva = len(train_S) // 5
    _, ep = train(perm[nva:], perm[:nva])
    net, _ = train(np.arange(len(train_S)), epochs=ep + 1)
    with torch.no_grad():
        out = net(*tens(nte, test_S)).cpu().numpy()
    return np.clip(out, -1, 1)


def chain_calibration(n, backend=None):
    """Per-qubit (T1, T2, sx error, readout error) and per-coupler (ECR error) of the chain, logical order."""
    import noise_layout as NL

    be = L.load_fake_backend(backend or NL.BACKEND)
    tgt = be.target
    phys = NL.CHAIN[:n]
    cq = np.array([[be.qubit_properties(q).t1 * 1e6, be.qubit_properties(q).t2 * 1e6, tgt["sx"][(q,)].error,
                    tgt["measure"][(q,)].error] for q in phys])
    ce = []
    for i in range(n - 1):
        a, b = phys[i], phys[i + 1]
        g = tgt[NL.TWOQ]
        ce.append([g[(a, b)].error if (a, b) in g else g[(b, a)].error])
    return cq, np.array(ce)
