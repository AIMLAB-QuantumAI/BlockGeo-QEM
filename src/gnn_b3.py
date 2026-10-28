"""GNN-QEM for three-qubit blocks: the message-passing baseline of baselines_v3.gnn_qem on the same qubit chain
(nodes: qubits with single-qubit estimates and calibration; edges: couplers with the nine nearest-neighbour
correlators and the two-qubit gate error), with a readout per three-qubit block that reads its three nodes, its two
edges and its own 63 measured Pauli coordinates (standardized), and predicts the residual of these coordinates (MSE).
The block's coordinates enter the readout because node and edge features omit the three-body and next-nearest-
neighbour terms; in the two-qubit model the nodes and edges already carry all 15 coordinates of a block.
Training is identical to gnn_qem.

In a b-qubit block over qubits k..k+b-1, Pauli-string position p acts on qubit k + (b - 1 - p) (qemlib convention).
"""
import numpy as np

import qemlib as L

_S3 = L.local_pauli_strings(3, include_identity=False)
_PAULI = "XYZ"


def _idx(*ops):
    """Index of the 3-qubit string with ops = (op on string position 0, 1, 2)."""
    return _S3.index(tuple(ops))


# single-qubit X, Y, Z at string positions 2 (qubit k), 1 (qubit k+1), 0 (qubit k+2)
_ONE = {2: [_idx("I", "I", P) for P in _PAULI], 1: [_idx("I", P, "I") for P in _PAULI],
        0: [_idx(P, "I", "I") for P in _PAULI]}
# nearest-neighbour correlators in the order of baselines_v3._X2 (first letter on the upper qubit of the pair)
_PAIR_LOW = [_idx("I", A, B) for A in _PAULI for B in _PAULI]     # qubits (k, k+1) of block k
_PAIR_HIGH = [_idx(A, B, "I") for A in _PAULI for B in _PAULI]    # qubits (k+1, k+2) of block k


def graph_feats_b3(S, calib_q, calib_e):
    """Node features [N, n, 3 + c_q] and edge features [N, n-1, 9 + c_e] from three-qubit block coordinates
    S [N, n-2, 63]."""
    N, nb, _ = S.shape
    n = nb + 2
    node = np.empty((N, n, 3))
    node[:, :nb] = S[:, :, _ONE[2]]
    node[:, nb] = S[:, nb - 1, _ONE[1]]
    node[:, nb + 1] = S[:, nb - 1, _ONE[0]]
    edge = np.empty((N, n - 1, 9))
    edge[:, :nb] = S[:, :, _PAIR_LOW]
    edge[:, nb] = S[:, nb - 1, _PAIR_HIGH]
    node = np.concatenate([node, np.broadcast_to(calib_q, (N,) + calib_q.shape)], -1)
    edge = np.concatenate([edge, np.broadcast_to(calib_e, (N,) + calib_e.shape)], -1)
    return node, edge


def gnn_qem_b3(train_S, train_Y, test_S, calib_q, calib_e, hidden=64, layers=3, lr=1e-3, wd=1e-4, max_epochs=3000,
               patience=300, seed=0):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ntr, nte = graph_feats_b3(train_S, calib_q, calib_e), graph_feats_b3(test_S, calib_q, calib_e)
    mu_n, sd_n = ntr[0].mean((0, 1)), ntr[0].std((0, 1)) + 1e-8
    mu_e, sd_e = ntr[1].mean((0, 1)), ntr[1].std((0, 1)) + 1e-8
    mu_s, sd_s = train_S.mean((0, 1)), train_S.std((0, 1)) + 1e-8
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    nc = train_S.shape[-1]

    def tens(g, S):
        return t((g[0] - mu_n) / sd_n), t((g[1] - mu_e) / sd_e), t(S), t((S - mu_s) / sd_s)

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, hidden), nn.SiLU(), nn.Linear(hidden, o))

    class Net(nn.Module):
        def __init__(self, dn, de):
            super().__init__()
            self.enc_n, self.enc_e = mlp(dn, hidden), mlp(de, hidden)
            self.upd_e = nn.ModuleList([mlp(3 * hidden, hidden) for _ in range(layers)])
            self.upd_n = nn.ModuleList([mlp(2 * hidden, hidden) for _ in range(layers)])
            self.out = mlp(5 * hidden + nc, nc)
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

        def forward(self, xn, xe, S, Sz):
            h, e = self.enc_n(xn), self.enc_e(xe)
            for ue, un in zip(self.upd_e, self.upd_n):
                e = e + ue(torch.cat([e, h[:, :-1], h[:, 1:]], -1))
                z = torch.zeros_like(e[:, :1])
                agg = 0.5 * (torch.cat([z, e], 1) + torch.cat([e, z], 1))   # incident edges of every node
                h = h + un(torch.cat([h, agg], -1))
            blk = torch.cat([h[:, :-2], h[:, 1:-1], h[:, 2:], e[:, :-1], e[:, 1:], Sz], -1)   # block j: qubits j..j+2
            return S + self.out(blk)

    Ytr = t(train_Y)

    def train(idx, val=None, epochs=max_epochs):
        torch.manual_seed(seed)
        xn, xe, S, Sz = tens((ntr[0][idx], ntr[1][idx]), train_S[idx])
        net = Net(xn.shape[-1], xe.shape[-1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        best, best_ep, last = np.inf, -1, 0
        for ep in range(epochs):
            loss = ((net(xn, xe, S, Sz) - Ytr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if val is not None and ep % 10 == 0:
                with torch.no_grad():
                    vn, ve, vS, vSz = tens((ntr[0][val], ntr[1][val]), train_S[val])
                    v = float(((net(vn, ve, vS, vSz) - Ytr[val]) ** 2).mean())
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
