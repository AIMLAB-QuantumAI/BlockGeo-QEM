"""GNN-QEM (Bures): the message-passing network of the GNN-QEM baseline used as the trunk of BlockGeo-QEM.

Everything except the trunk is that of BlockGeoQEM2: the per-coordinate ridge backbone, the coordinate head and the
Cholesky state head on top of the backbone estimate, the loss D_B^2 + beta/4 * MSE, the validation rule, and the
average over three seeds. The trunk reads the node features (single-qubit coordinates and qubit calibration) and edge
features (two-body correlators and two-qubit gate error) of baselines_v3._graph_feats, passes messages along the
chain for three rounds, and the readout of block k concatenates the edge state, the two node states and the
backbone estimate of the block. Isolates the contribution of the trunk from that of the loss and the heads.
The graph features are those of two-qubit blocks. run_gnnb.py runs the model on the run_v3 draws.
"""
import numpy as np
import torch
import torch.nn as nn

import baselines_v3 as B3
from v3 import Backbone, BlockGeoQEM2, mlp, stack_scales


class GNNTwoHead(nn.Module):
    def __init__(self, dn, de, ns, nch, hidden=64, layers=3):
        super().__init__()
        self.enc_n, self.enc_e = mlp(dn, hidden, hidden, 1), mlp(de, hidden, hidden, 1)
        self.upd_e = nn.ModuleList([mlp(3 * hidden, hidden, hidden, 1) for _ in range(layers)])
        self.upd_n = nn.ModuleList([mlp(2 * hidden, hidden, hidden, 1) for _ in range(layers)])
        self.out = mlp(3 * hidden + ns, hidden, ns + nch, 1)
        nn.init.zeros_(self.out[-1].weight)                   # untrained output = projected backbone, as in BlockGeo-QEM
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, xn, xe, base):
        h, e = self.enc_n(xn), self.enc_e(xe)
        for ue, un in zip(self.upd_e, self.upd_n):
            e = e + ue(torch.cat([e, h[:, :-1], h[:, 1:]], -1))
            z = torch.zeros_like(e[:, :1])
            agg = 0.5 * (torch.cat([z, e], 1) + torch.cat([e, z], 1))   # incident edges of every node
            h = h + un(torch.cat([h, agg], -1))
        return self.out(torch.cat([e, h[:, :-1], h[:, 1:], base], -1))


class GNNBures(BlockGeoQEM2):
    def __init__(self, calib_q, calib_e, gnn_hidden=64, gnn_layers=3, **kw):
        super().__init__(**kw)
        if self.b != 2:
            raise ValueError("GNNBures reads the graph features of two-qubit blocks (b = 2)")
        self.cq, self.ce, self.gh, self.gl = calib_q, calib_e, gnn_hidden, gnn_layers
        self._gstats = None

    def _graph(self, X):
        return B3._graph_feats(X[..., :self.ns], self.cq, self.ce)      # single noise level

    def _set_gstats(self, X):
        gn, ge = self._graph(X)
        self._gstats = (gn.mean((0, 1)), gn.std((0, 1)) + 1e-8, ge.mean((0, 1)), ge.std((0, 1)) + 1e-8)

    def _T(self, X, B, Y=None):
        T = super()._T(X, B, Y)
        gn, ge = self._graph(X)
        mn, sn, me, se = self._gstats
        f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        T["xn"], T["xe"] = f32((gn - mn) / sn), f32((ge - me) / se)
        return T

    def _net(self, T, seed):
        torch.manual_seed(seed)
        return GNNTwoHead(T["xn"].shape[-1], T["xe"].shape[-1], self.ns, self.D ** 2, self.gh, self.gl).to(self.device)

    def _forward(self, net, T):
        return net(T["xn"], T["xe"], T["base"])

    def fit(self, S_train, Y_train):
        X = stack_scales(S_train, self.scales)
        K = len(self.scales)
        self.backbone = Backbone(self.pos_mode, ns=self.ns).fit(X, Y_train, K)
        self.nets, self.best_eps = [], []
        for seed in self.seeds:
            perm = np.random.default_rng(seed).permutation(len(X))
            nva = int(round(self.val_frac * len(X)))
            va, tr = perm[:nva], perm[nva:]
            bb = Backbone(self.pos_mode, ns=self.ns).fit(X[tr], Y_train[tr], K)
            self.mu, self.sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-8
            self._set_gstats(X[tr])
            _, ep = self._train(self._T(X[tr], bb.predict(X[tr]), Y_train[tr]),
                                self._T(X[va], bb.predict(X[va]), Y_train[va]), seed=seed)
            self.best_eps.append(ep)
        self.mu, self.sd = X.mean((0, 1)), X.std((0, 1)) + 1e-8
        self._set_gstats(X)
        T = self._T(X, self.backbone.predict(X), Y_train)
        for seed, ep in zip(self.seeds, self.best_eps):
            self.nets.append(self._train(T, epochs=ep + 1, seed=seed)[0])
        return self

    def predict(self, S):
        X = stack_scales(S, self.scales)
        T = self._T(X, self.backbone.predict(X))
        with torch.no_grad():
            out = torch.stack([self._forward(net, T) for net in self.nets]).mean(0)
            _, rho = self._heads(out, T)
        return np.clip(self._coords(rho).double().cpu().numpy(), -1, 1)
