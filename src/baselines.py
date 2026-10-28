"""Baseline mitigators operating on block Pauli coordinates.

Shapes: S arrays are [N, nblocks, 4^b-1] (15 coordinates for two-qubit blocks); stacks over noise scales are
[K, N, nblocks, 4^b-1].
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

SCALES = (1, 3, 5)


# ---------------------------- ZNE ----------------------------

def zne_poly(Y, scales=SCALES, order=1):
    lam = np.asarray(scales, float)
    V = np.vander(lam, order + 1)
    coef = np.linalg.pinv(V) @ Y.reshape(len(lam), -1)
    return np.clip(coef[-1].reshape(Y.shape[1:]), -1, 1)


def zne_exp(Y, scales=SCALES):
    """y(lam) = A exp(-c lam) fitted in log space (asymptote 0); linear fallback where invalid."""
    lam = np.asarray(scales, float)
    flat = Y.reshape(len(lam), -1)
    out = zne_poly(Y, scales, 1).reshape(-1).copy()
    sgn = np.sign(flat)
    absy = np.abs(flat)
    ok = np.all(sgn == sgn[0], axis=0) & (sgn[0] != 0) & np.all(np.diff(absy, axis=0) < 0, axis=0)
    if ok.any():
        A = np.vstack([lam, np.ones_like(lam)]).T
        coef, *_ = np.linalg.lstsq(A, np.log(absy[:, ok]), rcond=None)
        out[ok] = sgn[0, ok] * np.exp(coef[1])
    return np.clip(out.reshape(Y.shape[1:]), -1, 1)


# ------------------------ learned, linear ------------------------

def _stack(Sk, use_scales):
    return np.stack([Sk[k] for k in use_scales], axis=-1)  # [N, nb, 15, K]


def ols_per_coord(train_S, train_Y, test_S, use_scales=(1,)):
    """CDR-style regression: each coordinate gets its own affine map of its noisy value(s).
    use_scales=(1,) is CDR-like, (1,3,5) is vnCDR-like."""
    Xtr, Xte = _stack(train_S, use_scales), _stack(test_S, use_scales)
    N, nb, P, K = Xtr.shape
    out = np.empty(test_S[1].shape)
    for k in range(nb):
        for j in range(P):
            A = np.hstack([Xtr[:, k, j, :], np.ones((N, 1))])
            w, *_ = np.linalg.lstsq(A, train_Y[:, k, j], rcond=None)
            out[:, k, j] = np.hstack([Xte[:, k, j, :], np.ones((len(Xte), 1))]) @ w
    return np.clip(out, -1, 1)


# ------------------------ learned, nonlinear ------------------------

def rf_block(train_S, train_Y, test_S, use_scales=(1,), seed=0):
    """Random-forest QEM (Liao et al. 2024 style) per block position, predicting the residual."""
    Xtr, Xte = _stack(train_S, use_scales), _stack(test_S, use_scales)
    N, nb, P, K = Xtr.shape
    out = np.empty(test_S[1].shape)
    for k in range(nb):
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=8, random_state=seed)
        m.fit(Xtr[:, k].reshape(N, -1), train_Y[:, k] - train_S[1][:, k])
        out[:, k] = test_S[1][:, k] + m.predict(Xte[:, k].reshape(len(Xte), -1))
    return np.clip(out, -1, 1)


def mlp_global(train_S, train_Y, test_S, use_scales=(1,), hidden=256, lr=1e-3, wd=1e-4, max_epochs=3000,
               patience=300, seed=0):
    """MLP-QEM (Liao et al. 2024 style): one network maps every block's noisy coordinates to the
    residual (ideal - noisy) of every block; MSE loss, epochs chosen on a 20% split, then refit."""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = np.concatenate([train_S[k] for k in use_scales], -1).reshape(len(train_Y), -1)
    Xte = np.concatenate([test_S[k] for k in use_scales], -1).reshape(len(test_S[1]), -1)
    R = (train_Y - train_S[1]).reshape(len(train_Y), -1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)

    def train(Xa, Ra, Xv=None, Rv=None, epochs=max_epochs):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xa.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                            nn.Linear(hidden, Ra.shape[1])).to(dev)
        nn.init.zeros_(net[-1].weight)   # start from the unmitigated input
        nn.init.zeros_(net[-1].bias)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        xa, ra = t((Xa - mu) / sd), t(Ra)
        best, best_ep, last = np.inf, 0, 0
        for ep in range(epochs):
            loss = ((net(xa) - ra) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if Xv is not None and ep % 10 == 0:
                with torch.no_grad():
                    v = float(((net(t((Xv - mu) / sd)) - t(Rv)) ** 2).mean())
                if v < best:
                    best, best_ep, last = v, ep, ep
                elif ep - last > patience:
                    break
        return net, best_ep

    perm = np.random.default_rng(seed).permutation(len(Xtr))
    nva = len(Xtr) // 5
    _, ep = train(Xtr[perm[nva:]], R[perm[nva:]], Xtr[perm[:nva]], R[perm[:nva]])
    net, _ = train(Xtr, R, epochs=ep + 1)
    with torch.no_grad():
        out = net(t((Xte - mu) / sd)).cpu().numpy().reshape(test_S[1].shape)
    return np.clip(test_S[1] + out, -1, 1)
