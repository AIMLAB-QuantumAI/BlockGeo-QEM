"""Analyses of the saved predictions (<results dir>/pred/*.npz of the runners), no new training.

usage: paper_review_checks.py [purity] [overlap] [holm]      (default: all three)

purity   gain of the Bures loss resolved by the ideal purity of each block: mean block Bures distance of the
         Hilbert-Schmidt-trained model and of BlockGeo-QEM in purity bins (uniform TFIM n=6, 8, 10, random n=10 and
         disordered TFIM n=8, 1e5 shots); writes <results dir>/purity_resolved_B100000.csv for paper_figs.py
         geometry. Weight-matched control: at the maximally mixed state both losses equal ||ds||^2/4.
overlap  overlapping blocks k and k+1 both estimate the single-qubit coordinates of qubit k+1; report their mean
         disagreement per method, and how much the nearest-state projection moves unphysical coordinate estimates.
holm     outcomes of the simulated grid of tables/tab_wtl.tex (both devices, 1e4-1e6 shots) after a Holm correction
         over all its paired tests.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import qemlib as L

P = os.path.join(paths.RES_DIR, "pred") + os.sep
STR = L.local_pauli_strings(2, include_identity=False)
SH_K = [STR.index((p, "I")) for p in "XYZ"]      # qubit k+1 as the second qubit of block k
SH_K1 = [STR.index(("I", p)) for p in "XYZ"]     # qubit k+1 as the first qubit of block k+1
DEPTHS = {"tfim": (1, 2, 3, 4, 6, 8), "random": (4, 8, 12, 16, 20, 24), "dtfim": (1, 2, 3, 4, 6, 8)}


def load(prefix, d):
    f = P + f"{prefix}_d{d}.npz"
    return dict(np.load(f)) if os.path.exists(f) else None


def block_bures(S, Y):
    """Bures distance of every block, [N, nb]."""
    N, nb, _ = S.shape
    return np.array([[np.sqrt(max(L.bures_sq(L.obs_to_rho(Y[i, k], 2), L.obs_to_rho(S[i, k], 2)), 0.0))
                      for k in range(nb)] for i in range(N)])


def purity_resolved(B=100000):
    sets = [("tfim", 6, "oursnl_tfim_n6_B{B}"), ("tfim", 8, "oursnl_tfim_n8_B{B}"), ("tfim", 10, "ours_tfim_n10_B{B}"),
            ("random", 10, "ours_random_n10_B{B}"), ("dtfim", 8, "ours_dtfim_n8_B{B}")]
    rows = []
    for fam, n, pre in sets:
        for d in DEPTHS[fam]:
            z = load(pre.format(B=B), d)
            if z is None:
                continue
            Y = z["__ideal__"]
            pur = (1 + (Y ** 2).sum(-1)) / 4
            ours, hs = block_bures(z["BlockGeo-QEM"], Y), block_bures(z["abl_HS_state_loss"], Y)
            rows.append(pd.DataFrame({"family": fam, "n": n, "depth": d, "purity": pur.ravel(),
                                      "ours": ours.ravel(), "hs": hs.ravel()}))
    df = pd.concat(rows, ignore_index=True)
    bins = [0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 1.0001]
    df["bin"] = pd.cut(df.purity, bins, right=False)
    g = df.groupby(["family", "bin"], observed=True).agg(blocks=("ours", "size"), ours=("ours", "mean"),
                                                         hs=("hs", "mean"))
    g["ratio_hs_over_ours"] = g.hs / g.ours
    return df, g


def overlap(prefixes, d_list):
    rows = []
    for pre in prefixes:
        for d in d_list:
            z = load(pre, d)
            if z is None:
                continue
            for name, S in z.items():
                dis = np.abs(S[:, :-1][:, :, SH_K] - S[:, 1:][:, :, SH_K1]).mean()
                proj = np.stack([[_proj(s) for s in Sk]
                                 for Sk in S])
                move = np.linalg.norm(proj - S, axis=-1)
                rows.append(dict(set=pre, depth=d, method=name, overlap_disagreement=dis,
                                 frac_unphysical=float((move > 1e-6).mean()), mean_projection_move=float(move.mean())))
    return pd.DataFrame(rows)


def _proj(s):
    """Pauli coordinates of the nearest density matrix to the coordinates s."""
    rho = L.obs_to_rho(s, 2)
    Pm = L.pauli_ops(2)
    return np.real(np.einsum("kij,ji->k", Pm, rho))


def holm():
    """Holm correction over every paired test of tables/tab_wtl.tex: the three circuit families on the first device
    and, when its results exist, uniform and disordered TFIM on the second device (results dir with the tag
    _v3_torino)."""
    import glob

    import analyze_v3 as A

    def tests(df, device):
        c = A.compare(df, "BlockGeo-QEM")
        c["budget"] = c.budget.astype(int)
        c = c[c.budget.isin([10000, 100000, 1000000]) & c.metric.isin(["local_bures", "local_trace"])].copy()
        c["device"] = device
        return c

    parts = [tests(A.load(), "first device")]
    tor = sorted(glob.glob(os.path.join(paths.results_dir("_v3_torino"), "*_samples.csv")))
    if tor:
        parts.append(tests(pd.concat([pd.read_csv(f) for f in tor], ignore_index=True), "second device"))
    c = pd.concat(parts, ignore_index=True)
    order = np.argsort(c.p.values)
    m = len(c)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * c.p.values[i])
        adj[i] = min(1.0, run)
    c["p_holm"] = adj
    c["outcome_holm"] = np.where((c.ratio < 1) & (c.p_holm < .05), "win",
                                 np.where((c.ratio > 1) & (c.p_holm < .05), "loss", "tie"))
    wtl = lambda s: f"{(s == 'win').sum()}/{(s == 'tie').sum()}/{(s == 'loss').sum()}"
    keys = ["device", "family", "budget", "metric"]
    out = pd.concat([c.groupby(keys).outcome.apply(wtl).rename("uncorrected"),
                     c.groupby(keys).outcome_holm.apply(wtl).rename("holm")], axis=1)
    return out, m


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    which = sys.argv[1:] or ["purity", "overlap", "holm"]
    if "purity" in which:
        df, g = purity_resolved()
        print(g.round(4).to_string())
        df.to_csv(os.path.join(paths.RES_DIR, "purity_resolved_B100000.csv"), index=False)
    if "overlap" in which:
        t = overlap(["basenl_tfim_n8_B100000", "oursnl_tfim_n8_B100000"], (1, 3, 6))
        print(t.groupby("method")[["overlap_disagreement", "frac_unphysical", "mean_projection_move"]].mean()
              .sort_values("overlap_disagreement").round(4).to_string())
    if "holm" in which:
        out, m = holm()
        print(f"Holm over {m} tests"); print(out.to_string())
