"""Site-resolved quench quantities from the saved predictions (<results dir>/quench_pred_*.npz of run_quench.py,
<hardware dir>/quench_<backend>/pred_d*.npz of quench_hw.py): per-block purity and per-bond concurrence, their RMS
error over blocks for every method, and the <X_i> profile. Prints a table per depth and writes the rows to
<results dir>/quench_blocks_tfim_n8_B10000.csv (sim) or <hardware dir>/quench_<backend>/blocks.csv (hw), which
paper_figs.py hardware reads.

usage: analyze_quench_blocks.py sim        (simulation, all depths found)
       analyze_quench_blocks.py hw ibm_fez
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import qemlib as L
from paths import HW, RES_DIR

STR = L.local_pauli_strings(2, include_identity=False)
IX, XI = STR.index(("I", "X")), STR.index(("X", "I"))
METHODS = ["Noisy", "OLS", "MLP-QEM", "GNN-QEM", "abl_HS_state_loss", "BlockGeo-QEM"]


def per_block(S):
    """S [..., n-1, 15] -> dict of per-block purity, concurrence [..., n-1] and per-site <X_i> [..., n]."""
    R = L.block_states(S, 2)
    return dict(purity=L.purity(R), concurrence=L.concurrence(R),
                X=np.concatenate([S[..., :, IX], S[..., -1:, XI]], axis=-1))


def table(files):
    rows = []
    for f in files:
        z = np.load(f)
        d = int(os.path.basename(f).split("_d")[-1].split(".")[0])
        ideal = per_block(z["__ideal__"][None])
        for m in METHODS:
            if m not in z:
                continue
            q = per_block(z[m])                                     # [R, blocks]
            for name in ("purity", "concurrence", "X"):
                err = np.abs(q[name] - ideal[name])                 # [R, blocks]
                rows.append(dict(depth=d, method=m.replace("_", " ").replace("abl HS", "abl: HS"), quantity=name,
                                 rms_over_blocks=float(np.sqrt((err ** 2).mean(-1)).mean()),
                                 max_over_blocks=float(err.max(-1).mean()),
                                 sd_of_rms=float(np.sqrt((err ** 2).mean(-1)).std())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    if sys.argv[1] == "sim":
        files = sorted(glob.glob(os.path.join(RES_DIR, "quench_pred_tfim_n8_B10000_d*.npz")))
        out = os.path.join(RES_DIR, "quench_blocks_tfim_n8_B10000.csv")
    else:
        files = sorted(glob.glob(os.path.join(HW, f"quench_{sys.argv[2]}", "pred_d*.npz")))
        out = os.path.join(HW, f"quench_{sys.argv[2]}", "blocks.csv")
    if not files:
        sys.exit(f"no quench predictions found for {out}")
    t = table(files)
    t.to_csv(out, index=False)
    for q in ("purity", "concurrence", "X"):
        print(f"== {q}: RMS error over blocks (mean over draws)")
        print(t[t.quantity == q].pivot(index="method", columns="depth", values="rms_over_blocks").round(4).to_string())
