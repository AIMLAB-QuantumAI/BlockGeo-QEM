"""Check sample_blocks.measure_blocks against measure_v3.measure (global multinomial sampling) at n = 8:
exact limit, and mean and standard deviation of every estimated coordinate over repeated draws.
Needs the gen_v3.py data of uniform TFIM n = 8, depth 3."""
import os

import numpy as np

import measure_v3 as M
import qemlib as L
import sample_blocks as SB
from paths import FEAT_DIR


def main(n=8, d=3, i=0, reps=300, B=10000):
    z = np.load(os.path.join(FEAT_DIR, f"tfim_n{n}_d{d}_test.npz"))
    S, Q = z["S_s1"][i], z["Q_s1"][i]
    ex = SB.measure_blocks(S[None], n, None, None)[0]
    print("exact limit max |dS|:", np.abs(ex - S).max())
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(2)
    a = np.stack([M.measure(Q[None], n, B // 9, rng1)[0][0] for _ in range(reps)])
    b = np.stack([SB.measure_blocks(S[None], n, B, rng2)[0] for _ in range(reps)])
    print("mean |bias| global vs blocks:", np.abs(a.mean(0) - S).mean(), np.abs(b.mean(0) - S).mean())
    ra = a.std(0) / np.maximum(b.std(0), 1e-12)
    print("std ratio global/blocks: median %.3f, 5-95%% [%.3f, %.3f]" % (np.median(ra), *np.percentile(ra, [5, 95])))
    # shared single-qubit coordinates of overlapping blocks must coincide within one draw, as for the device protocol
    STR = L.local_pauli_strings(2, include_identity=False)
    sh_k, sh_k1 = [STR.index((p, "I")) for p in "XYZ"], [STR.index(("I", p)) for p in "XYZ"]
    print("overlap disagreement global, blocks:", np.abs(a[:, :-1][:, :, sh_k] - a[:, 1:][:, :, sh_k1]).max(),
          np.abs(b[:, :-1][:, :, sh_k] - b[:, 1:][:, :, sh_k1]).max())


if __name__ == "__main__":
    main()
