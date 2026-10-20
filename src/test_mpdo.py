"""Check mpdo.py against the density-matrix data of gen_v3.py (uniform TFIM n = 8): ideal and noisy block coordinates
of test circuits at fold scales 1, 3 and 5.
usage: test_mpdo.py [n]"""
import os
import sys
import time

import numpy as np

import mpdo
import noise_layout as NL
import qemlib as L
from paths import FEAT_DIR


def main(n=8):
    dev_noisy, dev_ideal = mpdo.Device(NL.twirled_noise_model(n)), mpdo.Device(None)
    for d in (1, 3):
        z = np.load(os.path.join(FEAT_DIR, f"tfim_n{n}_d{d}_test.npz"))
        circs = L.make_circuits("tfim", n, d, 50, "test")
        for i in (0, 7):
            for s in (1, 3, 5):
                t0 = time.time()
                qc = NL.to_device(L.fold_global(circs[i], s), n)
                A, chi, disc = mpdo.simulate(qc, n, dev_noisy)
                S, norm = mpdo.block_coords(A)
                err = np.abs(S - z[f"S_s{s}"][i]).max()
                print(f"d={d} circuit {i} scale {s}: max |dS| {err:.2e}  trace {norm:.12f}  chi {chi}  "
                      f"{time.time() - t0:.1f}s")
            A, chi, _ = mpdo.simulate(NL.to_device(circs[i], n), n, dev_ideal)
            print(f"d={d} circuit {i} ideal: max |dS| {np.abs(mpdo.block_coords(A)[0] - z['S_ideal'][i]).max():.2e}"
                  f"  chi {chi}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
