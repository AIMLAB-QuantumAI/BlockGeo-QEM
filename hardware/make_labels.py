"""Ideal block coordinates of the circuits of a hardware run, written to <hardware dir>/<tag>/labels.npz, so that the
hardware analyses (analyze_hw.py, compare_hw.py, quench_hw.py) need no simulated data.

The values are taken from the uniform-TFIM data of gen_v3.py (n = chain length) in the feats directory: the circuits
of run_hw.py are the first n_train training and the first n_test test circuits of those ensembles.

run of run_hw.py (meta.json has "depth"):
    S_train       [n_train, n-1, 15]  S_ideal of the training circuits                  (compare_hw.py)
    S_test        [n_test, n-1, 15]   S_ideal of the test circuits                      (analyze_hw.py, compare_hw.py)
    S_test_s1     [n_test, n-1, 15]   simulated noisy coordinates (fold scale 1, exact) of the test circuits, the
                                      reference printed by analyze_hw.py
quench run of run_quench_hw.py (meta.json has "depths"), given the runs that quench_hw.py combines with it:
    S_train_d<d>  [n_train, n-1, 15]  S_ideal of the training circuits of the run used at depth d

usage: make_labels.py <tag>                                    e.g. ibm_fez_d1_tr100_te30_s1000
       make_labels.py quench_<backend> "<d>:<tag>,<d>:<tag>"   e.g. quench_ibm_fez "1:ibm_fez_d1_...,2:ibm_fez_d2_..."
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import numpy as np

from paths import FEAT_DIR, HW


def feats(n, d, split):
    return np.load(os.path.join(FEAT_DIR, f"tfim_n{n}_d{d}_{split}.npz"))


def meta(tag):
    with open(os.path.join(HW, tag, "meta.json")) as f:
        return json.load(f)


def run_labels(tag):
    m = meta(tag)
    n, d = len(m["chain"]), m["depth"]
    tr, te = feats(n, d, "train"), feats(n, d, "test")
    return dict(S_train=tr["S_ideal"][: m["n_train"]], S_test=te["S_ideal"][: m["n_test"]],
                S_test_s1=te["S_s1"][: m["n_test"]])


def quench_labels(qtag, runs):
    n = len(meta(qtag)["chain"])
    out = {}
    for d in meta(qtag)["depths"]:
        m = meta(runs[d])
        if len(m["chain"]) != n or m["depth"] != d:
            raise ValueError(f"run {runs[d]} does not match the quench circuit at depth {d}")
        out[f"S_train_d{d}"] = feats(n, d, "train")["S_ideal"][: m["n_train"]]
    return out


if __name__ == "__main__":
    tag = sys.argv[1]
    if tag.startswith("quench_"):
        runs = {int(k): v for k, v in (item.split(":") for item in sys.argv[2].split(","))}
        labels = quench_labels(tag, runs)
    else:
        labels = run_labels(tag)
    out = os.path.join(HW, tag, "labels.npz")
    np.savez_compressed(out, **labels)
    print("wrote", out, {k: v.shape for k, v in labels.items()})
