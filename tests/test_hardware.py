"""Shipped hardware runs (data/hardware/<run>): the block estimates rebuilt from the job files by
hardware/analyze_hw.py, together with the ideal labels in labels.npz, reproduce the stored results of compare_hw.py
(results.csv, pred.npz) and quench_hw.py (quench_ibm_fez/results.csv, pred_d<d>.npz) for the unmitigated estimates,
for OLS and for the exact values."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qiskit.quantum_info import DensityMatrix

import analyze_hw
import baselines as B
import qemlib as L
import quench_hw
from bench import metrics

HW_DATA = Path(__file__).resolve().parents[1] / "data" / "hardware"
RUNS = ["ibm_fez_d1_tr100_te30_s1000", "ibm_fez_d2_tr100_te30_s500"]
QUENCH, QUENCH_RUNS = "quench_ibm_fez", {1: RUNS[0], 2: RUNS[1]}       # quench run and the runs trained on per depth
METRICS = ["local_bures", "local_trace", "mae_diag", "mae_off", "err_purity", "err_renyi2", "err_concurrence"]


@pytest.fixture(scope="module", params=RUNS)
def run(request):
    """Block estimates without (S_raw) and with (S_rem) readout-error mitigation, confusion matrices, metadata,
    labels, and the stored per-sample results and predictions of one run."""
    tag = request.param
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(analyze_hw, "ROOT", str(HW_DATA))        # always the shipped measurements
        S_raw, S_rem, A, meta = analyze_hw.load_run(tag)
        labels = analyze_hw.load_labels(tag)
    return dict(S_raw=S_raw, S_rem=S_rem, A=A, meta=meta, labels=labels,
                results=pd.read_csv(HW_DATA / tag / "results.csv"), pred=dict(np.load(HW_DATA / tag / "pred.npz")))


def test_run_contents(run):
    """Every circuit is measured in all 9 covering settings; estimates, labels and the calibrated confusion matrices
    have the expected shapes and ranges."""
    meta = run["meta"]
    n, n_train, n_test = len(meta["chain"]), meta["n_train"], meta["n_test"]
    assert len(meta["pairs"]) == 9 * (n_train + n_test)
    for split, num in (("train", n_train), ("test", n_test)):
        for S in (run["S_raw"][split], run["S_rem"][split]):
            assert S.shape == (num, n - 1, 15)
            assert np.isfinite(S).all() and np.abs(S).max() <= 1
    lab = run["labels"]
    assert lab["S_train"].shape == (n_train, n - 1, 15)
    assert lab["S_test"].shape == lab["S_test_s1"].shape == (n_test, n - 1, 15)
    A = run["A"]
    assert A.shape == (n, 2, 2)
    np.testing.assert_allclose(A.sum(1), 1, atol=1e-6)                  # columns are outcome distributions
    assert A[:, 1, 0].max() < 0.2 and A[:, 0, 1].max() < 0.2            # P(1|0) and P(0|1) are small


def test_noisy_local_bures_reproduced(run):
    """The mean local Bures distance between the readout-mitigated test estimates and the labels reproduces that of
    the unmitigated method ("Noisy") in results.csv to 1e-6, and so does every sample."""
    Y = run["labels"]["S_test"][: run["meta"]["n_test"]]
    stored = run["results"][run["results"]["method"] == "Noisy"].sort_values("sample")["local_bures"].to_numpy()
    lb = L.local_bures(run["S_rem"]["test"], Y, 2)
    assert abs(lb.mean() - stored.mean()) < 1e-6
    np.testing.assert_allclose(lb, stored, rtol=0, atol=1e-6)


@pytest.mark.parametrize("method", ["Noisy", "Noisy (raw)", "OLS"])
def test_stored_results_reproduced(run, method):
    """All per-sample errors in results.csv and the predictions in pred.npz are reproduced for the estimates with and
    without readout-error mitigation and for OLS trained on the hardware training circuits."""
    meta = run["meta"]
    Y, Yte = run["labels"]["S_train"][: meta["n_train"]], run["labels"]["S_test"][: meta["n_test"]]
    X, Xte = run["S_rem"]["train"], run["S_rem"]["test"]
    if method == "OLS":
        est = B.ols_per_coord({1: X}, Y, {1: Xte}, (1,))
    else:
        est = Xte if method == "Noisy" else run["S_raw"]["test"]
    df = metrics(method, est, Yte, Xte)
    stored = run["results"][run["results"]["method"] == method].sort_values("sample")
    assert stored["sample"].tolist() == list(range(meta["n_test"]))
    for col in METRICS:
        np.testing.assert_allclose(df[col].to_numpy(), stored[col].to_numpy(), rtol=0, atol=1e-6, err_msg=col)
    np.testing.assert_allclose(est, run["pred"][method.replace(" ", "_")], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(run["pred"]["__ideal__"], Yte)


@pytest.mark.parametrize("depth", sorted(QUENCH_RUNS))
def test_quench_run_reproduced(depth, monkeypatch):
    """For the fixed-parameter quench circuit measured on hardware, the readout-mitigated multinomial resamples of the
    measured frequencies (as in quench_hw.py), OLS trained on the run of the same step count and the exact noiseless
    coordinates reproduce the "Noisy", "OLS" and "ideal" rows of results.csv and the arrays of pred_d<d>.npz."""
    monkeypatch.setattr(analyze_hw, "ROOT", str(HW_DATA))
    qd = HW_DATA / QUENCH
    meta = json.loads((qd / "meta.json").read_text())
    z = np.load(qd / "job0.npz")
    n, ns = len(meta["chain"]), len(meta["settings"])
    freq = np.zeros((ns, 2 ** n))
    for j, (d, s) in enumerate(meta["pairs"]):
        if d == depth:
            freq[s] = z["probs"][j]
    freq /= freq.sum(-1, keepdims=True)
    rng = np.random.default_rng([13, depth])
    draws = np.stack([[rng.multinomial(meta["shots"], f) / meta["shots"] for f in freq] for _ in range(quench_hw.R)])
    noisy = analyze_hw.estimate(draws, n, analyze_hw.confusion_from_cal(z["cal"], n))
    ideal = L.block_paulis(DensityMatrix(L.tfim_circuit(n, depth, quench_hw.TH_H, quench_hw.TH_J)).data, n, 2)
    _, S_rem, _, m = analyze_hw.load_run(QUENCH_RUNS[depth])
    Y = np.load(qd / "labels.npz")[f"S_train_d{depth}"][: m["n_train"]]
    ols = B.ols_per_coord({1: S_rem["train"]}, Y, {1: noisy}, (1,))
    pred = np.load(qd / f"pred_d{depth}.npz")
    stored = pd.read_csv(qd / "results.csv")
    for name, key, S in (("ideal", "__ideal__", ideal[None]), ("Noisy", "Noisy", noisy), ("OLS", "OLS", ols)):
        np.testing.assert_allclose(S, np.reshape(pred[key], S.shape), rtol=0, atol=1e-10, err_msg=name)
        rows = stored[(stored["depth"] == depth) & (stored["method"] == name)].sort_values("draw")
        assert len(rows) == len(S), name
        for col, values in quench_hw.quantities(S).items():
            np.testing.assert_allclose(rows[col].to_numpy(), values, rtol=0, atol=1e-10, err_msg=f"{name} {col}")
