"""End-to-end run on a tiny uniform-TFIM data set (n = 4, one Trotter step, 12 training and 4 test circuits, 3
near-Clifford circuits per test circuit): the generator gen_v3.py, the measurement model and baselines of run_v3.py,
BlockGeo-QEM with shortened training, the runner's command line and the statistics of the ideal block marginals
(paper_tables.py marginals). Data and results go to temporary directories."""
import numpy as np
import pandas as pd
import pytest
import torch

import measure_v3 as M
import qemlib as L
import run_v3
import shots as SH
import v3
from bench import metrics

N, DEPTH, N_TRAIN, N_TEST, K_CDR, BUDGET = 4, 1, 12, 4, 3, 10000
NB, NS, NSET, DIM = N - 1, 15, 9, 2 ** N
SCALES = (1, 3, 5)
FAST = dict(max_epochs=60, patience=20, seeds=(0, 1))     # shortened training of BlockGeo-QEM
BASELINES = ["Noisy (raw)", "Noisy", "ZNE-lin", "ZNE-Rich", "ZNE-exp", "PEC", "CDR", "vnCDR", "CDR-family", "OLS",
             "vnOLS", "RF", "MLP-QEM", "GNN-QEM", "BlockGeo-v1"]
METRICS = ["local_bures", "local_trace", "mae_diag", "mae_off", "err_purity", "err_renyi2", "err_concurrence"]


@pytest.fixture(scope="module")
def work(tmp_path_factory, run_python):
    """Working directory (BLOCKGEO_WORKDIR) holding the gen_v3.py output of the tiny data set."""
    work = tmp_path_factory.mktemp("work")
    run_python("src/gen_v3.py", "--family", "tfim", "--n", N, "--depths", DEPTH, "--n_train", N_TRAIN,
               "--n_test", N_TEST, "--K", K_CDR, "--workers", 2, workdir=work)
    return work


def load(work, part):
    return dict(np.load(work / "feats_v3" / f"tfim_n{N}_d{DEPTH}_{part}.npz"))


@pytest.fixture(scope="module")
def data(work):
    """Finite-shot measurements of all circuit sets at 10^4 shots per circuit (run_v3.draws)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_v3, "FEAT_DIR", str(work / "feats_v3"))
        return run_v3.draws("tfim", N, DEPTH, BUDGET)


def test_generator_output(work):
    """gen_v3.py writes the training, test, CDR and near-Clifford sets to BLOCKGEO_WORKDIR/feats_v3 with the
    documented arrays and shapes; coordinates lie in [-1, 1] and outcome distributions are normalised."""
    s, q = {f"S_s{k}": (NB, NS) for k in SCALES}, {f"Q_s{k}": (NSET, DIM) for k in SCALES}
    per_circuit = {"S_ideal": (NB, NS), "P_ideal": (DIM,), **s, **q}
    expected = {
        "train": {k: (N_TRAIN,) + v for k, v in per_circuit.items()},
        "test": {**{k: (N_TEST,) + v for k, v in per_circuit.items()}, "gamma": (N_TEST,)},
        "cdr": {k: (N_TEST, K_CDR) + v for k, v in per_circuit.items() if k == "S_ideal" or k.startswith("Q_")},
        "cliff": {"S_ideal": (N_TRAIN, NB, NS), "Q_s1": (N_TRAIN, NSET, DIM)},
    }
    assert sorted(p.name for p in (work / "feats_v3").iterdir()) == sorted(
        f"tfim_n{N}_d{DEPTH}_{part}.npz" for part in expected)
    for part, shapes in expected.items():
        z = load(work, part)
        assert {k: v.shape for k, v in z.items()} == shapes, part
        for k, v in z.items():
            assert np.isfinite(v).all(), (part, k)
            if k.startswith("S_"):
                assert np.abs(v).max() <= 1 + 1e-12, (part, k)
            if k.startswith(("Q_", "P_")):
                assert v.min() >= 0, (part, k)
                np.testing.assert_allclose(v.sum(-1), 1, atol=1e-5, err_msg=f"{part} {k}")
    assert (load(work, "test")["gamma"] > 1).all()           # PEC sampling overhead


def test_stored_representations_agree(work):
    """The covering-setting estimator applied to the exact setting distributions Q_s<k> returns the stored block
    coordinates S_s<k> of the same state, and the ideal outcome distribution P_ideal gives the Z-type coordinates of
    S_ideal."""
    signs, wcover, _, nb, ns = SH.tables(N)
    z_type = [j for j, st in enumerate(L.local_pauli_strings(2, include_identity=False)) if set(st) <= {"I", "Z"}]
    for part in ("train", "test"):
        z = load(work, part)
        for k in SCALES:
            est = np.einsum("isd,dm,sm->im", z[f"Q_s{k}"].astype(float), signs, wcover).reshape(-1, nb, ns)
            np.testing.assert_allclose(est, z[f"S_s{k}"], rtol=0, atol=1e-6)        # Q is stored as float32
        est = np.einsum("id,dm->im", z["P_ideal"], signs).reshape(-1, nb, ns)
        np.testing.assert_allclose(est[..., z_type], z["S_ideal"][..., z_type], rtol=0, atol=1e-10)


def test_measurement_model(work, data):
    """run_v3.draws measures every set with the documented shapes; the estimates scatter around the exact noisy
    coordinates, and readout-error mitigation removes the readout bias exactly in the infinite-shot limit."""
    shapes = {"tr1": (N_TRAIN, NB, NS), "te1": (N_TEST, NB, NS), "te_raw": (N_TEST, NB, NS),
              "cdr1": (N_TEST, K_CDR, NB, NS), "cl1": (N_TRAIN, NB, NS), "tr_p": (N_TRAIN, DIM), "te_p": (N_TEST, DIM)}
    for key, shape in shapes.items():
        assert data[key].shape == shape, key
        assert np.isfinite(data[key]).all(), key
    for key, shape in (("tr3", (N_TRAIN, NB, NS)), ("te3", (N_TEST, NB, NS)), ("cdr3", (N_TEST, K_CDR, NB, NS))):
        assert sorted(data[key]) == list(SCALES)
        assert all(data[key][k].shape == shape for k in SCALES), key
    np.testing.assert_allclose(data["tr_p"].sum(-1), 1, atol=1e-12)
    te = load(work, "test")
    err = np.abs(data["te1"] - te["S_s1"]).mean()
    assert 1e-3 < err < 0.05
    exact, _ = M.measure(te["Q_s1"], N, None, None)
    np.testing.assert_allclose(exact, te["S_s1"], rtol=0, atol=1e-6)
    raw, _ = M.measure(te["Q_s1"], N, None, None, rem=False)
    assert np.abs(raw - te["S_s1"]).max() > 1e-2


@pytest.fixture(scope="module")
def baselines(data):
    """Method table of the base part of run_v3.py on the tiny data set: name -> function returning the estimates."""
    return dict(run_v3.base_methods(data, N, BUDGET))


def test_baseline_table(baselines):
    """The base part of run_v3.py consists of the baselines tested below, in this order."""
    assert list(baselines) == BASELINES


@pytest.mark.parametrize("name", BASELINES)
def test_baseline(baselines, data, name):
    """Each baseline of the runner returns finite estimates of shape [N_test, n-1, 15], in [-1, 1] except for PEC
    (an unbiased estimator, not clipped), and all its error metrics are finite."""
    pred = baselines[name]()
    assert pred.shape == (N_TEST, NB, NS)
    assert np.isfinite(pred).all()
    if name != "PEC":
        assert np.abs(pred).max() <= 1
    df = metrics(name, pred, data["te"]["S_ideal"], data["te1"])
    assert len(df) == N_TEST
    assert np.isfinite(df[METRICS].to_numpy()).all()


OURS = {                                                   # BlockGeoQEM2 settings of run_v3.py --part ours
    "BlockGeo-QEM": {},
    "BlockGeo-QEM-ms": {"scales": SCALES},
    "abl: HS state loss": {"loss": "hs"},
    "abl: no circuit context": {"ctx": 0},
}


@pytest.fixture(scope="module")
def ours(data):
    """BlockGeo-QEM, its multi-scale version and two ablations, fitted with shortened training; name -> (model, test
    measurements at the model's fold scales)."""
    out = {}
    for name, kw in OURS.items():
        multi = kw.get("scales", (1,)) != (1,)
        tr, te = (data["tr3"], data["te3"]) if multi else ({1: data["tr1"]}, {1: data["te1"]})
        out[name] = v3.BlockGeoQEM2(beta=1.0, **FAST, **kw).fit(tr, data["tr"]["S_ideal"]), te
    return out


def block_states(model, S):
    """Block density matrices [N, n-1, 4, 4] of the state head of a fitted BlockGeoQEM2 (the states whose Pauli
    coordinates predict returns)."""
    X = v3.stack_scales(S, model.scales)
    T = model._T(X, model.backbone.predict(X))
    with torch.no_grad():
        out = torch.stack([net(T["x"], T["base"], T["pos"]) for net in model.nets]).mean(0)
        _, rho = model._heads(out, T)
    return rho.cpu().numpy()


@pytest.mark.parametrize("name", list(OURS))
def test_blockgeo_qem_outputs_are_states(ours, name):
    """The block states of BlockGeo-QEM are Hermitian with unit trace and eigenvalues >= -1e-8, and predict returns
    their Pauli coordinates (shape [N_test, n-1, 15])."""
    model, S = ours[name]
    rho = block_states(model, S)
    assert rho.shape == (N_TEST, NB, 4, 4)
    np.testing.assert_allclose(rho, rho.conj().swapaxes(-1, -2), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho, axis1=-2, axis2=-1), 1, rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(rho).min() >= -1e-8
    pred = model.predict(S)
    assert pred.shape == (N_TEST, NB, NS)
    coords = np.real(np.einsum("nbij,kji->nbk", rho, L.pauli_ops(2)))
    np.testing.assert_allclose(pred, coords, rtol=0, atol=1e-6)            # the coordinates pass through float32


@pytest.mark.parametrize("name", list(OURS))
def test_blockgeo_qem_metrics(ours, data, name):
    """BlockGeo-QEM keeps one network per training seed with a selected epoch within the training budget, and all
    error metrics of its predictions are finite."""
    model, S = ours[name]
    assert len(model.nets) == len(model.best_eps) == len(FAST["seeds"])
    assert all(-1 <= ep < FAST["max_epochs"] for ep in model.best_eps)
    df = metrics(name, model.predict(S), data["te"]["S_ideal"], data["te1"])
    assert len(df) == N_TEST
    assert np.isfinite(df[METRICS].to_numpy()).all()


def test_coordinate_head_ablation(data):
    """The coordinate-head ablation (BlockGeoQEM with output='coords') returns finite coordinates in [-1, 1]."""
    model = v3.BlockGeoQEM(output="coords", loss="hs", ctx=16, **FAST).fit({1: data["tr1"]}, data["tr"]["S_ideal"])
    pred = model.predict({1: data["te1"]})
    assert pred.shape == (N_TEST, NB, NS)
    assert np.isfinite(pred).all() and np.abs(pred).max() <= 1


def test_blockgeo_qem_falls_back_to_projected_backbone(data):
    """When no epoch passes the validation check (tol=-1), BlockGeo-QEM returns the nearest state to the ridge
    backbone (up to the 1e-6 regularisation of its Cholesky factor)."""
    model = v3.BlockGeoQEM2(beta=1.0, tol=-1.0, max_epochs=30, patience=10, seeds=(0,))
    model.fit({1: data["tr1"]}, data["tr"]["S_ideal"])
    assert model.best_eps == [-1]
    base = model.backbone.predict(v3.stack_scales({1: data["te1"]}, (1,)))
    nearest = L.block_states(base)
    expected = np.real(np.einsum("nbij,kji->nbk", nearest, L.pauli_ops(2)))
    np.testing.assert_allclose(model.predict({1: data["te1"]}), expected, rtol=0, atol=1e-5)


def test_runner_command_line(work, data, run_python):
    """run_v3.py --part base writes the per-sample CSV and the predictions to BLOCKGEO_WORKDIR/results_v3, and its
    measurements equal those of run_v3.draws in this process."""
    methods = ["Noisy", "ZNE-Rich", "OLS"]
    run_python("src/run_v3.py", "--part", "base", "--family", "tfim", "--n", N, "--budget", BUDGET, "--depths", DEPTH,
               "--methods", ",".join(methods), workdir=work)
    res = work / "results_v3"
    df = pd.read_csv(res / f"base_tfim_n{N}_B{BUDGET}_samples.csv")
    assert df["method"].unique().tolist() == methods
    assert len(df) == len(methods) * N_TEST
    assert np.isfinite(df[METRICS].to_numpy()).all()
    settings = set(df[["family", "n", "depth", "budget"]].itertuples(index=False, name=None))
    assert settings == {("tfim", N, DEPTH, BUDGET)}
    pred = dict(np.load(res / "pred" / f"base_tfim_n{N}_B{BUDGET}_d{DEPTH}.npz"))
    assert sorted(pred) == sorted(["__ideal__"] + methods)
    np.testing.assert_array_equal(pred["__ideal__"], data["te"]["S_ideal"])
    np.testing.assert_allclose(pred["Noisy"], data["te1"], rtol=0, atol=1e-12)


def test_marginal_statistics(work, run_python, monkeypatch):
    """paper_tables.py marginals: the median purity and the median smallest eigenvalue of the ideal block states of
    the test circuits agree with those of the density matrices that qemlib builds from the coordinates."""
    import paper_tables

    monkeypatch.setattr(paper_tables.paths, "FEAT_DIR", str(work / "feats_v3"))
    t = paper_tables.marginals()
    assert t[["family", "n", "depth"]].values.tolist() == [["tfim", N, DEPTH]]
    R = L.block_states(load(work, "test")["S_ideal"])                    # [N_TEST, NB, 4, 4]
    assert t.purity[0] == pytest.approx(np.median(L.purity(R)), abs=1e-12)
    assert t.lambda_min[0] == pytest.approx(np.median(np.linalg.eigvalsh(R)[..., 0]), abs=1e-12)
    assert 0.25 <= t.purity[0] <= 1 and -1e-12 <= t.lambda_min[0] <= 0.25
    out = run_python("src/paper_tables.py", "marginals", workdir=work).stdout
    assert "Uniform TFIM" in out and f"{t.purity[0]:.4f}" in out
