"""BlockGeo-v1, the direct network of blockgeo_qem_fixed.py (squared-error, classical-fidelity and Bures losses, no
backbone or physical parametrization), trained and evaluated on finite-shot block estimates (the BlockGeo-v1
baseline of run_v3.py)."""
import numpy as np


def paper_nn_shots(trS, trP, tr, teS, teP, te, n, variant, seed=0):
    import torch
    from blockgeo_qem_fixed import DatasetPack, predict_model, train_model
    import qemlib as L

    mk = lambda S, P, f: DatasetPack(noisy_p=P, ideal_p=f["P_ideal"], noisy_obs=S.reshape(len(S), -1),
                                     ideal_obs=f["S_ideal"].reshape(len(S), -1))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # The BlockGeo-v1 loss differentiates through eigenvectors (torch.linalg.eigh), which can raise on
    # (near-)degenerate spectra; retry with another seed before giving up.
    for attempt in range(3):
        torch.manual_seed(seed + attempt)
        np.random.seed(seed + attempt)
        try:
            model = train_model(mk(trS, trP, tr), L.contiguous_blocks(n, 2), variant, epochs=100, batch_size=16,
                                lr=1e-3, hidden=256, lambda_mse=1.0, lambda_fid=0.5, lambda_block=0.05,
                                lambda_obs=0.1, device=dev)
            break
        except RuntimeError as err:
            print(f"BlockGeo-v1 training failed (seed {seed + attempt}): {err}", flush=True)
    else:
        return np.full(teS.shape, np.nan)
    _, obs = predict_model(model, mk(teS, teP, te), dev)
    return obs.reshape(teS.shape)
