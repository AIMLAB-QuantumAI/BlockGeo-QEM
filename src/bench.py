"""Per-sample error metrics of mitigated block Pauli coordinates, shared by all runners."""
import numpy as np
import pandas as pd

import qemlib as L


def metrics(name, S_pred, S_ideal, S_noisy, b=2):
    """Per-sample errors: state level (Bures, trace distance), observable level (Pauli MAE), and nonlinear
    block quantities (purity, Renyi-2 entropy, concurrence) of the nearest density matrices, for b-qubit blocks.
    The concurrence is defined for two-qubit blocks only and is NaN otherwise."""
    diag, off = L.diag_offdiag_masks(b)
    lb = L.local_bures(S_pred, S_ideal, b)
    err = np.abs(S_pred - S_ideal)
    Rp, Ri = L.block_states(S_pred, b), L.block_states(S_ideal, b)
    Pp, Pi = L.purity(Rp), L.purity(Ri)
    return pd.DataFrame({
        "method": name,
        "sample": np.arange(len(S_pred)),
        "local_bures": lb,
        "local_trace": L.local_trace(S_pred, S_ideal, b),
        "mae_diag": err[:, :, diag].mean(axis=(1, 2)),
        "mae_off": err[:, :, off].mean(axis=(1, 2)),
        "err_purity": np.abs(Pp - Pi).mean(1),
        "err_renyi2": np.abs(np.log2(Pp) - np.log2(Pi)).mean(1),
        "err_concurrence": (np.abs(L.concurrence(Rp) - L.concurrence(Ri)).mean(1) if b == 2
                            else np.full(len(S_pred), np.nan)),
    })
