# Hardware measurements on ibm_fez

This folder holds the measurements of three runs on the IBM Heron processor ibm_fez, made with `hardware/run_hw.py` and `hardware/run_quench_hw.py` through the free IBM Quantum Open Plan. All runs use the 8-qubit path 129–118–109–110–111–98–91–92 with the calibration of 2026-10-17 00:02 (UTC+9), gate twirling and XY4 dynamical decoupling. Every circuit is measured in the nine covering settings, and every run includes two readout-calibration circuits (all qubits in |0⟩, all in |1⟩) with 4000 shots each.

| Folder | Circuits | Shots per setting | QPU time |
|---|---|---|---|
| `ibm_fez_d1_tr100_te30_s1000` | uniform TFIM, one Trotter step, 100 training and 30 test circuits | 1000 | 358 s in 4 jobs |
| `ibm_fez_d2_tr100_te30_s500` | uniform TFIM, two Trotter steps, 100 training and 30 test circuits | 500 | 186 s in 4 jobs |
| `quench_ibm_fez` | one quench circuit with θ_h = π/4 and θ_J = −π/4, one and two Trotter steps | 1000 | 10 s in 1 job |

The circuits of the first two runs are the first 100 training and the first 30 test circuits of the uniform-TFIM ensembles that `src/gen_v3.py --family tfim --n 8` draws, so they can be rebuilt with `qemlib.make_circuits`.

## Files

| File | Content |
|---|---|
| `meta.json` | backend, chain, Trotter step(s), shots, the nine settings, the measured (circuit, setting) pairs, the shuffled submission order (`order`, measurement runs only), the calibration snapshot of the chain (T1, T2, single-qubit and CZ errors, readout errors) and the jobs with their QPU time |
| `job<k>.npz` | `idx`: positions in the list of pairs; `probs`: measured outcome frequencies over the 256 bit strings, one row per pair; `cal`: the two calibration distributions (first job only) |
| `labels.npz` | ideal block coordinates of the circuits, `[circuits, 7 blocks, 15]`. Measurement runs: `S_train`, `S_test` and `S_test_s1` (simulated noisy coordinates of the test circuits, used as a reference by `analyze_hw.py`). Quench run: `S_train_d1` and `S_train_d2` (training labels of the run used at each step count) |
| `results.csv` | errors of every method per test circuit (measurement runs), or the quench quantities per resampled draw (quench run) |
| `pred.npz`, `pred_d<d>.npz` | mitigated block coordinates of every method and the ideal ones (`__ideal__`) |
| `blocks.csv` | site-resolved errors of the quench quantities (quench run) |

Bit strings follow the Qiskit convention: in the index of an outcome, bit k (counted from the least significant bit) is the result of chain qubit k.

## Loading a run

```python
import json
import numpy as np

run = "data/hardware/ibm_fez_d1_tr100_te30_s1000"
meta = json.load(open(f"{run}/meta.json"))
probs = {}
for k in range(len(meta["jobs"])):
    z = np.load(f"{run}/job{k}.npz")
    probs.update(zip(z["idx"].tolist(), z["probs"]))
split, circuit, setting = meta["pairs"][0]      # for example ["train", 0, 0]
frequencies = probs[0]                          # 256 outcome frequencies of that pair
```

`hardware/analyze_hw.py` turns the frequencies into readout-mitigated block coordinates, and the README describes how to rerun the analyses without IBM access.

The measurements were made with IBM Quantum services. The views expressed are those of the authors and do not reflect the official policy or position of IBM or the IBM Quantum team.
