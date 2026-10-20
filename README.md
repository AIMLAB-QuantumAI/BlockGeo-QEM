# BlockGeo-QEM

BlockGeo-QEM is a quantum error-mitigation method in which a neural network maps the measured two-qubit block marginals of a qubit chain to physical block density matrices and is trained with the Bures distance.

## Authors

Jioh Lee (Korea University), Sung Whan Yoon (UNIST) and Joongheon Kim (Korea University).

## Overview

On an n-qubit chain, every pair of neighbouring qubits (k, k+1) forms a block whose 15 Pauli coordinates are estimated from nine covering measurement settings. BlockGeo-QEM starts from a per-coordinate ridge regression of these noisy coordinates, and a neural network then refines the coordinates and corrects the Cholesky factor A of their nearest density matrix, so the output ρ = AA†/Tr(AA†) is always a valid state. This state output is trained with the squared Bures distance to the ideal block state plus a weighted coordinate error. Training epochs are chosen on held-out training circuits among those whose coordinate error stays within 2% of that of the projected linear estimate, and the method falls back to the projected linear estimate when no epoch qualifies. This repository contains the simulation pipeline, BlockGeo-QEM and the baselines, the scripts that write the win/tie/loss table, the figures and the printed statistics of every study, and the measurements of a hardware experiment on ibm_fez.

## Repository layout

```text
BlockGeo-QEM/
├── src/                           simulation, models, baselines and analyses (flat; run as python src/<script>.py)
│   ├── paths.py                   locations of generated data, results, figures, tables and hardware data
│   ├── qemlib.py                  circuit families, block Pauli coordinates, Bures and trace distances
│   ├── noise_layout.py            device chain, Pauli-twirled noise model of a fake backend, PEC overheads
│   ├── v3.py                      BlockGeo-QEM (class BlockGeoQEM2) and its ablations
│   ├── baselines.py               ZNE, per-coordinate linear regression, random forest, MLP-QEM
│   ├── baselines_v3.py            CDR, vnCDR, idealised PEC, GNN-QEM
│   ├── measure_v3.py              finite-shot measurement with readout errors and readout-error mitigation
│   ├── shots.py, readout.py       block Pauli estimates from the covering settings, tensored readout errors and REM
│   ├── geometry.py                Bures metric in Pauli coordinates
│   ├── bench.py, bench_shots.py   error metrics of every runner, runner of the BlockGeo-v1 baseline
│   ├── mpdo.py, sample_blocks.py  20-qubit simulator (matrix-product density operator) and block sampler
│   ├── gen_*.py, gen20.py         data generators
│   ├── run_*.py, run20.py         runners of the studies
│   ├── analyze_*.py, paper_*.py   analyses, the win/tie/loss table and the figures
│   ├── find_chain.py              lowest-error chain on the coupling graph of a fake backend
│   ├── test_mpdo.py, test_sample_blocks.py   checks of the 20-qubit simulator and the block sampler
│   └── blockgeo_qem_fixed.py      vendored code of BlockGeo-v1 (the earlier workshop model) and simulation primitives
├── hardware/                      IBM Quantum measurement, labels and analysis scripts
├── data/hardware/                 measurements of three runs on ibm_fez
├── scripts/                       reproduce_data.sh, reproduce_benchmarks.sh, reproduce_figures.sh
├── tests/                         fast pytest tests
├── docs/reproduce.md              full reproduction guide
├── examples/quickstart.ipynb      small end-to-end example with outputs
├── .github/workflows/tests.yml    continuous integration (pytest on push and pull request)
├── requirements.txt               pinned pip requirements
├── environment.yml                conda environment with the same pins
├── pytest.ini                     pytest settings (collects tests/ only)
├── CHANGELOG.md                   changes between versions
├── CITATION.cff
├── CONTRIBUTING.md                how to report problems and propose changes
└── LICENSE
```

## Installation

The code is tested with Python 3.10 on Linux. The data generators use the `fork` start method of `multiprocessing`, which Windows does not provide, so on Windows use WSL. macOS has not been tested.

With conda, create and activate the environment.

```bash
git clone https://github.com/AIMLAB-QuantumAI/BlockGeo-QEM.git
cd BlockGeo-QEM
conda env create -f environment.yml
conda activate blockgeo-qem
```

With a virtual environment, install the same pinned packages with pip.

```bash
git clone https://github.com/AIMLAB-QuantumAI/BlockGeo-QEM.git
cd BlockGeo-QEM
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Both install the CPU build of PyTorch 2.11.0. A GPU is optional and only speeds up training. To use one, replace PyTorch by the CUDA build of the same version. The commands below install the build for CUDA 13.0; pick the CUDA version that matches your driver on [pytorch.org](https://pytorch.org).

```bash
pip uninstall -y torch
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

The models run on the GPU whenever PyTorch finds one. `CUDA_VISIBLE_DEVICES=""` forces the CPU, and `GPU_FRAC` caps the GPU memory of each process.

## Quick start

The commands below run the tests, simulate a small data set and run every method on it. They took about one minute on four CPU cores. All data and results go to `BLOCKGEO_WORKDIR`, here `work/quickstart`. Apart from it, only the Python and pytest caches appear in the repository, and `.gitignore` excludes all of them. Run the commands from the repository root with the environment activated.

```bash
export BLOCKGEO_WORKDIR="$PWD/work/quickstart"
python -m pytest -q
python src/gen_v3.py --family tfim --n 4 --depths 1 --n_train 12 --n_test 4 --K 3 --workers 2
python src/run_v3.py --part base --family tfim --n 4 --budget 10000 --depths 1
python src/run_v3.py --part ours --family tfim --n 4 --budget 10000 --depths 1
ls "$BLOCKGEO_WORKDIR/results_v3"
```

`gen_v3.py` simulates 12 training and 4 test circuits of the uniform transverse-field Ising (TFIM) family on a 4-qubit chain with one Trotter step, under the noise model of the FakeSherbrooke snapshot, together with 3 near-Clifford circuits per test circuit for CDR. It writes `feats_v3/tfim_n4_d1_{train,test,cdr,cliff}.npz`. The first `run_v3.py` call runs the baselines with 10⁴ shots per circuit, and the second trains BlockGeo-QEM and its multi-scale variant BlockGeo-QEM-ms. Each call prints the mean errors of every method and writes

- `results_v3/base_tfim_n4_B10000_samples.csv` and `results_v3/ours_tfim_n4_B10000_samples.csv`, with one row per method and test circuit and the columns `local_bures` (RMS block Bures distance), `local_trace` (mean block trace distance), `mae_diag` and `mae_off` (mean absolute errors of the Z-type and of the other Pauli coordinates), and `err_purity`, `err_renyi2` and `err_concurrence` (errors of block purity, Rényi-2 entropy and concurrence),
- `results_v3/pred/base_tfim_n4_B10000_d1.npz` and `results_v3/pred/ours_tfim_n4_B10000_d1.npz`, with the mitigated block coordinates of every method and the ideal ones (`__ideal__`); keys are the method names with spaces replaced by `_` and colons removed, e.g. `Noisy_(raw)`.

With 12 training circuits the errors say nothing about the methods. The quick start only checks that the installation works.

The notebook `examples/quickstart.ipynb` runs the same steps with 100 training circuits in about a minute on a CPU. It compares the unmitigated estimates, OLS, GNN-QEM and BlockGeo-QEM, and counts the blocks whose mitigated coordinates do not belong to a density matrix.

## Reproducing the results

Three scripts run the full pipeline in order. They can be started from any directory and write everything to `BLOCKGEO_WORKDIR` (default `work/`). The generators skip finished files and the runners skip finished depths, so a rerun after an interruption continues where it stopped; only `run_quench.py` and the analyses start again.

```bash
scripts/reproduce_data.sh         # simulated circuit data, about 40 h and 9.0 GB
scripts/reproduce_benchmarks.sh   # every method and study, about 45 h
scripts/reproduce_figures.sh      # win/tie/loss table, figures and printed statistics, about 15 min
```

`reproduce_data.sh` runs `WORKERS` simulations in parallel (default 8), and its memory use grows with that number. The scripts log their output to `logs/` in `BLOCKGEO_WORKDIR`.

The table and the figures are written by the analysis scripts in `src/`. Paths are relative to `BLOCKGEO_WORKDIR`.

| Output | Content | Analysis | Runs it needs |
|---|---|---|---|
| `tables/tab_wtl.tex` | Win/tie/loss counts over the six depths and geometric-mean error ratio of BlockGeo-QEM against the best baseline, for the local Bures and trace distances at 10⁴, 10⁵ and 10⁶ shots, every circuit family and n = 6, 8, 10, and uniform TFIM on the second device | `paper_tables.py` | main grid, second device |
| `figures/main_tfim.pdf` | Uniform TFIM, local Bures and trace distance against Trotter steps at 10⁵ shots and Bures distance at 10⁴ shots, n = 6, 8, 10 | `paper_figs.py main` | main grid |
| `figures/unstructured_B100000.pdf` | Local Bures distance of disordered TFIM and random circuits at 10⁵ shots | `paper_figs.py unstructured` | main grid |
| `figures/geometry.pdf` | Ablations of the state head, block Bures distance in bins of ideal block purity for the Hilbert-Schmidt-trained and the Bures-trained model, and the sweep of the weight β (TFIM n = 8, 10⁵ shots) | `paper_review_checks.py purity`, then `paper_figs.py geometry` | main grid, `--tag nl` runs, `run_beta.py` |
| `figures/spectral.pdf` | Errors of block purity, Rényi-2 entropy and concurrence against Trotter steps (TFIM n = 8, 10⁵ shots) | `paper_figs.py spectral` | `--tag nl` runs |
| `figures/robust.pdf` | Size transfer to n = 10, training-set size and calibration drift (TFIM, 10⁵ shots, the last two at three Trotter steps) | `paper_figs.py robust` | `run_transfer.py`, `run_exp12.py`, main grid |
| `figures/n20_B100000.pdf` | 20-qubit chain, local Bures and trace distance for Trotter steps 1 to 3, including models transferred from n = 10 | `paper_figs.py n20` | `gen20.py`, `run20.py` |
| `figures/hardware.pdf` | ibm_fez runs (local Bures and trace distance of the unmitigated estimate and the learned methods at one and two Trotter steps) and the RMS block-purity error of the quench circuit in simulation and on hardware | `analyze_quench_blocks.py sim` and `hw ibm_fez`, then `paper_figs.py hardware` | `hardware/compare_hw.py`, `hardware/quench_hw.py`, `run_quench.py` |

The other studies are reported as printed statistics or as CSV files of the runners.

| Study | Runs | Analysis |
|---|---|---|
| Purity and smallest eigenvalue of the ideal block marginals (median over test circuits and blocks) | `gen_v3.py` | `paper_tables.py marginals` |
| Main benchmark (uniform TFIM, disordered TFIM, random circuits; n = 6, 8, 10; 10⁴, 10⁵, 10⁶ shots and exact) | `gen_v3.py`, `run_v3.py --part base` and `--part ours` | `paper_tables.py` and `paper_tables.py numbers`, per-setting comparisons in `results_v3/compare_BlockGeo-QEM.csv` from `analyze_v3.py` |
| Holm correction of the win/tie/loss tests | main grid | `paper_review_checks.py holm` |
| Ablations, exact limit, PEC sampling overhead, size transfer, network size | `run_v3.py --part ours`, `run_transfer.py` | `paper_tables.py numbers` |
| Second-order expansion of the Bures distance (metric G in Pauli coordinates against the exact distance) | none | `tests/test_geometry.py` |
| Spectral quantities | `run_v3.py --tag nl` | `analyze_nl.py`, `paper_tables.py numbers` |
| Block Bures distance by ideal block purity | `run_v3.py` (prediction files) | `paper_review_checks.py purity` |
| Overlapping blocks and the nearest-state projection | `run_v3.py --tag nl` | `paper_review_checks.py overlap` |
| CDR estimate as a second input | `gen_v3_trcdr.py`, `run_cdrbb.py` | `paper_tables.py cdr` |
| Tolerance of the validation check | `run_tol.py` | `results_v3/tol_*_samples.csv` (selected epochs in column `epochs`) |
| Weight β, training-set size, calibration drift | `run_beta.py`, `gen_exp12.py`, `run_exp12.py` | `results_v3/beta_*`, `datasize_*` and `drift_*` CSV files |
| Training seeds | `run_seeds.py` | `results_v3/seeds_tfim_n8_B100000_samples.csv` |
| Second device (FakeTorino) | `gen_v3.py` and `run_v3.py` with the FakeTorino settings | `paper_extra.py torino` |
| 20-qubit chain (MPDO simulator) | `gen20.py`, `run20.py` | `paper_extra.py n20` |
| Quench of a fixed-angle TFIM circuit | `run_quench.py`, `hardware/quench_hw.py` | printed chain averages, `analyze_quench_blocks.py` |
| Hardware experiment on ibm_fez | `hardware/compare_hw.py`, `hardware/analyze_hw.py` | printed error ratios, Wilcoxon p-values and readout errors |
| Checks of the 20-qubit simulator and the block sampler | `test_mpdo.py`, `test_sample_blocks.py` | printed (`gen20.py` prints the bond dimensions) |
| Chain of the second device | `find_chain.py torino cz 10 0.05` | printed |
| Two-qubit and readout errors of the FakeSherbrooke chain and of its 20-qubit extension | `find_chain.py sherbrooke ecr` with the 20-qubit chain | printed |

[docs/reproduce.md](docs/reproduce.md) lists every command in order with its outputs, the size of the generated data (about 9.0 GB, plus 0.2 GB of results) and the run times.

### Reproducibility

The simulated data are deterministic. The noise models come from the FakeSherbrooke (1.6.58) and FakeTorino (1.0.54) calibration snapshots in qiskit-ibm-runtime 0.48.0, and another version of that package can change them. Shot sampling, data splits and training seeds are fixed, so on the CPU a rerun with the same library versions repeats the values up to rounding. GPU training is not bitwise deterministic. The neural methods also change with the device (CPU or GPU) and with library versions, because rounding changes their training trajectories. In this code the nearest-state projection inside BlockGeo-QEM (`v3.project_states_t`) runs on the CPU, and earlier runs that computed it on a GPU differ from it by rounding. Expect reruns to agree with earlier results within the training variability that `run_seeds.py` measures, not bit for bit.

## Hardware experiment

### Data

`data/hardware` holds three runs on the 8-qubit chain 129, 118, 109, 110, 111, 98, 91, 92 of ibm_fez (logical qubit q is the q-th entry). Every circuit was measured in the nine covering settings, and every run adds two readout-calibration circuits (all qubits in |0⟩ and all in |1⟩, 4000 shots each). The Sampler used gate twirling and XY4 dynamical decoupling.

| Directory | Circuits | Shots per setting |
|---|---|---|
| `ibm_fez_d1_tr100_te30_s1000` | first 100 training and 30 test circuits of the uniform TFIM ensembles for n = 8, one Trotter step | 1000 |
| `ibm_fez_d2_tr100_te30_s500` | first 100 training and 30 test circuits of the uniform TFIM ensembles for n = 8, two Trotter steps | 500 |
| `quench_ibm_fez` | fixed-angle quench circuit (θ_h = π/4, θ_J = −π/4) at one and two Trotter steps | 1000 |

The raw measurements are `meta.json` and `job*.npz`. The other files are derived from them and from the ideal values, and the analysis scripts rewrite them.

| File | Written by | Content |
|---|---|---|
| `meta.json` | `run_hw.py`, `run_quench_hw.py` | `backend`, `chain`, `depth`, `n_train` and `n_test` (for the quench `depths`, `theta_h` and `theta_J` instead), `shots` per setting and circuit, `cal_shots`, `settings` (nine strings whose character q is the basis of qubit q), `pairs` (the measured `[split, circuit, setting]` triples, `[depth, setting]` for the quench), `order` (the shuffled order in which the pairs were submitted; `run_hw.py` runs only), `calibration` (`cq` with T1 and T2 in µs, sx error and readout error per qubit, `ce` with the CZ error per coupler, `calibrated` with the time of the device calibration) and `jobs` (IBM job id, number of PUBs, charged QPU seconds `charge_s`, wall time and timestamps of every job) |
| `job<k>.npz` | `run_hw.py`, `run_quench_hw.py` | `idx` (m indices into `pairs`), `probs` (m × 256 measured outcome frequencies, indexed by the integer value of the bit string with qubit 0 as the least significant bit) and `cal` (2 × 256 outcome frequencies of the calibration circuits in `job0.npz`, an empty array in the other job files) |
| `labels.npz` | `make_labels.py` | ideal block coordinates `S_train` (100 × 7 × 15) and `S_test` (30 × 7 × 15), and `S_test_s1`, the noisy coordinates of the test circuits under the simulated FakeSherbrooke model; in `quench_ibm_fez` the training labels `S_train_d1` and `S_train_d2` |
| `results.csv` | `compare_hw.py` | one row per method and test circuit with the error columns of the simulated runs |
| `pred.npz` | `compare_hw.py` | mitigated block coordinates (30 × 7 × 15) of every method and `__ideal__`; keys are method names with spaces replaced by `_` and colons removed |
| `quench_ibm_fez/results.csv` | `quench_hw.py` | chain averages of ⟨X_i⟩, ⟨Z_i Z_{i+1}⟩, block purity and concurrence for every method and each of 30 resamples of the measured frequencies, and the ideal values |
| `quench_ibm_fez/pred_d<d>.npz` | `quench_hw.py` | mitigated block coordinates (30 resamples × 7 × 15) of every method and `__ideal__` (7 × 15) |
| `quench_ibm_fez/blocks.csv` | `src/analyze_quench_blocks.py hw ibm_fez` | RMS and maximum error over blocks of block purity, concurrence and ⟨X_i⟩ for every method and depth |

Block coordinates are the 15 Pauli expectation values Tr(ρ_k P) of the 7 blocks (k, k+1), in the order of `qemlib.local_pauli_strings(2, include_identity=False)`, i.e. IX, IY, IZ, XI, XX, ..., ZZ as Qiskit labels whose right letter acts on qubit k. The three runs used 358 s, 186 s and 10 s of QPU time (sum of `charge_s`).

### Re-analysis without IBM access

The analyses need neither an IBM account nor simulated data, because `labels.npz` holds the ideal values. They rewrite the derived files, so the commands below work on a copy.

```bash
mkdir -p work
[ -d work/hardware ] || cp -R data/hardware work/hardware
export BLOCKGEO_HW_DIR="$PWD/work/hardware"
python hardware/analyze_hw.py ibm_fez_d1_tr100_te30_s1000    # readout errors, unmitigated errors, simulated reference
python hardware/analyze_hw.py ibm_fez_d2_tr100_te30_s500
python hardware/compare_hw.py ibm_fez_d1_tr100_te30_s1000    # learned methods trained on the measured training circuits
python hardware/compare_hw.py ibm_fez_d2_tr100_te30_s500
python hardware/quench_hw.py ibm_fez "1:ibm_fez_d1_tr100_te30_s1000,2:ibm_fez_d2_tr100_te30_s500"
python src/analyze_quench_blocks.py hw ibm_fez
```

`compare_hw.py` prints the error ratio of BlockGeo-QEM to the best baseline with a paired Wilcoxon p-value for every metric, and `quench_hw.py` prints the chain averages of every method.

### Running on IBM Quantum

The two measurement scripts, `hardware/run_hw.py` and `hardware/run_quench_hw.py`, reach IBM Quantum only through `hardware/ibm_guard.py`. Its `open_service` loads the account saved with `QiskitRuntimeService.save_account`, refuses to run if the account has any instance whose pricing type is not free, requires exactly one free Open Plan instance and binds the service to that instance. Before every job the measurement scripts check again that the active instance is that free Open Plan instance. They also estimate the QPU time of the job (250 µs per shot plus 20%, plus 5 s) and stop unless the remaining free time (`remaining_seconds`, from `service.usage()`) exceeds it by at least 5 s.

Save your own account once, in an interactive Python session in a terminal, so that the API key is not stored in a file of the repository or in the shell history.

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<your IBM Cloud API key>", set_as_default=True)
```

qiskit-ibm-runtime stores the account in `~/.qiskit/qiskit-ibm.json`. Never commit that file or an API key. `.gitignore` excludes `qiskit-ibm.json`, `.env` and `*.pem`.

A new measurement needs the ideal labels of the measured circuits, which come from the uniform TFIM data of `gen_v3.py`. `run_hw.py` saves every job to `job<k>.npz` as soon as it finishes and skips jobs whose file exists, so an interrupted run resumes. With the default `BLOCKGEO_HW_DIR` it would find the shipped files and skip every job, and `run_quench_hw.py` and `make_labels.py`, which skip nothing, would overwrite the shipped `quench_ibm_fez` measurement and `labels.npz` files. A new measurement therefore always needs a new `BLOCKGEO_HW_DIR`.

```bash
python src/gen_v3.py --family tfim --n 8 --depths 1,2 --workers 8
export BLOCKGEO_HW_DIR="$PWD/work/hardware_new"
python hardware/run_hw.py --backend ibm_fez --chain 129,118,109,110,111,98,91,92 --depth 1 --n-train 100 --n-test 30 --shots 1000
python hardware/run_hw.py --backend ibm_fez --chain 129,118,109,110,111,98,91,92 --depth 2 --n-train 100 --n-test 30 --shots 500
python hardware/run_quench_hw.py --backend ibm_fez --chain 129,118,109,110,111,98,91,92 --depths 1,2 --shots 1000
python hardware/make_labels.py ibm_fez_d1_tr100_te30_s1000
python hardware/make_labels.py ibm_fez_d2_tr100_te30_s500
python hardware/make_labels.py quench_ibm_fez "1:ibm_fez_d1_tr100_te30_s1000,2:ibm_fez_d2_tr100_te30_s500"
```

The analyses then follow as in the previous section. The chain above was chosen from the live calibration of ibm_fez on 17 October 2026, which `meta.json` records. Choose a chain from the current calibration of your device, since `find_chain.py` only searches the fake-backend snapshots.

## Environment variables

These are the variables that the code in `src/` and `hardware/` reads.

| Variable | Read in | Default | Effect |
|---|---|---|---|
| `BLOCKGEO_WORKDIR` | `src/paths.py` | `<repository>/work` | directory of all generated data, results, figures, tables and logs |
| `BLOCKGEO_HW_DIR` | `src/paths.py` | `<repository>/data/hardware` | hardware data read and written by `hardware/*.py`, `analyze_quench_blocks.py hw` and `paper_figs.py hardware` |
| `BLOCKGEO_TAG` | `src/paths.py` | `_v3` | suffix of the data and results directories (`feats<TAG>`, `results<TAG>`); `_v3_torino` for the second device. Leave it unset for the analysis scripts. |
| `BLOCKGEO_BACKEND` | `src/noise_layout.py` | `sherbrooke` | fake backend whose calibration defines the noise model (`torino` for the second device) |
| `BLOCKGEO_TWOQ` | `src/noise_layout.py` | `ecr` | native two-qubit gate of that backend (`cz` for the second device) |
| `BLOCKGEO_CHAIN` | `src/noise_layout.py` | `33,39,40,41,53,60,59,58,71,77` | physical qubits of the chain; a run with n qubits uses the first n |
| `GPU_FRAC` | `src/v3.py` | unset | if set, the fraction of GPU memory one process may allocate |
| `OMP_NUM_THREADS` | `src/gen_v3.py`, `gen_v3_trcdr.py`, `gen_exp12.py`, `gen20.py` | `1` if unset | threads of the linear-algebra libraries in the data generators, which set it before numpy is imported |

Importing `src/blockgeo_qem_fixed.py` (through `qemlib.py`) sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 4 unless they are already set, so values chosen by the user are kept. The second device uses `BLOCKGEO_BACKEND=torino BLOCKGEO_TWOQ=cz BLOCKGEO_CHAIN=11,10,9,8,17,27,28,29,36,48 BLOCKGEO_TAG=_v3_torino`, and the 20-qubit study uses `BLOCKGEO_CHAIN=33,39,40,41,53,60,59,58,71,77,78,79,91,98,99,100,110,118,119,120`.

## Citation

If you use this code, please cite it using [CITATION.cff](CITATION.cff).

```bibtex
@software{lee2026blockgeoqem,
  author  = {Lee, Jioh and Yoon, Sung Whan and Kim, Joongheon},
  title   = {{BlockGeo-QEM}},
  year    = {2026},
  month   = oct,
  version = {0.1.0},
  url     = {https://github.com/AIMLAB-QuantumAI/BlockGeo-QEM}
}
```

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgments

We acknowledge the use of IBM Quantum services for this work. The views expressed are those of the authors and do not reflect the official policy or position of IBM or the IBM Quantum team. This research was supported by IITP (Institute for Information & Communications Technology Planning & Evaluation) funded by MSIT (Ministry of Science and ICT), Korea, (RS-2024-00439803, SW Star Lab) for Quantum AI Empowered Second-Life Platform Technology.

## Contact

Questions and bug reports are welcome as GitHub issues. `CONTRIBUTING.md` describes how to report a problem and how to propose a change through a pull request.
