# Reproduction guide

This guide lists, in order, every command that regenerates the simulated data, runs every method and writes the win/tie/loss table, the figures and the printed statistics. The scripts `scripts/reproduce_data.sh`, `scripts/reproduce_benchmarks.sh` and `scripts/reproduce_figures.sh` run the same commands in the same order and also write their output to `logs/` in the working directory.

## Setup

**Paths.** Run every command from the repository root. Generated files go to `BLOCKGEO_WORKDIR` (default `work/` in the repository root), and the paths below are relative to it. Hardware data are read from and written to `BLOCKGEO_HW_DIR` (default `data/hardware`).

**Data version.** Leave `BLOCKGEO_TAG` unset. The default `_v3` selects `feats_v3/` and `results_v3/`. Only the second-device commands set `BLOCKGEO_TAG=_v3_torino`, which selects `feats_v3_torino/` and `results_v3_torino/`, and the three-qubit-block commands set `BLOCKGEO_TAG=_v3b3 BLOCKGEO_B=3`, which selects `feats_v3b3/`, `results_v3b3/` and blocks of three neighbouring qubits. The analysis scripts read `results_v3/`, `results_v3_torino/` and `results_v3b3/` and must run without `BLOCKGEO_TAG`.

**Resources.** The generated data take about 12.0 GB of disk space and the results about 0.2 GB. A data generator needs 3 to 7.5 GB of RAM depending on n and on the number of worker processes, and a training job about 2.5 to 3 GB. A GPU is optional and only speeds up training.

**Resuming.** The generators skip files that already exist. The runners append to their CSV file after every depth and skip depths that are already in it, except `run_quench.py`, which recomputes all depths. To recompute a setting, delete its files first.

**Device and block settings.** The commands below use three shell arrays (bash or zsh).

```bash
TORINO=(BLOCKGEO_BACKEND=torino BLOCKGEO_TWOQ=cz "BLOCKGEO_CHAIN=11,10,9,8,17,27,28,29,36,48" BLOCKGEO_TAG=_v3_torino)
CHAIN20=("BLOCKGEO_CHAIN=33,39,40,41,53,60,59,58,71,77,78,79,91,98,99,100,110,118,119,120")
B3=(BLOCKGEO_TAG=_v3b3 BLOCKGEO_B=3)
```

`TORINO` selects the noise model of the FakeTorino snapshot (CZ gates) on a 10-qubit chain, which `find_chain.py torino cz 10 0.05` finds as the lowest-error path among qubits with readout error at most 5%. `CHAIN20` extends the default FakeSherbrooke chain to 20 qubits, and its first 10 qubits are the default chain. `B3` selects blocks of three neighbouring qubits with their own data and results directories.

## Run times and data size

The times below come from the logs of the original runs on one workstation with 16 GB of RAM and one GPU, with up to two jobs at a time. They are orders of magnitude. CPU-only training takes longer.

| Step | Time per call | Total |
|---|---|---|
| `gen_v3.py`, three families (10 to 12 workers) | 4–8 min (n = 6), 10–16 min (n = 8), about 5 h (n = 10) | about 16 h |
| `gen_v3_trcdr.py` (6 to 10 workers) | 4–7 min (n = 6), 10–16 min (n = 8), about 8.5 h (n = 10) | about 18 h |
| `gen_exp12.py` | | about 6 min |
| `gen20.py`, Trotter steps 1 to 3 (8 workers) | 1.5–4 min per family | about 6 min |
| `gen_v3.py`, second device (6 workers) | uniform TFIM 6 min (n = 6), 35 min (n = 8), 5.6 h (n = 10); disordered TFIM 6 min, 15 min, 5.5 h | about 12.5 h |
| `gen_v3.py`, three-qubit blocks (6 workers) | 13–17 min per family | about 30 min |
| `run_v3.py --part base` | 6–28 min per (family, n, budget) | |
| `run_v3.py --part ours` at 10⁵ shots | uniform TFIM 1–2 h, disordered TFIM 12–36 min, random 7–15 min | |
| `run_v3.py --part ours` at other budgets | uniform TFIM 11–65 min, others 3–25 min | |
| main grid (both parts, all settings) | | about one day |
| `run_v3.py --tag nl` | 5–25 min | about 2 h |
| `run_transfer.py` | 20–46 min | about 1.5 h |
| `run_beta.py` | | about 17 min |
| `run_cdrbb.py` | 2–22 min | about 3 h |
| `run_tol.py` | 7–69 min | about 2.5 h |
| `run_seeds.py` | | about 1.3 h |
| `run_exp12.py` | data size about 70 min, drift about 25 min | about 1.5 h |
| `run20.py` | per budget about 1.6 h (uniform TFIM) and 14–20 min (disordered TFIM) | about 3.9 h |
| `run_quench.py` | | about 36 min |
| second-device runs | base 3–20 min, ours 2–19 min | about 6 h |
| three-qubit-block runs | base and `run_gnn_b3.py` 1.5–3 min, ours 3–35 min | about 1.3 h |
| `run_gnnb.py` | 3–16 min | about 40 min |
| hardware analyses (`compare_hw.py`, `quench_hw.py`) | 3–4 min per call on four CPU cores | about 10 min |
| analyses, table and figures | seconds to 2 min | about 15 min |

In total this is about 96 hours of computing, about 47 h of data generation and about 49 h of runs and analyses. With two jobs in parallel it took 2 to 3 days.

| Directory | Content | Size |
|---|---|---|
| `feats_v3/` | main data (`_train`, `_test`, `_cdr`, `_cliff`) | 4.1 GB |
| `feats_v3/` | near-Clifford training sets (`_trcdr`) | 3.3 GB |
| `feats_v3/` | extra training circuits (`_trainx`) and `drift/` | 0.16 GB |
| `feats_v3_torino/` | second device, both TFIM families | 2.7 GB |
| `feats_v3_n20/` | 20-qubit chain, both TFIM families | 0.04 GB |
| `feats_v3b3/` | three-qubit blocks | 1.6 GB |
| `results_v3/`, `results_v3_torino/`, `results_v3b3/` | per-sample CSV files and predictions | 0.2 GB |

## Conventions

**Circuit families.** `tfim` is the uniform transverse-field Ising model with one field angle and one coupling drawn per circuit, at 1, 2, 3, 4, 6 and 8 Trotter steps. `dtfim` is the disordered model with a field per qubit and a coupling per bond, at the same step counts. `random` are the random circuits of the BlockGeo-v1 code, at depths 4 to 24. Every setting (family, n, depth) has 200 training circuits, 50 test circuits, 24 near-Clifford circuits per test circuit for CDR, and a near-Clifford copy of every training circuit for the Clifford-trained variant.

**Noise and measurement.** Gate errors of the fake backend's calibration on the chosen chain are Pauli-twirled. Readout errors are applied at measurement time and removed by tensored readout-error mitigation (REM) with the calibrated confusion matrices. Every circuit is measured in the 9 covering settings of two-qubit blocks, or in the 27 of three-qubit blocks.

**Shot budgets.** `--budget B` is the total number of shots per circuit. The single-scale protocol measures each setting with B/9 shots. The multi-scale protocol of ZNE, vnCDR, vnOLS and BlockGeo-QEM-ms measures each setting at fold scales 1, 3 and 5 with B/27 shots. For three-qubit blocks the shots per setting are B/27 and B/81. Budget 0 gives exact expectation values, with readout errors applied exactly and removed by REM.

**Methods.** `run_v3.py --part base` runs the unmitigated estimate without and with REM (`Noisy (raw)`, `Noisy`), ZNE with linear, Richardson and exponential extrapolation, idealised PEC with the exact twirled noise, CDR, vnCDR, `CDR-family` (a per-coordinate linear map trained on the near-Clifford copies of the training circuits), OLS, vnOLS, RF, MLP-QEM, GNN-QEM and BlockGeo-v1, a direct network that maps the measured outcome distribution and block coordinates to their ideal values, trained with squared-error, classical-fidelity and Bures losses, without backbone or physical parametrization. `run_v3.py --part ours` runs BlockGeo-QEM and BlockGeo-QEM-ms, and at 10⁵ shots also `BlockGeo-QEM (Clifford-trained)` and the ablations `abl: HS state loss`, `abl: coordinate head only` and `abl: no circuit context`. With three-qubit blocks `--part base` leaves out GNN-QEM and BlockGeo-v1, which read two-qubit blocks, and `run_gnn_b3.py` runs GNN-QEM for three-qubit blocks.

**Result files.** Most runners write one row per method and test circuit to `results_v3/<stem>_samples.csv`, with the columns `method`, `sample`, `local_bures`, `local_trace`, `mae_diag`, `mae_off`, `err_purity`, `err_renyi2`, `err_concurrence`, the fit time (`fit_time_s`) and the setting. `run_beta.py` and the drift study of `run_exp12.py` record no fit time, and the drift study adds the column `drift`. `run_quench.py` writes per-draw chain averages instead (2.10). Most runners also save the predicted block coordinates of every method to `results_v3/pred/<stem>_d<depth>.npz`.

## 1. Data generation

`scripts/reproduce_data.sh` runs these commands, with `WORKERS` worker processes (default 8) for every generator instead of the counts below, which are those of the original runs. The number of worker processes only sets the parallelism and does not change the data.

### 1.1 Main benchmark

```bash
for n in 6 8 10; do python src/gen_v3.py --family tfim --n $n --depths 1,2,3,4,6,8 --workers 12; done
for n in 6 8 10; do python src/gen_v3.py --family dtfim --n $n --depths 1,2,3,4,6,8 --workers 12; done
for n in 6 8 10; do python src/gen_v3.py --family random --n $n --depths 4,8,12,16,20,24 --workers 12; done
```

Writes `feats_v3/{family}_n{n}_d{depth}_{train,test,cdr,cliff}.npz`. The `_train` and `_test` files hold the ideal block coordinates `S_ideal`, the ideal outcome distribution `P_ideal`, the noisy block coordinates `S_s1`, `S_s3`, `S_s5` and the outcome distributions of the 9 settings `Q_s1`, `Q_s3`, `Q_s5` at fold scales 1, 3 and 5, and `_test` also holds the PEC overhead `gamma`. The `_cdr` files hold `S_ideal` and `Q_s1`, `Q_s3`, `Q_s5` of the 24 near-Clifford circuits of every test circuit, and the `_cliff` files `S_ideal` and `Q_s1` of the near-Clifford copy of every training circuit.

### 1.2 Near-Clifford training sets for the CDR-input study

```bash
for n in 6 8 10; do python src/gen_v3_trcdr.py --family tfim --n $n --depths 1,2,3,4,6,8 --workers 10; done
for n in 6 8; do python src/gen_v3_trcdr.py --family dtfim --n $n --depths 1,2,3,4,6,8 --workers 10; done
for n in 6 8 10; do python src/gen_v3_trcdr.py --family random --n $n --depths 4,8,12,16,20,24 --workers 10; done
```

Writes `feats_v3/{family}_n{n}_d{depth}_trcdr.npz` with 24 near-Clifford circuits for every training circuit.

### 1.3 Training-set size and calibration drift

```bash
python src/gen_exp12.py --family tfim --n 8 --depths 1,3,6 --drift 0.7,0.85,1.15,1.3 --K 24 --workers 8
```

Writes `feats_v3/tfim_n8_d{1,3,6}_trainx.npz` (training circuits 200 to 399 of the same ensembles) and `feats_v3/drift/tfim_n8_d{depth}_f{factor}.npz` (test circuits and their CDR circuits with every gate error probability scaled by the factor).

### 1.4 20-qubit chain

```bash
for fam in tfim dtfim; do env "${CHAIN20[@]}" python src/gen20.py --family $fam --n 20 --depths 1,2,3 --workers 8; done
```

Writes `feats_v3_n20/{tfim,dtfim}_n20_d{1,2,3}_{test,train,cdr}.npz` and prints the largest bond dimension of every depth. The data come from the MPDO simulator (`mpdo.py`), and `gen20.py` stops if the simulation of a noisy circuit discards more than 10⁻¹⁰ of the weight. Only fold scale 1 is simulated. More Trotter steps can be requested with `--depths`, but the bond dimension grows quickly and a fourth step takes hours.

### 1.5 Second device

```bash
for n in 6 8 10; do env "${TORINO[@]}" python src/gen_v3.py --family tfim --n $n --depths 1,2,3,4,6,8 --workers 6; done
for n in 6 8 10; do env "${TORINO[@]}" python src/gen_v3.py --family dtfim --n $n --depths 1,2,3,4,6,8 --workers 6; done
```

Writes `feats_v3_torino/{tfim,dtfim}_n{n}_d{depth}_*.npz`.

### 1.6 Three-qubit blocks

```bash
for fam in tfim dtfim; do env "${B3[@]}" python src/gen_v3.py --family $fam --n 8 --depths 1,2,3,4,6,8 --workers 6; done
```

Writes `feats_v3b3/{tfim,dtfim}_n8_d{depth}_*.npz` for the circuits of 1.1, with the 63 Pauli coordinates of each of the six three-qubit blocks and the outcome distributions of the 27 covering settings.

## 2. Simulation runs

`scripts/reproduce_benchmarks.sh` runs these commands.

### 2.1 Main grid

```bash
for n in 6 8 10; do
  for B in 100000 10000 1000000 0; do
    for part in base ours; do
      python src/run_v3.py --part $part --family tfim --n $n --budget $B --depths 1,2,3,4,6,8
      python src/run_v3.py --part $part --family dtfim --n $n --budget $B --depths 1,2,3,4,6,8
      python src/run_v3.py --part $part --family random --n $n --budget $B --depths 4,8,12,16,20,24
    done
  done
done
```

Writes `results_v3/{base,ours}_{family}_n{n}_B{budget}_samples.csv` and `results_v3/pred/{base,ours}_{family}_n{n}_B{budget}_d{depth}.npz`.

### 2.2 Repeated runs for the spectral, purity-bin and overlap analyses

```bash
for n in 6 8; do
  for B in 100000 10000; do
    python src/run_v3.py --part base --family tfim --n $n --budget $B --depths 1,2,3,4,6,8 --tag nl
    python src/run_v3.py --part ours --family tfim --n $n --budget $B --depths 1,2,3,4,6,8 --tag nl \
      --methods "BlockGeo-QEM,abl: HS state loss,abl: coordinate head only"
  done
done
```

Writes `results_v3/basenl_*` and `results_v3/oursnl_*`. These files repeat the uniform TFIM runs of 2.1 for n = 6 and 8 with the same draws and seeds, and `analyze_nl.py`, `paper_figs.py spectral` and `paper_review_checks.py` read them under these prefixes.

### 2.3 Size transfer

```bash
python src/run_transfer.py --family tfim --src 8 --tgt 10 --budget 100000
python src/run_transfer.py --family tfim --src 6 --tgt 10 --budget 100000
python src/run_transfer.py --family tfim --src 8 --tgt 10 --budget 10000
```

Writes `results_v3/transfer_tfim_n{8,6}to10_B{budget}_samples.csv`. BlockGeo-QEM, GNN-QEM and a pooled OLS map are trained at the source size and applied at n = 10.

### 2.4 Weight β of the coordinate term

```bash
python src/run_beta.py --family tfim --n 8 --budget 100000 --depths 1,3,6 --betas 0.1,0.3,3,10
```

Writes `results_v3/beta_tfim_n8_B100000_samples.csv` with methods `beta=<value>`. The value β = 1 is the BlockGeo-QEM row of the main grid.

### 2.5 CDR estimate as a second input

```bash
for B in 10000 100000; do
  for n in 6 8 10; do
    python src/run_cdrbb.py --family tfim --n $n --budget $B --depths 1,2,3,4,6,8
    python src/run_cdrbb.py --family random --n $n --budget $B --depths 4,8,12,16,20,24
  done
  for n in 6 8; do python src/run_cdrbb.py --family dtfim --n $n --budget $B --depths 1,2,3,4,6,8; done
done
```

Writes `results_v3/cdrbb_{family}_n{n}_B{budget}_samples.csv` with CDR, OLS+CDR and BlockGeo-QEM+CDR. It needs the `_trcdr` files of 1.2.

### 2.6 Tolerance of the validation check

```bash
python src/run_tol.py --family dtfim --n 8 --budget 100000 --depths 1,2,3,4,6,8
python src/run_tol.py --family dtfim --n 6 --budget 100000 --depths 1,2,3,4,6,8
python src/run_tol.py --family random --n 8 --budget 100000 --depths 4,8,12,16,20,24
python src/run_tol.py --family tfim --n 8 --budget 100000 --depths 1,2,3,4,6,8
```

Writes `results_v3/tol_{family}_n{n}_B100000_samples.csv` with a stricter check that also requires the held-out coordinate error to stay within 2% of that of the backbone (`BlockGeo-QEM (tol 0.02)`) and with the default rule, the held-out loss alone (`BlockGeo-QEM (no check)`). The column `epochs` holds the selected epoch of each of the three seeds, and -1 means that no epoch qualified.

### 2.7 Training seeds

```bash
python src/run_seeds.py --family tfim --n 8 --budget 100000 --depths 1,2,3,4,6,8
```

Writes `results_v3/seeds_tfim_n8_B100000_samples.csv` with BlockGeo-QEM retrained with seeds 3 to 5 and 6 to 8.

### 2.8 Training-set size and calibration drift

```bash
python src/run_exp12.py --exp datasize --family tfim --n 8 --budget 100000 --depths 1,3,6
python src/run_exp12.py --exp drift --family tfim --n 8 --budget 100000 --depths 1,3,6
```

Writes `results_v3/datasize_tfim_n8_B100000_samples.csv` (learned methods trained on 25 to 400 circuits, column `n_train`) and `results_v3/drift_tfim_n8_B100000_samples.csv` (test data with scaled gate errors, column `drift`).

### 2.9 20-qubit chain

```bash
for fam in tfim dtfim; do
  for B in 100000 10000; do
    env "${CHAIN20[@]}" python src/run20.py --family $fam --budget $B --depths 1,2,3
  done
done
```

Writes `results_v3/n20_{family}_B{budget}_samples.csv`. The in-size learned methods are trained on 200 twenty-qubit circuits, and BlockGeo-QEM, GNN-QEM and pooled OLS are also transferred from n = 8 and n = 10, which needs `feats_v3/{family}_n{8,10}_d{1,2,3}_train.npz` from 1.1. Finite-shot measurements are drawn with the block sampler (`sample_blocks.py`). ZNE and the other multi-noise-level methods are left out, because the folded 20-qubit circuits are not simulated.

### 2.10 Quench of a fixed-angle circuit

```bash
python src/run_quench.py --n 8 --budget 10000 --depths 1,2,3,4,6,8 --draws 30
```

Writes `results_v3/quench_tfim_n8_B10000.csv` (chain averages of ⟨X_i⟩, ⟨Z_i Z_{i+1}⟩, block purity and concurrence for 30 shot draws) and `results_v3/quench_pred_tfim_n8_B10000_d{depth}.npz`. The models are trained on the uniform TFIM training circuits of the same step count.

### 2.11 Second device

```bash
for n in 6 8 10; do
  for B in 100000 10000 1000000; do
    env "${TORINO[@]}" python src/run_v3.py --part base --family tfim --n $n --budget $B --depths 1,2,3,4,6,8
  done
done
for n in 6 10; do
  for B in 100000 10000 1000000; do
    env "${TORINO[@]}" python src/run_v3.py --part ours --family tfim --n $n --budget $B --depths 1,2,3,4,6,8 --methods BlockGeo-QEM
  done
done
for B in 10000 1000000; do
  env "${TORINO[@]}" python src/run_v3.py --part ours --family tfim --n 8 --budget $B --depths 1,2,3,4,6,8 --methods BlockGeo-QEM
done
env "${TORINO[@]}" python src/run_v3.py --part ours --family tfim --n 8 --budget 100000 --depths 1,2,3,4,6,8 \
  --methods "BlockGeo-QEM,abl: HS state loss"
for n in 6 8 10; do
  for B in 100000 10000 1000000; do
    env "${TORINO[@]}" python src/run_v3.py --part base --family dtfim --n $n --budget $B --depths 1,2,3,4,6,8
    env "${TORINO[@]}" python src/run_v3.py --part ours --family dtfim --n $n --budget $B --depths 1,2,3,4,6,8 --methods BlockGeo-QEM
  done
done
```

Writes `results_v3_torino/{base,ours}_{family}_n{n}_B{budget}_samples.csv`. For uniform TFIM with n = 8 at 10⁵ shots, run the command with both methods (`BlockGeo-QEM,abl: HS state loss`) from the start, because the resume logic skips depths that are already in the CSV file.

### 2.12 Three-qubit blocks

```bash
for fam in tfim dtfim; do
  for B in 100000 10000; do
    env "${B3[@]}" python src/run_v3.py --part base --family $fam --n 8 --budget $B --depths 1,2,3,4,6,8
    env "${B3[@]}" python src/run_v3.py --part ours --family $fam --n 8 --budget $B --depths 1,2,3,4,6,8 \
      --methods "BlockGeo-QEM,abl: HS state loss,abl: coordinate head only"
    env "${B3[@]}" python src/run_gnn_b3.py --family $fam --n 8 --budget $B --depths 1,2,3,4,6,8
  done
done
```

Writes `results_v3b3/{base,ours,gnn}_{family}_n8_B{budget}_samples.csv` and the prediction files. The ablations run at 10⁵ shots only. The concurrence is defined for two qubits, so `err_concurrence` is empty for three-qubit blocks.

### 2.13 GNN trunk

```bash
for fam in tfim dtfim; do
  for B in 100000 10000; do
    python src/run_gnnb.py --family $fam --n 8 --budget $B --depths 1,2,3,4,6,8
  done
done
```

Writes `results_v3/gnnb_{family}_n8_B{budget}_samples.csv` (method `GNN-QEM (Bures)`, the selected epochs of the three seeds in column `epochs`) and the prediction files. The model (`gnn_bures.py`) replaces the trunk of BlockGeo-QEM by the message-passing network of GNN-QEM with its node and edge features, and keeps the backbone, the heads, the loss and the training of BlockGeo-QEM.

## 3. Hardware experiment

The measurements are shipped in `data/hardware` together with the ideal labels (`labels.npz`), so the QPU steps are optional. The offline analyses at the end of `scripts/reproduce_benchmarks.sh` work on a copy in `$BLOCKGEO_WORKDIR/hardware` (default `work/hardware`).

### 3.1 Measurement on IBM Quantum (optional)

Save an IBM Quantum account first, as described in the README. The measurement scripts submit jobs only to a free Open Plan instance (`hardware/ibm_guard.py`) and only if the remaining free QPU time covers the job. Point `BLOCKGEO_HW_DIR` to a new directory, since `run_hw.py` skips every job whose file already exists, and `run_quench_hw.py` and `make_labels.py` overwrite `quench_<backend>/` and `labels.npz` without checking.

```bash
python src/gen_v3.py --family tfim --n 8 --depths 1,2 --workers 8          # needed only for the labels; part of 1.1
export BLOCKGEO_HW_DIR="$PWD/work/hardware_new"
python hardware/run_hw.py --backend ibm_fez --chain 129,118,109,110,111,98,91,92 --depth 1 --n-train 100 --n-test 30 --shots 1000
python hardware/run_hw.py --backend ibm_fez --chain 129,118,109,110,111,98,91,92 --depth 2 --n-train 100 --n-test 30 --shots 500
python hardware/run_quench_hw.py --backend ibm_fez --chain 129,118,109,110,111,98,91,92 --depths 1,2 --shots 1000
python hardware/make_labels.py ibm_fez_d1_tr100_te30_s1000
python hardware/make_labels.py ibm_fez_d2_tr100_te30_s500
python hardware/make_labels.py quench_ibm_fez "1:ibm_fez_d1_tr100_te30_s1000,2:ibm_fez_d2_tr100_te30_s500"
```

`run_hw.py` writes `<tag>/meta.json` and `<tag>/job<k>.npz`, with the tag `ibm_fez_d{depth}_tr{n_train}_te{n_test}_s{shots}`. `run_quench_hw.py` writes `quench_ibm_fez/meta.json` and `job0.npz`. `make_labels.py` writes `labels.npz` from the `gen_v3.py` data. The shipped runs used 358 s, 186 s and 10 s of QPU time, with wall times of about 12, 9 and 1 min.

### 3.2 Offline analysis

```bash
mkdir -p work
[ -d work/hardware ] || cp -R data/hardware work/hardware
export BLOCKGEO_HW_DIR="$PWD/work/hardware"
python hardware/analyze_hw.py ibm_fez_d1_tr100_te30_s1000
python hardware/analyze_hw.py ibm_fez_d2_tr100_te30_s500
python hardware/compare_hw.py ibm_fez_d1_tr100_te30_s1000
python hardware/compare_hw.py ibm_fez_d2_tr100_te30_s500
python hardware/quench_hw.py ibm_fez "1:ibm_fez_d1_tr100_te30_s1000,2:ibm_fez_d2_tr100_te30_s500"
```

`analyze_hw.py` prints the readout errors from the calibration circuits, the errors of the unmitigated test-circuit estimates and, as a reference, those of the simulated FakeSherbrooke model for the same circuits. `compare_hw.py` trains OLS, RF, MLP-QEM, GNN-QEM, the HS control and BlockGeo-QEM on the measured training circuits, writes `<tag>/results.csv` and `<tag>/pred.npz`, and prints the error ratio of BlockGeo-QEM to the best baseline with a paired Wilcoxon p-value and the ratio of the HS control to BlockGeo-QEM for every metric. ZNE, CDR and PEC need circuits that were not measured, so they are not part of the hardware comparison. `quench_hw.py` mitigates the measured quench circuit with models trained on the hardware runs of the same step count, writes `quench_ibm_fez/results.csv` and `quench_ibm_fez/pred_d{1,2}.npz`, and prints the chain averages.

## 4. Analyses, table and figures

`scripts/reproduce_figures.sh` runs these commands. Run them without `BLOCKGEO_TAG` and, for the hardware parts, with the same `BLOCKGEO_HW_DIR` as in 3.2. The order matters, because the hardware figure reads the files of `analyze_quench_blocks.py` and the geometry figure reads the CSV file of `paper_review_checks.py purity`.

```bash
python src/analyze_quench_blocks.py sim
python src/analyze_quench_blocks.py hw ibm_fez
python src/paper_review_checks.py purity overlap holm
python src/paper_tables.py
python src/paper_tables.py numbers
python src/paper_tables.py cdr
python src/paper_tables.py marginals
python src/paper_extra.py torino n20 gnnb
python src/paper_b3.py
python src/analyze_nl.py
python src/analyze_v3.py
python src/paper_figs.py main budget geometry spectral robust n20 hardware unstructured
python src/fig_metric_dome.py
python src/test_mpdo.py 8
python src/test_sample_blocks.py
python src/find_chain.py torino cz 10 0.05
python src/find_chain.py sherbrooke ecr 33,39,40,41,53,60,59,58,71,77,78,79,91,98,99,100,110,118,119,120
```

### Files

| File | Content | Written by | Inputs |
|---|---|---|---|
| `results_v3/quench_blocks_tfim_n8_B10000.csv` | RMS and maximum error over blocks of block purity, concurrence and ⟨X_i⟩, simulated quench | `analyze_quench_blocks.py sim` | `quench_pred_tfim_n8_B10000_d*.npz` (2.10) |
| `<hardware dir>/quench_ibm_fez/blocks.csv` | the same for the hardware quench | `analyze_quench_blocks.py hw ibm_fez` | `quench_ibm_fez/pred_d*.npz` (3.2) |
| `results_v3/purity_resolved_B100000.csv` | block Bures distance of BlockGeo-QEM and of the HS-trained model for every block, with its ideal purity | `paper_review_checks.py purity` | prediction files of `oursnl_tfim_n{6,8}`, `ours_tfim_n10`, `ours_random_n10` and `ours_dtfim_n8` at 10⁵ shots (2.1, 2.2) |
| `tables/tab_wtl.tex` | LaTeX rows of the win/tie/loss table, one row per family and n, with the win/tie/loss counts over the six depths and the geometric-mean error ratio of BlockGeo-QEM against the best baseline for the local Bures and trace distances at 10⁴, 10⁵ and 10⁶ shots; the last two blocks are uniform and disordered TFIM on the second device | `paper_tables.py` | `base_*` and `ours_*` of 2.1, `results_v3_torino/` (2.11) |
| `results_v3/compare_BlockGeo-QEM.csv` | best baseline, error ratio, Wilcoxon p-value and outcome for every family, n, budget, depth and metric | `analyze_v3.py` | 2.1 |
| `results_v3b3/compare_b2_b3_{family}_n8.csv` | error ratio of BlockGeo-QEM to the best baseline, the best baseline, the outcome and the ratio of the HS-trained model to BlockGeo-QEM, per block size, budget, depth and metric (uniform and disordered TFIM, n = 8) | `paper_b3.py` | 2.1, 2.2, 2.12 |
| `results_v3b3/purity_resolved_b2_b3_{family}_n8_B100000.csv` | block Bures distance of BlockGeo-QEM and of the HS-trained model for every two- and three-qubit block, with its ideal purity and smallest eigenvalue | `paper_b3.py` | prediction files of 2.1, 2.2 and 2.12 at 10⁵ shots |
| `tables/tab_b3.tex`, `tables/tab_b3_dtfim.tex` | LaTeX rows of the comparison of two- and three-qubit blocks for uniform and disordered TFIM, n = 8: win/tie/loss counts over the six depths, geometric-mean and per-depth error ratio of BlockGeo-QEM against the best baseline for the local Bures and trace distances at 10⁴ and 10⁵ shots | `paper_b3.py` | `compare_b2_b3_*.csv` |
| `tables/tab_b3_gain_both.tex` | LaTeX rows of the number of blocks and the Bures-distance ratio of the HS-trained model to BlockGeo-QEM in bins of the ideal block purity and of the smallest eigenvalue, two- and three-qubit blocks, both TFIM families, 10⁵ shots (from `tab_b3_gain.tex` and `tab_b3_gain_dtfim.tex`) | `paper_b3.py` | `purity_resolved_b2_b3_*.csv` |
| `figures/metric_dome.pdf` | sets of qubit states at Euclidean coordinate distance 0.1 and at Bures distance 0.05 from a grid of states in a section of the Bloch ball, drawn in the section and on the hemisphere that the section becomes under the Bures metric | `fig_metric_dome.py` | none |
| `figures/main_tfim.pdf` | uniform TFIM, local Bures and trace distance against Trotter steps at 10⁵ and 10⁴ shots, n = 6, 8, 10 | `paper_figs.py main` | 2.1 |
| `figures/budget.pdf` | geometric mean over depths of the error ratio of BlockGeo-QEM to the best baseline against the shot budget (10⁴, 10⁵, 10⁶ and exact), per family, for the local Bures and trace distances, with the geometric mean over n = 6, 8, 10 and the range over n | `paper_figs.py budget` | 2.1 |
| `figures/geometry.pdf` | ablations of the state head and the GNN trunk (Bures and trace distance, n = 8, 10⁵ shots), block Bures distance in bins of ideal block purity for the HS-trained and the Bures-trained model, and the β sweep at 1, 3 and 6 steps | `paper_figs.py geometry` | 2.1, 2.13, `purity_resolved_B100000.csv`, 2.4 |
| `figures/spectral.pdf` | errors of block purity, Rényi-2 entropy and concurrence against Trotter steps, TFIM n = 8, 10⁵ shots | `paper_figs.py spectral` | 2.2 |
| `figures/robust.pdf` | size transfer to n = 10, training-set size and calibration drift at three Trotter steps, TFIM, 10⁵ shots | `paper_figs.py robust` | 2.1, 2.3, 2.8 |
| `figures/n20_B100000.pdf` | 20-qubit chain, local Bures and trace distance at Trotter steps 1 to 3, including the transfer from n = 10 | `paper_figs.py n20` | 2.9 |
| `figures/hardware.pdf` | ibm_fez runs (local Bures and trace distance at one and two Trotter steps) and the RMS block-purity error of the quench circuit, simulation lines and hardware markers | `paper_figs.py hardware` | 3.2, both `blocks.csv` files |
| `figures/unstructured_B100000.pdf` | local Bures distance of disordered TFIM and random circuits at 10⁵ shots, n = 6, 8, 10 | `paper_figs.py unstructured` | 2.1 |

### Printed statistics

| Command | Prints |
|---|---|
| `analyze_quench_blocks.py sim`, `hw ibm_fez` | site-resolved RMS errors of block purity, concurrence and ⟨X_i⟩ per method and depth |
| `paper_review_checks.py purity` | mean block Bures distance of both models and their ratio per family and purity bin |
| `paper_review_checks.py overlap` | disagreement of overlapping blocks on their shared qubit, fraction of unphysical estimates and size of the nearest-state projection, per method (TFIM n = 8, 10⁵ shots, depths 1, 3, 6) |
| `paper_review_checks.py holm` | win/tie/loss counts per device at 10⁴ to 10⁶ shots before and after a Holm correction over all paired tests of the table (both devices when the second-device results exist) |
| `paper_tables.py` | the rows of `tab_wtl.tex` |
| `paper_tables.py numbers` | one- and eight-step comparison with GNN-QEM (uniform TFIM), median PEC sampling overhead γ², exact limit, spectral quantities, ablation ratios, size transfer and the number of trainable parameters of the BlockGeo-QEM network for n = 6, 8, 10, 20 |
| `paper_tables.py cdr` | BlockGeo-QEM+CDR against OLS+CDR, BlockGeo-QEM against OLS+CDR, and BlockGeo-QEM+CDR against BlockGeo-QEM, as win/tie/loss counts and geometric-mean ratios |
| `paper_tables.py marginals` | median over test circuits and blocks of the purity and of the smallest eigenvalue of the ideal block marginals, per family, n and depth, and the range of the median purity per family; reads only the `_test` files of 1.1 |
| `paper_extra.py torino` | second device, per family: win/tie/loss counts and ratios for six metrics pooled over n and per n, per-depth ratios and best baselines, and the HS control where it was run |
| `paper_extra.py n20` | 20-qubit chain, per family: win/tie/loss counts and ratios of every BlockGeo-QEM variant, per-depth ratios and outcomes of BlockGeo-QEM trained at n = 20, the mean Bures distance per method and depth, and the errors of the model transferred from n = 10 on the qubits of the 10-qubit training chain and on the added qubits |
| `paper_extra.py gnnb` | GNN trunk (n = 8), ratio of its mean local Bures and trace distances to those of BlockGeo-QEM and ratio of those of GNN-QEM to it, per family, budget and depth, with their ranges |
| `paper_b3.py` | per family, the error ratios and best baselines per block size, budget and depth, the HS control per depth, the median smallest eigenvalue and the mean Bures distance of BlockGeo-QEM per purity bin, and the Bures distances of the unmitigated estimate, GNN-QEM and BlockGeo-QEM, followed by the written table rows |
| `analyze_nl.py` | spectral quantities per setting against the best baseline and the two Euclidean ablations |
| `analyze_v3.py` | win/tie/loss counts per n and budget for uniform TFIM and random circuits |
| `test_mpdo.py 8` | largest deviation between MPDO and density-matrix block coordinates for two test circuits at depths 1 and 3 and fold scales 1, 3, 5 |
| `test_sample_blocks.py` | bias, spread and overlap consistency of the block sampler against global multinomial sampling (n = 8, depth 3, 10⁴ shots, 300 repetitions) |
| `find_chain.py torino cz 10 0.05` | the second-device chain with its two-qubit and readout errors |
| `find_chain.py sherbrooke ecr <chain>` | two-qubit and readout errors of a given chain, here the 20-qubit chain, whose first 10 qubits and 9 couplers are the default chain |

The values of the tolerance, β, seed, training-set size and drift studies are the means of the corresponding CSV files of section 2. `run_tol.py`, `run_beta.py` and `run_seeds.py` also print them per depth, while `run_exp12.py` prints only its progress, so the training-set size and drift values are in its CSV files only. The check of the second-order Bures expansion is `tests/test_geometry.py`. The chain, the shots and the QPU times of the hardware runs are in their `meta.json` files.

## Notes

- **20-qubit study.** The data come from the MPDO simulator and the block sampler instead of density matrices, and the study has no ZNE and no other multi-noise-level method. `test_mpdo.py` and `test_sample_blocks.py` compare both tools with the density-matrix data at n = 8.
- **Second device.** Every command of the second device needs `BLOCKGEO_BACKEND=torino BLOCKGEO_TWOQ=cz "BLOCKGEO_CHAIN=11,10,9,8,17,27,28,29,36,48" BLOCKGEO_TAG=_v3_torino`. `paper_tables.py` and `paper_extra.py torino` read `results_v3_torino/` directly.
- **GPU.** A GPU is optional. The models use it whenever PyTorch finds one, and `GPU_FRAC` caps the memory of each process so that two jobs can share one card. `CUDA_VISIBLE_DEVICES=""` forces the CPU.
- **Reproducibility.** The simulated data are deterministic for fixed library versions. The noise models come from the FakeSherbrooke and FakeTorino snapshots of qiskit-ibm-runtime 0.48.0. The neural methods depend on the device and on library versions through rounding, so expect agreement within the training variability that `run_seeds.py` measures, not bit for bit.
- **Hardware files.** `compare_hw.py`, `quench_hw.py` and `analyze_quench_blocks.py hw` rewrite `results.csv`, `pred*.npz` and `blocks.csv` in `BLOCKGEO_HW_DIR`. Work on a copy to keep the shipped files unchanged.
