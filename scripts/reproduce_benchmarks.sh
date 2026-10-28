#!/usr/bin/env bash
# Runs every mitigation method on the data of reproduce_data.sh and re-analyses the shipped hardware measurements.
#
# It takes about 40 hours of computing. The times below were measured with one GPU for training. Training on a
# CPU takes longer. Every runner except run_quench.py appends to its CSV file after each depth and skips depths
# that are already there, so the script can be restarted after an interruption.
#
# Environment
#   BLOCKGEO_WORKDIR  directory of the generated data and results (default <repository>/work)
#   BLOCKGEO_HW_DIR   hardware data (default $BLOCKGEO_WORKDIR/hardware, a copy of data/hardware made here so that
#                     the shipped files stay unchanged)
#   GPU_FRAC          optional cap on the fraction of GPU memory per process
# The output is also appended to $BLOCKGEO_WORKDIR/logs/reproduce_benchmarks.log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${BLOCKGEO_WORKDIR:-$REPO/work}"
WORK="${WORK/#\~/$HOME}"
mkdir -p "$WORK/logs"
WORK="$(cd "$WORK" && pwd)"
export BLOCKGEO_WORKDIR="$WORK"
cd "$REPO"
# BlockGeo-QEM selects its training epochs by the held-out loss alone (the default of src/v3.py).
unset BLOCKGEO_TOL

# Default device and data version (FakeSherbrooke, WORK/results_v3). The second device and the 20-qubit chain are
# selected per command below.
unset BLOCKGEO_TAG BLOCKGEO_BACKEND BLOCKGEO_TWOQ BLOCKGEO_CHAIN
TORINO=(BLOCKGEO_BACKEND=torino BLOCKGEO_TWOQ=cz "BLOCKGEO_CHAIN=11,10,9,8,17,27,28,29,36,48" BLOCKGEO_TAG=_v3_torino)
CHAIN20=("BLOCKGEO_CHAIN=33,39,40,41,53,60,59,58,71,77,78,79,91,98,99,100,110,118,119,120")

# The hardware analyses below rewrite results.csv and pred*.npz, so they work on a copy of data/hardware.
export BLOCKGEO_HW_DIR="${BLOCKGEO_HW_DIR:-$WORK/hardware}"
if [ ! -d "$BLOCKGEO_HW_DIR" ]; then
  mkdir -p "$(dirname "$BLOCKGEO_HW_DIR")"
  cp -R data/hardware "$BLOCKGEO_HW_DIR"
fi

exec > >(tee -a "$WORK/logs/reproduce_benchmarks.log") 2>&1
run() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; "$@"; }

# 1. Main grid with the baselines (--part base) and BlockGeo-QEM with its variants and ablations (--part ours) for three
#    circuit families, n = 6, 8, 10, at 1e5, 1e4 and 1e6 shots and in the exact limit (budget 0). About one day.
#    One call covers six depths and takes 6 to 28 min (base) or 3 to 65 min (ours), and 1 to 2 h for ours with
#    uniform TFIM at 1e5.
for n in 6 8 10; do
  for B in 100000 10000 1000000 0; do
    for part in base ours; do
      run python src/run_v3.py --part "$part" --family tfim --n "$n" --budget "$B" --depths 1,2,3,4,6,8
      run python src/run_v3.py --part "$part" --family dtfim --n "$n" --budget "$B" --depths 1,2,3,4,6,8
      run python src/run_v3.py --part "$part" --family random --n "$n" --budget "$B" --depths 4,8,12,16,20,24
    done
  done
done

# 2. The uniform TFIM runs for n = 6, 8 under the file prefixes basenl_ and oursnl_, which the spectral, purity-bin
#    and overlap analyses read. They repeat the base_ and ours_ values of these settings (about 2 h).
for n in 6 8; do
  for B in 100000 10000; do
    run python src/run_v3.py --part base --family tfim --n "$n" --budget "$B" --depths 1,2,3,4,6,8 --tag nl
    run python src/run_v3.py --part ours --family tfim --n "$n" --budget "$B" --depths 1,2,3,4,6,8 --tag nl \
      --methods "BlockGeo-QEM,abl: HS state loss,abl: coordinate head only"
  done
done

# 3. Size transfer to n = 10 (about 1.5 h).
run python src/run_transfer.py --family tfim --src 8 --tgt 10 --budget 100000
run python src/run_transfer.py --family tfim --src 6 --tgt 10 --budget 100000
run python src/run_transfer.py --family tfim --src 8 --tgt 10 --budget 10000

# 4. Weight beta of the coordinate term (about 17 min).
run python src/run_beta.py --family tfim --n 8 --budget 100000 --depths 1,3,6 --betas 0.1,0.3,3,10

# 5. Per-circuit CDR estimate as a second input (about 3 h).
for B in 10000 100000; do
  for n in 6 8 10; do
    run python src/run_cdrbb.py --family tfim --n "$n" --budget "$B" --depths 1,2,3,4,6,8
    run python src/run_cdrbb.py --family random --n "$n" --budget "$B" --depths 4,8,12,16,20,24
  done
  for n in 6 8; do
    run python src/run_cdrbb.py --family dtfim --n "$n" --budget "$B" --depths 1,2,3,4,6,8
  done
done

# 6. Tolerance of the validation check (about 2.5 h).
run python src/run_tol.py --family dtfim --n 8 --budget 100000 --depths 1,2,3,4,6,8
run python src/run_tol.py --family dtfim --n 6 --budget 100000 --depths 1,2,3,4,6,8
run python src/run_tol.py --family random --n 8 --budget 100000 --depths 4,8,12,16,20,24
run python src/run_tol.py --family tfim --n 8 --budget 100000 --depths 1,2,3,4,6,8

# 7. Retraining with two further sets of three seeds (about 1.3 h).
run python src/run_seeds.py --family tfim --n 8 --budget 100000 --depths 1,2,3,4,6,8

# 8. Training-set size and calibration drift, TFIM n = 8 (about 1.5 h).
run python src/run_exp12.py --exp datasize --family tfim --n 8 --budget 100000 --depths 1,3,6
run python src/run_exp12.py --exp drift --family tfim --n 8 --budget 100000 --depths 1,3,6

# 9. 20-qubit chain, with models transferred from n = 8 and 10 (about 3.3 h).
run env "${CHAIN20[@]}" python src/run20.py --budget 100000 --depths 1,2,3
run env "${CHAIN20[@]}" python src/run20.py --budget 10000 --depths 1,2,3

# 10. Quench of a fixed-angle TFIM circuit, n = 8, 1e4 shots, 30 shot draws (about 36 min).
run python src/run_quench.py --n 8 --budget 10000 --depths 1,2,3,4,6,8 --draws 30

# 11. Second device (FakeTorino), uniform TFIM (about 4.5 h).
for n in 6 8 10; do
  for B in 100000 10000 1000000; do
    run env "${TORINO[@]}" python src/run_v3.py --part base --family tfim --n "$n" --budget "$B" --depths 1,2,3,4,6,8
  done
done
for n in 6 10; do
  for B in 100000 10000 1000000; do
    run env "${TORINO[@]}" python src/run_v3.py --part ours --family tfim --n "$n" --budget "$B" --depths 1,2,3,4,6,8 \
      --methods BlockGeo-QEM
  done
done
for B in 10000 1000000; do
  run env "${TORINO[@]}" python src/run_v3.py --part ours --family tfim --n 8 --budget "$B" --depths 1,2,3,4,6,8 \
    --methods BlockGeo-QEM
done
run env "${TORINO[@]}" python src/run_v3.py --part ours --family tfim --n 8 --budget 100000 --depths 1,2,3,4,6,8 \
  --methods "BlockGeo-QEM,abl: HS state loss"

# 12. Hardware data of ibm_fez (no IBM account needed). Readout errors and unmitigated errors, the learned methods
#     trained on the measured training circuits, and the hardware quench (minutes).
run python hardware/analyze_hw.py ibm_fez_d1_tr100_te30_s1000
run python hardware/analyze_hw.py ibm_fez_d2_tr100_te30_s500
run python hardware/compare_hw.py ibm_fez_d1_tr100_te30_s1000
run python hardware/compare_hw.py ibm_fez_d2_tr100_te30_s500
run python hardware/quench_hw.py ibm_fez "1:ibm_fez_d1_tr100_te30_s1000,2:ibm_fez_d2_tr100_te30_s500"

printf '\n[%s] benchmarks finished; results in %s and %s\n' "$(date '+%F %T')" "$WORK" "$BLOCKGEO_HW_DIR"
