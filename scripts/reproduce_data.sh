#!/usr/bin/env bash
# Generates the simulated circuit data of every study (density-matrix simulation with the noise models of the
# FakeSherbrooke and FakeTorino snapshots, and the MPDO simulator for the 20-qubit chain).
#
# It takes about 40 hours of computing and 9.0 GB of disk space. The times below were measured with 6 to 12
# worker processes. The generators skip files that already exist, so the script can be restarted after an
# interruption. The worker count does not change the data.
#
# Environment
#   BLOCKGEO_WORKDIR  directory for all generated data (default <repository>/work)
#   WORKERS           worker processes per generator (default 8). Memory grows with the number of workers; the
#                     n = 10 runs used 6 to 7.5 GB of RAM with 10 to 12 workers.
# The output is also appended to $BLOCKGEO_WORKDIR/logs/reproduce_data.log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${BLOCKGEO_WORKDIR:-$REPO/work}"
WORK="${WORK/#\~/$HOME}"
mkdir -p "$WORK/logs"
WORK="$(cd "$WORK" && pwd)"
export BLOCKGEO_WORKDIR="$WORK"
cd "$REPO"
WORKERS="${WORKERS:-8}"

# Default device and data version (FakeSherbrooke, WORK/feats_v3). The second device and the 20-qubit chain are
# selected per command below.
unset BLOCKGEO_TAG BLOCKGEO_BACKEND BLOCKGEO_TWOQ BLOCKGEO_CHAIN
TORINO=(BLOCKGEO_BACKEND=torino BLOCKGEO_TWOQ=cz "BLOCKGEO_CHAIN=11,10,9,8,17,27,28,29,36,48" BLOCKGEO_TAG=_v3_torino)
CHAIN20=("BLOCKGEO_CHAIN=33,39,40,41,53,60,59,58,71,77,78,79,91,98,99,100,110,118,119,120")

exec > >(tee -a "$WORK/logs/reproduce_data.log") 2>&1
run() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; "$@"; }

# 1. Main benchmark with uniform TFIM, disordered TFIM and random circuits, n = 6, 8, 10 (about 16 h).
#    n = 6 and 8 take 4 to 16 min per family, n = 10 about 5 h per family.
for n in 6 8 10; do
  run python src/gen_v3.py --family tfim --n "$n" --depths 1,2,3,4,6,8 --workers "$WORKERS"
done
for n in 6 8 10; do
  run python src/gen_v3.py --family dtfim --n "$n" --depths 1,2,3,4,6,8 --workers "$WORKERS"
done
for n in 6 8 10; do
  run python src/gen_v3.py --family random --n "$n" --depths 4,8,12,16,20,24 --workers "$WORKERS"
done

# 2. Near-Clifford training sets of the training circuits for the CDR-input study (about 18 h).
#    n = 6 and 8 take 4 to 16 min each, n = 10 about 8.5 h per family.
for n in 6 8 10; do
  run python src/gen_v3_trcdr.py --family tfim --n "$n" --depths 1,2,3,4,6,8 --workers "$WORKERS"
done
for n in 6 8; do
  run python src/gen_v3_trcdr.py --family dtfim --n "$n" --depths 1,2,3,4,6,8 --workers "$WORKERS"
done
for n in 6 8 10; do
  run python src/gen_v3_trcdr.py --family random --n "$n" --depths 4,8,12,16,20,24 --workers "$WORKERS"
done

# 3. Training circuits 200 to 399 (training-set size) and drifted test data (calibration drift), TFIM n = 8
#    (about 6 min).
run python src/gen_exp12.py --family tfim --n 8 --depths 1,3,6 --drift 0.7,0.85,1.15,1.3 --K 24 --workers "$WORKERS"

# 4. 20-qubit chain with the MPDO simulator, Trotter steps 1 to 3 (about 4 min).
run env "${CHAIN20[@]}" python src/gen20.py --n 20 --depths 1,2,3 --workers "$WORKERS"

# 5. Second device (FakeTorino), uniform TFIM, n = 6, 8, 10 (about 6.5 h; n = 10 takes about 5.6 h).
for n in 6 8 10; do
  run env "${TORINO[@]}" python src/gen_v3.py --family tfim --n "$n" --depths 1,2,3,4,6,8 --workers "$WORKERS"
done

printf '\n[%s] data generation finished in %s\n' "$(date '+%F %T')" "$WORK"
