#!/usr/bin/env bash
# Writes the win/tie/loss table, the figures and the printed statistics from the results of reproduce_benchmarks.sh.
#
# Takes about 10 to 15 minutes in total and writes
#   $BLOCKGEO_WORKDIR/tables/tab_wtl.tex     win/tie/loss table
#   $BLOCKGEO_WORKDIR/figures/*.pdf          main_tfim, geometry, spectral, robust, n20_B100000, hardware,
#                                            unstructured_B100000
#   $BLOCKGEO_WORKDIR/logs/reproduce_figures.log   every printed statistic
#
# Environment
#   BLOCKGEO_WORKDIR  directory of the generated data and results (default <repository>/work)
#   BLOCKGEO_HW_DIR   hardware data (default $BLOCKGEO_WORKDIR/hardware, the copy made by reproduce_benchmarks.sh;
#                     it is copied from data/hardware here if it does not exist)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${BLOCKGEO_WORKDIR:-$REPO/work}"
WORK="${WORK/#\~/$HOME}"
mkdir -p "$WORK/logs"
WORK="$(cd "$WORK" && pwd)"
export BLOCKGEO_WORKDIR="$WORK"
cd "$REPO"

# The analyses read the default data version (WORK/results_v3) and WORK/results_v3_torino.
unset BLOCKGEO_TAG BLOCKGEO_BACKEND BLOCKGEO_TWOQ BLOCKGEO_CHAIN

export BLOCKGEO_HW_DIR="${BLOCKGEO_HW_DIR:-$WORK/hardware}"
if [ ! -d "$BLOCKGEO_HW_DIR" ]; then
  mkdir -p "$(dirname "$BLOCKGEO_HW_DIR")"
  cp -R data/hardware "$BLOCKGEO_HW_DIR"
fi

exec > >(tee -a "$WORK/logs/reproduce_figures.log") 2>&1
run() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; "$@"; }

# 1. Site-resolved quench errors of the simulation and of the hardware run (seconds). They write
#    results_v3/quench_blocks_tfim_n8_B10000.csv and quench_ibm_fez/blocks.csv, which the hardware figure reads.
run python src/analyze_quench_blocks.py sim
run python src/analyze_quench_blocks.py hw ibm_fez

# 2. Purity bins (writes results_v3/purity_resolved_B100000.csv for the geometry figure), disagreement of
#    overlapping blocks, Holm correction of the win/tie/loss tests (a few minutes).
run python src/paper_review_checks.py purity overlap holm

# 3. Win/tie/loss table and printed summary statistics, including the median purity and smallest eigenvalue of the
#    ideal block marginals (marginals), a few minutes.
run python src/paper_tables.py
run python src/paper_tables.py numbers
run python src/paper_tables.py cdr
run python src/paper_tables.py marginals
run python src/paper_extra.py torino n20
run python src/analyze_nl.py
run python src/analyze_v3.py      # per-setting comparison with the best baseline, results_v3/compare_BlockGeo-QEM.csv

# 4. Figures (about a minute).
run python src/paper_figs.py main geometry spectral robust n20 hardware unstructured

# 5. Checks of the MPDO simulator and of the block sampler against the density-matrix data, the chain search on
#    the FakeTorino coupling graph, and the two-qubit and readout errors of the FakeSherbrooke chain and its
#    20-qubit extension (a few minutes).
run python src/test_mpdo.py 8
run python src/test_sample_blocks.py
run python src/find_chain.py torino cz 10 0.05
run python src/find_chain.py sherbrooke ecr 33,39,40,41,53,60,59,58,71,77,78,79,91,98,99,100,110,118,119,120

printf '\n[%s] table in %s, figures in %s\n' "$(date '+%F %T')" "$WORK/tables" "$WORK/figures"
