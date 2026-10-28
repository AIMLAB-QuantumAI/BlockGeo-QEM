# Changelog

All notable changes to this project are listed here. Version numbers follow [semantic versioning](https://semver.org).

## Unreleased

### Added
- Example notebook `examples/quickstart.ipynb` that runs a small benchmark end to end, with stored outputs (#10).
- Contributing guidelines, issue templates and a pull-request template (#14).
- This changelog (#15).
- A badge in the README that shows the status of the tests on `main`.
- `data/hardware/README.md`, which describes the hardware runs and their files and shows how to load them.
- `examples/mitigate_own_data.py`, which applies BlockGeo-QEM to block coordinates supplied by the user.
- A `Makefile` with targets for the tests, the quick start, the example and the reproduction scripts.
- A CI job that checks the shell scripts with `bash -n` and ShellCheck.
- Blocks of three neighbouring qubits (`BLOCKGEO_B=3`, 27 covering settings) in the data generator, the measurement, the baselines and BlockGeo-QEM, with GNN-QEM for three-qubit blocks (`run_gnn_b3.py`) and the comparison of two- and three-qubit blocks (`paper_b3.py`).

### Changed
- BlockGeo-QEM chooses the number of training epochs by the held-out loss alone, the rule of the final results. The earlier default, which also required the held-out coordinate error to stay within 2% of that of the projected linear estimate, is available with `BLOCKGEO_TOL=0.02` or `tol=0.02`, and `run_tol.py` runs it explicitly.
- CI runs on pushes to `main`, on pull requests and on demand, and cancels runs that a newer push supersedes; pushes to other branches no longer start a separate run.
- Importing `qemlib` keeps thread settings chosen by the user; 4 threads remain the default (#11).
- The data generators use at most as many worker processes as there are CPUs by default (#12).
- The runners stop with a message that names a missing generated file and the command that writes it (#13).

### Fixed
- The hardware quench test compares the purity and the concurrence with a tolerance that does not depend on the CPU (#8).

## 0.1.0 (2026-10-20)

First public release.

- Simulation of Trotterized transverse-field Ising (uniform and disordered) and random circuits on a device chain with a Pauli-twirled noise model, finite-shot measurement in the covering settings, readout errors and readout-error mitigation.
- BlockGeo-QEM: per-coordinate linear backbone, Cholesky-factor state head trained with the Bures distance, validation check and ablations.
- Fourteen baselines: zero-noise extrapolation, idealised PEC, CDR, vnCDR, per-coordinate and multi-noise-level regression, random forest, MLP-QEM, GNN-QEM and BlockGeo-v1.
- Studies: CDR input, size transfer, training-set size, calibration drift, validation tolerance, training seeds, loss weight, a second device, a 20-qubit chain simulated as a matrix-product density operator, and a quench.
- IBM Quantum hardware scripts restricted to the free Open Plan, and the measurements of three runs on ibm_fez.
- Analysis scripts for the win/tie/loss table, the figures and the printed statistics, a reproduction guide and scripts, tests and continuous integration.
