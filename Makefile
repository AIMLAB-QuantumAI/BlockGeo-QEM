# Common tasks. Run from the repository root with the environment activated; "make help" lists the targets.
PYTHON ?= python
QUICK := $(CURDIR)/work/quickstart

.PHONY: help install test quickstart example data benchmarks figures clean

help:
	@echo "make install     install the pinned packages into the active environment (pip)"
	@echo "make test        run the test suite (about one minute on a CPU)"
	@echo "make quickstart  tiny end-to-end run, results in work/quickstart (about one minute)"
	@echo "make example     BlockGeo-QEM on a small simulated data set (examples/mitigate_own_data.py)"
	@echo "make data        simulated data of all studies (about 47 h, 12 GB)"
	@echo "make benchmarks  every method and study (about 49 h)"
	@echo "make figures     tables, figures and printed statistics (about 15 min)"
	@echo "make clean       remove Python and pytest caches (generated data in work/ are kept)"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

quickstart:
	BLOCKGEO_WORKDIR="$(QUICK)" $(PYTHON) src/gen_v3.py --family tfim --n 4 --depths 1 --n_train 12 --n_test 4 --K 3 --workers 2
	BLOCKGEO_WORKDIR="$(QUICK)" $(PYTHON) src/run_v3.py --part base --family tfim --n 4 --budget 10000 --depths 1
	BLOCKGEO_WORKDIR="$(QUICK)" $(PYTHON) src/run_v3.py --part ours --family tfim --n 4 --budget 10000 --depths 1

example:
	$(PYTHON) examples/mitigate_own_data.py

data:
	scripts/reproduce_data.sh

benchmarks:
	scripts/reproduce_benchmarks.sh

figures:
	scripts/reproduce_figures.sh

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
