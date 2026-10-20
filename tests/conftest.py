"""Shared set-up of the test suite.

The tests run on the CPU with the default device model and write no data into the repository (only the gitignored
Python and pytest caches appear there). Before any module of src/ is imported, this file
  - removes the variables that select another data version, device or hardware directory (BLOCKGEO_TAG,
    BLOCKGEO_BACKEND, BLOCKGEO_TWOQ, BLOCKGEO_CHAIN, BLOCKGEO_HW_DIR), so the defaults of src/paths.py and
    src/noise_layout.py apply,
  - hides CUDA devices,
  - points BLOCKGEO_WORKDIR and MPLCONFIGDIR at a temporary directory that is removed after the session,
  - puts src/ and hardware/ on sys.path, as the scripts there do for themselves.
Tests that write data run the scripts in a fresh interpreter with their own temporary BLOCKGEO_WORKDIR (fixture
run_python).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC, HARDWARE = REPO / "src", REPO / "hardware"

for _name in ("BLOCKGEO_TAG", "BLOCKGEO_BACKEND", "BLOCKGEO_TWOQ", "BLOCKGEO_CHAIN", "BLOCKGEO_HW_DIR"):
    os.environ.pop(_name, None)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_SESSION_DIR = Path(tempfile.mkdtemp(prefix="blockgeo-qem-tests-"))
os.environ["BLOCKGEO_WORKDIR"] = str(_SESSION_DIR / "work")
os.environ["MPLCONFIGDIR"] = str(_SESSION_DIR / "matplotlib")
# Environment of the subprocesses, taken before src/blockgeo_qem_fixed.py sets its thread counts in this process.
_BASE_ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
sys.path[:0] = [str(SRC), str(HARDWARE)]


def pytest_unconfigure(config):
    """Remove the session's temporary directory."""
    shutil.rmtree(_SESSION_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def run_python():
    """run(*args, workdir, env=None): run `python *args` from the repository root in a fresh interpreter with
    BLOCKGEO_WORKDIR=workdir (unset if workdir is None) and the variables in env (None removes one). Fails the test
    with the captured output if the process fails; returns the CompletedProcess."""

    def run(*args, workdir, env=None, timeout=600):
        full = dict(_BASE_ENV)
        full.pop("BLOCKGEO_WORKDIR")
        if workdir is not None:
            full["BLOCKGEO_WORKDIR"] = str(workdir)
        for key, value in (env or {}).items():
            if value is None:
                full.pop(key, None)
            else:
                full[key] = str(value)
        cmd = [sys.executable, *map(str, args)]
        proc = subprocess.run(cmd, cwd=REPO, env=full, capture_output=True, text=True, timeout=timeout)
        assert proc.returncode == 0, (f"{' '.join(cmd[1:])} exited with {proc.returncode}\n"
                                      f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        return proc

    return run
