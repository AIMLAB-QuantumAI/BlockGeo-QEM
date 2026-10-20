"""Importing qemlib (which imports the vendored blockgeo_qem_fixed module) keeps thread settings chosen by the user
and fills in 4 threads only where none is set."""
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CODE = ("import os, sys; sys.path.insert(0, sys.argv[1]); import qemlib; "
        f"print(' '.join(os.environ.get(v, '-') for v in {VARS!r}))")


def _threads_after_import(env):
    env = {k: v for k, v in os.environ.items() if k not in VARS} | env | {"CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", CODE, str(SRC)], env=env, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_user_settings_are_kept():
    """Thread counts set before the import survive it."""
    assert _threads_after_import({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2"}) == ["1", "2", "4", "4"]


def test_default_is_four_threads():
    """Without any setting, every library gets 4 threads, as before."""
    assert _threads_after_import({}) == ["4", "4", "4", "4"]
