"""The data generators default to at most as many worker processes as there are CPUs."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize("script, cap", [("gen_v3.py", 14), ("gen_v3_trcdr.py", 10), ("gen_exp12.py", 8),
                                         ("gen20.py", 8)])
def test_default_workers(script, cap):
    """--help reports min(cap, number of CPUs) as the default number of workers."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, str(SRC / script), "--help"], capture_output=True, text=True, env=env,
                         check=True).stdout
    expected = min(cap, os.cpu_count() or 1)
    assert f"{expected} here" in " ".join(out.split())
