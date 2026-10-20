"""Runners stop with a message naming the missing generated file and the generator command, instead of a traceback."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("args, command", [
    (["src/run_v3.py", "--part", "base", "--family", "tfim", "--n", "4", "--budget", "10000", "--depths", "1"],
     "python src/gen_v3.py --family tfim --n 4 --depths 1"),
    (["src/run_cdrbb.py", "--family", "dtfim", "--n", "4", "--budget", "10000", "--depths", "1"],
     "python src/gen_v3.py --family dtfim --n 4 --depths 1"),
])
def test_missing_data_message(tmp_path, args, command):
    """With an empty working directory the runner exits with status 1 and names the generator command."""
    env = dict(os.environ, BLOCKGEO_WORKDIR=str(tmp_path), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 1
    assert "Missing input file" in r.stderr and command in r.stderr
    assert "Traceback" not in r.stderr
