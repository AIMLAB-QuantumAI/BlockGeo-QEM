"""Path handling (src/paths.py): the working, data-version and hardware directories follow the environment variables
and their documented defaults, and importing the modules of src/ and hardware/ writes nothing."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC, HARDWARE = REPO / "src", REPO / "hardware"
MODULES = [p.stem for p in sorted(SRC.glob("*.py"))] + [p.stem for p in sorted(HARDWARE.glob("*.py"))]
IGNORED = {".git", "__pycache__", ".pytest_cache"}

PATHS_CODE = f"""
import json, sys
sys.path.insert(0, {str(SRC)!r})
import paths
names = ["WORK", "TAG", "FEAT_DIR", "RES_DIR", "CACHE_DIR", "FIG_DIR", "TAB_DIR", "LOG_DIR", "HW"]
out = {{name: getattr(paths, name) for name in names}}
out.update(results_v3=paths.results_dir("_v3"), feats_v3_torino=paths.feats_dir("_v3_torino"))
print(json.dumps(out))
"""


def snapshot(root):
    """Every file and directory below root (without .git and caches) with the size and modification time of files.
    Of the default working directory work/ only the existence is recorded, so that runs writing there meanwhile do not
    interfere."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED]
        if dirpath == os.fspath(root) and "work" in dirnames:
            dirnames.remove("work")
            out["work"] = "dir"
        for name in dirnames:
            out[os.path.relpath(os.path.join(dirpath, name), root)] = "dir"
        for name in filenames:
            st = os.lstat(os.path.join(dirpath, name))
            out[os.path.relpath(os.path.join(dirpath, name), root)] = (st.st_size, st.st_mtime_ns)
    return out


def changes(before, after):
    return sorted(set(before) ^ set(after)) + sorted(k for k in set(before) & set(after) if before[k] != after[k])


def expected_paths(work, tag, hw):
    return dict(WORK=str(work), TAG=tag, FEAT_DIR=str(work / f"feats{tag}"), RES_DIR=str(work / f"results{tag}"),
                CACHE_DIR=str(work / f"cache{tag}"), FIG_DIR=str(work / "figures"), TAB_DIR=str(work / "tables"),
                LOG_DIR=str(work / "logs"), HW=str(hw), results_v3=str(work / "results_v3"),
                feats_v3_torino=str(work / "feats_v3_torino"))


def test_importing_modules_writes_nothing(tmp_path, run_python):
    """Importing every module of src/ and hardware/ with BLOCKGEO_WORKDIR=tmp_path/work changes nothing in the
    repository and does not create the working directory."""
    work = tmp_path / "work"
    code = (f"import importlib, sys\nsys.path[:0] = [{str(SRC)!r}, {str(HARDWARE)!r}]\n"
            f"for name in {MODULES!r}:\n    importlib.import_module(name)\n"
            "import paths\nprint(paths.WORK)\n")
    before = snapshot(REPO)
    proc = run_python("-c", code, workdir=work)
    assert not changes(before, snapshot(REPO))
    assert proc.stdout.splitlines()[-1] == str(work)
    assert not work.exists()


def test_default_paths(run_python):
    """Without BLOCKGEO_WORKDIR, BLOCKGEO_TAG and BLOCKGEO_HW_DIR the data go to work/ in the repository root with the
    tag _v3, and the hardware data are read from data/hardware; importing paths creates nothing."""
    before = snapshot(REPO)
    got = json.loads(run_python("-c", PATHS_CODE, workdir=None).stdout)
    assert got == expected_paths(REPO / "work", "_v3", REPO / "data" / "hardware")
    assert not changes(before, snapshot(REPO))


def test_paths_follow_environment(tmp_path, run_python):
    """BLOCKGEO_WORKDIR (with ~ expanded), BLOCKGEO_TAG and BLOCKGEO_HW_DIR move the working, data-version and hardware
    directories; none of them is created on import."""
    env = dict(HOME=tmp_path, BLOCKGEO_TAG="_v3_torino", BLOCKGEO_HW_DIR=tmp_path / "hw")
    got = json.loads(run_python("-c", PATHS_CODE, workdir="~/work", env=env).stdout)
    assert got == expected_paths(tmp_path / "work", "_v3_torino", tmp_path / "hw")
    assert sorted(os.listdir(tmp_path)) == []
