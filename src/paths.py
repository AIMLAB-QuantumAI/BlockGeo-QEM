"""Locations of everything the scripts read and write.

REPO       repository root (the parent directory of src/)
WORK       working directory for all generated data: $BLOCKGEO_WORKDIR, default REPO/work
TAG        data version, $BLOCKGEO_TAG (default "_v3"; the second-device runs use "_v3_torino")
FEAT_DIR   WORK/feats<TAG>      simulated circuit data of the gen_*.py scripts
RES_DIR    WORK/results<TAG>    per-sample results of the runners and analyses (predictions in RES_DIR/pred)
CACHE_DIR  WORK/cache<TAG>      reserved for caches (no script writes it at present)
FIG_DIR    WORK/figures         figures of paper_figs.py
TAB_DIR    WORK/tables          tables of paper_tables.py
LOG_DIR    WORK/logs            logs of scripts/reproduce_*.sh (no Python script writes it)
HW         hardware measurements: $BLOCKGEO_HW_DIR, default REPO/data/hardware (read and written by hardware/*.py)

feats_dir(tag), results_dir(tag) and cache_dir(tag) give the directories of another data version, e.g.
results_dir("_v3_torino"). Nothing is created on import; scripts create directories where they write files.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.abspath(os.path.expanduser(os.environ.get("BLOCKGEO_WORKDIR") or os.path.join(REPO, "work")))
TAG = os.environ.get("BLOCKGEO_TAG", "_v3")
HW = os.path.abspath(os.path.expanduser(os.environ.get("BLOCKGEO_HW_DIR") or os.path.join(REPO, "data", "hardware")))


def feats_dir(tag=None):
    return os.path.join(WORK, "feats" + (TAG if tag is None else tag))


def results_dir(tag=None):
    return os.path.join(WORK, "results" + (TAG if tag is None else tag))


def cache_dir(tag=None):
    return os.path.join(WORK, "cache" + (TAG if tag is None else tag))


FEAT_DIR = feats_dir()
RES_DIR = results_dir()
CACHE_DIR = cache_dir()
FIG_DIR = os.path.join(WORK, "figures")
TAB_DIR = os.path.join(WORK, "tables")
LOG_DIR = os.path.join(WORK, "logs")
