"""v44 build step (reports/ctx_seedavg_preregistration.md, post-hoc exception): ctxsel combination of the one-process
seed-42 and 3-seed-mean correctors from tests/seedavg_one_fit.py. Holdout/OOF files are copied from ctxc1sel (the chain
only uses them to fit stack weights, which therefore match v42).

Run:  .venv/Scripts/python.exe -u tests/seedavg_sel.py
"""
import os
import shutil
import sys
from pathlib import Path
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import ctx_adsb_stable_test as ST
SUB, EVAL, OOF = ROOT / "data/submissions", ROOT / "cache/eval", ROOT / "cache/oof"
apt = ST._airports()
for t in ("s42", "s3"):
    name = f"ctxc1sel_1fit{t}"
    rk, n = ST._select(pl.read_parquet(SUB / f"ctxc1r_1fit{t}.parquet"), pl.read_parquet(SUB / f"ctxc1rb_1fit{t}.parquet"), "TAXITIME_SEC_mvt", apt)
    rk.write_parquet(SUB / f"{name}.parquet")
    shutil.copy(EVAL / "ctxc1sel_holdout_ev.parquet", EVAL / f"{name}_holdout_ev.parquet")
    shutil.copytree(OOF / "ctxc1sel", OOF / name, dirs_exist_ok=True)
    print(f"{name}: ranking {n:,} of {rk.height:,} rows from rb; holdout/OOF copied from ctxc1sel")
