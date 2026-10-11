"""v64's delta: the hour-window corrector (ctxc1sel_roh) in place of ctxc1sel_ro, through the full chain plus the
identity-arm deltas, both chains run on this machine (reports/ctx_hour_preregistration.md, as src/post/v54_delta.py).

  delta = (chain_ctxc1sel_roh_final + id deltas on it) - (chain_ctxc1sel_ro_final + id deltas on it)

Then  FREEZE_LIRF=1 v50_assemble.py smart-jigsaw_v54.parquet P64 DELTA  and  day_alloc_build.py P64 smart-jigsaw_v62.parquet v64.

Run:  .venv/Scripts/python.exe src/post/v64_delta.py OUT.parquet [NEW_CORRECTOR, default ctxc1sel_roh] [ID_DIR, default cache/v64]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUB, V64 = ROOT / "data" / "submissions", ROOT / "cache" / "v64"
out_f = sys.argv[1]
NEW = sys.argv[2] if len(sys.argv) > 2 else "ctxc1sel_roh"
NEW_ID = Path(sys.argv[3]) if len(sys.argv) > 3 else V64   # v65: cache/v65 (id_<NEW> there)
tpl = pl.read_parquet(ROOT / "data" / "ranking" / "submitting.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64))


def col(path, src):
    return tpl.join(pl.read_parquet(path).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col(src).cast(pl.Float64).alias("x")),
                    on="MVT_ID_mvt", how="left")["x"].fill_null(0.0).to_numpy()


def ids(n, d=V64):
    return col(d / f"id_{n}" / "comb_delta_all_id.parquet", "delta") + col(d / f"id_{n}" / "nb_delta_ro_id.parquet", "delta")


new, old = col(SUB / f"chain_{NEW}_final.parquet", "TAXITIME_SEC_mvt"), col(SUB / "chain_ctxc1sel_ro_final.parquet", "TAXITIME_SEC_mvt")
dn, do = ids(NEW, NEW_ID), ids("ctxc1sel_ro")
d = (new + dn) - (old + do)
print(f"chain change RMS {np.sqrt(np.mean((new - old) ** 2)):.2f}; id deltas new RMS {np.sqrt(np.mean(dn ** 2)):.2f}, "
      f"old {np.sqrt(np.mean(do ** 2)):.2f}; total delta RMS {np.sqrt(np.mean(d ** 2)):.2f}, mean {d.mean():+.3f}")
pl.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"], "delta": d}).write_parquet(out_f)
print("wrote", out_f)
