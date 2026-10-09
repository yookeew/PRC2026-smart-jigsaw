"""v54's delta: the stand-aware context corrector (ctxc1sel_ro) in place of ctxc1sel, through the full chain plus v53's
identity-arm post-stage deltas (PROGRESS §124).

  delta = (chain_ctxc1sel_ro_final + id deltas on it) - (chain_ctxc1selcp_final + id deltas on it)

chain_ctxc1selcp is the production chain rerun on this machine from a byte copy of ctxc1sel (it reproduces
chain_ctxc1sel_final exactly). The identity deltas on the old chain are v53's (cache/v53/), on the new chain they come
from src/post/v50_comb_delta.py / v50_nb_delta.py with PRC_LGB=ctxc1sel_ro (ARM=all_id, NB_ARM=ro_id).
Write MVT_ID_mvt + delta; then  FREEZE_LIRF=1 assemble.py smart-jigsaw_v53.parquet smart-jigsaw_v54.parquet DELTA.

Run:  .venv/Scripts/python.exe src/post/v54_delta.py NEW_DELTA_DIR OUT.parquet
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "data" / "submissions"
new_dir, out_f = Path(sys.argv[1]), sys.argv[2]
tpl = pl.read_parquet(ROOT / "data" / "ranking" / "submitting.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64))


def col(path, c, src):
    return tpl.join(pl.read_parquet(path).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col(src).cast(pl.Float64).alias(c)),
                    on="MVT_ID_mvt", how="left")[c].fill_null(0.0).to_numpy()


new = col(SUB / "chain_ctxc1sel_ro_final.parquet", "x", "TAXITIME_SEC_mvt")
old = col(SUB / "chain_ctxc1selcp_final.parquet", "x", "TAXITIME_SEC_mvt")
dn = col(new_dir / "comb_delta_all_id.parquet", "x", "delta") + col(new_dir / "nb_delta_ro_id.parquet", "x", "delta")
do = (col(ROOT / "cache" / "v53" / "comb_delta_all_id.parquet", "x", "delta")
      + col(ROOT / "cache" / "v53" / "nb_delta_ro_id.parquet", "x", "delta"))
d = (new + dn) - (old + do)
print(f"chain change RMS {np.sqrt(np.mean((new - old) ** 2)):.2f}; id deltas new RMS {np.sqrt(np.mean(dn ** 2)):.2f}, "
      f"old {np.sqrt(np.mean(do ** 2)):.2f}; total delta RMS {np.sqrt(np.mean(d ** 2)):.2f}, mean {d.mean():+.3f}")
pl.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"], "delta": d}).write_parquet(out_f)
print("wrote", out_f)
