"""LOBT window projection (reports/lobt_window_preregistration.md).

Every 2025 departure with a LOBT has |BLOCK - LOBT| <= 3606 s, so its taxi lies in
[MVT - LOBT - 3606, MVT - LOBT + 3606]. Each prediction is clipped into that window
(floored at 0); rows without LOBT are unchanged.

Run:  .venv/Scripts/python.exe src/post/lobt_window.py IN.parquet OUT.parquet
  e.g. data/submissions/smart-jigsaw_v31.parquet -> data/submissions/smart-jigsaw_v32.parquet
PRC_SCOPE=final4m: windows from final_ranking.parquet, checked against final_submitting.parquet
(src/scope.py).
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import scope  # noqa: E402

RANKING = scope.RANKING
TEMPLATE = scope.TEMPLATE
HALF = 3606


def window(src) -> pl.DataFrame:
    """MVT_ID_mvt, ADEP_mvt and the window bounds lo/hi (null without LOBT)."""
    lt = (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64)
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(pl.col("MVT_ID_mvt").cast(pl.Float64), "ADEP_mvt",
                    lo=(lt - HALF).clip(lower_bound=0), hi=lt + HALF).collect())


def project(pred: pl.Expr) -> pl.Expr:
    return pl.when(pl.col("lo").is_null()).then(pred).otherwise(pred.clip(pl.col("lo"), pl.col("hi")))


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    sub = pl.read_parquet(src)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    d = (sub.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id"))
         .join(window(RANKING).rename({"MVT_ID_mvt": "_id"}), on="_id", how="left"))
    assert d["ADEP_mvt"].null_count() == 0, "submission ids missing from ranking.parquet"
    p = pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)
    d = d.with_columns(new=project(p).round())
    moved = d.filter(pl.col("new") != p)
    print(f"rows moved: {moved.height} of {d.height:,}; up {(moved['new'] > moved['TAXITIME_SEC_mvt']).sum()}, "
          f"mean |shift| {(moved['new'] - moved['TAXITIME_SEC_mvt']).abs().mean():.0f} s")
    print("by airport:", dict(moved.group_by("ADEP_mvt").len().sort("ADEP_mvt").iter_rows()))
    out = d.select("MVT_ID_mvt", pl.col("new").cast(dtype).alias("TAXITIME_SEC_mvt"))
    tpl = pl.read_parquet(TEMPLATE)
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
    assert (out["TAXITIME_SEC_mvt"] >= 0).all()
    out.write_parquet(dst)
    print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
