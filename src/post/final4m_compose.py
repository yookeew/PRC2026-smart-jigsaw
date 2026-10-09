"""Blind-board file: Jan/Jul rows from the live board's best file, Feb/Jun rows from the four-month build (PROGRESS.md §126).

Until the rebuilt chain's Jan/Jul rows have been checked against the live board's best file, the blind-board file keeps
that file's Jan/Jul predictions (the IDs of submitting.parquet) and takes every other row of final_submitting.parquet
from the four-month build. Checks: exact final_submitting IDs, no nulls or duplicates, integer seconds in [0, 140,000];
prints the rebuilt-vs-kept comparison on Jan/Jul (RMS / mean per airport and month) and the md5 of the output.

Run:  .venv/Scripts/python.exe src/post/final4m_compose.py JANJUL.parquet BUILD4M.parquet OUT.parquet
      e.g. data/submissions/smart-jigsaw_v55.parquet data/submissions/final4m/f4m_full.parquet data/submissions/final4m/blind_v1.parquet
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import scope  # noqa: E402

CEIL = 140000
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)


def main() -> None:
    jj_f, b_f, out_f = (Path(a) for a in sys.argv[1:4])
    tpl = pl.read_parquet(scope._RANK_DIR / "final_submitting.parquet")
    tpl_jj = pl.read_parquet(scope.TEMPLATE_JJ).select(I64)
    jj = pl.read_parquet(jj_f).select(I64, jj=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    b = pl.read_parquet(b_f).select(I64, b=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    assert jj.height == tpl_jj.height and set(jj["MVT_ID_mvt"].to_list()) == set(tpl_jj["MVT_ID_mvt"].to_list())
    mv = (pl.read_parquet(scope._RANK_DIR / "final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
          .select(I64, apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month()))
    x = tpl.select(I64).join(b, on="MVT_ID_mvt", how="left").join(jj, on="MVT_ID_mvt", how="left").join(mv, on="MVT_ID_mvt", how="left")
    assert x["b"].null_count() == 0, "build misses rows of final_submitting"
    assert x["apt"].null_count() == 0
    in_jj = x["jj"].is_not_null()
    missing = set(tpl_jj["MVT_ID_mvt"].to_list()) - set(x.filter(in_jj)["MVT_ID_mvt"].to_list())
    assert not missing, f"{len(missing)} live-board IDs are not in final_submitting"
    print("rows by month:", dict(x.group_by("mo").len().sort("mo").iter_rows()),
          f"| from {jj_f.name}: {int(in_jj.sum()):,}, from {b_f.name}: {int((~in_jj).sum()):,}")
    print("Jan/Jul months of kept rows:", sorted(x.filter(in_jj)["mo"].unique().to_list()),
          "| months of build rows:", sorted(x.filter(~in_jj)["mo"].unique().to_list()))

    # rebuilt vs kept on Jan/Jul
    j = x.filter(in_jj).with_columns(d=pl.col("b") - pl.col("jj"))
    print(f"Jan/Jul rebuilt vs kept: RMS {np.sqrt((j['d'] ** 2).mean()):.1f} s, mean {j['d'].mean():+.2f}, "
          f"median |d| {j['d'].abs().median():.0f}, rows |d| > 300 s {(j['d'].abs() > 300).sum():,}")
    with pl.Config(tbl_rows=30, tbl_formatting="ASCII_MARKDOWN", float_precision=1):
        print(j.group_by("apt", "mo").agg(n=pl.len(), rms=(pl.col("d") ** 2).mean().sqrt(), mean=pl.col("d").mean())
              .sort("apt", "mo"))

    val = pl.when(in_jj).then(pl.col("jj")).otherwise(pl.col("b"))
    out = x.select(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype),
                   val.round().cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
    out = tpl.select("MVT_ID_mvt").join(out, on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height and out["MVT_ID_mvt"].n_unique() == out.height
    assert out["TAXITIME_SEC_mvt"].null_count() == 0
    lo, hi = int(out["TAXITIME_SEC_mvt"].min()), int(out["TAXITIME_SEC_mvt"].max())
    assert lo >= 0 and hi <= CEIL, (lo, hi)
    chk = out.select(I64, "TAXITIME_SEC_mvt").join(pl.read_parquet(jj_f).select(I64, k="TAXITIME_SEC_mvt"), on="MVT_ID_mvt")
    assert chk.height == tpl_jj.height and (chk["TAXITIME_SEC_mvt"] == chk["k"].cast(pl.Int32)).all()
    s = out["TAXITIME_SEC_mvt"]
    print(f"output: {out.height:,} rows, range [{lo}, {hi}], median {s.median():.0f}, mean {s.mean():.1f}; "
          f"Jan/Jul rows identical to {jj_f.name}: True")
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_f)
    print(f"wrote {out_f}  md5 {hashlib.md5(out_f.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
