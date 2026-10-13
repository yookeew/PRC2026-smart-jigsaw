"""Average of two independent builds on Jan / Jul (reports/twin_build_average_preregistration.md, PROGRESS.md §163).

The live board's file and the four-month build's Jan / Jul rows are the same recipe fitted twice (base refits, CatBoost
GPU rerun, other corrector seeds). On NM-matched rows at the nine non-LIRF airports the output is the rounded mean of
the two, projected into the live LOBT window; every other row keeps LIVE's value.

Run:  .venv/Scripts/python.exe src/post/twin_average.py LIVE.parquet BUILD4M.parquet OUT.parquet
      e.g. data/submissions/smart-jigsaw_v75.parquet data/submissions/final4m/f4m_v75.parquet data/submissions/smart-jigsaw_v76.parquet
Live scope only (submitting.parquet / ranking.parquet).
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
from post.lobt_window import window  # noqa: E402

assert not scope.FINAL, "live scope only"
F64 = pl.col("MVT_ID_mvt").cast(pl.Float64)


def main() -> None:
    live_f, b_f, out_f = (Path(a) for a in sys.argv[1:4])
    live = pl.read_parquet(live_f)
    tpl = pl.read_parquet(scope.TEMPLATE)
    assert live.height == tpl.height and set(live["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
    rk = (pl.scan_parquet(scope.RANKING).filter(pl.col("PHASE_mvt") == "DEP")
          .select(F64, nmu=pl.col("FLIGHT_ID_mvt").is_null()).collect())
    x = (live.with_columns(_id=F64, p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
         .join(rk.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
         .join(window(scope.RANKING).rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
         .join(pl.read_parquet(b_f).select(_id=F64, q=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="_id", how="left"))
    assert x["ADEP_mvt"].null_count() == 0 and x["q"].null_count() == 0, "rows missing from ranking or the build"
    k = (x["ADEP_mvt"] != "LIRF") & ~x["nmu"]
    p, q = x["p"].to_numpy(), x["q"].to_numpy()
    lo, hi = x["lo"].fill_null(-np.inf).to_numpy(), x["hi"].fill_null(np.inf).to_numpy()
    ks = k.to_numpy()
    a = np.where(ks, np.rint(np.clip((p + q) / 2, lo, hi)), p)
    d = q[ks] - p[ks]
    ch = a[ks] - p[ks]
    print(f"scope rows {ks.sum():,} of {len(ks):,}; build - live on scope: RMS {np.sqrt(np.mean(d ** 2)):.2f}, mean {d.mean():+.2f}")
    print(f"change on scope rows: RMS {np.sqrt(np.mean(ch ** 2)):.2f}, mean {ch.mean():+.3f}, max |change| {np.abs(ch).max():.0f}; "
          f"rows moved by the window {(np.rint((p + q) / 2)[ks] != a[ks]).sum()}")
    print(f"expected gain at equal quality: {np.sum(d ** 2) / 4 / len(p) / (2 * 234.51):.3f} s")

    # stop rules (reports/twin_build_average_preregistration.md)
    assert np.isfinite(a).all() and (a >= 0).all(), "rule 1: nulls / negatives"
    assert np.array_equal(a[~ks], p[~ks]), "rule 2: rows outside the scope changed"
    rms = np.sqrt(np.mean(ch ** 2))
    assert 11.7 <= rms <= 13.7 and abs(ch.mean()) < 0.5, f"rule 3: change RMS {rms:.2f} / mean {ch.mean():+.3f}"
    assert ((a[ks] >= lo[ks]) & (a[ks] <= hi[ks])).all(), "rule 4: outside the live LOBT window"

    dtype = live["TAXITIME_SEC_mvt"].dtype
    out = x.select("MVT_ID_mvt").with_columns(TAXITIME_SEC_mvt=pl.Series(a).cast(dtype))
    assert out["MVT_ID_mvt"].to_list() == live["MVT_ID_mvt"].to_list()
    by = (x.with_columns(c=pl.Series(a) - pl.col("p")).filter(pl.Series(ks))
          .group_by("ADEP_mvt").agg(n=pl.len(), rms=(pl.col("c") ** 2).mean().sqrt().round(2), mean=pl.col("c").mean().round(3))
          .sort("ADEP_mvt"))
    print(by)
    out.write_parquet(out_f)
    print(f"wrote {out_f}  md5 {hashlib.md5(out_f.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
