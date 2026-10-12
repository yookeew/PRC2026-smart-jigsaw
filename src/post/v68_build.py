"""v68 (reports/ctx_huber_preregistration.md): a corrector change carried onto the current best file.

  OUT = BASE + (B_treat - B_ctl), then the LOBT window, clip at 0, round (final4m_v62.finish).

B_treat and B_ctl are full builds (v54 + chain delta, assembly, v63's day recipe) of the treatment corrector and of its
control, both built on this machine, so the difference carries the treatment only (refit noise of the shared parts
cancels). LIRF / LSZH rows are equal in both builds (taken from v65), so they keep BASE's value.

Run:  .venv/Scripts/python.exe src/post/v68_build.py BASE B_TREAT B_CTL OUT
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "post"))
import final4m_v62 as F  # noqa: E402
import scope  # noqa: E402


def main(base_f, bt_f, bc_f, out_f):
    tpl = pl.read_parquet(scope.TEMPLATE).select(F.F64)
    rk = pl.read_parquet(scope.RANKING).filter(pl.col("PHASE_mvt") == "DEP").with_columns(F.F64)
    rd = lambda f, c: pl.read_parquet(f).select(F.F64, pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
    x = (tpl.join(rk, on="MVT_ID_mvt", how="left").join(rd(base_f, "b"), on="MVT_ID_mvt", how="left")
         .join(rd(bt_f, "t"), on="MVT_ID_mvt", how="left").join(rd(bc_f, "c"), on="MVT_ID_mvt", how="left"))
    assert x.height == tpl.height and all(x[c].null_count() == 0 for c in ("b", "t", "c", "ADEP_mvt"))
    d = (x["t"] - x["c"]).to_numpy()
    apt = x["ADEP_mvt"].to_numpy()
    assert np.all(d[np.isin(apt, ["LIRF", "LSZH"])] == 0), "LIRF / LSZH must be equal in both builds"
    q = F.finish(x, x["b"].to_numpy() + d)
    out = q.astype(np.int64) - x["b"].to_numpy().astype(np.int64)
    print(f"build delta RMS {np.sqrt(np.mean(d ** 2)):.2f}, mean {d.mean():+.3f}; vs BASE rows changed {(out != 0).sum():,}, "
          f"RMS {np.sqrt(np.mean(out ** 2.0)):.2f}, mean {out.mean():+.3f}, max |d| {np.abs(out).max()}")
    for a in sorted(set(apt)):
        m = apt == a
        print(f"  {a}: RMS {np.sqrt(np.mean(out[m] ** 2.0)):6.2f}  mean {out[m].mean():+6.2f}")
    sub = pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": q})
    t0 = pl.read_parquet(scope.TEMPLATE)
    sub = t0.select("MVT_ID_mvt").join(sub.with_columns(pl.col("MVT_ID_mvt").cast(t0["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
    assert sub.height == t0.height and sub["TAXITIME_SEC_mvt"].null_count() == 0 and (sub["TAXITIME_SEC_mvt"] >= 0).all()
    sub.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(*sys.argv[1:5])
