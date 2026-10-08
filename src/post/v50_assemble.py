"""Assemble a candidate: BASE + sum of per-row deltas, LOBT window re-applied, integrity checks.

Usage: assemble.py BASE.parquet OUT.parquet DELTA1.parquet [DELTA2.parquet ...]
Each delta file has MVT_ID_mvt + delta (seconds). Deltas are zeroed on rows the NM-unmatched lines
override (NM-unmatched, d > 6 h; src/post/nm_lines.py) and on LIRF NM-unmatched rows (team rule).
The LOBT window |BLOCK - LOBT| <= 3606 s is applied last (it held on 100% of 2025 NM-matched departures).
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
base_f, out_f, deltas = sys.argv[1], sys.argv[2], sys.argv[3:]
tpl = pl.read_parquet("data/ranking/submitting.parquet")
rk = (pl.read_parquet("data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
      .select("MVT_ID_mvt", apt="ADEP_mvt", nm=pl.col("FLIGHT_ID_mvt").is_null(),
              d=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
              lt=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64)))
b = pl.read_parquet(base_f).select("MVT_ID_mvt", base=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
x = tpl.select("MVT_ID_mvt").join(b, on="MVT_ID_mvt", how="left").join(rk, on="MVT_ID_mvt", how="left")
assert x["base"].null_count() == 0 and x["apt"].null_count() == 0
tot = np.zeros(x.height)
for f in deltas:
    dl = pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("delta").cast(pl.Float64))
    v = x.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(dl, on="MVT_ID_mvt", how="left")["delta"].fill_null(0).to_numpy()
    print(f"{os.path.basename(f)}: rows {np.sum(v != 0):,}, RMS {np.sqrt(np.mean(v ** 2)):.2f}")
    tot += v
nm, d, apt = x["nm"].to_numpy(), x["d"].fill_null(-1e9).to_numpy(), x["apt"].to_numpy()
frozen = (nm & (d > 6 * 3600)) | (nm & (apt == "LIRF"))
if os.environ.get("FREEZE_LIRF") == "1":          # every LIRF row keeps the base value
    frozen = frozen | (apt == "LIRF")
tot[frozen] = 0.0
p = x["base"].to_numpy() + tot
lt = x["lt"].to_numpy()
win = ~np.isnan(lt)
p[win] = np.clip(p[win], np.clip(lt[win] - 3606, 0, None), lt[win] + 3606)
p = np.clip(np.round(p), 0, None).astype(np.int32)
out = pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": p})
assert out.height == tpl.height == 344841
assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
assert out["TAXITIME_SEC_mvt"].null_count() == 0 and (out["TAXITIME_SEC_mvt"] >= 0).all()
dd = p - x["base"].to_numpy()
print(f"vs base: RMS {np.sqrt(np.mean(dd ** 2)):.2f} s, mean {dd.mean():+.3f}, rows changed {(dd != 0).sum():,}, "
      f"max |d| {np.abs(dd).max():.0f}; frozen rows {frozen.sum()}; LIRF NM-unmatched identical "
      f"{(dd[nm & (apt == 'LIRF')] == 0).all()}")
for a in sorted(set(apt)):
    k = apt == a
    print(f"  {a} RMS {np.sqrt(np.mean(dd[k] ** 2)):7.2f} mean {dd[k].mean():+7.2f}")
out.write_parquet(out_f)
print("wrote", out_f)
