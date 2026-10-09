"""(a) ADS-B witness on the extreme 2026 days; (b) which LIRF days blow up the z-score and why."""
import os, sys, glob
sys.path.insert(0, "src")
import numpy as np, polars as pl
import post.pru_daily_recon as R
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(16)
SCR = os.environ["SCR"]
det = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts"]) for p in sorted(glob.glob("cache/adsb_pos_v4/day=2026-01-0*.parquet"))]).filter(
    pl.col("adsb_tier").is_in(["dwell", "appear"])).select(pl.col("MVT_ID_mvt").cast(pl.Float64), "adsb_tier", "adsb_pushback_ts")
rk = pl.read_parquet("data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Float64), apt="ADEP_mvt",
    T="MVT_TIME_UTC_mvt", day=pl.col("MVT_TIME_UTC_mvt").dt.date(), nm=pl.col("FLIGHT_ID_mvt").is_not_null())
v = lambda f, c: pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
x = rk.join(det, on="MVT_ID_mvt").join(v("data/submissions/smart-jigsaw_v54.parquet", "v54"), on="MVT_ID_mvt").join(
    v("data/submissions/smart-jigsaw_v55.parquet", "v55"), on="MVT_ID_mvt").with_columns(at=pl.col("T").dt.epoch("s").cast(pl.Float64) - pl.col("adsb_pushback_ts"))
x = x.filter(pl.col("at").is_between(60, 7200))
# 2025 offsets truth - adsb per airport x tier (January)
files = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f]
h = pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Float64), apt="ADEP_mvt", T="MVT_TIME_UTC_mvt", y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
d25 = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts"]) for p in sorted(glob.glob("cache/adsb_pos_v4/day=2025-01*.parquet"))]).filter(
    pl.col("adsb_tier").is_in(["dwell", "appear"])).select(pl.col("MVT_ID_mvt").cast(pl.Float64), "adsb_tier", "adsb_pushback_ts")
h = h.join(d25, on="MVT_ID_mvt").with_columns(at=pl.col("T").dt.epoch("s").cast(pl.Float64) - pl.col("adsb_pushback_ts")).filter(pl.col("at").is_between(60, 7200))
off = h.group_by("apt", "adsb_tier").agg(o25=(pl.col("y") - pl.col("at")).median())
x = x.join(off, on=["apt", "adsb_tier"])
g = x.filter(pl.col("apt").is_in(["EHAM", "LSZH", "LFPG", "EDDF", "EDDM", "EGLL", "LEBL"])).group_by("apt", "day").agg(
    n=pl.len(), e54=(pl.col("v54") - pl.col("at") - pl.col("o25")).median(), e55=(pl.col("v55") - pl.col("at") - pl.col("o25")).median(),
    rms54=((pl.col("v54") - pl.col("at") - pl.col("o25")) ** 2).mean().sqrt(), rms55=((pl.col("v55") - pl.col("at") - pl.col("o25")) ** 2).mean().sqrt())
print("ADS-B witness by day, Jan 1-9 2026 (pred - adsb_taxi - 2025 offset; 0 = consistent with 2025 truth):")
print(g.filter((pl.col("n") >= 20) & pl.col("apt").is_in(["EHAM", "EGLL"])).sort("apt", "day").with_columns(pl.col("e54", "e55", "rms54", "rms55").round(0)))
# mean (not median) offset and RMS on tracked rows, Jan 2026, per airport
print(x.group_by("apt").agg(n=pl.len(), mean54=(pl.col("v54") - pl.col("at") - pl.col("o25")).mean(), mean55=(pl.col("v55") - pl.col("at") - pl.col("o25")).mean(),
      rms54=((pl.col("v54") - pl.col("at") - pl.col("o25")) ** 2).mean().sqrt(), rms55=((pl.col("v55") - pl.col("at") - pl.col("o25")) ** 2).mean().sqrt()).sort("apt"))
