"""Independent 2026 witnesses for the day shift: (1) ADS-B taxi on pushback-tier rows, (2) anchor T - AOBT_3 on NM-matched rows.
Offset = mean(pred - witness) in 2026 minus mean(truth - witness) in the same airport x month of 2025 (median-robust too).
A correct level change moves v55's offset toward 0 relative to v54's."""
import os, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
SCR = os.environ["SCR"]
def det(pat):
    x = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts"]) for p in sorted(glob.glob(f"cache/adsb_pos_v4/{pat}"))])
    return x.filter(pl.col("adsb_tier").is_in(["dwell", "appear"])).select(pl.col("MVT_ID_mvt").cast(pl.Float64), "adsb_tier", "adsb_pushback_ts")
def base(df):
    return df.filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Float64), apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        T="MVT_TIME_UTC_mvt", nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
        anc=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
files = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f or "2025-07-01" in f]
h = base(pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")).join(det("day=2025-0[17]*.parquet"), on="MVT_ID_mvt", how="left")
r = base(pl.read_parquet("data/ranking/ranking.parquet")).join(det("day=2026-*.parquet"), on="MVT_ID_mvt", how="left")
for nm, f in (("v54", "data/submissions/smart-jigsaw_v54.parquet"), ("v55", f"{SCR}/v55_candidate.parquet")):
    r = r.join(pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(nm)), on="MVT_ID_mvt", how="left")
def adsb_taxi(df):
    return (df["T"].cast(pl.Int64) // 1_000_000 - df["adsb_pushback_ts"]).cast(pl.Float64) if df["adsb_pushback_ts"].dtype != pl.Datetime else (df["T"] - df["adsb_pushback_ts"]).dt.total_seconds()
print("pushback ts dtype:", h["adsb_pushback_ts"].dtype, " T dtype:", h["T"].dtype)
h = h.with_columns(at=pl.col("T").dt.epoch("s").cast(pl.Float64) - pl.col("adsb_pushback_ts"))
r = r.with_columns(at=pl.col("T").dt.epoch("s").cast(pl.Float64) - pl.col("adsb_pushback_ts"))
ok = lambda c: pl.col(c).is_not_null() & (pl.col(c) > 60) & (pl.col(c) < 3600)
# ADS-B witness, per airport x month x tier, robust (median of differences)
w25 = h.filter(ok("at") & (pl.col("y") <= 3600)).group_by("apt", "mo", "adsb_tier").agg(o25=(pl.col("y") - pl.col("at")).median(), n25=pl.len())
w26 = r.filter(ok("at")).group_by("apt", "mo", "adsb_tier").agg(o54=(pl.col("v54") - pl.col("at")).median(), o55=(pl.col("v55") - pl.col("at")).median(), n26=pl.len())
a = w26.join(w25, on=["apt", "mo", "adsb_tier"]).filter((pl.col("n25") >= 100) & (pl.col("n26") >= 100))
a = a.with_columns(e54=pl.col("o54") - pl.col("o25"), e55=pl.col("o55") - pl.col("o25"))
agg = a.group_by("apt", "mo").agg(n26=pl.col("n26").sum(), e54=(pl.col("e54") * pl.col("n26")).sum() / pl.col("n26").sum(),
                                   e55=(pl.col("e55") * pl.col("n26")).sum() / pl.col("n26").sum()).sort("apt", "mo")
print("ADS-B witness (pushback tier; pred - adsb_taxi in 2026 minus truth - adsb_taxi in 2025, median, s):")
print(agg.with_columns(pl.col("e54", "e55").round(1), closer=(pl.col("e55").abs() < pl.col("e54").abs())))
# anchor witness, NM-matched rows with 60 < anchor < 3600
b25 = h.filter(pl.col("nm") & ok("anc") & (pl.col("y") <= 3600)).group_by("apt", "mo").agg(o25=(pl.col("y") - pl.col("anc")).median())
b26 = r.filter(pl.col("nm") & ok("anc")).group_by("apt", "mo").agg(o54=(pl.col("v54") - pl.col("anc")).median(), o55=(pl.col("v55") - pl.col("anc")).median(), n26=pl.len())
b = b26.join(b25, on=["apt", "mo"]).with_columns(e54=pl.col("o54") - pl.col("o25"), e55=pl.col("o55") - pl.col("o25")).sort("apt", "mo")
print("Anchor witness (T - AOBT_3; NM-matched; median):")
print(b.select("apt", "mo", "n26", pl.col("e54").round(1), pl.col("e55").round(1), closer=(pl.col("e55").abs() < pl.col("e54").abs())))
