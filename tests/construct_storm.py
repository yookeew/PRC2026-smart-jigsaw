import polars as pl, glob, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
I64=pl.col("MVT_ID_mvt").cast(pl.Int64)
ad = pl.concat([pl.read_parquet(f) for f in glob.glob("cache/adsb_pushback_v6/day=2026-01-*.parquet")+glob.glob("cache/adsb_pushback_v6/day=2025-01-*.parquet")], how="diagonal_relaxed").filter(pl.col("airport")=="EHAM")
def frame(src): return pl.scan_parquet(src).filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="EHAM")).select(I64, nmu=pl.col("FLIGHT_ID_mvt").is_null(), T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"),
       y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), day=pl.col("MVT_TIME_UTC_mvt").dt.day(), mo=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m")).collect()
r26 = frame("data/ranking/ranking.parquet").filter(pl.col("mo")=="2026-01").join(ad, on="MVT_ID_mvt").join(pl.read_parquet("data/submissions/ctxc1sel.parquet").select(I64, C=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt").join(
       pl.read_parquet("data/submissions/smart-jigsaw_v48.parquet").select(I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
r25 = frame("data/raw/training_2025-01-01_2025-02-01.parquet").join(ad, on="MVT_ID_mvt").join(pl.read_parquet("cache/eval/ctxc1sel_holdout_ev.parquet").select(I64, C=pl.col("pred").cast(pl.Float64)), on="MVT_ID_mvt", how="left")
good = pl.col("adsb_tier").is_in(["dwell","pass"])
for nm, x in (("2025-01", r25), ("2026-01", r26)):
    x = x.filter(good).with_columns(at=pl.col("T")-pl.col("adsb_pushback_ts"))
    storm = [2,3,4,5,6,9,29] if nm=="2026-01" else []
    x = x.with_columns(storm=pl.col("day").is_in(storm))
    agg = dict(n=pl.len(), C_minus_adsb=(pl.col("C")-pl.col("at")).median(), medC=pl.col("C").median(), medadsb=pl.col("at").median())
    if nm=="2025-01": agg["y_minus_adsb"]=(pl.col("y")-pl.col("at")).median()
    print(nm); print(x.group_by("nmu","storm").agg(**agg).sort("nmu","storm"))
# untracked NMU storm rows in 2026
u = frame("data/ranking/ranking.parquet").filter((pl.col("mo")=="2026-01") & pl.col("nmu")).join(ad.select("MVT_ID_mvt","adsb_tier"), on="MVT_ID_mvt", how="left").with_columns(storm=pl.col("day").is_in([2,3,4,5,6,9,29]))
print("2026-01 EHAM NMU rows by tier availability:", u.group_by("storm", tracked=pl.col("adsb_tier").is_in(["dwell","pass","appear"])).len().sort("storm","tracked").rows())
