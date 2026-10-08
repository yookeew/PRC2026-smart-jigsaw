"""Scoping (PROGRESS §117): airline NM-matched rate among late LIRF departures as an echo signal."""
import sys; sys.path.insert(0,"src")
import polars as pl, numpy as np, glob
from post.lirf_signals import p_2025, p_2026, I64
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
def frame(src):
    return pl.scan_parquet(src).filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="LIRF")).select(I64, al=pl.col("FLIGHT_mvt").str.slice(0,3),
        nmu=pl.col("FLIGHT_ID_mvt").is_null(), off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), mt=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m")).collect()
d = frame("data/raw/training_*.parquet").join(p_2025(), on="MVT_ID_mvt", how="left")
late = d.filter(pl.col("off")>=3600)
# per (airline, month) counts -> leave-own-month-out matched rate among delayed departures
g = late.group_by("al","mt").agg(n=pl.len(), nm=(~pl.col("nmu")).sum())
tot = g.group_by("al").agg(N=pl.col("n").sum(), NM=pl.col("nm").sum())
x = late.join(tot, on="al", how="left").join(g, on=["al","mt"], how="left").with_columns(
    rate=((pl.col("NM")-pl.col("nm"))/(pl.col("N")-pl.col("n"))), cnt=pl.col("N")-pl.col("n"))
u = x.filter(pl.col("nmu") & pl.col("y").is_not_null()).with_columns(echo=(pl.col("off")-pl.col("y")).abs()<30)
u = u.with_columns(band=pl.when(pl.col("cnt").fill_null(0)<5).then(pl.lit("few")).otherwise(pl.col("rate").cut([0.3,0.6,0.8,0.9,0.95]).cast(pl.Utf8)))
print(u.group_by("band").agg(n=pl.len(), echo=pl.col("echo").mean(), p=pl.col("p").mean()).sort("band"))
print(u.filter(pl.col("cnt")>=5).select(pl.corr("rate","echo"), pl.corr("p","echo")))
# same-period version (label-free in 2026): rate within the same month incl. own row's month (matched rate is label-free)
x2 = late.join(g, on=["al","mt"], how="left").with_columns(rate_m=pl.col("nm")/pl.col("n"))
u2 = x2.filter(pl.col("nmu")).with_columns(echo=(pl.col("off")-pl.col("y")).abs()<30, band=pl.col("rate_m").cut([0.3,0.6,0.8,0.9,0.95]).cast(pl.Utf8))
print("same-month rate:"); print(u2.group_by("band").agg(n=pl.len(), echo=pl.col("echo").mean(), p=pl.col("p").mean()).sort("band"))
