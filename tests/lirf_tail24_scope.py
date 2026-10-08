"""Scoping (PROGRESS §118): echo vs +24h for LIRF NM-unmatched >= 12 h rows, label-free signals."""
import sys; sys.path.insert(0,"src")
import polars as pl, numpy as np
import post.lirf_signals as L, post.lirf_hedge2 as H
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(16)
src = str(L.RAW/"training_*.parquet"); hist = H.airline_late(src)
d = H.with_loo_rate(H.frame25(), hist).filter(pl.col("nmu") & (pl.col("off")>=43200))
d = d.with_columns(kind=pl.when((pl.col("off")-pl.col("y")).abs()<60).then(pl.lit("echo")).otherwise(pl.lit("+24h")),
                   hS=pl.from_epoch("S").dt.convert_time_zone("Europe/Rome").dt.hour(), hT=pl.from_epoch("T").dt.convert_time_zone("Europe/Rome").dt.hour(),
                   dd=(pl.from_epoch("T").dt.convert_time_zone("Europe/Rome").dt.date()-pl.from_epoch("S").dt.convert_time_zone("Europe/Rome").dt.date()).dt.total_days())
print(d.select("kind","FLIGHT_mvt","off","y","rate","N","nF","nD","nA","p","hS","hT","dd","STAND_mvt").sort("kind","off"))
print(d.group_by("kind").agg(n=pl.len(), rate=pl.col("rate").mean(), few=(pl.col("N")<5).mean(), p=pl.col("p").mean(), nF0=(pl.col("nF")==0).mean(), dd1=(pl.col("dd")>=1).mean()))
rk = str(L.RANKING)
z = L.frame(rk).join(H.airline_late(rk).select("MVT_ID_mvt","al"), on="MVT_ID_mvt", how="left").join(L.p_2026(), on="MVT_ID_mvt", how="left")
z = H.add_rate(z, H.rate_table(hist)).filter(pl.col("nmu") & (pl.col("off")>=43200)).with_columns(
    dd=(pl.from_epoch("T").dt.convert_time_zone("Europe/Rome").dt.date()-pl.from_epoch("S").dt.convert_time_zone("Europe/Rome").dt.date()).dt.total_days())
print(z.select("FLIGHT_mvt","off","rate","N","nF","nD","nA","p","dd").sort("off"))
