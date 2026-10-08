import polars as pl, glob, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
I64=pl.col("MVT_ID_mvt").cast(pl.Int64)
ad = pl.concat([pl.read_parquet(f) for f in glob.glob("cache/adsb_pushback_v6/day=202[56]-0[17]-*.parquet")], how="diagonal_relaxed").filter(pl.col("adsb_tier").is_in(["dwell","pass","appear"]))
def frame(src): return pl.scan_parquet(src).filter(pl.col("PHASE_mvt")=="DEP").select(I64, apt="ADEP_mvt", T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
     mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), nmu=pl.col("FLIGHT_ID_mvt").is_null()).collect()
r25 = frame([f"data/raw/training_2025-{a}-01_2025-{b}-01.parquet" for a,b in (("01","02"),("07","08"))]).join(ad, on="MVT_ID_mvt").join(pl.read_parquet("cache/eval/v42_chain_holdout.parquet").select(I64, P="v42"), on="MVT_ID_mvt")
r26 = frame("data/ranking/ranking.parquet").join(ad, on="MVT_ID_mvt").join(pl.read_parquet("data/submissions/smart-jigsaw_v48.parquet").select(I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
k = lambda x: x.filter(~pl.col("nmu")).with_columns(at=pl.col("T")-pl.col("adsb_pushback_ts")).filter(pl.col("at").is_between(60, 7200))
a25 = k(r25).group_by("apt","mo","adsb_tier").agg(n25=pl.len(), truth_m_adsb=(pl.col("y")-pl.col("at")).clip(-1200,1200).mean(), v42_m_adsb=(pl.col("P")-pl.col("at")).clip(-1200,1200).mean())
a26 = k(r26).group_by("apt","mo","adsb_tier").agg(n26=pl.len(), v48_m_adsb=(pl.col("P")-pl.col("at")).clip(-1200,1200).mean())
x = a26.join(a25, on=["apt","mo","adsb_tier"], how="left").with_columns(implied_bias=pl.col("v48_m_adsb")-pl.col("truth_m_adsb"), holdout_bias=pl.col("v42_m_adsb")-pl.col("truth_m_adsb")).filter(pl.col("n26")>=100).sort("apt","mo","adsb_tier")
print(x)
