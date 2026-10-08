import polars as pl, glob, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
I64=pl.col("MVT_ID_mvt").cast(pl.Int64)
ts=lambda c: pl.col(c).dt.epoch("s")
def frame(src): return pl.scan_parquet(src).filter((pl.col("PHASE_mvt")=="DEP") & pl.col("FLIGHT_ID_mvt").is_not_null()).select(I64, apt="ADEP_mvt", T=ts("MVT_TIME_UTC_mvt"),
     a=(ts("MVT_TIME_UTC_mvt")-ts("AOBT_3_flt")).cast(pl.Float64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), mo=pl.col("MVT_TIME_UTC_mvt").dt.month()).collect()
ad = pl.concat([pl.read_parquet(f) for f in glob.glob("cache/adsb_pushback_v6/day=202[56]-0[17]-*.parquet")], how="diagonal_relaxed").filter(pl.col("adsb_tier").is_in(["dwell","pass","appear"])).select("MVT_ID_mvt","adsb_tier","adsb_pushback_ts")
d25 = frame(sorted(glob.glob("data/raw/training_*.parquet"))).filter(pl.col("mo").is_in([1,7]))
r26 = frame("data/ranking/ranking.parquet").join(pl.read_parquet("data/submissions/smart-jigsaw_v49.parquet").select(I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
cm = lambda e: e.clip(-300,300).mean()
A25 = d25.filter(pl.col("a").is_not_null()).group_by("apt","mo").agg(t_a=cm(pl.col("y")-pl.col("a")))
A26 = r26.filter(pl.col("a").is_not_null()).group_by("apt","mo").agg(n=pl.len(), p_a=cm(pl.col("P")-pl.col("a")))
# 2025 monthly range of the truth-anchor offset (all 12 months) for context
rng = frame(sorted(glob.glob("data/raw/training_*.parquet"))).filter(pl.col("a").is_not_null()).group_by("apt","mo").agg(t=cm(pl.col("y")-pl.col("a"))).group_by("apt").agg(t_min=pl.col("t").min(), t_max=pl.col("t").max())
def adw(x, col):
    x = x.join(ad, on="MVT_ID_mvt").with_columns(at=pl.col("T")-pl.col("adsb_pushback_ts")).filter(pl.col("at").is_between(60,7200))
    return x.group_by("apt","mo","adsb_tier").agg(n=pl.len(), v=cm(pl.col(col)-pl.col("at")))
B25 = adw(d25, "y").rename({"v":"t_b","n":"n25"}); B26 = adw(r26, "P").rename({"v":"p_b","n":"n26"})
B = B26.join(B25, on=["apt","mo","adsb_tier"]).filter((pl.col("n26")>=300)&(pl.col("n25")>=100)).with_columns(bias=pl.col("p_b")-pl.col("t_b"))
Bm = B.group_by("apt","mo").agg(adsb_bias=(pl.col("bias")*pl.col("n26")).sum()/pl.col("n26").sum(), adsb_n=pl.col("n26").sum())
x = A26.join(A25, on=["apt","mo"]).join(rng, on="apt").with_columns(anchor_bias=pl.col("p_a")-pl.col("t_a")).join(Bm, on=["apt","mo"], how="left").sort("apt","mo")
x = x.with_columns(agree=(pl.col("anchor_bias")*pl.col("adsb_bias")>0) & (pl.col("anchor_bias").abs()>15) & (pl.col("adsb_bias").abs()>15),
                   gain_s=(pl.col("n")/344841)*pl.min_horizontal(pl.col("anchor_bias").abs(), pl.col("adsb_bias").abs().fill_null(0))**2/(2*245.38))
print(x.select("apt","mo","t_min","t_max","anchor_bias","adsb_bias","agree"))
