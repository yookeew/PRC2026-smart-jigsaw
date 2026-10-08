import polars as pl, glob, numpy as np
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
I64=pl.col("MVT_ID_mvt").cast(pl.Int64); ts=lambda c: pl.col(c).dt.epoch("s")
def frame(src): return pl.scan_parquet(src).filter(pl.col("PHASE_mvt")=="DEP").select(I64, apt="ADEP_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null(), T=ts("MVT_TIME_UTC_mvt"),
    off=(ts("MVT_TIME_UTC_mvt")-ts("SCHED_TIME_UTC_mvt")).cast(pl.Float64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), day=pl.col("MVT_TIME_UTC_mvt").dt.date()).collect()
ad = pl.concat([pl.read_parquet(f) for f in glob.glob("cache/adsb_pushback_v6/day=2025-0[17]-*.parquet")], how="diagonal_relaxed").select("MVT_ID_mvt","adsb_tier","adsb_pushback_ts")
good = pl.col("adsb_tier").is_in(["dwell","pass"])
r = frame([f"data/raw/training_2025-{a}-01_2025-{b}-01.parquet" for a,b in (("01","02"),("07","08"))]).join(ad, on="MVT_ID_mvt", how="left").join(
    pl.read_parquet("cache/eval/ctxc1sel_holdout_ev.parquet").select(I64, C=pl.col("pred").cast(pl.Float64)), on="MVT_ID_mvt").join(pl.read_parquet("cache/eval/v42_chain_holdout.parquet").select(I64, P="v42"), on="MVT_ID_mvt")
r = r.with_columns(at=pl.col("T")-pl.col("adsb_pushback_ts"), late=pl.col("off")>=3600)
dd = r.group_by("apt","day").agg(md=pl.col("off").median()); dd = dd.join(dd.group_by("apt").agg(q=pl.col("md").quantile(0.85)), on="apt").with_columns(disr=pl.col("md")>pl.col("q")).select("apt","day","disr")
r = r.join(dd, on=["apt","day"])
seg = ["apt","mo","nmu","disr","late"]
tot_old = float(((r["y"]-r["P"])**2).sum()); tot_d = 0; rows=[]
for m, other in ((1,7),(7,1)):
    lag = r.filter((pl.col("mo")==other) & good & pl.col("at").is_between(60,7200)).group_by("apt").agg(lag=(pl.col("y")-pl.col("at")).median())
    x = r.filter(pl.col("mo")==m).join(lag, on="apt", how="left")
    b = x.filter(good & pl.col("at").is_between(60,7200)).with_columns(bC=pl.col("at")+pl.col("lag")-pl.col("C")).group_by(seg).agg(n_tr=pl.len(), biasC=pl.col("bC").median())
    u = x.filter(pl.col("adsb_tier").is_null()).join(b.filter(pl.col("n_tr")>=50), on=seg, how="inner")
    PmC = u.group_by(seg).agg(PmC=(pl.col("P")-pl.col("C")).median())
    u = u.join(PmC, on=seg).with_columns(shift=pl.col("biasC")-pl.col("PmC")).filter(pl.col("shift").is_not_null())
    for shrink in (1.0, 0.5):
        Q = u["P"] + shrink*u["shift"]
        d = float(((u["y"]-Q)**2).sum() - ((u["y"]-u["P"])**2).sum())
        rows.append((m, shrink, u.height, d))
        if shrink==1.0: tot_d += d
    # truth check of the segment bias on untracked rows
    chk = u.group_by(seg).agg(n=pl.len(), est=pl.col("shift").first(), true_resid=(pl.col("y")-pl.col("P")).clip(-1500,1500).mean()).sort("n", descending=True).head(8)
    print(f"month {m}:"); print(chk)
print(pl.DataFrame(rows, schema=["month","shrink","rows","dSSE"], orient="row"))
print(f"holdout full RMSE {np.sqrt(tot_old/r.height):.3f} -> {np.sqrt((tot_old+tot_d)/r.height):.3f} (shrink 1)")
