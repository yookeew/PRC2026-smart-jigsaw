exec(open(os.environ["SCR"] + "/recon2.py").read().split("for nm_, L, wv")[0])
q_day = np.maximum(ts - (dh + np.nan_to_num(0.75 * L1)), 0.0)
rep("day a0.75", q_day)
# monthly taxi-out level: published mean taxi per airport-month vs label mean, offset calibrated on the 10 non-holdout months
tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="vertical_relaxed").filter(
    (pl.col("PHASE_mvt") == "DEP") & pl.col("TAXITIME_SEC_mvt").is_not_null())
lab = tr.group_by(apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month()).agg(lm=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).mean(),
      lm_cap=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).filter(pl.col("TAXITIME_SEC_mvt") <= 3600).mean())
to = pl.read_csv(f"{SCR}/ext/taxi_out_additional_time_2025.csv", infer_schema_length=0).select(
    apt="APT_ICAO", mo=pl.col("MONTH_NUM").cast(pl.Int8), pm=(pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64) + pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64)) / pl.col("TOTAL_REF_NB_FL").cast(pl.Float64) * 60)
x = lab.with_columns(pl.col("mo").cast(pl.Int8)).join(to, on=["apt", "mo"])
off = x.filter(~pl.col("mo").is_in([1, 7])).group_by("apt").agg(off=(pl.col("lm_cap") - pl.col("pm")).mean(), sd=(pl.col("lm_cap") - pl.col("pm")).std())
est = x.filter(pl.col("mo").is_in([1, 7])).join(off, on="apt").with_columns(est=pl.col("pm") + pl.col("off"))
print(est.sort("apt", "mo").select("apt", "mo", "lm_cap", "est", "sd"))
for base_name, qb in (("plain", p), ("after day", q_day)):
    for beta in (0.5, 1.0):
        q = qb.copy()
        for row in est.iter_rows(named=True):
            m = (apt == row["apt"]) & (mo == row["mo"])
            body_pred = m & (qb <= 3600)
            shift = row["est"] - qb[body_pred].mean()
            q[m] = np.maximum(qb[m] + beta * shift, 0)
        rep(f"month b{beta} on {base_name}", q)
