"""How well do EUROCONTROL's published series track our 2025 labels?  (analysis only, scratch)
(1) monthly taxi-out (ref + additional time) per airport vs our label mean;  (2) daily all-cause pre-departure delay
per airport vs our daily mean of d = BLOCK - SCHED (several definitions)."""
import os, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(20)
SCR = os.environ["SCR"]
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="vertical_relaxed")
dep = (tr.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("TAXITIME_SEC_mvt").is_not_null())
       .with_columns(y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), m=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                     day=pl.col("BLOCK_TIME_UTC_mvt").dt.date(),
                     d=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() / 60,
                     ifr=pl.coalesce("FLIGHT_RULE_flt", "FLIGHT_RULE_mvt")))
to = pl.read_csv(f"{SCR}/ext/taxi_out_additional_time_2025.csv", infer_schema_length=0).filter(pl.col("APT_ICAO").is_in(APT)).select(
    apt="APT_ICAO", m=pl.col("MONTH_NUM").cast(pl.Int32), tf=pl.col("TF").cast(pl.Float64), valid=pl.col("VALID_FL").cast(pl.Float64),
    nref=pl.col("TOTAL_REF_NB_FL").cast(pl.Float64), ref=pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64), add=pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64))
to = to.with_columns(pru_taxi=(pl.col("ref") + pl.col("add")) / pl.col("nref") * 60, pru_add=pl.col("add") / pl.col("nref") * 60)
ours = dep.group_by(apt="ADEP_mvt", m=pl.col("m").cast(pl.Int32)).agg(n=pl.len(), mean_y=pl.col("y").mean(),
    mean_y_cap=pl.col("y").filter(pl.col("y") <= 3600).mean(), med_y=pl.col("y").median())
j = ours.join(to, on=["apt", "m"], how="inner").with_columns(diff=pl.col("mean_y_cap") - pl.col("pru_taxi"))
print(j.sort("apt", "m").select("apt", "m", "n", "tf", "valid", "nref", "mean_y", "mean_y_cap", "pru_taxi", "pru_add", "diff"))
print(j.group_by("apt").agg(diff_mean=pl.col("diff").mean(), diff_sd=pl.col("diff").std(),
      r=pl.corr("mean_y_cap", "pru_taxi")).sort("apt"))
pdd = pl.read_csv(f"{SCR}/ext/all_pre_departure_delays_2025.csv", infer_schema_length=0).filter(pl.col("APT_ICAO").is_in(APT)).select(
    apt="APT_ICAO", day=pl.col("FLT_DATE").str.to_date(), n1=pl.col("FLT_DEP_1").cast(pl.Float64),
    n2=pl.col("FLT_DEP_IFR_2").cast(pl.Float64), dly=pl.col("DLY_ALL_PRE_2").cast(pl.Float64))
print("pre-dep rows per airport with values:", pdd.filter(pl.col("dly").is_not_null()).group_by("apt").len().sort("apt"))
od = dep.group_by(apt="ADEP_mvt", day="day").agg(n=pl.len(), d_mean=pl.col("d").mean(), d_pos=pl.col("d").clip(0, None).mean(),
    d_pos_sum=pl.col("d").clip(0, None).sum(), d_cap=pl.col("d").clip(-60, 600).clip(0, None).mean())
jd = od.join(pdd, on=["apt", "day"], how="inner").with_columns(pru=pl.col("dly") / pl.col("n2"))
print(jd.group_by("apt").agg(days=pl.len(), n_ratio=(pl.col("n2") / pl.col("n")).median(),
      r_mean=pl.corr("d_mean", "pru"), r_pos=pl.corr("d_pos", "pru"), r_cap=pl.corr("d_cap", "pru"),
      sum_ratio=(pl.col("d_pos_sum") / pl.col("dly")).median(), pru_med=pl.col("pru").median(), dpos_med=pl.col("d_pos").median()).sort("apt"))
jd.write_parquet(f"{SCR}/ext/daily_join_2025.parquet")
