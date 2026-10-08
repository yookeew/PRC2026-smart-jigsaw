import polars as pl, glob, numpy as np
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(16)
I64=pl.col("MVT_ID_mvt").cast(pl.Int64)
d = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="diagonal_relaxed").filter((pl.col("PHASE_mvt")=="DEP") & pl.col("FLIGHT_ID_mvt").is_not_null())
ts=lambda c: pl.col(c).dt.epoch("s")
d = d.with_columns(I64, T=ts("MVT_TIME_UTC_mvt"), B=ts("BLOCK_TIME_UTC_mvt"), S=ts("SCHED_TIME_UTC_mvt"), L=ts("LOBT_flt"), Io=ts("IOBT_flt"), E=ts("EOBT_1_flt"), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), m=pl.col("MVT_TIME_UTC_mvt").dt.month())
d = d.with_columns(cL=(pl.col("B")==pl.col("L")), cS=(pl.col("B")==pl.col("S")), cI=(pl.col("B")==pl.col("Io")), cE=(pl.col("B")==pl.col("E")), LneS=(pl.col("L")!=pl.col("S")))
print("exact (to the second) copies, NM-matched departures:")
print(d.group_by("ADEP_mvt").agg(n=pl.len(), B_eq_L=pl.col("cL").mean(), B_eq_S=pl.col("cS").mean(), B_eq_I=pl.col("cI").mean(), B_eq_E=pl.col("cE").mean(),
     B_eq_L_and_LneS=(pl.col("cL")&pl.col("LneS")).mean(), y_copyL=pl.col("y").filter(pl.col("cL")).median()).sort("ADEP_mvt"))
# holdout cost of exact-LOBT-copy rows
h = d.join(pl.read_parquet("cache/eval/v42_chain_holdout.parquet").select(I64,"v42"), on="MVT_ID_mvt")
tot = float(((h["y"]-h["v42"])**2).sum())
for a in ["LEBL","LEMD","LFPG","EDDF","EHAM","LSZH"]:
    z = h.filter((pl.col("ADEP_mvt")==a) & pl.col("cL"))
    zz = z.filter(pl.col("LneS"))
    print(f"{a}: holdout B==LOBT rows {z.height} (LOBT!=SCHED {zz.height}); SSE share {float(((z['y']-z['v42'])**2).sum())/tot:.4f}; mean y {z['y'].mean():.0f} vs v42 {z['v42'].mean():.0f}; "
          f"on LOBT!=SCHED: mean y {zz['y'].mean() if zz.height else 0:.0f} vs v42 {zz['v42'].mean() if zz.height else 0:.0f}, mean T-LOBT {(zz['T']-zz['L']).mean() if zz.height else 0:.0f}")
