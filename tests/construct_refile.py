import polars as pl, numpy as np
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(14)
APT = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH","LTFM"]
ts = lambda c: pl.col(c).dt.epoch("s")
for nm, src in (("2025","data/raw/training_*.parquet"),("2026","data/ranking/ranking.parquet")):
    m = pl.scan_parquet(src).select(pl.col("MVT_ID_mvt").cast(pl.Int64),"PHASE_mvt","ADEP_mvt","ADES_mvt","FLIGHT_mvt","FLIGHT_ID_mvt", T=ts("MVT_TIME_UTC_mvt"), B=ts("BLOCK_TIME_UTC_mvt"), L=ts("LOBT_flt"), y=pl.col("TAXITIME_SEC_mvt")).collect()
    dep = m.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("ADES_mvt").is_in(APT) & (pl.col("ADES_mvt")!=pl.col("ADEP_mvt")))
    arr = m.filter((pl.col("PHASE_mvt")=="ARR") & pl.col("FLIGHT_ID_mvt").is_not_null()).select("ADEP_mvt","ADES_mvt","FLIGHT_mvt", Ta="T", La="L", fa="FLIGHT_ID_mvt")
    j = dep.join(arr, on=["ADEP_mvt","ADES_mvt","FLIGHT_mvt"]).filter((pl.col("Ta")>pl.col("T")) & (pl.col("Ta")<pl.col("T")+8*3600)).sort("Ta").group_by("MVT_ID_mvt").first()
    diff = j.filter(pl.col("fa")!=pl.col("FLIGHT_ID_mvt"))
    print(f"{nm}: matched DEP to the 10 with matched ARR twin {j.height}; twin with different FLIGHT_ID {diff.height}; of those |La-L|>3606: {diff.filter((pl.col('La')-pl.col('L')).abs()>3606).height}")
    if nm=="2025" and diff.height:
        dd = diff.with_columns(inA=(pl.col("B")-pl.col("La")).abs()<=3606, LmL=(pl.col("La")-pl.col("L")))
        print("  share with block inside the twin's (other) LOBT window:", float(dd["inA"].mean()))
        print(dd.select("ADEP_mvt","FLIGHT_mvt","LmL","inA", bL=pl.col("B")-pl.col("L"), bLa=pl.col("B")-pl.col("La")).head(12))
