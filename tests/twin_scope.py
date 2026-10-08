"""Scoping of NM-matched arrival twins for NM-unmatched departures (PROGRESS §116): join-window structure, 2026 reach."""
import polars as pl, glob, numpy as np
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
APT = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH","LTFM"]
def load(src):
    ts = lambda c: pl.col(c).dt.epoch("s")
    return pl.scan_parquet(src).select(pl.col("MVT_ID_mvt").cast(pl.Int64),"PHASE_mvt","ADEP_mvt","ADES_mvt","FLIGHT_mvt","FLIGHT_ID_mvt",
        T=ts("MVT_TIME_UTC_mvt"), S=ts("SCHED_TIME_UTC_mvt"), B=ts("BLOCK_TIME_UTC_mvt"), L=ts("LOBT_flt"), A3=ts("AOBT_3_flt"), E1=ts("EOBT_1_flt"),
        y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).collect()
def twins(m):
    dep = m.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("FLIGHT_ID_mvt").is_null() & pl.col("ADES_mvt").is_in(APT) & pl.col("FLIGHT_mvt").is_not_null())
    arr = m.filter((pl.col("PHASE_mvt")=="ARR") & pl.col("FLIGHT_ID_mvt").is_not_null()).select(
        pl.col("ADEP_mvt"), pl.col("ADES_mvt"), "FLIGHT_mvt", Ta="T", La="L", A3a="A3", E1a="E1", fid="FLIGHT_ID_mvt")
    j = dep.join(arr, on=["ADEP_mvt","ADES_mvt","FLIGHT_mvt"]).filter((pl.col("Ta")>pl.col("T")) & (pl.col("Ta")<pl.col("T")+8*3600))
    j = j.sort("Ta").group_by("MVT_ID_mvt").first()
    return dep, j
m = load("data/raw/training_*.parquet")
dep, j = twins(m)
print(f"2025 NM-unmatched DEP: {m.filter((pl.col('PHASE_mvt')=='DEP') & pl.col('FLIGHT_ID_mvt').is_null()).height}; to one of the 10 airports {dep.height}; with NM-matched arrival twin {j.height}")
j = j.with_columns(off=pl.col("T")-pl.col("S"), tl=pl.col("T")-pl.col("La"), ta=pl.col("T")-pl.col("A3a"), bl=pl.col("B")-pl.col("La"))
print("share |B - LOBT_twin| <= 3606:", float((j["bl"].abs()<=3606).mean()))
print(j.group_by("ADEP_mvt").agg(n=pl.len(), in_win=(pl.col("bl").abs()<=3606).mean(), med_y=pl.col("y").median(), rmse_anchor=((pl.col("y")-pl.col("ta"))**2).mean().sqrt(), sd_y=pl.col("y").std(), corr=pl.corr("y","ta")).sort("n",descending=True))
print(j.select("ADEP_mvt","ADES_mvt","FLIGHT_mvt","off","y","tl","ta","bl").sort("off",descending=True).head(25))
rk = load("data/ranking/ranking.parquet"); d26, j26 = twins(rk)
print(f"2026 NM-unmatched DEP to the 10: {d26.height}; with twin {j26.height}; by airport", j26.group_by("ADEP_mvt").len().sort("len",descending=True).rows())
j.write_parquet(r"cache/eval/twin25.parquet")
