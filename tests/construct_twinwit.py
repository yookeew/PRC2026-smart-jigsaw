import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
exec(open(r"C:/Users/yuqiw/AppData/Local/Temp/claude/c--Users-yuqiw-prc2025/3cfb4df4-9874-4828-bc10-eee943f0b11f/scratchpad/twin.py").read().split("m = load(")[0])
I64=pl.col("MVT_ID_mvt").cast(pl.Int64)
def prep(j):
    return j.with_columns(off=pl.col("T")-pl.col("S"), tl=pl.col("T")-pl.col("La"), ta=(pl.col("T")-pl.col("A3a")).cast(pl.Float64), mo=pl.from_epoch("T").dt.month()).filter(
        (pl.col("ADEP_mvt")!="LIRF") & (pl.col("ADES_mvt")!=pl.col("ADEP_mvt")) & pl.col("ta").is_not_null() & (pl.col("tl")>=3726))
_, j25 = twins(load("data/raw/training_*.parquet")); j25 = prep(j25)
_, j26 = twins(load("data/ranking/ranking.parquet")); j26 = prep(j26)
j26 = j26.join(pl.read_parquet("data/submissions/smart-jigsaw_v48.parquet").select(I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
j25h = j25.join(pl.read_parquet("cache/eval/v42_chain_holdout.parquet").select(I64, P="v42"), on="MVT_ID_mvt", how="left")
print("2025 (truth): y - ta by airport (late-branch twins, non-LIRF)")
print(j25h.group_by("ADEP_mvt").agg(n=pl.len(), y_minus_ta=(pl.col("y")-pl.col("ta")).median(), y_minus_ta_mean=(pl.col("y")-pl.col("ta")).clip(-3000,3000).mean(),
      P_minus_ta=(pl.col("P")-pl.col("ta")).median(), nP=pl.col("P").is_not_null().sum()).sort("ADEP_mvt"))
print("2026 (v48): P - ta by airport x month")
print(j26.group_by("ADEP_mvt","mo").agg(n=pl.len(), P_minus_ta=(pl.col("P")-pl.col("ta")).median(), P_minus_ta_mean=(pl.col("P")-pl.col("ta")).clip(-3000,3000).mean(), medP=pl.col("P").median(), medta=pl.col("ta").median()).sort("ADEP_mvt","mo"))
print(j26.filter((pl.col("ADEP_mvt")=="EHAM")&(pl.col("mo")==1)).select("FLIGHT_mvt","ADES_mvt", offh=pl.col("off")/3600, tl=pl.col("tl"), ta=pl.col("ta"), P=pl.col("P"), day=pl.from_epoch("T").dt.day()).sort("day"))
r = pl.read_parquet("data/ranking/ranking.parquet").filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="EHAM")).with_columns(I64, mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), day=pl.col("MVT_TIME_UTC_mvt").dt.day(),
    nmu=pl.col("FLIGHT_ID_mvt").is_null(), off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(), a=(pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds())
r = r.join(pl.read_parquet("data/submissions/smart-jigsaw_v48.parquet").select(I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
print("EHAM Jan 2026 by day: NM-unmatched count, NMU median P, NMU median off; matched median P and median anchor (T-AOBT3)")
print(r.filter(pl.col("mo")==1).group_by("day").agg(nmu=pl.col("nmu").sum(), P_nmu=pl.col("P").filter(pl.col("nmu")).median(), off_nmu=pl.col("off").filter(pl.col("nmu")).median(),
      P_m=pl.col("P").filter(~pl.col("nmu")).median(), a_m=pl.col("a").filter(~pl.col("nmu")).median(), n=pl.len()).sort("day"))
