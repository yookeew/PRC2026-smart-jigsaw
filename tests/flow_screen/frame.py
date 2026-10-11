import glob, os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
ROOT="C:/Users/zhang/smart-jigsaw"; S="C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"
raw=pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(ROOT+"/data/raw/training_2025-*.parquet"))],how="vertical_relaxed")

ep=lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
dep=raw.filter(pl.col("PHASE_mvt")=="DEP").select(
  pl.col("MVT_ID_mvt").cast(pl.Int64),apt="ADEP_mvt",ades="ADES_mvt",
  y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),nm=pl.col("FLIGHT_ID_mvt").is_null(),
  mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),T=ep("MVT_TIME_UTC_mvt"),S=ep("SCHED_TIME_UTC_mvt"),
  LOBT=ep("LOBT_flt"),AOBT3=ep("AOBT_3_flt"),EOBT1=ep("EOBT_1_flt"),IOBT=ep("IOBT_flt"),
  rwy="RUNWAY_mvt",stand="STAND_mvt",typ="AIRCRAFT_TYPE_mvt",op="AIRCRAFT_OPERATOR_flt",fno="FLIGHT_mvt",cs="CALLSIGN_flt",
  ms="MARKET_SEGMENT_flt")
arr=raw.filter(pl.col("PHASE_mvt")=="ARR").select(pl.col("MVT_ID_mvt").cast(pl.Int64),apt="ADES_mvt",adep="ADEP_mvt",
  T=ep("MVT_TIME_UTC_mvt"),S=ep("SCHED_TIME_UTC_mvt"),B=ep("BLOCK_TIME_UTC_mvt"),y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
rd=lambda f: pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Int64),pl.col("pred").cast(pl.Float64))
oof=pl.concat([rd(f) for f in sorted(glob.glob(ROOT+"/cache/oof/ctxc1sel_rh2/fold=*.parquet"))])
ho=pl.read_parquet(ROOT+"/cache/eval/ctxc1sel_rh2_holdout_ev.parquet")
ho=ho.select(pl.col("MVT_ID_mvt").cast(pl.Int64),pl.col("pred").cast(pl.Float64))
p=pl.concat([oof,ho]).unique("MVT_ID_mvt")
f=dep.join(p,on="MVT_ID_mvt",how="left")
print(f.group_by("mo").agg(n=pl.len(),has=pl.col("pred").is_not_null().sum(),yl=pl.col("y").is_not_null().sum()).sort("mo"))
f.write_parquet(S+"/dep25.parquet"); arr.write_parquet(S+"/arr25.parquet")
