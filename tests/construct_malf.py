import polars as pl, glob, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(16)
d = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="diagonal_relaxed")
F = pl.col("FLIGHT_mvt")
mal = (F.str.slice(3,1)==F.str.slice(2,1)) | (F.str.slice(3,2)==F.str.slice(1,2))
d = d.with_columns(mal=mal.fill_null(False), nmu=pl.col("FLIGHT_ID_mvt").is_null(), fnull=F.is_null(),
                   off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(), y=pl.col("TAXITIME_SEC_mvt"))
dep = d.filter(pl.col("PHASE_mvt")=="DEP")
print(dep.group_by("ADEP_mvt").agg(n=pl.len(), mal=pl.col("mal").mean(), nmu_if_mal=pl.col("nmu").filter(pl.col("mal")).mean(), nmu_if_ok=pl.col("nmu").filter(~pl.col("mal")).mean()).sort("ADEP_mvt"))
# matched rows: does FLIGHT_mvt equal CALLSIGN_flt?
m = dep.filter(~pl.col("nmu"))
print("matched DEP: FLIGHT_mvt == CALLSIGN_flt share", float((m["FLIGHT_mvt"]==m["CALLSIGN_flt"]).mean()), "; malformed among matched", float(m["mal"].mean()))
print(m.filter(pl.col("mal")).select("ADEP_mvt","FLIGHT_mvt","CALLSIGN_flt").head(12))
print(m.filter(pl.col("FLIGHT_mvt")!=pl.col("CALLSIGN_flt")).select("ADEP_mvt","FLIGHT_mvt","CALLSIGN_flt").sample(12, seed=1))
# LIRF NMU delayed: echo & late-branch validity by malformed
z = dep.filter((pl.col("ADEP_mvt")=="LIRF") & pl.col("nmu") & (pl.col("off")>=3600)).with_columns(echo=(pl.col("off")-pl.col("y")).abs()<30, late=pl.col("y")<=pl.col("off")-3606)
print(z.group_by("mal").agg(n=pl.len(), echo=pl.col("echo").mean(), late=pl.col("late").mean(), neither=(~pl.col("echo") & ~pl.col("late")).mean()))
