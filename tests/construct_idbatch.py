import sys; sys.path.insert(0,"src")
import polars as pl, numpy as np, glob
import post.lirf_signals as L
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
d = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="diagonal_relaxed").filter(pl.col("PHASE_mvt")=="DEP")
d = d.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), S=pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s"), T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), B=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s"))
# look at structure: per airport, sort by ID, check how S evolves; find "batches" where ID jumps
z = d.filter(pl.col("ADEP_mvt")=="LIRF").sort("MVT_ID_mvt")
ids, S = z["MVT_ID_mvt"].to_numpy(), z["S"].to_numpy()
# expected: within a batch S nondecreasing. Flag rows where S < running max of S over previous 50 IDs - 2h (inserted late)
run = pl.Series(S).rolling_max(window_size=50, min_samples=1).shift(1).to_numpy()
oo = (S < run - 3*3600)
z = z.with_columns(oo=pl.Series(oo), gap=pl.Series(np.r_[0, np.diff(ids)]))
print("LIRF out-of-order share", oo.mean())
z = z.with_columns(off=pl.col("T")-pl.col("S"), echo=(pl.col("B")-pl.col("S")).abs()<30, nmu=pl.col("FLIGHT_ID_mvt").is_null()).join(L.p_2025(), on="MVT_ID_mvt", how="left")
print(z.filter(pl.col("off")>=3600).group_by("nmu","oo").agg(n=pl.len(), echo=pl.col("echo").mean(), p=pl.col("p").mean()).sort("nmu","oo"))
# where do IDs come from? print a slice around an out-of-order row
i = int(np.where(oo)[0][100]); print(z.slice(i-5, 10).select("MVT_ID_mvt","SCHED_TIME_UTC_mvt","MVT_TIME_UTC_mvt","FLIGHT_mvt","nmu","oo"))
