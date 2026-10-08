"""Label-free remaining echo-hedge cost of v48 by segment (PROGRESS §118)."""
import sys; sys.path.insert(0,"src")
import polars as pl, numpy as np
import post.lirf_signals as L
from post.lirf_hedge2 import airline_late, rate_table, add_rate, fit2, frame25, with_loo_rate
from post.lirf_twin_bounds import twin_frame
rk = str(L.RANKING)
hist = airline_late(str(L.RAW/"training_*.parquet"))
d25 = with_loo_rate(frame25(), hist); clf2 = fit2(d25)
r = (pl.scan_parquet(rk).filter(pl.col("PHASE_mvt")=="DEP").select(L.I64,"ADEP_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null(),
     off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64)).collect()
     .join(L.p_2026(), on="MVT_ID_mvt").join(pl.read_parquet("data/submissions/smart-jigsaw_v48.parquet").select(L.I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt"))
# q for LIRF NMU scope rows = q2 (v48), else p
d26 = L.frame(rk).join(L.p_2026(), on="MVT_ID_mvt", how="left").join(airline_late(rk).select("MVT_ID_mvt","al"), on="MVT_ID_mvt", how="left")
d26 = add_rate(d26, rate_table(hist)); k = L.in_scope(d26)
tw = set(twin_frame(rk)["MVT_ID_mvt"].to_list())
k = k & ~np.array([i in tw for i in d26["MVT_ID_mvt"].to_list()])
q2 = pl.DataFrame({"MVT_ID_mvt": d26.filter(pl.Series(k))["MVT_ID_mvt"], "q": clf2.predict_proba(__import__("post.lirf_hedge2",fromlist=["design2"]).design2(d26.filter(pl.Series(k))))[:,1]})
r = r.join(q2, on="MVT_ID_mvt", how="left").with_columns(q=pl.coalesce("q","p"), tw=pl.col("MVT_ID_mvt").is_in(list(tw)))
off, P, q = r["off"].to_numpy(), r["P"].to_numpy(), r["q"].to_numpy(); offc=np.clip(off,0,140000)
with np.errstate(divide="ignore", invalid="ignore"): pne = np.where(q<0.99,(P-q*offc)/(1-q),P)
H = np.where(r["tw"].to_numpy(), 0, q*(1-q)*(offc-np.clip(pne,0,None))**2)
TOT = 245.8701**2*344841
r = r.with_columns(H=pl.Series(H), seg=pl.when((pl.col("ADEP_mvt")=="LIRF")&pl.col("nmu")).then(pl.lit("LIRF NMU")).when(pl.col("ADEP_mvt")=="LIRF").then(pl.lit("LIRF NM-m")).otherwise(pl.lit("other")),
   band=pl.col("off").cut([3600,3*3600,6*3600,12*3600]))
print(f"board SSE {TOT:.4g}; total label-free hedge cost {H.sum():.4g} ({100*H.sum()/TOT:.1f}%)")
print(r.group_by("seg","band").agg(n=pl.len(), H=pl.col("H").sum(), Hpct=(100*pl.col("H").sum()/TOT), mq=pl.col("q").mean()).filter(pl.col("Hpct")>0.2).sort("Hpct",descending=True))
