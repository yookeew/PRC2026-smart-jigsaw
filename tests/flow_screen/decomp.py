import glob, os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
ROOT="C:/Users/zhang/smart-jigsaw"
cols=["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt","SCHED_TIME_UTC_mvt","AOBT_3_flt","LOBT_flt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_OPERATOR_flt"]
raw=pl.concat([pl.read_parquet(f,columns=cols) for f in sorted(glob.glob(ROOT+"/data/raw/training_2025-*.parquet"))],how="vertical_relaxed")
lab=raw.filter((pl.col("PHASE_mvt")=="DEP")&pl.col("TAXITIME_SEC_mvt").is_not_null()).select(
 pl.col("MVT_ID_mvt").cast(pl.Int64),apt="ADEP_mvt",y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
 nm=pl.col("FLIGHT_ID_mvt").is_null(),mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
 off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
 anc=(pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
p=pl.read_parquet(ROOT+"/cache/v65/gate_rh2/chain_all.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64),pl.col("pred").cast(pl.Float64))
x=p.join(lab,on="MVT_ID_mvt",how="inner")
y,t=x["y"].to_numpy(),x["pred"].to_numpy(); e=y-t
apt,nm,mo,off,anc=x["apt"].to_numpy(),x["nm"].to_numpy(),x["mo"].to_numpy(),x["off"].to_numpy(),x["anc"].to_numpy()
trim=y<=18000
N=trim.sum(); SSE=(e[trim]**2).sum()
print("rows",len(y),"trimmed",N,"RMSE",np.sqrt(SSE/N))
def seg(name,m):
    m=m&trim
    print(f"{name:34s} n={m.sum():7d} ({100*m.sum()/N:5.2f}%) RMSE={np.sqrt(np.mean(e[m]**2)) if m.sum() else 0:8.1f} bias={np.mean(e[m]) if m.sum() else 0:+7.1f} SSE%={100*(e[m]**2).sum()/SSE:5.1f}")
print("--- segments")
seg("NM-matched y<=15m",(~nm)&(y<=900))
seg("NM-matched 15-30m",(~nm)&(y>900)&(y<=1800))
seg("NM-matched 30-60m",(~nm)&(y>1800)&(y<=3600))
seg("NM-matched 1-3h",(~nm)&(y>3600)&(y<=10800))
seg("NM-matched 3-5h",(~nm)&(y>10800))
seg("LIRF NM-unmatched",nm&(apt=="LIRF"))
seg("other NM-unmatched",nm&(apt!="LIRF"))
seg("LIRF NM-matched",(~nm)&(apt=="LIRF"))
print("--- by airport")
for a in sorted(set(apt)): seg(a,apt==a)
print("--- by airport, body (NM-matched y<=1h)")
for a in sorted(set(apt)): seg(a+" body",(apt==a)&(~nm)&(y<=3600))
print("--- by offset band (T-SOBT), NM-matched")
for lo,hi in [(-1e9,600),(600,1200),(1200,1800),(1800,3600),(3600,7200),(7200,1e9)]:
    seg(f"off {lo/60:.0f}-{hi/60:.0f} min",(~nm)&(off>lo)&(off<=hi))
print("--- prediction-band (pred) NM-matched")
for lo,hi in [(0,600),(600,900),(900,1200),(1200,1800),(1800,3600),(3600,1e9)]:
    seg(f"pred {lo/60:.0f}-{hi/60:.0f} min",(~nm)&(t>lo)&(t<=hi))
# what body RMSE would be needed for 212 overall
tb=212**2*N; 
bm=trim&(~nm)&(y<=3600)
other=SSE-(e[bm]**2).sum()
print("body RMSE now",np.sqrt(np.mean(e[bm]**2)),"needed for 212 (rest fixed):",np.sqrt(max(tb-other,0)/bm.sum()))
x.with_columns(pl.Series("e",e)).write_parquet("C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen/hold_v65.parquet")
