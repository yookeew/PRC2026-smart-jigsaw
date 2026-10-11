import os, numpy as np, polars as pl
S="C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"
f=pl.read_parquet(S+"/dep25.parquet").filter(pl.col("pred").is_not_null()&(pl.col("y")<=18000))
f=f.with_columns(r=pl.col("y")-pl.col("pred"),day=(pl.col("T")//86400).cast(pl.Int64),hr=(pl.col("T")//3600).cast(pl.Int64),
                 h30=(pl.col("T")//1800).cast(pl.Int64),reg=pl.col("ades").str.slice(0,2))
r=f["r"].to_numpy(); base=np.sqrt(np.mean(r**2)); apt=f["apt"].to_numpy(); nl=apt!="LIRF"
print("base",round(base,2))
def orc(keys,k=3,name=None):
    x=f.select(keys+["r"]).with_columns(s=pl.col("r").sum().over(keys),n=pl.len().over(keys))
    adj=((x["s"]-x["r"])/(x["n"]-1+k)).to_numpy()
    e=r-adj
    print(f"{(name or '+'.join(keys)):34s} all {np.sqrt(np.mean(e**2))-base:+7.2f}  nonLIRF {np.sqrt(np.mean(e[nl]**2))-np.sqrt(np.mean(r[nl]**2)):+7.2f}")
    return adj
a1=orc(["apt","hr"])
orc(["apt","h30","rwy"])
orc(["apt","hr","reg"])
orc(["apt","hr","ades"])
orc(["hr","ades"],name="hr+ades (cross-origin)")
orc(["day","ades"],name="day+ades (cross-origin)")
orc(["apt","hr","op"])
# incremental: dest cross-origin on top of apt-hr oracle
f=f.with_columns(r=pl.Series(r-a1)); r2=f["r"].to_numpy()
x=f.select(["hr","ades","apt","r"])
# leave-own-airport-out cross-origin ades-hour mean
s_all=x.group_by(["hr","ades"]).agg(sa=pl.col("r").sum(),na=pl.len()); s_own=x.group_by(["hr","ades","apt"]).agg(so=pl.col("r").sum(),no=pl.len())
y=x.join(s_all,on=["hr","ades"]).join(s_own,on=["hr","ades","apt"])
adj=((y["sa"]-y["so"])/(y["na"]-y["no"]+3)).to_numpy()
print("cross-origin (other airports only) hr+ades on top of apt-hr oracle:", round(np.sqrt(np.mean((y['r'].to_numpy()-adj)**2))-np.sqrt(np.mean(y['r'].to_numpy()**2)),2))
