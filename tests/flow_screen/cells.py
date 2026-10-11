import os, numpy as np, polars as pl
S="C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"
f=pl.read_parquet(S+"/dep25.parquet").filter(pl.col("pred").is_not_null()&(pl.col("y")<=18000))
f=f.with_columns(r=pl.col("y")-pl.col("pred"),hr=((pl.col("T")%86400)//3600).cast(pl.Int32),
   dow=(((pl.col("T")//86400)+4)%7).cast(pl.Int32),off=((pl.col("T")-pl.col("S"))//600).clip(-3,12).cast(pl.Int32))
base=float(np.sqrt((f["r"]**2).mean())); print("trimmed RMSE 12 months corrector-level:",round(base,2), "rows",f.height)
nl=f.filter(pl.col("apt")!="LIRF"); bnl=float(np.sqrt((nl["r"]**2).mean()))
def screen(keys,k=30,name=None, sub=None):
    g=f if sub is None else sub
    a=g.group_by(keys).agg(sa=pl.col("r").sum(),na=pl.len())
    m=g.group_by(keys+["mo"]).agg(sm=pl.col("r").sum(),nm_=pl.len())
    x=g.join(a,on=keys,how="left").join(m,on=keys+["mo"],how="left")
    x=x.with_columns(n=pl.col("na")-pl.col("nm_"),s=pl.col("sa")-pl.col("sm"))
    x=x.with_columns(adj=pl.when(pl.col("n")>0).then(pl.col("s")/(pl.col("n")+k)).otherwise(0.0))
    r=x["r"].to_numpy(); adj=x["adj"].to_numpy(); apt=x["apt"].to_numpy(); mo=x["mo"].to_numpy()
    b=np.sqrt(np.mean(r**2)); t=np.sqrt(np.mean((r-adj)**2))
    mnl=apt!="LIRF"
    worse=sum(np.mean((r-adj)[mo==mm]**2)>np.mean(r[mo==mm]**2) for mm in range(1,13))
    print(f"{(name or '+'.join(keys)):38s} all {t-b:+7.3f}  nonLIRF {np.sqrt(np.mean((r-adj)[mnl]**2))-np.sqrt(np.mean(r[mnl]**2)):+7.3f}  months worse {worse}/12  cells {a.height}")
for keys in [["apt","stand"],["apt","stand","rwy"],["apt","op"],["apt","op","stand"],["apt","typ"],["apt","typ","rwy"],
             ["apt","fno"],["apt","cs"],["apt","ades"],["apt","ades","rwy"],["apt","ades","hr"],["apt","rwy","hr"],["apt","dow","hr"],
             ["apt","op","hr"],["apt","off"],["apt","op","off"],["apt","stand","off"],["apt","ms","off"],["apt","rwy","off"],["apt","fno","off"]]:
    for k in [30]:
        screen(keys,k)
