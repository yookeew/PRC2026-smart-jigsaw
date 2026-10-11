import os, numpy as np, polars as pl
S="C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"; ROOT="C:/Users/zhang/smart-jigsaw"
ep=lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
rk=pl.read_parquet(ROOT+"/data/ranking/final_ranking.parquet").filter(pl.col("PHASE_mvt")=="DEP").select(
  pl.col("MVT_ID_mvt").cast(pl.Int64),apt="ADEP_mvt",ades="ADES_mvt",mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),T=ep("MVT_TIME_UTC_mvt"),S=ep("SCHED_TIME_UTC_mvt"),LOBT=ep("LOBT_flt"),nm=pl.col("FLIGHT_ID_mvt").is_null())
tr=pl.read_parquet(S+"/dep25.parquet").select("MVT_ID_mvt","apt","ades","mo","T","S","LOBT","nm")
def feats(f):
    f=f.with_columns(off=pl.col("T")-pl.col("S"),dl=pl.col("T")-pl.col("LOBT"),reg=pl.col("ades").str.slice(0,2))
    W=3600
    a=f.select("MVT_ID_mvt","apt","ades","reg","T").with_columns(bk=(pl.col("T")//W).cast(pl.Int64))
    b=f.select(pl.col("apt").alias("apt_j"),"ades","reg",pl.col("ades").alias("ades_j"),pl.col("T").alias("Tj"),pl.col("off").alias("off_j"),pl.col("MVT_ID_mvt").alias("id_j")).with_columns(bk0=(pl.col("Tj")//W).cast(pl.Int64))
    x=pl.concat([a.join(b.drop("reg").with_columns(bk=pl.col("bk0")-d),on=["ades","bk"],how="inner") for d in (-1,0,1)]).filter((pl.col("id_j")!=pl.col("MVT_ID_mvt"))&((pl.col("Tj")-pl.col("T")).abs()<=W)&(pl.col("apt_j")!=pl.col("apt")))
    g=x.group_by("MVT_ID_mvt").agg(xo_n=pl.len(),xo_moff=pl.col("off_j").median(),xo_f30=(pl.col("off_j")>1800).mean())
    return f.join(g,on="MVT_ID_mvt",how="left")
A=feats(tr); B=feats(rk)
for c in ["xo_n","xo_moff","xo_f30"]:
    for m in [1,2,6,7]:
        a=A.filter((pl.col("mo")==m)&~pl.col("nm"))[c]; b=B.filter((pl.col("mo")==m)&~pl.col("nm"))[c]
        sd=A.filter(~pl.col("nm"))[c].std()
        print(f"{c:8s} month {m}: cov 2025 {a.is_not_null().mean():.3f} 2026 {b.is_not_null().mean():.3f}  mean 2025 {a.mean():8.2f} 2026 {b.mean():8.2f}  SMD {(b.mean()-a.mean())/sd:+.3f}")
