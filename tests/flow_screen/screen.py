"""Generic paired residual screen: LightGBM on r = y - pred, leave-period-out (7 periods), control vs control+new."""
import os, sys, numpy as np, polars as pl, lightgbm as lgb
S="C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"
PER={1:0,7:1,2:2,3:2,4:3,5:3,6:4,8:4,9:5,10:5,11:6,12:6}
def run(f, new_cols, ctl_extra=(), label="", rounds=300, trim=18000):
    f=f.filter(pl.col("pred").is_not_null()&(pl.col("y")<=trim))
    f=f.with_columns(r=pl.col("y")-pl.col("pred"),hr=((pl.col("T")%86400)/3600.0),
        off=pl.col("T")-pl.col("S"),dl=pl.col("T")-pl.col("LOBT"),da=pl.col("T")-pl.col("AOBT3"),
        aptc=pl.col("apt").cast(pl.Categorical).to_physical().cast(pl.Int32),nmi=pl.col("nm").cast(pl.Int8),
        per=pl.col("mo").replace_strict(PER,return_dtype=pl.Int8))
    ctl=["pred","aptc","off","dl","da","hr","nmi"]+list(ctl_extra)
    y=f["r"].to_numpy(); per=f["per"].to_numpy(); yy=f["y"].to_numpy(); nm=f["nm"].to_numpy(); apt=f["apt"].to_numpy(); mo=f["mo"].to_numpy()
    out={}
    for arm,cols in [("ctl",ctl),("trt",ctl+list(new_cols))]:
        X=f.select(cols).to_numpy().astype(np.float32); p=np.zeros(len(y))
        for k in range(7):
            tr=per!=k
            m=lgb.train(dict(objective="huber",alpha=300.0,learning_rate=0.05,num_leaves=31,min_data_in_leaf=1000,
                feature_fraction=0.9,bagging_fraction=0.7,bagging_freq=1,verbose=-1,seed=1,num_threads=8),
                lgb.Dataset(X[tr],y[tr],categorical_feature=[1]),rounds)
            p[~tr]=m.predict(X[~tr])
        out[arm]=y-p
    rm=lambda e: np.sqrt(np.mean(e**2))
    e0=y; ec,et=out["ctl"],out["trt"]
    body=(~nm)&(yy<=3600)
    print(f"== {label}: rows {len(y)}  base {rm(e0):.2f}  ctl {rm(ec):.2f}  trt {rm(et):.2f}  PAIRED trt-ctl {rm(et)-rm(ec):+.3f}  body {rm(et[body])-rm(ec[body]):+.3f}")
    print("   by period:", [round(rm(et[per==k])-rm(ec[per==k]),2) for k in range(7)])
    print("   by apt   :", {a: round(rm(et[apt==a])-rm(ec[apt==a]),2) for a in sorted(set(apt))})
    return out
if __name__=="__main__":
    f=pl.read_parquet(S+"/dep25_flow.parquet")
    new=[c for c in f.columns if c.startswith(("xo60_","so90_"))]
    run(f,new,label="dest flow")
