import polars as pl, glob, numpy as np
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
d = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="diagonal_relaxed").filter(pl.col("PHASE_mvt")=="DEP")
ts = lambda c: pl.col(c).dt.epoch("s")
rnd = lambda e: ((e+30)//60)*60
d = d.with_columns(T=ts("MVT_TIME_UTC_mvt"), B=ts("BLOCK_TIME_UTC_mvt"), S=ts("SCHED_TIME_UTC_mvt"), L=ts("LOBT_flt"), E=ts("EOBT_1_flt"), I=ts("IOBT_flt"), A=ts("AOBT_3_flt"), R3=ts("ARVT_3_flt"), R1=ts("ARVT_1_flt"))
d = d.with_columns(Bm=rnd(pl.col("B")), Tm=rnd(pl.col("T"))).filter((pl.col("TAXITIME_SEC_mvt")>=1800) & pl.col("FLIGHT_ID_mvt").is_not_null())
rows=[]
for apt in sorted(d["ADEP_mvt"].unique().to_list()):
    z = d.filter(pl.col("ADEP_mvt")==apt)
    for lab, a, b in (("B-S","Bm","S"),("B-LOBT","Bm","L"),("B-EOBT1","Bm","E"),("B-IOBT","Bm","I"),("B-AOBT3","Bm","A"),("T-AOBT3","Tm","A"),("B-ARVT1","Bm","R1")):
        x = (z[a]-z[b]).drop_nulls().to_numpy()//60
        n = len(x)
        if n==0: continue
        c0 = np.mean(x==0); cn = np.mean([np.mean(x==k) for k in (-3,-2,2,3)])
        rows.append((apt, lab, n, round(100*c0,2), round(100*cn,2), round(c0/max(cn,1e-9),1), round(100*(c0-cn),2)))
r = pl.DataFrame(rows, schema=["apt","pair","n","%=0","%nbr","ratio","excess%"], orient="row")
print(r.filter(pl.col("ratio")>1.3).sort("excess%",descending=True))
