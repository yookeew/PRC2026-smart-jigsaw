import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
I64=pl.col("MVT_ID_mvt").cast(pl.Int64)
d = pl.concat([pl.read_parquet(f) for f in ["data/raw/training_2025-01-01_2025-02-01.parquet","data/raw/training_2025-07-01_2025-08-01.parquet"]], how="diagonal_relaxed").filter(pl.col("PHASE_mvt")=="DEP")
d = d.with_columns(I64).join(pl.read_parquet("cache/eval/v48_like.parquet") if False else pl.read_parquet("cache/eval/v42_chain_holdout.parquet").select(I64,"v42"), on="MVT_ID_mvt")
ts=lambda c: pl.col(c).dt.epoch("s"); rnd=lambda e: ((e+30)//60)*60
d = d.with_columns(r=(pl.col("TAXITIME_SEC_mvt")-pl.col("v42")).cast(pl.Float64), m=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    T=ts("MVT_TIME_UTC_mvt"), S=ts("SCHED_TIME_UTC_mvt"), L=ts("LOBT_flt"), E=ts("EOBT_1_flt"), Io=ts("IOBT_flt"), A=ts("AOBT_3_flt"), R1=ts("ARVT_1_flt"), R3=ts("ARVT_3_flt"))
F=pl.col("FLIGHT_mvt"); C=pl.col("CALLSIGN_flt")
flags = {
 "nm_null": pl.col("FLIGHT_ID_mvt").is_null(),
 **{f"null_{c}": pl.col(c).is_null() for c in ["LOBT_flt","AOBT_3_flt","EOBT_1_flt","IOBT_flt","ARVT_1_flt","ARVT_3_flt","CALLSIGN_flt","MARKET_SEGMENT_flt","WK_TBL_CAT_flt","AIRCRAFT_TYPE_mvt","AIRCRAFT_TYPE_flt","STAND_mvt","RUNWAY_mvt","FLIGHT_mvt","AIRCRAFT_OPERATOR_flt","ADES_FILED_flt"]},
 "L==S": pl.col("L")==pl.col("S"), "E==S": pl.col("E")==pl.col("S"), "I==S": pl.col("Io")==pl.col("S"), "L==E": pl.col("L")==pl.col("E"), "L==I": pl.col("L")==pl.col("Io"), "E==I": pl.col("E")==pl.col("Io"),
 "A==L": pl.col("A")==pl.col("L"), "A==E": pl.col("A")==pl.col("E"), "A==S": pl.col("A")==pl.col("S"), "Tm==A": rnd(pl.col("T"))==pl.col("A"),
 "T%60==0": pl.col("T")%60==0, "A%60!=0": pl.col("A")%60!=0, "L%300==0": pl.col("L")%300==0, "S%300!=0": pl.col("S")%300!=0,
 "F==C": F==C, "Fpre==Cpre": F.str.slice(0,3)==C.str.slice(0,3), "mal": (F.str.slice(3,1)==F.str.slice(2,1))|(F.str.slice(3,2)==F.str.slice(1,2)),
 "type_mismatch": pl.col("AIRCRAFT_TYPE_mvt")!=pl.col("AIRCRAFT_TYPE_flt"), "ades_mismatch": pl.col("ADES_mvt")!=pl.col("ADES_flt"), "diverted": pl.col("ADES_FILED_flt")!=pl.col("ADES_flt"),
 "rule_mismatch": pl.col("FLIGHT_RULE_mvt")!=pl.col("FLIGHT_RULE_flt"), "adep_mismatch": pl.col("ADEP_mvt")!=pl.col("ADEP_flt"),
 "R3<R1": pl.col("R3")<pl.col("R1"), "A<L-3600": pl.col("A")<pl.col("L")-3600, "A>T": pl.col("A")>pl.col("T"),
}
for k,v in flags.items(): d = d.with_columns(v.fill_null(False).alias(k))
tot = float((d["r"]**2).sum()); out=[]
for k in flags:
    g = d.group_by("ADEP_mvt", k, "m").agg(n=pl.len(), s=pl.col("r").sum(), ss=(pl.col("r")**2).sum())
    gain=0; gains_m={1:0,7:0}
    for tr, te in ((1,7),(7,1)):
        a = g.filter(pl.col("m")==tr).select("ADEP_mvt",k, mu=pl.col("s")/(pl.col("n")+50))   # shrunk mean
        b = g.filter(pl.col("m")==te).join(a, on=["ADEP_mvt",k], how="left").with_columns(pl.col("mu").fill_null(0))
        # SSE change applying mu: sum (r-mu)^2 - r^2 = n mu^2 - 2 mu s
        ch = float((b["n"]*b["mu"]**2 - 2*b["mu"]*b["s"]).sum()); gain += ch; gains_m[te]=ch
    nflag = int(d[k].sum())
    out.append((k, nflag, gain, gains_m[1], gains_m[7]))
r = pl.DataFrame(out, schema=["flag","n_true","dSSE","dJan","dJul"], orient="row").with_columns(pct=100*pl.col("dSSE")/tot).sort("dSSE")
print(f"holdout SSE {tot:.4g}"); print(r)
