"""Board A/B consistency check for LIRF NM-unmatched rows >= 12 h late (reports/lirf_echo_tail_preregistration.md).
v19 and v20 differ only on those 11 rows; their board delta is tested against echo / +24 h / normal hypotheses."""
import json, polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(20)
d=json.load(open('logs/leaderboard_2026-10-07.json'))
sc={r['filename'].replace('smart-jigsaw_','').replace('.parquet',''):r['score'] for r in d if r['teamName']=='smart-jigsaw'}
print(sorted(sc.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 0))
json.dump(sc, open("cache/eval/board_scores.json","w"))
N=344841
r = pl.read_parquet("data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt")=="DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64),"ADEP_mvt","FLIGHT_mvt","STAND_mvt",
      nmu=pl.col("FLIGHT_ID_mvt").is_null(), off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
def sub(v): return pl.read_parquet(f"data/submissions/smart-jigsaw_{v}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(v))
def pair(a,b):
    x = r.join(sub(a),on="MVT_ID_mvt").join(sub(b),on="MVT_ID_mvt").filter(pl.col(a)!=pl.col(b))
    dS = N*(sc[b]**2-sc[a]**2)
    print(f"\n{a}->{b}: board {sc[a]:.4f}->{sc[b]:.4f}  dSSE {dS:.4g}; rows differing {x.height}")
    return x, dS
x, dS = pair("v19","v20")
print(x.select("ADEP_mvt","nmu","FLIGHT_mvt","off","v19","v20"))
A, B, off = x["v19"].to_numpy(), x["v20"].to_numpy(), x["off"].to_numpy()
f = lambda y: np.sum((B-y)**2-(A-y)**2)
for nm, y in (("all echo y=off", off), ("all +24h y=87610", np.full(len(A),87610.)), ("all normal y=1500", np.full(len(A),1500.))):
    print(f"  hypothesis {nm:20s}: predicted dSSE {f(y):.4g}   (observed {dS:.4g})")
