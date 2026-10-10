"""Per-airport comparison with outgoing-boat on v50's holdout chain (rebuilt: cache/v50/chain_ctl / chain_all).
v50-like = chain_all with every LIRF row from chain_ctl (v50 kept LIRF rows). Their numbers are copied from their
public experiments.jsonl (best run e196 per-airport, all rows) and docs/research/2026-09-29-base-sem-adsb.md
(v28 body = NM-matched, y <= 1 h)."""
import os, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
OB_ALL = {"EDDF": 146.0, "EDDM": 130.8, "EGLL": 246.9, "EHAM": 130.1, "LEBL": 190.0, "LEMD": 171.4, "LFPG": 564.2,
          "LIRF": 540.8, "LSZH": 166.7, "LTFM": 239.6}
OB_BODY28 = {"LTFM": 232.48, "LIRF": 307.39, "LFPG": 225.28, "EGLL": 206.74, "LEMD": 175.74, "LEBL": 191.99,
             "EDDF": 143.00, "EHAM": 102.90, "LSZH": 136.22, "EDDM": 120.77}
fs = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f or "2025-07-01" in f]
raw = pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP").select(
    pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
    nmu=pl.col("FLIGHT_ID_mvt").is_null())
c = pl.read_parquet("cache/v50/chain_ctl.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), v42="pred")
a = pl.read_parquet("cache/v50/chain_all.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), allp="pred")
d = raw.join(c, on="MVT_ID_mvt").join(a, on="MVT_ID_mvt").with_columns(
    v50=pl.when(pl.col("ADEP_mvt") == "LIRF").then("v42").otherwise("allp"))
d = d.with_columns(e42=(pl.col("y") - pl.col("v42")) ** 2, e50=(pl.col("y") - pl.col("v50")) ** 2)
body = ~pl.col("nmu") & (pl.col("y") <= 3600)
t = (d.group_by("ADEP_mvt").agg(all_v42=pl.col("e42").mean().sqrt(), all_v50=pl.col("e50").mean().sqrt(),
                                body_v42=pl.col("e42").filter(body).mean().sqrt(), body_v50=pl.col("e50").filter(body).mean().sqrt(),
                                n_body=body.sum())
     .with_columns(ob_all=pl.col("ADEP_mvt").replace_strict(OB_ALL), ob_body_v28=pl.col("ADEP_mvt").replace_strict(OB_BODY28))
     .with_columns(gap_all=pl.col("all_v50") - pl.col("ob_all"), gap_body=pl.col("body_v50") - pl.col("ob_body_v28"))
     .sort("ADEP_mvt"))
with pl.Config(tbl_rows=20, float_precision=1, tbl_formatting="ASCII_MARKDOWN", tbl_width_chars=200, tbl_cols=20):
    print(t)
b = d.filter(body)
print(f"pooled: full v42 {np.sqrt(d['e42'].mean()):.2f} v50 {np.sqrt(d['e50'].mean()):.2f} (OB best 295.80); "
      f"body v42 {np.sqrt(b['e42'].mean()):.2f} v50 {np.sqrt(b['e50'].mean()):.2f} (OB best normais_nm 189.16, v28 193.45)")
# what closing each airport's all-rows gap to OB would be worth on the pooled full RMSE
tot = d["e50"].sum(); n = d.height
for r in t.iter_rows(named=True):
    if r["gap_all"] > 0:
        k = d.filter(pl.col("ADEP_mvt") == r["ADEP_mvt"]).height
        new = tot - k * (r["all_v50"] ** 2 - r["ob_all"] ** 2)
        print(f"  closing {r['ADEP_mvt']} all-rows gap ({r['all_v50']:.1f} -> {r['ob_all']:.1f}) = full {np.sqrt(new / n) - np.sqrt(tot / n):+.2f} s")
