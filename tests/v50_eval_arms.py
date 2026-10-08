"""Compare chain arms against the control of the same process. Usage: eval_arms.py DIR arm1 [arm2 ...]"""
import os, sys, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
d0 = sys.argv[1]
files = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f or "2025-07-01" in f]
raw = (pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
       .select(pl.col("MVT_ID_mvt").cast(pl.Int64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
               nm=pl.col("FLIGHT_ID_mvt").is_null(), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), apt=pl.col("ADEP_mvt"),
               day=pl.col("MVT_TIME_UTC_mvt").dt.date()))
ctl = pl.read_parquet(f"{d0}/chain_ctl.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), ctl="pred")
v42 = pl.read_parquet("cache/eval/v42_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "v42")
d = ctl.join(raw, on="MVT_ID_mvt").join(v42, on="MVT_ID_mvt")
print(f"control vs cache v42: max |diff| {np.abs(d['ctl'] - d['v42']).max():.4f}, RMS {np.sqrt(np.mean((d['ctl'] - d['v42']).to_numpy() ** 2)):.4f}")
y, mo, apt, nmu = d["y"].to_numpy(), d["mo"].to_numpy(), d["apt"].to_numpy(), d["nm"].to_numpy()
body = (~nmu) & (y <= 3600)
r = lambda p, k: float(np.sqrt(np.mean((y - p)[k] ** 2)))  # noqa: E731
b = d["ctl"].to_numpy()
for arm in sys.argv[2:]:
    t = pl.read_parquet(f"{d0}/chain_{arm}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p="pred")
    p = d.select("MVT_ID_mvt").join(t, on="MVT_ID_mvt", how="left")["p"].to_numpy()
    print(f"\n=== {arm} vs ctl  (rows changed {(np.abs(p - b) > 0.5).sum():,}, RMS change {np.sqrt(np.mean((p - b) ** 2)):.2f})")
    for nme, k in [("full", np.ones_like(body)), ("trimmed y<=5h", y <= 18000), ("body pooled", body),
                   ("body Jan", body & (mo == 1)), ("body Jul", body & (mo == 7)), ("full Jan", mo == 1), ("full Jul", mo == 7)]:
        print(f"  {nme:16s} {r(b, k):9.3f} -> {r(p, k):9.3f}  ({r(p, k) - r(b, k):+.3f})")
    # exact per-row decomposition of the full-RMSE gain (outgoing-boat transferencia): body vs tail
    dd = (y - b) ** 2 - (y - p) ** 2
    denom = len(y) * (r(b, np.ones_like(body)) + r(p, np.ones_like(body)))
    gain = dd.sum() / denom
    print(f"  full gain {gain:+.3f} s = body(y<=1h) {dd[y <= 3600].sum() / denom:+.3f} + tail {dd[y > 3600].sum() / denom:+.3f};"
          f" top-20 rows hold {np.sort(dd)[::-1][:20].sum() / max(dd[y <= 3600].sum(), 1e-9) * 100:.0f}% of body gain")
    print("  per airport (all rows):", {a: round(r(p, apt == a) - r(b, apt == a), 2) for a in sorted(set(apt))})
    # day-cluster bootstrap of the body gain
    rng = np.random.default_rng(0)
    keys = pl.Series(d["apt"] + "|" + d["day"].cast(pl.Utf8)).to_numpy()
    uk, inv = np.unique(keys, return_inverse=True)
    sb = np.bincount(inv, weights=np.where(body, (y - b) ** 2, 0.0)); sn = np.bincount(inv, weights=np.where(body, (y - p) ** 2, 0.0))
    nb = np.bincount(inv, weights=body.astype(float))
    g = []
    for _ in range(2000):
        i = rng.integers(0, len(uk), len(uk))
        g.append(np.sqrt(sn[i].sum() / nb[i].sum()) - np.sqrt(sb[i].sum() / nb[i].sum()))
    g = np.array(g)
    print(f"  body delta bootstrap: mean {g.mean():+.3f}, 95% CI [{np.quantile(g, 0.025):+.3f}, {np.quantile(g, 0.975):+.3f}], P(worse) {np.mean(g > 0):.3f}")
