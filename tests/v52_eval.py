"""v52 gate evaluation (reports/v52_opdi_gate_preregistration.md). Control = SCR/chain_all (v50's chain);
arms: 'opdi' (unscoped) and 'opdi_scoped' (opdi on airports passing the label-free scope rule, control elsewhere).

Usage: python tests/v52_eval.py [SCR]
"""
import os, sys, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

d0 = sys.argv[1] if len(sys.argv) > 1 else "cache/v52"
of = pl.read_parquet("cache/opdi_feats/opdi_feats.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))

# ---- label-free scope rule and 2025-vs-2026 input check
files = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f or "2025-07-01" in f]
keys = lambda fs: pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]) for f in fs],  # noqa: E731
                            how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP").select(
    pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
ho_k, rk_k = keys(files), keys(["data/ranking/ranking.parquet"])
def cov(k):
    x = k.join(of, on="MVT_ID_mvt", how="left").with_columns(c=pl.col("o_lead_first").fill_null(0) >= 60)
    return x.group_by("apt").agg(share=pl.col("c").mean(), day=pl.col("o_day").mean(), n_cov=pl.col("c").sum(),
                                 m=pl.col("o_lead_first").filter("c").mean(), sd=pl.col("o_lead_first").filter("c").std())
cv = cov(ho_k).join(cov(rk_k), on="apt", suffix="_26").with_columns(
    ratio=pl.col("share_26") / pl.col("share"),
    smd=(pl.col("m_26") - pl.col("m")).abs() / ((pl.col("sd") ** 2 + pl.col("sd_26") ** 2) / 2).sqrt())
cv = cv.with_columns(keep=(pl.col("share") >= 0.20) & pl.col("ratio").is_between(0.5, 2.0) & (pl.col("smd") <= 0.25)
                     & (pl.col("apt") != "LIRF")).sort("apt")
with pl.Config(tbl_rows=20, float_precision=3, tbl_width_chars=220, tbl_cols=20):
    print("scope rule / input check (OPDI ground event >= 60 s before T; holdout Jan+Jul 2025 vs 2026 ranking):")
    print(cv.select("apt", "day", "share", "day_26", "share_26", "ratio", "m", "m_26", "smd", "keep"))
SCOPE = set(cv.filter("keep")["apt"].to_list())
print("scoped airports:", sorted(SCOPE))

# ---- gate
raw = (pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
       .select(pl.col("MVT_ID_mvt").cast(pl.Int64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
               nm=pl.col("FLIGHT_ID_mvt").is_null(), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), apt=pl.col("ADEP_mvt"),
               day=pl.col("MVT_TIME_UTC_mvt").dt.date()))
ctl = pl.read_parquet(f"{d0}/chain_all.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), ctl="pred")
trt = pl.read_parquet(f"{d0}/chain_opdi.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), opdi="pred")
d = ctl.join(raw, on="MVT_ID_mvt").join(trt, on="MVT_ID_mvt")
d = d.with_columns(opdi_scoped=pl.when(pl.col("apt").is_in(list(SCOPE))).then("opdi").otherwise("ctl"))
y, mo, apt, nmu = d["y"].to_numpy(), d["mo"].to_numpy(), d["apt"].to_numpy(), d["nm"].to_numpy()
body = (~nmu) & (y <= 3600)
r = lambda p, k: float(np.sqrt(np.mean((y - p)[k] ** 2)))  # noqa: E731
b = d["ctl"].to_numpy()
for arm in ("opdi_scoped", "opdi"):
    p = d[arm].to_numpy()
    print(f"\n=== {arm} vs ctl (v50 chain)  (rows changed {(np.abs(p - b) > 0.5).sum():,}, RMS change {np.sqrt(np.mean((p - b) ** 2)):.2f})")
    for nme, k in [("full", np.ones_like(body)), ("trimmed y<=5h", y <= 18000), ("body pooled", body),
                   ("body Jan", body & (mo == 1)), ("body Jul", body & (mo == 7)), ("full Jan", mo == 1), ("full Jul", mo == 7)]:
        print(f"  {nme:16s} {r(b, k):9.3f} -> {r(p, k):9.3f}  ({r(p, k) - r(b, k):+.3f})")
    dd = (y - b) ** 2 - (y - p) ** 2
    denom = len(y) * (r(b, np.ones_like(body)) + r(p, np.ones_like(body)))
    print(f"  full gain {dd.sum() / denom:+.3f} s = body(y<=1h) {dd[y <= 3600].sum() / denom:+.3f} + tail {dd[y > 3600].sum() / denom:+.3f};"
          f" top-20 rows hold {np.sort(dd)[::-1][:20].sum() / max(dd[y <= 3600].sum(), 1e-9) * 100:.0f}% of body gain")
    print("  per airport body:", {a: round(r(p, body & (apt == a)) - r(b, body & (apt == a)), 2) for a in sorted(set(apt))})
    print("  per airport all rows:", {a: round(r(p, apt == a) - r(b, apt == a), 2) for a in sorted(set(apt))})
    rng = np.random.default_rng(0)
    kk = pl.Series(d["apt"] + "|" + d["day"].cast(pl.Utf8)).to_numpy()
    uk, inv = np.unique(kk, return_inverse=True)
    sb = np.bincount(inv, weights=np.where(body, (y - b) ** 2, 0.0)); sn = np.bincount(inv, weights=np.where(body, (y - p) ** 2, 0.0))
    nb = np.bincount(inv, weights=body.astype(float))
    g = []
    for _ in range(2000):
        i = rng.integers(0, len(uk), len(uk))
        g.append(np.sqrt(sn[i].sum() / nb[i].sum()) - np.sqrt(sb[i].sum() / nb[i].sum()))
    g = np.array(g)
    print(f"  body delta bootstrap: mean {g.mean():+.3f}, 95% CI [{np.quantile(g, 0.025):+.3f}, {np.quantile(g, 0.975):+.3f}], P(worse) {np.mean(g > 0):.3f}")
    if arm == "opdi_scoped":
        ok = [r(p, body) - r(b, body) <= -0.5, r(p, body & (mo == 1)) < r(b, body & (mo == 1)),
              r(p, body & (mo == 7)) < r(b, body & (mo == 7)), np.quantile(g, 0.975) < 0,
              r(p, np.ones_like(body)) <= r(b, np.ones_like(body))]
        print(f"  GATE (body<=-0.5, Jan<0, Jul<0, CI<0, full<=0): {ok} -> {'PASS' if all(ok) else 'FAIL'}")
