"""Third-round checks for v55/v56: (1) extreme 2026 airport-day shifts, EHAM storm days;
(2) LIRF model calibration: standardised day surprise z = (R - E[contrib]) / sqrt(sig^2 + var) in the 2025 holdout vs 2026."""
import os, sys
sys.path.insert(0, "src")
import numpy as np, polars as pl
import post.pru_daily_recon as R
pl.Config.set_tbl_rows(25); pl.Config.set_tbl_width_chars(200)
ROOT = R.ROOT
tpl = pl.read_parquet(ROOT / "data/ranking/submitting.parquet")
b = pl.read_parquet(ROOT / "data/submissions/smart-jigsaw_v54.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64), p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
x = R.frame(tpl.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(rk, on="MVT_ID_mvt").join(b, on="MVT_ID_mvt"), 2026)
# (1) per airport-day lambda in 2026
p, ts = x["p"].to_numpy(), x["ts"].to_numpy(); dh = ts - p
x = x.with_columns(dh=pl.Series(dh))
rows = []
for (a, dday), g in x.group_by(["apt", "day"]):
    if a == "LIRF" or g["D"][0] is None: continue
    m = np.arange(g.height); lam = R.lam_for(m, g["dh"].to_numpy(), np.ones(g.height), g["D"][0])
    rows.append((a, dday, g.height, g["D"][0], float(np.maximum(0, g["dh"].to_numpy()).sum() / 60), lam))
L = pl.DataFrame(rows, schema=["apt", "day", "n", "D", "Dhat", "lam"], orient="row")
print("largest |lambda| airport-days 2026 (shift applied = 0.75 lambda; taxi moves by -0.75 lambda):")
print(L.sort(pl.col("lam").abs(), descending=True).head(12))
print("EHAM January 2026:"); print(L.filter((pl.col("apt") == "EHAM") & (pl.col("day").dt.month() == 1)).sort("day").head(10))
# (2) LIRF z-scores
def zscores(fr, cal):
    iso, sig, bias = cal
    out = []
    for dday, g in fr.filter(pl.col("apt") == "LIRF").group_by("day"):
        if g["D"][0] is None: continue
        p, ts, nmu = g["p"].to_numpy(), g["ts"].to_numpy(), g["nmu"].to_numpy()
        H = R.hedged(p, ts)
        base = np.maximum(0, (ts - p)[~H]).sum() / 60
        pi = iso.predict(R.ratio(p[H], ts[H]) + 10 * nmu[H]).clip(0.002, 0.998)
        N = (p[H] - pi * ts[H]) / (1 - pi); fl = N < R.N_MIN; N = np.where(fl, R.N_MIN, N)
        pi = np.where(fl, (p[H] - R.N_MIN) / np.maximum(ts[H] - R.N_MIN, 1), pi).clip(0.002, 0.998)
        c = np.maximum(ts[H] - N, 0) / 60
        Rr = g["D"][0] - base - bias
        out.append((Rr - ((1 - pi) * c).sum()) / np.sqrt(sig ** 2 + (pi * (1 - pi) * c ** 2).sum()))
    return np.array(out)
h = R.holdout_frame("cache/v54/ctxc1sel_ro_chain_all_holdout.parquet", "cache/v54/v48m_chain_holdout.parquet")
cal = R.calibrate(h.filter(pl.col("apt") == "LIRF"))
z25 = zscores(h, cal); z26 = zscores(x, cal)
for nm, z in (("2025 holdout (in-sample cal)", z25), ("2026", z26)):
    print(f"{nm}: days {len(z)}, mean z {z.mean():+.2f}, sd z {z.std():.2f}, |z|>2: {(np.abs(z) > 2).sum()}, max |z| {np.abs(z).max():.2f}")
