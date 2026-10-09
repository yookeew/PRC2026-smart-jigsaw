import os, sys
sys.path.insert(0, "src")
import numpy as np, polars as pl
import post.pru_daily_recon as R
pl.Config.set_tbl_rows(15); pl.Config.set_tbl_width_chars(220)
tpl = pl.read_parquet(R.ROOT / "data/ranking/submitting.parquet")
b = pl.read_parquet(R.ROOT / "data/submissions/smart-jigsaw_v54.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64), p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
rk = pl.read_parquet(R.ROOT / "data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
x = R.frame(tpl.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(rk, on="MVT_ID_mvt").join(b, on="MVT_ID_mvt"), 2026).filter(pl.col("apt") == "LIRF")
h = R.holdout_frame("cache/v54/ctxc1sel_ro_chain_all_holdout.parquet", "cache/v54/v48m_chain_holdout.parquet")
iso, sig, bias = R.calibrate(h.filter(pl.col("apt") == "LIRF"))
rows = []
for (dday,), g in x.group_by(["day"]):
    if g["D"][0] is None or abs(g.height / g["nD"][0] - 1) > 0.05: continue
    p, ts, nmu = g["p"].to_numpy(), g["ts"].to_numpy(), g["nmu"].to_numpy()
    H = R.hedged(p, ts)
    dpos = np.maximum(0, ts - p) / 60
    long12 = (ts >= 12 * 3600)
    pi = iso.predict(R.ratio(p[H], ts[H]) + 10 * nmu[H]).clip(0.002, 0.998)
    N = (p[H] - pi * ts[H]) / (1 - pi); fl = N < R.N_MIN; N = np.where(fl, R.N_MIN, N)
    pi = np.where(fl, (p[H] - R.N_MIN) / np.maximum(ts[H] - R.N_MIN, 1), pi).clip(0.002, 0.998)
    c = np.maximum(ts[H] - N, 0) / 60
    E = dpos[~H].sum() + bias + ((1 - pi) * c).sum()
    z = (g["D"][0] - E) / np.sqrt(sig ** 2 + (pi * (1 - pi) * c ** 2).sum())
    rows.append((dday, g.height, g["D"][0], round(E), round(z, 1), int(H.sum()), int(long12.sum()), round(dpos[long12].sum()), round((ts[long12] / 60).sum()),
                 round(dpos[nmu & ~H].sum())))
Z = pl.DataFrame(rows, schema=["day", "n", "D", "E", "z", "nH", "n12h", "pred_d_12h", "if_real_12h", "pred_d_nmu_nonH"], orient="row")
print(Z.sort(pl.col("z").abs(), descending=True).head(8))
z = Z["z"].to_numpy(); print(f"2026 LIRF days {len(z)}: mean z {z.mean():+.2f}, sd {z.std():.2f}, |z|>2: {(abs(z)>2).sum()}, |z|>3: {(abs(z)>3).sum()}")
for m in (1, 7):
    zz = Z.filter(pl.col("day").dt.month() == m)["z"].to_numpy(); print(f"  month {m}: mean {zz.mean():+.2f} sd {zz.std():.2f}")
