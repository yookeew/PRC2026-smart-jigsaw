exec(open(os.environ["SCR"] + "/recon2.py").read().split("for nm_, L, wv")[0])
det = pl.concat([pl.read_parquet(x, columns=["MVT_ID_mvt", "adsb_matched", "adsb_tier"]) for x in sorted(glob.glob("cache/adsb_pos_v4/day=2025-0[17]*.parquet"))])
det = det.select(pl.col("MVT_ID_mvt").cast(pl.Int64), has=(pl.col("adsb_matched").fill_null(False) | pl.col("adsb_tier").is_in(["fb_dwell", "fb_appear"]).fill_null(False)),
                 pb=pl.col("adsb_tier").is_in(["appear", "dwell", "fb_dwell", "fb_appear"]).fill_null(False))
dd = d.select(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(det, on="MVT_ID_mvt", how="left")
has = dd["has"].fill_null(False).to_numpy(); pb = dd["pb"].fill_null(False).to_numpy()
print(f"has ADS-B info {has.mean():.3f}, pushback tier {pb.mean():.3f}")
nl = apt != "LIRF"
def run(w, a, label):
    Lw = solve(w)
    q = np.maximum(ts - (dh + np.nan_to_num(a * Lw) * w), 0.0)
    qq = p.copy(); qq[nl] = q[nl]
    rep(label, qq)
    return qq
run(np.ones(len(y)), 0.75, "A all rows a0.75")
for a in (0.75, 1.0):
    run((~pb).astype(float), a, f"B pushback-tier fixed a{a}")
    run((~has).astype(float), a, f"C any-ADS-B fixed a{a}")
run(np.where(pb, 0.3, 1.0), 0.75, "D pushback-tier w0.3 a0.75")
r0 = y - p
for lab, m in (("pushback tier", pb & nl), ("other ADS-B info", has & ~pb & nl), ("no ADS-B", ~has & nl)):
    print(f"  {lab:18s} rows {m.sum():7,}  RMSE {np.sqrt(np.mean(r0[m]**2)):7.1f}")
