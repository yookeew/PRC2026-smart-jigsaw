"""Three new label-free input families, screened on top of v66's corrector (ctxc1sel_rf OOF + holdout), PROGRESS.md §140.

D (direction flow): departures from the SAME origin whose initial great-circle bearing to ADES is in the same 30-degree
  sector (an exit-point / SID proxy, pooling destinations an en-route regulation hits together), +-60 min: n, median
  T - SOBT, median T - LOBT, share > 30 min late; plus great-circle distance. Coordinates: OurAirports (public domain).
L (LOBT order): departures on the same runway with LOBT within +-20 min: count, overtakers (later LOBT, earlier
  takeoff), overtaken, net share, median takeoff gap; the same at airport level with AOBT_3 order.
C (configuration change ahead): dominant departure runway per 15-min bin; time since the last change and until the next
  (both capped 3 h), and whether the flight's runway was the dominant one 20 / 40 min before its takeoff.
"""
import glob, os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "C:/Users/zhang/smart-jigsaw/src")
from screen import run
from post.flow_feats import _win_join
ROOT = "C:/Users/zhang/smart-jigsaw"; S = ROOT + "/cache/analysis/flow_screen"

f = pl.read_parquet(S + "/dep25_air_rf.parquet")
raw = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "RUNWAY_mvt"]) for p in sorted(glob.glob(ROOT + "/data/raw/training_2025-*.parquet"))])
f = f.drop("rwy").join(raw.select(pl.col("MVT_ID_mvt").cast(pl.Int64), rwy=pl.col("RUNWAY_mvt").fill_null("?")), on="MVT_ID_mvt", how="left")

# ---------------- D: direction sector
ap = (pl.read_csv(ROOT + "/external-data/ourairports/airports.csv", columns=["ident", "gps_code", "icao_code", "latitude_deg", "longitude_deg"],
                  infer_schema_length=0)
      .with_columns(pl.col("latitude_deg").cast(pl.Float64), pl.col("longitude_deg").cast(pl.Float64)))
coord = {}
for col in ("ident", "gps_code", "icao_code"):     # icao_code wins, then gps_code, then ident
    for k, la, lo in ap.select(col, "latitude_deg", "longitude_deg").drop_nulls().iter_rows():
        coord[k] = (la, lo)
def ll(codes):
    a = np.array([coord.get(c, (np.nan, np.nan)) if c is not None else (np.nan, np.nan) for c in codes], dtype=float)
    return a[:, 0], a[:, 1]
la1, lo1 = ll(f["apt"].to_list()); la2, lo2 = ll(f["ades"].to_list())
r = np.radians
y_ = np.sin(r(lo2 - lo1)) * np.cos(r(la2)); x_ = np.cos(r(la1)) * np.sin(r(la2)) - np.sin(r(la1)) * np.cos(r(la2)) * np.cos(r(lo2 - lo1))
brg = (np.degrees(np.arctan2(y_, x_)) + 360) % 360
dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin(r(la2 - la1) / 2) ** 2 + np.cos(r(la1)) * np.cos(r(la2)) * np.sin(r(lo2 - lo1) / 2) ** 2))
print("ADES with coordinates:", np.mean(~np.isnan(brg)).round(4), " same-airport ADES (dist<5km):", np.mean(dist < 5).round(4))
sec = np.where(np.isnan(brg) | (dist < 5), -1, (brg // 30).astype(int))
f = f.with_columns(sec=pl.Series(sec).cast(pl.Int32), dkm=pl.Series(dist))
d = f.select("MVT_ID_mvt", "apt", "sec", t="T", off="off", dl="dl").filter(pl.col("sec") >= 0)
parts = []
for a_ in d["apt"].unique().sort().to_list():
    a = d.filter(pl.col("apt") == a_)
    b = a.select("apt", "sec", id_j="MVT_ID_mvt", tj="t", off_j="off", dl_j="dl")
    x = _win_join(a.select("MVT_ID_mvt", "apt", "sec", "t"), b, ["apt", "sec"], "t", "tj", 3600.0)
    parts.append(x.group_by("MVT_ID_mvt").agg(ds_n=pl.len(), ds_moff=pl.col("off_j").median(), ds_mdl=pl.col("dl_j").median(),
                                              ds_f30=(pl.col("off_j") > 1800).mean()))
g = pl.concat(parts)
f = f.join(g, on="MVT_ID_mvt", how="left")
D = ["ds_n", "ds_moff", "ds_mdl", "ds_f30", "dkm"]

# ---------------- L: LOBT order on the runway, AOBT_3 order at the airport
def order_feats(keys, tcol, tag, w=1200.0):
    a = f.select("MVT_ID_mvt", *keys, t="T", s=tcol).filter(pl.col("s").is_not_null())
    b = a.select(*keys, id_j="MVT_ID_mvt", tj="t", sj="s")
    out = []
    for a_ in a["apt"].unique().sort().to_list():
        x = _win_join(a.filter(pl.col("apt") == a_), b.filter(pl.col("apt") == a_), keys, "s", "sj", w)
        out.append(x.group_by("MVT_ID_mvt").agg(**{
            f"{tag}_n": pl.len(), f"{tag}_over": ((pl.col("sj") > pl.col("s")) & (pl.col("tj") < pl.col("t"))).sum(),
            f"{tag}_under": ((pl.col("sj") < pl.col("s")) & (pl.col("tj") > pl.col("t"))).sum(),
            f"{tag}_mdT": (pl.col("tj") - pl.col("t")).median()}))
    o = pl.concat(out).with_columns(**{f"{tag}_net": (pl.col(f"{tag}_over").cast(pl.Float64) - pl.col(f"{tag}_under").cast(pl.Float64)) / pl.col(f"{tag}_n").cast(pl.Float64)})
    return o
f = f.join(order_feats(["apt", "rwy"], "LOBT", "lq"), on="MVT_ID_mvt", how="left")
f = f.join(order_feats(["apt"], "AOBT3", "aq"), on="MVT_ID_mvt", how="left")
L = [c for c in f.columns if c.startswith(("lq_", "aq_"))]

# ---------------- C: configuration change ahead / behind
T = f["T"].to_numpy(); apt = f["apt"].to_numpy(); rw = f["rwy"].to_numpy().astype(str)
cs = {k: np.full(f.height, np.nan) for k in ("cf_since", "cf_until", "cf_dom20", "cf_dom40", "cf_share_now")}
B = 900
for a_ in np.unique(apt):
    ka = np.flatnonzero(apt == a_)
    bins = (T[ka] // B).astype(np.int64)
    t = pl.DataFrame({"b": bins, "r": rw[ka]}).group_by("b", "r").agg(c=pl.len()).sort(["b", "c", "r"], descending=[False, True, False])
    dom = t.unique("b", keep="first", maintain_order=True).sort("b")
    db, dr = dom["b"].to_numpy(), dom["r"].to_numpy().astype(str)
    chg = np.flatnonzero(dr[1:] != dr[:-1]) + 1          # index of the first bin of each new regime
    cb = db[chg] * B                                     # change times (bin starts)
    i = np.searchsorted(cb, T[ka], "right")
    cs["cf_since"][ka] = np.where(i > 0, T[ka] - cb[np.maximum(i - 1, 0)], np.nan)
    cs["cf_until"][ka] = np.where(i < len(cb), cb[np.minimum(i, len(cb) - 1)] - T[ka], np.nan)
    m = dict(zip(db.tolist(), dr.tolist()))
    for lag, k in ((1200, "cf_dom20"), (2400, "cf_dom40")):
        lb = ((T[ka] - lag) // B).astype(np.int64)
        cs[k][ka] = [float(m.get(x) == r_) if m.get(x) is not None else np.nan for x, r_ in zip(lb.tolist(), rw[ka].tolist())]
    tot = t.group_by("b").agg(n=pl.col("c").sum())
    sh = t.join(tot, on="b").with_columns(s=pl.col("c") / pl.col("n"))
    shm = {(b_, r_): s_ for b_, r_, s_ in sh.select("b", "r", "s").iter_rows()}
    cs["cf_share_now"][ka] = [shm.get((x, r_), np.nan) for x, r_ in zip(bins.tolist(), rw[ka].tolist())]
for k in ("cf_since", "cf_until"):
    cs[k] = np.minimum(cs[k], 10800.0)
f = f.with_columns([pl.Series(k, v) for k, v in cs.items()])
C = list(cs)

f.write_parquet(S + "/dep25_dir_rf.parquet")
print("null shares:", {c: round(f[c].null_count() / f.height, 3) for c in D + L + C})
which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "D"):
    run(f, D, label="D direction-sector flow (on rf)")
if which in ("all", "L"):
    run(f, L, label="L LOBT/AOBT3 order displacement (on rf)")
if which in ("all", "C"):
    run(f, C, label="C configuration change ahead (on rf)")
if which in ("all", "DLC"):
    run(f, D + L + C, label="D+L+C (on rf)")
