"""ADS-B takeoff runs of ALL aircraft (not only the matched own track), per raw day -> cache/adsb_rwyruns/.

Label-free. For every adsb.lol surface run that ends in a takeoff (src/link/adsb_pushback.surface_runs):
runway (centreline <= 150 m and heading <= 25 deg at the end of the run), roll-start point, visibility, time since
entering 1 km / 500 m of the roll start (only if seen outside first), slow time near the runway, and long stops
> 1 km from the roll start (de-icing candidates). Runs are matched to departures (|run end - MVT_TIME| <= 120 s,
greedy nearest, same airport and runway). See reports/adsb_queue_scoping.md.

Run:  python tests/adsb_queue_runs.py [--workers 8]
"""
import datetime as dt
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingest.normalise_adsb import available_days, load_day  # noqa: E402
from src.link.adsb_pushback import SURFACE_GS, surface_runs, to_xy  # noqa: E402

OUT = ROOT / "cache" / "adsb_rwyruns"
RWY = pl.read_csv(ROOT / "data" / "external" / "runways.csv")
LAT_M, HDG_DEG, ONRWY_M = 150.0, 25.0, 60.0
STOP_GS, STOP_S, MID_FAR_M = 1.0, 180.0, 1000.0
MATCH_S = 120.0


def runway_segments(ap: str, lat0: float) -> list[tuple]:
    r = RWY.filter(pl.col("airport") == ap)
    by = {x["runway"]: x for x in r.iter_rows(named=True)}
    segs = []
    for name, x in by.items():
        o = by.get(x["opposite"])
        if o is None:
            continue
        a = to_xy([x["lat"]], [x["lon"]], lat0)[0]
        b = to_xy([o["lat"]], [o["lon"]], lat0)[0]
        v = b - a
        hdg = (np.degrees(np.arctan2(v[0], v[1])) + 360) % 360   # bearing a -> b (x east, y north)
        segs.append((name, a, v, float(np.hypot(*v)), hdg))
    return segs


def lateral(p: np.ndarray, a: np.ndarray, v: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Signed along-track position (m from a) and absolute lateral distance of points p to the line a + t v."""
    u = v / L
    w = p - a
    along = w @ u
    lat = np.abs(w[:, 0] * u[1] - w[:, 1] * u[0])
    return along, lat


def one_run(ts, gs, xy, trk, i, j, segs):
    e = j - 1
    # runway: end-of-run position and the climb-out sample's track
    hdg = trk[j] if not np.isnan(trk[j]) else trk[e]
    best = None
    for name, a, v, L, h in segs:
        if np.isnan(hdg) or abs((hdg - h + 180) % 360 - 180) > HDG_DEG:
            continue
        along, lat = lateral(xy[e:e + 1], a, v, L)
        if lat[0] <= LAT_M and -300 <= along[0] <= L + 300 and (best is None or lat[0] < best[1]):
            best = (name, lat[0], a, v, L)
    seg = xy[i:j]
    k0 = e
    if best is not None:
        _, _, a, v, L = best
        _, lat = lateral(seg, a, v, L)
        k = e - i
        while k - 1 >= 0 and lat[k - 1] <= ONRWY_M and ts[i + k] - ts[i + k - 1] <= 60:
            k -= 1
        k0 = i + k
    p0 = xy[k0]
    d = np.hypot(*(seg - p0).T)
    t_end = ts[e]
    out = dict(runway=None if best is None else best[0], t_end=float(t_end), t_air=float(ts[j]),
               t_roll=float(ts[k0]), n_pts=int(j - i), obs_back=float(t_end - ts[i]),
               d_first=float(d[0]), x0=float(p0[0]), y0=float(p0[1]))
    tt = ts[i:j]
    for rad, key in ((1000.0, "t_1km"), (500.0, "t_500")):
        far = np.flatnonzero(d > rad)
        if len(far) and far[-1] + 1 < len(d):
            a_ = far[-1]
            out[key] = float(t_end - tt[a_ + 1])
            out[key + "_gap"] = float(tt[a_ + 1] - tt[a_])
        else:
            out[key], out[key + "_gap"] = None, None
    dt_ = np.minimum(np.diff(tt), 60.0)
    near_slow = (d[:-1] <= 1000) & (gs[i:j - 1] < 3.0)
    out["slow_1km"] = float(dt_[near_slow].sum())
    last10 = tt >= t_end - 600
    out["max_gap10"] = float(np.diff(tt[last10]).max()) if last10.sum() >= 2 else None
    # long stops away from the runway that are not the run's first stop (de-icing candidates)
    st = gs[i:j] <= STOP_GS
    n_mid, mid_s, mid_x, mid_y = 0, 0.0, None, None
    k = 0
    n = len(st)
    first = True
    while k < n:
        if not st[k]:
            k += 1
            continue
        m = k
        while m + 1 < n and st[m + 1] and tt[m + 1] - tt[m] <= 120:
            m += 1
        dur = tt[m] - tt[k]
        if dur >= STOP_S:
            if not (first and tt[k] - tt[0] <= 60):
                cx, cy = seg[k:m + 1].mean(axis=0)
                if np.hypot(cx - p0[0], cy - p0[1]) > MID_FAR_M:
                    n_mid += 1
                    mid_s += dur
                    if mid_x is None:
                        mid_x, mid_y = float(cx), float(cy)
            first = False
        elif k == 0:
            first = False
        k = m + 1
    out.update(n_mid=n_mid, mid_s=mid_s, mid_x=mid_x, mid_y=mid_y)
    return out


def do_day(day: dt.date) -> str:
    f = OUT / f"day={day.isoformat()}.parquet"
    if f.exists():
        return f"{day} cached"
    adsb = load_day(day).sort("airport", "hex", "ts")
    src = (ROOT / "data" / "ranking" / "ranking.parquet" if day.year >= 2026
           else next((ROOT / "data" / "raw").glob(f"training_{day:%Y-%m}-01_*.parquet")))
    mv = (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == day))
          .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt").collect())
    rows = []
    for ap in sorted(adsb["airport"].unique().to_list()):
        sub = adsb.filter(pl.col("airport") == ap)
        r = RWY.filter(pl.col("airport") == ap)
        if r.height == 0:
            continue
        lat0 = float(r["lat"].mean())
        segs = runway_segments(ap, lat0)
        for (hx,), g in sub.group_by("hex", maintain_order=True):
            ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
            gs = np.where(np.isnan(gs), SURFACE_GS, gs)
            xy = to_xy(g["lat"].to_numpy(), g["lon"].to_numpy(), lat0)
            trk = g["track"].to_numpy().astype(float)
            for i, j in surface_runs(ts, gs):
                if j - i < 2:
                    continue
                rows.append(dict(airport=ap, hex=hx, **one_run(ts, gs, xy, trk, i, j, segs)))
    runs = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    if runs.height == 0:
        pl.DataFrame({"airport": [], "MVT_ID_mvt": []}).write_parquet(f)
        return f"{day} empty"
    runs = runs.with_row_index("run_id")
    # greedy nearest match to departures
    mvd = mv.select("MVT_ID_mvt", "ADEP_mvt", mts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0)
    mvd = mvd.with_columns(mv["RUNWAY_mvt"])
    pairs = (runs.select("run_id", "airport", "runway", "t_end").join(mvd, left_on=["airport", "runway"],
                                                                       right_on=["ADEP_mvt", "RUNWAY_mvt"])
             .with_columns(dd=(pl.col("t_end") - pl.col("mts")).abs()).filter(pl.col("dd") <= MATCH_S).sort("dd"))
    used_r, used_m, match = set(), set(), {}
    for rid, mid in pairs.select("run_id", "MVT_ID_mvt").iter_rows():
        if rid in used_r or mid in used_m:
            continue
        used_r.add(rid); used_m.add(mid); match[rid] = mid
    runs = runs.with_columns(MVT_ID_mvt=pl.col("run_id").replace_strict(match, default=None, return_dtype=pl.Int64),
                             day=pl.lit(day))
    runs.write_parquet(f)
    return f"{day} runs {runs.height:,} matched {len(match):,} of {mv.height:,} departures"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    days = available_days()
    if "--day" in sys.argv:
        days = [dt.date.fromisoformat(sys.argv[sys.argv.index("--day") + 1])]
    with Pool(w) as p:
        for msg in p.imap_unordered(do_day, days):
            print(msg, flush=True)
