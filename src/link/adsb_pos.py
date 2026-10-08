"""v4 detector + first-sighting position (scratch; production detector untouched).

Adds, for every matched run: adsb_first_x / adsb_first_y (metres, airport frame centred on the
mean stand position) and adsb_first_dx / adsb_first_dy (offset from the own stand, metres).
Everything else is the v4 logic verbatim (inferred stands, fallback for matched untiered rows).

Run:  python adsb_pos.py [YYYY-MM-DD ...]   -> cache/adsb_pos_v4/day=*.parquet
      python adsb_pos.py --check 2025-01-15 (compare shared columns with cache/adsb_pushback_v4)
"""
import os, sys, time
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import datetime as dt
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import src.link.adsb_pushback as A  # noqa: E402
from src.ingest.normalise_adsb import available_days, load_day  # noqa: E402
from src.ingest.stands import attach_stand_coords, load_stands, stand_positions  # noqa: E402
from src.ingest.stand_infer import load_inferred  # noqa: E402

OUT = ROOT / "cache" / "adsb_pos_v4"
EXTRA = {"adsb_first_x": pl.Float64, "adsb_first_y": pl.Float64,
         "adsb_first_dx": pl.Float64, "adsb_first_dy": pl.Float64}


def detect_day(day, pos):
    adsb = load_day(day)
    mv = attach_stand_coords(A.departures(day), pos)
    cov = dict(adsb.filter(pl.col("gs") < A.SURFACE_GS).group_by("airport").agg(pl.col("hex").n_unique()).iter_rows())
    out = []
    for (ap,), m_ap in mv.group_by("ADEP_mvt"):
        recs = m_ap.to_dicts()
        base = {"airport": ap, "adsb_day_coverage": cov.get(ap, 0) >= A.MIN_COVERAGE_AIRCRAFT}
        res = {r["MVT_ID_mvt"]: dict(MVT_ID_mvt=r["MVT_ID_mvt"], **base, adsb_matched=False,
                                     adsb_tier=None, adsb_pushback_ts=None, adsb_takeoff_gap_s=None, adsb_n_pts=None,
                                     adsb_first_ts=None, adsb_first_gs=None, adsb_first_own_m=None, adsb_min_own_m=None,
                                     adsb_pb_gap_s=None, adsb_pb_dist_m=None, adsb_pb_gs=None,
                                     **{k: None for k in EXTRA}) for r in recs}
        sub = adsb.filter(pl.col("airport") == ap)
        ap_pos = pos.filter(pl.col("airport") == ap)
        if sub.height and ap_pos.height:
            lat0 = ap_pos["lat"].mean()
            ref = A.to_xy([ap_pos["lat"].mean()], [ap_pos["lon"].mean()], lat0)[0]
            runs = []
            for _, g in sub.group_by("hex", maintain_order=True):
                ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
                gs = np.where(np.isnan(gs), A.SURFACE_GS, gs)
                xy = A.to_xy(g["lat"].to_numpy(), g["lon"].to_numpy(), lat0)
                for i, j in A.surface_runs(ts, gs):
                    runs.append((ts[i:j], gs[i:j], xy[i:j], ts[j - 1]))
            if runs:
                run_to = np.array([r[3] for r in runs])
                order = np.argsort(run_to)
                run_sorted = run_to[order]
                pairs = []
                for r in recs:
                    lo, hi = np.searchsorted(run_sorted, [r["mvt_ts"] - A.MATCH_TOL_S, r["mvt_ts"] + A.MATCH_TOL_S])
                    sxy = None if r["stand_lat"] is None else A.to_xy([r["stand_lat"]], [r["stand_lon"]], lat0)[0]
                    for k in range(lo, hi):
                        ri = int(order[k])
                        d_own = None if sxy is None else np.hypot(*(runs[ri][2] - sxy).T)
                        visits = d_own is not None and bool((d_own < A.R_STAND).any())
                        pairs.append((not visits, abs(run_to[ri] - r["mvt_ts"]), r["MVT_ID_mvt"], ri, d_own, r["mvt_ts"], sxy))
                pairs.sort(key=lambda p: (p[0], p[1]))
                used_m, used_r = set(), set()
                for _, _, mid, ri, d_own, mts, sxy in pairs:
                    if mid in used_m or ri in used_r:
                        continue
                    used_m.add(mid); used_r.add(ri)
                    ts, gs, xy, to = runs[ri]
                    row = res[mid]
                    row.update(adsb_matched=True, adsb_takeoff_gap_s=float(to - mts), adsb_n_pts=len(ts),
                               adsb_first_ts=float(ts[0]), adsb_first_gs=float(gs[0]),
                               adsb_first_x=float(xy[0][0] - ref[0]), adsb_first_y=float(xy[0][1] - ref[1]))
                    if d_own is not None:
                        row.update(adsb_first_own_m=float(d_own[0]), adsb_min_own_m=float(d_own.min()),
                                   adsb_first_dx=float(xy[0][0] - sxy[0]), adsb_first_dy=float(xy[0][1] - sxy[1]))
                        pb, tier, k = A.pushback(ts, gs, d_own)
                        if tier is not None:
                            row.update(adsb_tier=tier, adsb_pushback_ts=float(pb),
                                       adsb_pb_gap_s=float(ts[k + 1] - ts[k]) if k + 1 < len(ts) else None,
                                       adsb_pb_dist_m=float(d_own[k]), adsb_pb_gs=float(gs[k]))
        for row in res.values():
            row["adsb_fallback"] = False
        if sub.height and ap_pos.height:
            A._fallback(recs, res, sub, lat0, True)
        out.extend(res.values())
    return pl.DataFrame(out, schema={**A.SCHEMA_V3, **EXTRA})


def positions():
    pos = stand_positions(load_stands())
    inf = load_inferred().join(pos.select("airport", "key"), on=["airport", "key"], how="anti")
    return pl.concat([pos, inf])


if __name__ == "__main__":
    pos = positions()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--check" in sys.argv:
        day = dt.date.fromisoformat(args[0])
        t0 = time.time(); new = detect_day(day, pos); print(f"{day}: {time.time() - t0:.1f}s, {new.height} rows")
        old = pl.read_parquet(ROOT / "cache" / "adsb_pushback_v4" / f"day={day.isoformat()}.parquet")
        cols = [c for c in old.columns]
        a = old.sort("MVT_ID_mvt").select(cols); b = new.sort("MVT_ID_mvt").select(cols)
        print("shared columns identical:", a.equals(b, null_equal=True))
        for c in cols:
            if not a[c].equals(b[c], check_dtypes=False, null_equal=True):
                print("  differs:", c)
        print(new.select([pl.col(c).is_not_null().mean().alias(c) for c in EXTRA]))
        sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    days = [dt.date.fromisoformat(a) for a in args] or available_days()
    for day in days:
        f = OUT / f"day={day.isoformat()}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        df = detect_day(day, pos)
        df.write_parquet(f)
        print(f"{day}: {df.height} DEP matched {int(df['adsb_matched'].sum())} {time.time() - t0:.1f}s", flush=True)
