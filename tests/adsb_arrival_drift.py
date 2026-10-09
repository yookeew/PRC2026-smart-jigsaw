"""Fallback 1 of reports/adsb_queue_scoping.md: does adsb.lol surface timing drift from 2025 to 2026?

2026 arrivals keep their block times, so the true taxi-in is known in both years. For each arrival: the adsb.lol
landing run (gs falls below 40 kt within 120 s of a faster sample), matched on airport + runway + |run start - MVT|
<= 120 s; the ADS-B in-block = first stationary (gs <= 1) sample within 100 m of the own stand ("stop"), else the
first sample within 100 m ("reach"). lag = (BLOCK - MVT) - (in-block_adsb - MVT), per airport x month x tier.
A 2025 -> 2026 lag shift means the receiver timing at stands changed, which the 2025-fitted departure lags in the
combiner cannot see. Writes cache/analysis/adsb_arrival_drift.parquet.

Run:  python tests/adsb_arrival_drift.py [--workers 10]
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from src.ingest.normalise_adsb import available_days, load_day  # noqa: E402
from src.ingest.stand_infer import load_inferred  # noqa: E402
from src.ingest.stands import attach_stand_coords, load_stands, stand_positions  # noqa: E402
from src.link.adsb_pushback import SURFACE_GS, to_xy  # noqa: E402
from adsb_queue_runs import RWY, lateral, runway_segments  # noqa: E402

OUT = ROOT / "cache" / "analysis" / "adsb_arrival_days"
R_STAND, MATCH_S, GAP_S = 100.0, 120.0, 600.0


def positions() -> pl.DataFrame:
    pos = stand_positions(load_stands())
    inf = load_inferred().join(pos.select("airport", "key"), on=["airport", "key"], how="anti")
    return pl.concat([pos, inf])


def landing_runs(ts, gs):
    n = len(ts)
    k = 1
    while k < n:
        if gs[k] < SURFACE_GS and gs[k - 1] >= SURFACE_GS and ts[k] - ts[k - 1] <= 120:
            j = k + 1
            while j < n and gs[j] < SURFACE_GS and ts[j] - ts[j - 1] <= GAP_S:
                j += 1
            yield k, j
            k = j
        else:
            k += 1


def do_day(day: dt.date) -> str:
    f = OUT / f"day={day.isoformat()}.parquet"
    if f.exists():
        return f"{day} cached"
    src = (ROOT / "data" / "ranking" / "ranking.parquet" if day.year >= 2026
           else next((ROOT / "data" / "raw").glob(f"training_{day:%Y-%m}-01_*.parquet")))
    mv = (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == day))
          .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  mts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
                  y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).collect())
    mv = attach_stand_coords(mv, POS, airport_col="ADES_mvt")
    adsb = load_day(day).sort("airport", "hex", "ts")
    rows = []
    for ap in sorted(adsb["airport"].unique().to_list()):
        r = RWY.filter(pl.col("airport") == ap)
        if r.height == 0:
            continue
        lat0 = float(r["lat"].mean())
        segs = runway_segments(ap, lat0)
        sub = adsb.filter(pl.col("airport") == ap)
        for (hx,), g in sub.group_by("hex", maintain_order=True):
            ts, gs = g["ts"].to_numpy(), g["gs"].to_numpy()
            gs = np.where(np.isnan(gs), SURFACE_GS, gs)
            xy = to_xy(g["lat"].to_numpy(), g["lon"].to_numpy(), lat0)
            trk = g["track"].to_numpy().astype(float)
            for i, j in landing_runs(ts, gs):
                hdg = trk[i - 1]
                best = None
                for name, a, v, L, h in segs:
                    if np.isnan(hdg) or abs((hdg - h + 180) % 360 - 180) > 25:
                        continue
                    along, lat = lateral(xy[i:i + 1], a, v, L)
                    if lat[0] <= 150 and -300 <= along[0] <= L + 300 and (best is None or lat[0] < best[1]):
                        best = (name, lat[0])
                if best is None:
                    continue
                rows.append(dict(airport=ap, hex=hx, runway=best[0], t_land=float(ts[i]), i=i, j=j,
                                 _ts=ts[i:j], _gs=gs[i:j], _xy=xy[i:j], lat0=lat0))
    if not rows:
        pl.DataFrame({"MVT_ID_mvt": []}).write_parquet(f)
        return f"{day} empty"
    # greedy nearest match on airport + runway
    cand = []
    by_key = {}
    for ri, rr in enumerate(rows):
        by_key.setdefault((rr["airport"], rr["runway"]), []).append(ri)
    for m in mv.iter_rows(named=True):
        for ri in by_key.get((m["ADES_mvt"], m["RUNWAY_mvt"]), []):
            dd = abs(rows[ri]["t_land"] - m["mts"])
            if dd <= MATCH_S:
                cand.append((dd, ri, m["MVT_ID_mvt"]))
    cand.sort()
    used_r, used_m, out = set(), set(), []
    mvd = {m["MVT_ID_mvt"]: m for m in mv.iter_rows(named=True)}
    for dd, ri, mid in cand:
        if ri in used_r or mid in used_m:
            continue
        used_r.add(ri); used_m.add(mid)
        rr, m = rows[ri], mvd[mid]
        rec = dict(MVT_ID_mvt=mid, airport=rr["airport"], land_gap=float(rr["t_land"] - m["mts"]), tier=None,
                   in_adsb=None, last_own_m=None, n_pts=len(rr["_ts"]), obs_after=float(rr["_ts"][-1] - m["mts"]))
        if m["stand_lat"] is not None:
            sxy = to_xy([m["stand_lat"]], [m["stand_lon"]], rr["lat0"])[0]
            d = np.hypot(*(rr["_xy"] - sxy).T)
            rec["last_own_m"] = float(d[-1])
            at = np.flatnonzero(d < R_STAND)
            if len(at):
                st = at[rr["_gs"][at] <= 1.0]
                k, tier = (st[0], "stop") if len(st) else (at[0], "reach")
                rec.update(tier=tier, in_adsb=float(rr["_ts"][k] - m["mts"]))
        out.append(rec)
    res = mv.select("MVT_ID_mvt", "ADES_mvt", "y", has_stand=pl.col("stand_lat").is_not_null()).join(
        pl.DataFrame(out, infer_schema_length=None).drop("airport"), on="MVT_ID_mvt", how="left").with_columns(day=pl.lit(day))
    res.write_parquet(f)
    return f"{day}: {mv.height:,} ARR, matched {len(out):,}, stop {sum(o['tier'] == 'stop' for o in out):,}"


POS = positions()


def report():
    d = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("day=*.parquet"))], how="diagonal_relaxed")
    d = d.with_columns(per=pl.col("day").dt.strftime("%Y-%m"), lag=pl.col("y") - pl.col("in_adsb"))
    d.write_parquet(ROOT / "cache" / "analysis" / "adsb_arrival_drift.parquet")
    ok = d.filter(pl.col("y").is_not_null() & (pl.col("y") <= 3600))
    agg = ok.group_by("ADES_mvt", "per").agg(
        n=pl.len(), matched=pl.col("land_gap").is_not_null().mean(), stop=(pl.col("tier") == "stop").mean(),
        lag_stop_med=pl.col("lag").filter(pl.col("tier") == "stop").median(),
        lag_stop_iqr=pl.col("lag").filter(pl.col("tier") == "stop").quantile(0.75) - pl.col("lag").filter(pl.col("tier") == "stop").quantile(0.25),
        within60=(pl.col("lag").filter(pl.col("tier") == "stop").abs() <= 60).mean(),
        lag_reach_med=pl.col("lag").filter(pl.col("tier") == "reach").median(),
        land_gap_med=pl.col("land_gap").median(), y_med=pl.col("y").median()).sort("ADES_mvt", "per")
    with pl.Config(tbl_rows=60, tbl_formatting="ASCII_MARKDOWN", float_precision=1, tbl_width_chars=220, tbl_cols=20):
        print("Arrivals with y <= 1 h. lag = true taxi-in - ADS-B taxi-in (s); stop = stationary within 100 m of own stand")
        print(agg)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--report" not in sys.argv:
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 10
        days = [d for d in available_days() if d.month in (1, 7)]
        with Pool(w) as p:
            for msg in p.imap_unordered(do_day, days):
                print(msg, flush=True)
    report()
