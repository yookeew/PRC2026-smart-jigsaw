"""OPDI scoping S1 (reports/opdi_scoping.md): label-free coverage of OPDI ground events on departures, and the
reach of information adsb.lol does not already give. Four 10-day windows (Jan / Jul 2025 holdout, Jan / Jul 2026).

Departure match: OPDI flight (flight_id within a file) at ADEP whose callsign equals CALLSIGN_flt (else FLIGHT_mvt)
and whose last exit-runway event is within 300 s of the takeoff time T. Ground events are the same flight's events
at ADEP in [T - 3 h, T + 60 s]."""
import os, glob, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

GROUND = ["exit-parking_position", "entry-apron", "exit-apron", "entry-taxiway", "exit-taxiway", "entry-hangar",
          "exit-hangar", "entry-threshold", "exit-threshold"]


def opdi_departures(files: list[str]) -> pl.DataFrame:
    """One row per (file, flight_id, airport) with a runway exit: callsign, t_rwy_exit, ground-event times."""
    out = []
    for f in files:
        e = pl.read_parquet(f).with_columns(src=pl.lit(os.path.basename(f)))
        g = e.group_by("src", "flight_id", "airport").agg(
            callsign=pl.col("callsign").drop_nulls().mode().first(),
            icao24=pl.col("icao24").drop_nulls().first(),
            t_rwy_exit=pl.col("t").filter(pl.col("type") == "exit-runway").max(),
            t_rwy_entry=pl.col("t").filter(pl.col("type") == "entry-runway").max(),
            ev_t=pl.col("t"), ev_type=pl.col("type"), ev_ref=pl.col("ref"))
        out.append(g.filter(pl.col("t_rwy_exit").is_not_null()))
    return pl.concat(out)


def match(dep: pl.DataFrame, od: pl.DataFrame, tol: float = 300.0) -> pl.DataFrame:
    """dep: MVT_ID_mvt, apt, T, cs (callsign key). Returns dep + matched OPDI columns (nearest runway exit)."""
    od = od.rename({"airport": "apt", "callsign": "cs"}).sort("t_rwy_exit")
    m = dep.sort("T").join_asof(od, left_on="T", right_on="t_rwy_exit", by=["apt", "cs"], strategy="nearest",
                                tolerance=tol)
    return m


def ground_feats(m: pl.DataFrame) -> pl.DataFrame:
    x = m.filter(pl.col("flight_id").is_not_null()).select("MVT_ID_mvt", "T", "ev_t", "ev_type").explode("ev_t", "ev_type")
    x = x.filter((pl.col("ev_t") >= pl.col("T") - 3 * 3600) & (pl.col("ev_t") <= pl.col("T") + 60))
    f = x.group_by("MVT_ID_mvt").agg(
        t_first_ground=pl.col("ev_t").filter(pl.col("ev_type").is_in(GROUND)).min(),
        t_exit_park=pl.col("ev_t").filter(pl.col("ev_type") == "exit-parking_position").max(),
        t_entry_park=pl.col("ev_t").filter(pl.col("ev_type") == "entry-parking_position").min(),
        t_first_taxi=pl.col("ev_t").filter(pl.col("ev_type") == "entry-taxiway").min(),
        t_rwy_entry=pl.col("ev_t").filter(pl.col("ev_type") == "entry-runway").max(),
        n_ground=pl.col("ev_type").is_in(GROUND).sum())
    return m.select("MVT_ID_mvt").join(f, on="MVT_ID_mvt", how="left")


def departures_2025(months=("2025-01-01", "2025-07-01")) -> pl.DataFrame:
    fs = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if any(m in f for m in months)]
    raw = pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed")
    return raw.filter(pl.col("PHASE_mvt") == "DEP")


def dep_keys(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.select(pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt",
                      T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms").cast(pl.Float64) / 1000.0,
                      cs=pl.coalesce(pl.col("CALLSIGN_flt").str.strip_chars(), pl.col("FLIGHT_mvt").str.strip_chars()),
                      nmu=pl.col("FLIGHT_ID_mvt").is_null())


def adsb(days: list[str]) -> pl.DataFrame:
    fs = [f for d in days for f in glob.glob(f"cache/adsb_pushback_v4/day={d}*.parquet")]
    return pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_matched", "adsb_tier", "adsb_first_ts")


if __name__ == "__main__":
    os.chdir(r"C:\Users\zhang\smart-jigsaw")
    windows = {"2025-01": ("flight_events_20250105_20250115", 2025), "2025-07": ("flight_events_20250704_20250714", 2025),
               "2026-01": ("flight_events_20260110_20260120", 2026), "2026-07": ("flight_events_20260709_20260719", 2026)}
    r25 = dep_keys(departures_2025())
    r26 = dep_keys(pl.read_parquet("data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP"))
    a = adsb(["2025-01", "2025-07", "2026-01", "2026-07"])
    rows = []
    for w, (fname, yr) in windows.items():
        od = opdi_departures([f"cache/opdi_events/{fname}.parquet"])
        lo, hi = od["t_rwy_exit"].min() + 3 * 3600, od["t_rwy_exit"].max() - 3600   # full-coverage interior
        dep = (r25 if yr == 2025 else r26).filter((pl.col("T") >= lo) & (pl.col("T") <= hi))
        m = match(dep, od)
        g = ground_feats(m)
        d = m.select("MVT_ID_mvt", "apt", "T", "nmu", "flight_id").join(g, on="MVT_ID_mvt", how="left").join(a, on="MVT_ID_mvt", how="left")
        d = d.with_columns(
            lead_first=(pl.col("T") - pl.col("t_first_ground")) / 60, lead_park=(pl.col("T") - pl.col("t_exit_park")) / 60,
            lead_adsb=((pl.col("T") - pl.col("adsb_first_ts")) / 60).fill_null(0.0),
            tiered=pl.col("adsb_tier").is_in(["dwell", "pass"]).fill_null(False))
        d = d.with_columns(new=(~pl.col("tiered")) & (pl.col("lead_first").fill_null(0) >= pl.col("lead_adsb") + 2)
                           & (pl.col("lead_first").fill_null(0) >= 1))
        t = d.group_by("apt").agg(n=pl.len(), opdi_matched=pl.col("flight_id").is_not_null().mean(),
                                  any_ground=(pl.col("lead_first").fill_null(0) >= 1).mean(),
                                  exit_park=(pl.col("lead_park").fill_null(0) >= 1).mean(),
                                  lead_first_med=pl.col("lead_first").median(), lead_adsb_med=pl.col("lead_adsb").median(),
                                  adsb_tiered=pl.col("tiered").mean(), new_reach=pl.col("new").mean()).with_columns(win=pl.lit(w))
        rows.append(t)
        print(f"{w}: departures {dep.height:,}, OPDI matched {d['flight_id'].is_not_null().mean():.3f}, new reach {d['new'].mean():.3f}")
    t = pl.concat(rows).sort("apt", "win")
    with pl.Config(tbl_rows=60, float_precision=3, tbl_width_chars=250, tbl_cols=20):
        print(t)
    piv = t.pivot(index="apt", on="win", values="new_reach")
    with pl.Config(tbl_rows=20, float_precision=3):
        print("new-information reach by window"); print(piv)
