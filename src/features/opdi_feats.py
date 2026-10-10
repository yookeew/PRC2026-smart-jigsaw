"""Per-departure OPDI ground-event features (label-free), for the post-stages (PROGRESS §126).

Reads the airport-filtered OPDI flight events in cache/opdi_events/ (src/ingest/fetch_opdi_events.py) and the
departures of the 2025 training files and the 2026 ranking file, with identical code for both.

Match: an OPDI flight (flight_id within one 10-day file) at ADEP whose callsign equals CALLSIGN_flt (else
FLIGHT_mvt) and whose last exit-runway event at ADEP is the nearest to the takeoff time T, within 300 s.
Events of that flight at ADEP in [T - 3 h, T + 60 s] give the features (seconds before T, null when absent):
  o_matched, o_lead_first (earliest ground event), o_lead_exitpark (last exit-parking_position),
  o_lead_entrypark (first entry-parking_position), o_lead_taxi (first entry-taxiway), o_lead_apron (first apron
  event), o_lead_rwy (last entry-runway), o_deice (entry-deicing_pad seen), o_n_ground,
  o_day (OPDI file covering T exists; distinguishes "no OPDI data" from "not seen").

Output: cache/opdi_feats/opdi_feats.parquet (MVT_ID_mvt + the columns above).
Run:  python src/features/opdi_feats.py
      PRC_SCOPE=final4m: 2026 rows from final_ranking.parquet -> cache/opdi_feats/final4m/opdi_feats.parquet
      (2025 rows unchanged; PROGRESS.md §132).
"""

from __future__ import annotations

import datetime as dt
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scope  # noqa: E402

EVENTS = Path("cache/opdi_events")
OUT = scope.out_dir(Path("cache/opdi_feats")) / "opdi_feats.parquet"
GROUND = ["exit-parking_position", "entry-apron", "exit-apron", "entry-taxiway", "exit-taxiway", "entry-hangar",
          "exit-hangar", "entry-threshold", "exit-threshold", "entry-deicing_pad", "exit-deicing_pad"]
O_COLS = ["o_matched", "o_lead_first", "o_lead_exitpark", "o_lead_entrypark", "o_lead_taxi", "o_lead_apron",
          "o_lead_rwy", "o_deice", "o_n_ground"]
TOL_S = 300.0
WINDOW_S = 3 * 3600


def opdi_flights(files: list[str]) -> pl.DataFrame:
    """One row per (file, flight_id, airport) with a runway exit: callsign, t_rwy_exit and the event lists."""
    out = []
    for f in files:
        e = pl.read_parquet(f).with_columns(src=pl.lit(os.path.basename(f)))
        g = e.group_by("src", "flight_id", "airport").agg(
            callsign=pl.col("callsign").drop_nulls().mode().first(),
            t_rwy_exit=pl.col("t").filter(pl.col("type") == "exit-runway").max(),
            ev_t=pl.col("t"), ev_type=pl.col("type"))
        out.append(g.filter(pl.col("t_rwy_exit").is_not_null()))
    return pl.concat(out)


def departures() -> pl.DataFrame:
    """Departure keys from the 2025 training files and the 2026 ranking file (no label columns read)."""
    cols = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "PHASE_mvt"]
    fs = sorted(glob.glob("data/raw/training_*.parquet")) + [str(scope.RANKING)]
    d = pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
    return d.select(pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt",
                    T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms").cast(pl.Float64) / 1000.0,
                    cs=pl.coalesce(pl.col("CALLSIGN_flt").str.strip_chars(), pl.col("FLIGHT_mvt").str.strip_chars()))


def file_days(files: list[str]) -> pl.DataFrame:
    """Days covered by the downloaded files (file names carry [d0, d0 + 10))."""
    days = set()
    for f in files:
        a, b = os.path.basename(f).replace(".parquet", "").split("_")[2:4]
        d0, d1 = dt.datetime.strptime(a, "%Y%m%d").date(), dt.datetime.strptime(b, "%Y%m%d").date()
        days.update(d0 + dt.timedelta(days=i) for i in range((d1 - d0).days))
    return pl.DataFrame({"day": sorted(days)})


def build(dep: pl.DataFrame, files: list[str]) -> pl.DataFrame:
    od = opdi_flights(files).rename({"airport": "apt", "callsign": "cs"}).sort("t_rwy_exit")
    m = dep.sort("T").join_asof(od, left_on="T", right_on="t_rwy_exit", by=["apt", "cs"], strategy="nearest",
                                tolerance=TOL_S, check_sortedness=False)
    x = (m.filter(pl.col("flight_id").is_not_null()).select("MVT_ID_mvt", "T", "ev_t", "ev_type")
         .explode("ev_t", "ev_type")
         .filter((pl.col("ev_t") >= pl.col("T") - WINDOW_S) & (pl.col("ev_t") <= pl.col("T") + 60))
         .with_columns(lead=pl.col("T") - pl.col("ev_t")))
    f = x.group_by("MVT_ID_mvt").agg(
        o_lead_first=pl.col("lead").filter(pl.col("ev_type").is_in(GROUND)).max(),
        o_lead_exitpark=pl.col("lead").filter(pl.col("ev_type") == "exit-parking_position").min(),
        o_lead_entrypark=pl.col("lead").filter(pl.col("ev_type") == "entry-parking_position").max(),
        o_lead_taxi=pl.col("lead").filter(pl.col("ev_type") == "entry-taxiway").max(),
        o_lead_apron=pl.col("lead").filter(pl.col("ev_type").is_in(["entry-apron", "exit-apron"])).max(),
        o_lead_rwy=pl.col("lead").filter(pl.col("ev_type") == "entry-runway").min(),
        o_deice=(pl.col("ev_type") == "entry-deicing_pad").any().cast(pl.Float64),
        o_n_ground=pl.col("ev_type").is_in(GROUND).sum().cast(pl.Float64))
    days = file_days(files).with_columns(o_day=pl.lit(1.0))
    out = (m.select("MVT_ID_mvt", "T", o_matched=pl.col("flight_id").is_not_null().cast(pl.Float64))
           .join(f, on="MVT_ID_mvt", how="left")
           .with_columns(day=pl.from_epoch(pl.col("T").cast(pl.Int64), "s").dt.date())
           .join(days, on="day", how="left")
           .with_columns(pl.col("o_day").fill_null(0.0)))
    # no OPDI file for the day: every OPDI column null (not "not seen")
    return out.with_columns([pl.when(pl.col("o_day") > 0).then(pl.col(c)).otherwise(None).alias(c) for c in O_COLS]) \
              .select("MVT_ID_mvt", *O_COLS, "o_day")


def main() -> None:
    files = sorted(str(p) for p in EVENTS.glob("flight_events_*.parquet"))
    dep = departures()
    out = build(dep, files)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    print(f"{len(files)} OPDI files; {out.height:,} departures; OPDI day coverage {out['o_day'].mean():.3f}; "
          f"matched {out['o_matched'].mean():.3f}; ground event >= 60 s {(out['o_lead_first'].fill_null(0) >= 60).mean():.3f}")


if __name__ == "__main__":
    main()
