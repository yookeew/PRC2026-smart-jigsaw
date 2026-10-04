"""Model-free neighbour features for the base models (reports/base_neighbour_preregistration.md).

Departures close together in time share errors the base models miss (PROGRESS.md §70).
These columns describe the airport's / runway's live state from other movements only,
with no model prediction and no label in them, so they can be base-model inputs:

  nbnm_{apt,rwy}_{900,1800}   mean NM taxi (T - AOBT_3, 0..5400 s) of the other departures
                              taking off within +-W of T
  nboff_{apt,rwy}_{900,1800}  mean delay (T - SOBT, clipped -3600..14400) of those departures
  nbnm_anom_900, nboff_anom_900
                              the airport +-15 min means minus the same airport's UTC-day mean
                              (the within-day part only; the level drifts between years, §71)
  arrx_{900,1800}, arrx_prev  mean arrival taxi-in excess (in-block - landing - the 2025 median
                              for that airport/stand/runway) of arrivals landing within +-W of T,
                              and in [T - 1800, T - 900]

Each frame is processed on its own (train2025: all 2025 departures; holdout_gap2025: Jan+Jul
2025; ranking: Jan+Jul 2026), so neighbours come from the same frame. Departure off-block times
are never read. Arrival block times are never blanked, in training or ranking data.

Run:  .venv/Scripts/python.exe src/features/neighbour_feats.py
  -> cache/features_nb/{train2025,holdout_gap2025,ranking,labels2025}.parquet
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import shutil
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC, OUT = ROOT / "cache" / "features", ROOT / "cache" / "features_nb"
RAW, RANKING = ROOT / "data" / "raw", ROOT / "data" / "ranking" / "ranking.parquet"
WINDOWS = (900, 1800)
NB_COLS = ([f"{k}_{sc}_{w}" for k in ("nbnm", "nboff") for sc in ("apt", "rwy") for w in WINDOWS]
           + ["nbnm_anom_900", "nboff_anom_900"] + [f"arrx_{w}" for w in WINDOWS] + ["arrx_prev"])


def _win_mean(t: np.ndarray, v: np.ndarray, tq: np.ndarray, lo: float, hi: float, excl_self: bool) -> np.ndarray:
    ok = ~np.isnan(v)
    vv = np.where(ok, v, 0.0)
    cs, cn = np.concatenate([[0.0], np.cumsum(vv)]), np.concatenate([[0], np.cumsum(ok)])
    a, b = np.searchsorted(t, tq + lo, "left"), np.searchsorted(t, tq + hi, "right")
    s, n = cs[b] - cs[a], cn[b] - cn[a]
    if excl_self:
        s, n = s - vv, n - ok
    return np.where(n >= 2, s / np.maximum(n, 1), np.nan)


def arrivals(paths: list[Path]) -> pl.DataFrame:
    cols = ["PHASE_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
    a = pl.concat([pl.read_parquet(p, columns=cols) for p in paths]).filter(
        (pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    return a.with_columns(tin=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()
                          .cast(pl.Float64), t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0
                          ).filter(pl.col("tin").is_between(0, 3600))


def neighbour_columns(f: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """NB_COLS for every row of the feature frame f (row order kept)."""
    d = f.select("ADEP_mvt", "RUNWAY_mvt", t=pl.col("T").dt.epoch("ms") / 1000.0,
                 nm=pl.when(pl.col("aobt3_taxi").is_between(0, 5400)).then(pl.col("aobt3_taxi")).cast(pl.Float64),
                 off=pl.col("sched_takeoff_offset").cast(pl.Float64).clip(-3600, 14400),
                 day=pl.col("T").dt.date()).with_columns(_i=pl.int_range(pl.len()))
    out = {c: np.full(d.height, np.nan) for c in NB_COLS}
    for scope, keys in (("apt", ["ADEP_mvt"]), ("rwy", ["ADEP_mvt", "RUNWAY_mvt"])):
        for _, g in d.group_by(keys):
            g = g.sort("t")
            t, idx = g["t"].to_numpy(), g["_i"].to_numpy()
            for k, col in (("nbnm", "nm"), ("nboff", "off")):
                v = g[col].fill_null(np.nan).to_numpy()
                for w in WINDOWS:
                    out[f"{k}_{scope}_{w}"][idx] = _win_mean(t, v, t, -w, w, True)
    dm = d.select(nm_day=pl.col("nm").mean().over("ADEP_mvt", "day"), off_day=pl.col("off").mean().over("ADEP_mvt", "day"))
    out["nbnm_anom_900"] = out["nbnm_apt_900"] - dm["nm_day"].fill_null(np.nan).to_numpy()
    out["nboff_anom_900"] = out["nboff_apt_900"] - dm["off_day"].fill_null(np.nan).to_numpy()
    for (apt,), g in d.group_by(["ADEP_mvt"]):
        a = arr.filter(pl.col("ADES_mvt") == apt).sort("t")
        g = g.sort("t")
        t, idx, ta, xa = g["t"].to_numpy(), g["_i"].to_numpy(), a["t"].to_numpy(), a["xin"].to_numpy()
        for w in WINDOWS:
            out[f"arrx_{w}"][idx] = _win_mean(ta, xa, t, -w, w, False)
        out["arrx_prev"][idx] = _win_mean(ta, xa, t, -1800, -900, False)
    return pl.DataFrame({c: pl.Series(v, dtype=pl.Float32) for c, v in out.items()})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a25 = arrivals(sorted(RAW.glob("training_*.parquet")))
    med = a25.group_by("ADES_mvt", "STAND_mvt", "RUNWAY_mvt").agg(med=pl.col("tin").median())
    xin = lambda a: a.join(med, on=["ADES_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left").select(  # noqa: E731
        "ADES_mvt", "t", xin=pl.col("tin") - pl.col("med")).drop_nulls()
    arr = {"2025": xin(a25), "2026": xin(arrivals([RANKING]))}
    for name, year in (("train2025", "2025"), ("holdout_gap2025", "2025"), ("ranking", "2026")):
        f = pl.read_parquet(SRC / f"{name}.parquet")
        nb = neighbour_columns(f, arr[year])
        f = pl.concat([f, nb], how="horizontal")
        f.write_parquet(OUT / f"{name}.parquet")
        print(f"{name}: {f.height:,} rows; null share "
              f"{ {c: round(float(nb[c].is_nan().mean()), 3) for c in NB_COLS[:2] + ['nbnm_rwy_900', 'arrx_900']} }")
    shutil.copy(SRC / "labels2025.parquet", OUT / "labels2025.parquet")


if __name__ == "__main__":
    main()
