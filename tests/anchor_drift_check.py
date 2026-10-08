"""Label-free drift check of v42's 2026 predictions against the AOBT_3 anchor (PROGRESS.md §98). No training.

anchor a = MVT_TIME - AOBT_3 (seconds), NM-matched departures with AOBT_3 present.
  1. 2025 truth: per airport x month, median(y - a) and share |y - a| <= 120 s. Is it stable over the 12 months?
  2. 2025 holdout, v42 chain (Jan/Jul): median(v42 - a) -- does the model reproduce the truth's offset?
  3. 2026 ranking, v42 submission (Jan/Jul): median(pred - a) and share |pred - a| <= 120 s.
A 2026 cell whose prediction offset sits outside the 2025 truth's monthly range (and away from the model's
own 2025 offset) is a candidate for a 2026-only misreading; the anchor's own median is printed too, since a
shift in AOBT_3 recording would move both.

Run:  .venv/Scripts/python.exe -u tests/anchor_drift_check.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TOL = 120
CLIP = 300


def deps(src) -> pl.DataFrame:
    return (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "DEP") & pl.col("AOBT_3_flt").is_not_null()
                                        & pl.col("FLIGHT_ID_mvt").is_not_null())
            .select(pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                    a=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
            .collect())


def stats(d: pl.DataFrame, col: str) -> pl.DataFrame:
    e = pl.col(col) - pl.col("a")
    # the median sits on minute-quantised mass points (AOBT_3 is whole minutes), so a clipped mean is used
    return d.group_by("apt", "mo").agg(n=pl.len(), med=e.clip(-CLIP, CLIP).mean(), agree=(e.abs() <= TOL).mean(),
                                       a_med=pl.col("a").median()).sort("apt", "mo")


def main() -> None:
    tr = deps(str(RAW / "training_*.parquet"))
    t25 = stats(tr, "y")
    ho = tr.filter(pl.col("mo").is_in([1, 7])).join(
        pl.read_parquet(ROOT / "cache" / "eval" / "v42_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "v42"),
        on="MVT_ID_mvt")
    m25 = stats(ho, "v42")
    rk = deps(str(ROOT / "data" / "ranking" / "ranking.parquet")).join(
        pl.read_parquet(ROOT / "data" / "submissions" / "smart-jigsaw_v42.parquet")
        .select(pl.col("MVT_ID_mvt").cast(pl.Int64), p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
    m26 = stats(rk, "p")

    print(f"1. 2025 truth, mean of clip(y - a, +-{CLIP}) by month (s); last cols: 12-month range, agreement share range")
    for apt in sorted(t25["apt"].unique()):
        s = t25.filter(pl.col("apt") == apt)
        meds = s["med"].to_list()
        print(f"  {apt}  " + " ".join(f"{v:5.0f}" for v in meds)
              + f"  | range {min(meds):.0f}..{max(meds):.0f}  agree {s['agree'].min():.2f}..{s['agree'].max():.2f}")

    print("\n2-3. offset clipped-mean(x - a) and agreement share, Jan / Jul")
    print(f"  {'apt':5s} {'mo':>2s} | {'truth25':>7s} {'v42 25':>7s} | {'v42 26':>7s} {'n26':>6s} | "
          f"{'agr t25':>7s} {'agr m25':>7s} {'agr m26':>7s} | {'a med25':>7s} {'a med26':>7s}  flag")
    for apt in sorted(m26["apt"].unique()):
        rng = t25.filter(pl.col("apt") == apt)["med"]
        lo, hi = rng.min(), rng.max()
        for mo in (1, 7):
            g = lambda f: f.filter((pl.col("apt") == apt) & (pl.col("mo") == mo)).row(0, named=True)
            t, m, r = g(t25), g(m25), g(m26)
            off = r["med"] < lo - 10 or r["med"] > hi + 10
            gap = abs(r["med"] - m["med"]) > 15
            flag = ("OUT-OF-RANGE " if off else "") + ("MOVED vs model-25" if gap else "")
            print(f"  {apt:5s} {mo:2d} | {t['med']:7.0f} {m['med']:7.0f} | {r['med']:7.0f} {r['n']:6d} | "
                  f"{t['agree']:7.2f} {m['agree']:7.2f} {r['agree']:7.2f} | {t['a_med']:7.0f} {r['a_med']:7.0f}  {flag}")


if __name__ == "__main__":
    main()
