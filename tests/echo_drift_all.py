"""2026 echo-recording drift at every airport, label-free via arrivals (extends PROGRESS §67, which did LIRF).

Echo = |BLOCK - SCHED| < 30 s. Among movements more than 30 min off schedule (|MVT - SCHED| > 1800 s), where an
echo makes the recorded taxi time absurd, per airport x month: arrival echo rate 2025 vs 2026 (arrivals keep their
block time in 2026), and the 2025 departure echo rate for reference. A jump in 2026 arrival echo at an airport whose
2025 departures never echo at large offsets would mean 2026 departure echoes the model prices at p ~ 0.

Run:  .venv/Scripts/python.exe -u tests/echo_drift_all.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def frame(src, yr):
    return (pl.scan_parquet(src).filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
            .select("PHASE_mvt", apt=pl.when(pl.col("PHASE_mvt") == "DEP").then("ADEP_mvt").otherwise("ADES_mvt"),
                    mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                    late=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() > 1800,
                    echo=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() < 30,
                    has_blk=pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
            .with_columns(yr=pl.lit(yr)).collect())


d = pl.concat([frame(str(ROOT / "data" / "raw" / "training_*.parquet"), 2025), frame(str(ROOT / "data" / "ranking" / "ranking.parquet"), 2026)])
g = (d.filter(pl.col("late") & pl.col("has_blk")).group_by("apt", "PHASE_mvt", "yr", "mo")
     .agg(n=pl.len(), rate=pl.col("echo").mean()).sort("apt", "PHASE_mvt", "yr", "mo"))
print("echo rate among movements > 30 min off schedule (n in brackets)")
print(f"{'apt':5s} | {'ARR Jan25':>14s} {'ARR Jan26':>14s} {'ARR Jul25':>14s} {'ARR Jul26':>14s} | {'DEP Jan25':>14s} {'DEP Jul25':>14s}")
for apt in sorted(d["apt"].unique()):
    def c(ph, yr, mo):
        r = g.filter((pl.col("apt") == apt) & (pl.col("PHASE_mvt") == ph) & (pl.col("yr") == yr) & (pl.col("mo") == mo))
        return f"{r['rate'][0]:.4f} ({r['n'][0]:5d})" if r.height else "             -"
    print(f"{apt:5s} | {c('ARR', 2025, 1)} {c('ARR', 2026, 1)} {c('ARR', 2025, 7)} {c('ARR', 2026, 7)} | {c('DEP', 2025, 1)} {c('DEP', 2025, 7)}")
