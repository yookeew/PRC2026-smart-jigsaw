"""2026 stand-identity drift per airport (PROGRESS.md §105, label-free). The drift map (§89) put 46% of the adversarial
gain on stand prefix x airport. Per airport, departures: share of 2026 rows whose stand / stand prefix never occurs in
2025 departures; total-variation distance between the 2025 Jan+Jul and 2026 Jan+Jul stand-prefix distributions (and,
as a yardstick, between 2025 Jan and 2025 Jul); the prefixes that moved most.

Run:  .venv/Scripts/python.exe -u tests/stand_drift_audit.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def dep(src, yr):
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(apt="ADEP_mvt", stand=pl.col("STAND_mvt").fill_null("?"), mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
            .with_columns(pfx=pl.col("stand").str.extract(r"^([A-Za-z]+)", 1).fill_null(pl.col("stand").str.slice(0, 1)), yr=pl.lit(yr))
            .collect())


d = pl.concat([dep(str(ROOT / "data" / "raw" / "training_*.parquet"), 2025), dep(str(ROOT / "data" / "ranking" / "ranking.parquet"), 2026)])
d25, d26 = d.filter(pl.col("yr") == 2025), d.filter(pl.col("yr") == 2026)


def tv(a: pl.DataFrame, b: pl.DataFrame, key: str) -> float:
    pa = a.group_by(key).len().with_columns(pa=pl.col("len") / a.height).drop("len")
    pb = b.group_by(key).len().with_columns(pb=pl.col("len") / b.height).drop("len")
    j = pa.join(pb, on=key, how="full", coalesce=True).fill_null(0)
    return float(0.5 * (j["pa"] - j["pb"]).abs().sum())


print(f"{'apt':5s} {'n26':>7s} {'new stand':>9s} {'new pfx':>8s} {'null stand 25/26':>16s} | {'TV pfx 25->26':>13s} {'TV pfx Jan->Jul 25':>18s} | {'TV stand 25->26':>15s} {'Jan->Jul 25':>11s}")
for apt in sorted(d26["apt"].unique()):
    a25, a26 = d25.filter(pl.col("apt") == apt), d26.filter(pl.col("apt") == apt)
    s25, p25 = set(a25["stand"]), set(a25["pfx"])
    j25, l25 = a25.filter(pl.col("mo") == 1), a25.filter(pl.col("mo") == 7)
    a25jj = a25.filter(pl.col("mo").is_in([1, 7]))
    print(f"{apt:5s} {a26.height:7,} {(~a26['stand'].is_in(list(s25))).mean():9.3f} {(~a26['pfx'].is_in(list(p25))).mean():8.3f} "
          f"{(a25['stand'] == '?').mean():7.3f}/{(a26['stand'] == '?').mean():.3f} | {tv(a25jj, a26, 'pfx'):13.3f} {tv(j25, l25, 'pfx'):18.3f} | "
          f"{tv(a25jj, a26, 'stand'):15.3f} {tv(j25, l25, 'stand'):11.3f}")
print("\nlargest prefix share moves (2025 Jan+Jul -> 2026 Jan+Jul), |delta| >= 2 pp:")
for apt in sorted(d26["apt"].unique()):
    a = d25.filter((pl.col("apt") == apt) & pl.col("mo").is_in([1, 7])); b = d26.filter(pl.col("apt") == apt)
    pa = a.group_by("pfx").len().with_columns(s25=pl.col("len") / a.height).drop("len")
    pb = b.group_by("pfx").len().with_columns(s26=pl.col("len") / b.height).drop("len")
    j = pa.join(pb, on="pfx", how="full", coalesce=True).fill_null(0).with_columns(dl=pl.col("s26") - pl.col("s25")).filter(pl.col("dl").abs() >= 0.02).sort("dl")
    if j.height:
        print(f"  {apt}: " + ", ".join(f"{r['pfx']} {100 * r['s25']:.1f}->{100 * r['s26']:.1f}%" for r in j.iter_rows(named=True)))
