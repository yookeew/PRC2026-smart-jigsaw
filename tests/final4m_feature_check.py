"""Jan/Jul rows of a four-month feature frame vs the two-month one (PROGRESS.md §126).

Adding Feb and Jun 2026 to the ranking frame may only change Jan/Jul features near the month
edges (31 Jan evening sees 1 Feb, 1 Jul morning sees 30 Jun). For every column this reports how
many Jan/Jul rows differ, and how many of those lie outside the edge window (takeoff on
31 Jan >= EDGE_H hours before midnight, or 1 Jul within EDGE_H hours after it, counts as edge).

Run:  .venv/Scripts/python.exe tests/final4m_feature_check.py OLD.parquet NEW.parquet [EDGE_H]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

old_f, new_f = sys.argv[1], sys.argv[2]
EDGE_H = float(sys.argv[3]) if len(sys.argv) > 3 else 36.0

old = pl.read_parquet(old_f)
new = pl.read_parquet(new_f)
ids = old.select("MVT_ID_mvt")
new = ids.join(new, on="MVT_ID_mvt", how="left")
assert new.height == old.height, "row count changed"
missing = new.filter(pl.col("T").is_null()).height if "T" in new.columns else 0
print(f"Jan/Jul rows: {old.height:,}; missing from the new frame: {missing}")

t = old["T"].cast(pl.Datetime("us", "UTC")) if old["T"].dtype != pl.Datetime else old["T"]
ts = (t.dt.epoch("s")).to_numpy().astype(float)
jan_end = pl.datetime(2026, 2, 1, time_zone="UTC")
edge_lo_jan = np.datetime64("2026-02-01T00:00:00").astype("datetime64[s]").astype(float) - EDGE_H * 3600
edge_hi_jul = np.datetime64("2026-07-01T00:00:00").astype("datetime64[s]").astype(float) + EDGE_H * 3600
edge = (ts >= edge_lo_jan) & (ts < edge_lo_jan + EDGE_H * 3600) | (ts < edge_hi_jul) & (ts >= edge_hi_jul - EDGE_H * 3600)
print(f"edge rows (within {EDGE_H:.0f} h of 1 Feb / 1 Jul 00:00 UTC): {edge.sum():,}")

rows = []
for c in old.columns:
    if c == "MVT_ID_mvt" or c not in new.columns:
        if c not in new.columns:
            rows.append((c, -1, -1, "MISSING IN NEW"))
        continue
    a, b = old[c], new[c]
    if a.dtype != b.dtype:
        try:
            b = b.cast(a.dtype)
        except Exception:
            rows.append((c, -1, -1, f"dtype {a.dtype} vs {b.dtype}"))
            continue
    if a.dtype.is_numeric():
        av, bv = a.cast(pl.Float64).to_numpy(), b.cast(pl.Float64).to_numpy()
        diff = ~((av == bv) | (np.isnan(av) & np.isnan(bv)) | np.isclose(av, bv, rtol=1e-6, atol=1e-6, equal_nan=True))
    else:
        diff = ~((a == b).fill_null(False) | (a.is_null() & b.is_null())).to_numpy()
    n, n_out = int(diff.sum()), int((diff & ~edge).sum())
    if n:
        rows.append((c, n, n_out, ""))
extra = [c for c in new.columns if c not in old.columns]
print(f"columns: {len(old.columns)} old, {len(new.columns)} new; only in new: {extra}")
if not rows:
    print("every column identical on every Jan/Jul row")
for c, n, n_out, note in sorted(rows, key=lambda r: -r[2]):
    print(f"  {c:40s} differ {n:>8,}   outside edge {n_out:>8,}  {note}")
bad = [r for r in rows if r[2] != 0]
print("RESULT:", "PASS (differences confined to the month edges)" if not bad else f"{len(bad)} column(s) differ outside the edges")
