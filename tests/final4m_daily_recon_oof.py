"""Second gate for the day-level reconciliation on Feb/Jun-type months (PROGRESS.md §132).

The production stage (src/post/pru_daily_recon.py, non-LIRF day shift, LIRF untouched as in v55, LSZH skipped) was
gated on the Jan+Jul 2025 holdout only. Here the same code runs on the context corrector's out-of-fold predictions
(cache/oof/ctxc1sel_ro, two-month blocks fitted without the block) for every OOF month, with the 2025 EUROCONTROL file.
The input is the corrector stage, not the full chain, so the level of the gain is indicative; what is checked is that
the sign and size hold in February and June as in the other months.

Run:  .venv/Scripts/python.exe tests/final4m_daily_recon_oof.py
"""
import os
import sys
import glob
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ["PRU_LIRF"] = "0"                       # production v55 setting
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from post import pru_daily_recon as R  # noqa: E402

I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)


def main() -> None:
    oof = pl.concat([pl.read_parquet(f).select(I64, p=pl.col("pred").cast(pl.Float64))
                     for f in sorted(glob.glob(str(ROOT / "cache" / "oof" / "ctxc1sel_ro" / "*.parquet")))])
    raw = (pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "training_*.parquet")))],
                     how="vertical_relaxed")
           .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("TAXITIME_SEC_mvt").is_not_null()).with_columns(I64))
    h = R.frame(raw.join(oof, on="MVT_ID_mvt").with_columns(y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), 2025)
    q, _, _, _ = R.reconcile(h, None)
    y, p = h["y"].to_numpy(), h["p"].to_numpy()
    apt, mo, nmu = h["apt"].to_numpy(), h["mo"].to_numpy(), h["nmu"].to_numpy()
    r0, r1 = y - p, y - q
    body = ~nmu & (y <= 3600)
    d = lambda k: np.sqrt(np.mean(r1[k] ** 2)) - np.sqrt(np.mean(r0[k] ** 2))  # noqa: E731
    days = (h["apt"] + "_" + h["day"].cast(pl.Utf8)).to_numpy()
    rng = np.random.default_rng(0)
    print(f"OOF rows {len(y):,} (ctxc1sel_ro, LIRF untouched, skip {sorted(R.SKIP)})")
    print("month  rows     full base  full delta  [95% CI airport-day]   body delta   airports better/total")
    for m in sorted(set(mo)):
        k = mo == m
        ud, di = np.unique(days[k], return_inverse=True)
        sse0, dsse, n = np.bincount(di, r0[k] ** 2), np.bincount(di, r1[k] ** 2 - r0[k] ** 2), np.bincount(di)
        bs = []
        for _ in range(200):
            s = rng.integers(0, len(ud), len(ud))
            bs.append(np.sqrt((sse0[s].sum() + dsse[s].sum()) / n[s].sum()) - np.sqrt(sse0[s].sum() / n[s].sum()))
        moved = [a for a in sorted(set(apt[k])) if a != "LIRF" and a not in R.SKIP]
        better = sum(d(k & (apt == a)) < 0 for a in moved)
        print(f"2025-{m:02d} {k.sum():8,}  {np.sqrt(np.mean(r0[k] ** 2)):8.2f}  {d(k):+9.3f}  [{np.percentile(bs, 2.5):+.3f}, "
              f"{np.percentile(bs, 97.5):+.3f}]  {d(k & body):+9.3f}   {better}/{len(moved)}")
    for m in (2, 6):
        k = mo == m
        print(f"2025-{m:02d} per airport (full):", {a: round(d(k & (apt == a)), 2) for a in sorted(set(apt[k]))})


if __name__ == "__main__":
    main()
