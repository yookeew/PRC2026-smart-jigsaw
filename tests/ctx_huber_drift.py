"""Output-spread stop of reports/ctx_huber_preregistration.md: RMS of (treatment - control) on NM-matched rows, per month
pooled over airports, 2026 production files vs 2025 holdout files. Stop if Jan or Jul 2026 > 1.25 x the 2025 same month.

Run:  .venv/Scripts/python.exe tests/ctx_huber_drift.py [TREAT] [CTL]   (default ctxc1sel_rh2hy ctxc1sel_rh2)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RATIO = 1.25


def meta(src: str) -> pl.DataFrame:
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(pl.col("MVT_ID_mvt").cast(pl.Float64), apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                    nmm=pl.col("FLIGHT_ID_mvt").is_not_null()).collect())


def diff(f_t: Path, f_c: Path, col: str, m: pl.DataFrame) -> pl.DataFrame:
    rd = lambda f, c: pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col(col).cast(pl.Float64).alias(c))  # noqa: E731
    x = rd(f_t, "t").join(rd(f_c, "c"), on="MVT_ID_mvt").join(m, on="MVT_ID_mvt", how="inner").filter(pl.col("nmm"))
    return x.with_columns(d=pl.col("t") - pl.col("c"))


def main(treat: str = "ctxc1sel_rh2hy", ctl: str = "ctxc1sel_rh2") -> None:
    a = diff(ROOT / "cache/eval" / f"{treat}_holdout_ev.parquet", ROOT / "cache/eval" / f"{ctl}_holdout_ev.parquet", "pred",
             meta(str(ROOT / "data/raw/training_2025-0[17]-*.parquet")))
    b = diff(ROOT / "data/submissions" / f"{treat}.parquet", ROOT / "data/submissions" / f"{ctl}.parquet", "TAXITIME_SEC_mvt",
             meta(str(ROOT / "data/ranking/ranking.parquet")))
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")  # noqa: E731
    stop = False
    for mo in (1, 7):
        x, z = a.filter(pl.col("mo") == mo)["d"].to_numpy(), b.filter(pl.col("mo") == mo)["d"].to_numpy()
        r = rms(z) / rms(x)
        stop |= r > RATIO
        print(f"month {mo}: 2025 holdout RMS {rms(x):6.2f} (p99 |d| {np.percentile(np.abs(x), 99):6.1f}, mean {x.mean():+.2f}, n {len(x):,}) | "
              f"2026 RMS {rms(z):6.2f} (p99 {np.percentile(np.abs(z), 99):6.1f}, mean {z.mean():+.2f}, n {len(z):,}) | ratio {r:.3f}")
    print("per airport (2025 Jan / Jul -> 2026 Jan / Jul RMS):")
    for ap in sorted(set(a["apt"].to_list())):
        v = [rms(f.filter((pl.col("apt") == ap) & (pl.col("mo") == mo))["d"].to_numpy()) for f in (a, b) for mo in (1, 7)]
        print(f"  {ap}: {v[0]:6.2f} / {v[1]:6.2f} -> {v[2]:6.2f} / {v[3]:6.2f}")
    print("STOP (output spread)" if stop else "pass (output spread)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
