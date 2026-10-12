"""Drift stop of reports/ctx_abwin_preregistration.md: block-13 inputs, 2025 rows on fetched ADS-B days vs 2026 ranking rows,
per in-scope airport (as tests/ctx_hour2_drift.py: OOF / holdout stack on 2025, production stack on the ranking rows).
SMD = (mean 2026 - mean 2025) / sd 2025. Stop if any in-scope airport has |SMD| > 0.25.

Run:  .venv/Scripts/python.exe tests/ctx_abwin_drift.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import context_corrector_test as CC  # noqa: E402
import ctx_hour2_drift as H  # noqa: E402
import oof_context_corrector_test as O  # noqa: E402


def main() -> None:
    dep_all, _ = CC.movements(str(CC.RAW / "training_*.parquet"))
    pr = O.preds()
    dep = pr.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="left")
    s25 = H.win(pr.select("lgb", "cat").to_numpy() @ H.W, (dep["T"] - dep["t_LOBT_flt"]).to_numpy())
    rd = lambda f, c: pl.read_parquet(ROOT / "data/submissions" / f).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
    dep_r, _ = CC.movements(str(CC.RANKING))
    rk = dep_r.select("MVT_ID_mvt").join(rd("lgb_mixed_nb.parquet", "lgb"), on="MVT_ID_mvt", how="left").join(rd("cat_mixed.parquet", "cat"), on="MVT_ID_mvt", how="left")
    assert rk["lgb"].null_count() == 0 and rk["cat"].null_count() == 0
    s26 = H.win(rk.select("lgb", "cat").to_numpy() @ H.W, (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy())
    a = pl.concat([dep.select("apt", day=(pl.col("T") // 86400).cast(pl.Int64)), CC.abwin_block(dep, s25)], how="horizontal")
    b = pl.concat([dep_r.select("apt"), CC.abwin_block(dep_r, s26)], how="horizontal")
    fetched = {int(Path(f).stem.split("=")[1].replace("-", "")) for f in CC.scope.day_files(CC.DET4)}
    days = a["day"].to_numpy()
    dd =pl.from_epoch(pl.Series(days * 86400), time_unit="s").dt.strftime("%Y%m%d").cast(pl.Int64).to_numpy()
    a = a.filter(pl.Series(np.isin(dd, list(fetched))))
    worst = 0.0
    print(f"{'airport':8s} {'input':10s} {'mean25':>8s} {'mean26':>8s} {'sd25':>7s} {'SMD':>7s} {'null25':>7s} {'null26':>7s}")
    for ap in CC.ABWIN_APTS:
        xa, xb = a.filter(pl.col("apt") == ap), b.filter(pl.col("apt") == ap)
        for c in CC.AB_WIN_COLS:
            x, z = xa[c].drop_nulls().to_numpy(), xb[c].drop_nulls().to_numpy()
            smd = (z.mean() - x.mean()) / x.std()
            worst = max(worst, abs(smd))
            print(f"{ap:8s} {c:10s} {x.mean():8.1f} {z.mean():8.1f} {x.std():7.1f} {smd:+7.3f} "
                  f"{xa[c].null_count() / max(xa.height, 1):7.3f} {xb[c].null_count() / max(xb.height, 1):7.3f}")
    print("out-of-scope airports all null:", all(b.filter(~pl.col("apt").is_in(CC.ABWIN_APTS))[c].null_count()
                                                == b.filter(~pl.col("apt").is_in(CC.ABWIN_APTS)).height for c in CC.AB_WIN_COLS))
    print(f"max in-scope |SMD| {worst:.3f} -> drift stop {'not triggered (PASS)' if worst <= 0.25 else 'TRIGGERED (FAIL)'}")


if __name__ == "__main__":
    main()
