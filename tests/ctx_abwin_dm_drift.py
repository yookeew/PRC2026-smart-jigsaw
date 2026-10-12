"""Drift check of the demeaned ADS-B neighbour window (block 13d, reports/ctx_abwin_dm_preregistration.md): the two tier means
of tests/context_corrector_test.abwin_block minus their airport-day mean, no count. 2025 rows on fetched ADS-B days (OOF /
holdout stack) vs 2026 ranking rows (production stack), per in-scope airport: mean, SD, SD ratio 2026 / 2025, non-null share.

Run:  .venv/Scripts/python.exe tests/ctx_abwin_dm_drift.py
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

COLS = CC.AB_WIN_COLS[:2]


def demeaned(dep: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    b = pl.concat([dep.select("apt", day=(pl.col("T") // 86400).cast(pl.Int64)), CC.abwin_block(dep, p).select(COLS)], how="horizontal")
    return b.with_columns([(pl.col(c) - pl.col(c).mean().over("apt", "day")).alias(c) for c in COLS])


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
    a, b = demeaned(dep, s25), demeaned(dep_r, s26)
    fetched = {int(Path(f).stem.split("=")[1].replace("-", "")) for f in CC.scope.day_files(CC.DET4)}
    dd = pl.from_epoch(pl.Series(a["day"].to_numpy() * 86400), time_unit="s").dt.strftime("%Y%m%d").cast(pl.Int64).to_numpy()
    a = a.filter(pl.Series(np.isin(dd, list(fetched))))
    print(f"{'airport':8s} {'input':10s} {'mean25':>7s} {'mean26':>7s} {'sd25':>7s} {'sd26':>7s} {'ratio':>6s} {'SMD':>7s} {'nn25':>5s} {'nn26':>5s}")
    for ap in CC.ABWIN_APTS:
        xa, xb = a.filter(pl.col("apt") == ap), b.filter(pl.col("apt") == ap)
        for c in COLS:
            x, z = xa[c].drop_nulls().to_numpy(), xb[c].drop_nulls().to_numpy()
            print(f"{ap:8s} {c:10s} {x.mean():7.1f} {z.mean():7.1f} {x.std():7.1f} {z.std():7.1f} {z.std() / x.std():6.2f} "
                  f"{(z.mean() - x.mean()) / x.std():+7.3f} {len(x) / max(xa.height, 1):5.2f} {len(z) / max(xb.height, 1):5.2f}")


if __name__ == "__main__":
    main()
