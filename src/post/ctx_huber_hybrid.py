"""Huber / L2 corrector hybrid (reports/ctx_huber_preregistration.md).

  out = HUBER on NM-matched rows (FLIGHT_ID_mvt present), L2 on NM-unmatched rows,
row by row on holdout, OOF folds and (if both exist) ranking rows. Both inputs are already clipped at 0 and
LOBT-windowed, so the pick stays inside the window.

Run:  .venv/Scripts/python.exe src/post/ctx_huber_hybrid.py L2_SEL HUBER_SEL OUT
      e.g. ctxc1sel_rh2 ctxc1sel_rh2hu ctxc1sel_rh2hy
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import scope  # noqa: E402

EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"


def nm_matched() -> pl.DataFrame:
    srcs = [str(ROOT / "data" / "raw" / "training_*.parquet"), str(scope.RANKING)]
    return (pl.concat([pl.scan_parquet(s).filter(pl.col("PHASE_mvt") == "DEP")
                       .select(pl.col("MVT_ID_mvt").cast(pl.Float64), _nmm=pl.col("FLIGHT_ID_mvt").is_not_null()) for s in srcs])
            .collect().unique("MVT_ID_mvt"))


def pick(a: pl.DataFrame, b: pl.DataFrame, col: str, nmm: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Frame a (L2) with `col` taken from b (Huber) on NM-matched rows."""
    j = (a.with_columns(_id=pl.col("MVT_ID_mvt").cast(pl.Float64)).join(nmm, left_on="_id", right_on="MVT_ID_mvt", how="left")
         .join(b.select(_id=pl.col("MVT_ID_mvt").cast(pl.Float64), _b=pl.col(col)), on="_id", how="left"))
    assert j.height == a.height and j["_nmm"].null_count() == 0 and j["_b"].null_count() == 0
    n = int(j["_nmm"].sum())
    return j.with_columns(pl.when(pl.col("_nmm")).then(pl.col("_b").cast(a.schema[col])).otherwise(pl.col(col)).alias(col)).select(a.columns), n


def main(l2: str, hub: str, out: str) -> None:
    nmm = nm_matched()
    ev, n = pick(pl.read_parquet(EVAL / f"{l2}_holdout_ev.parquet"), pl.read_parquet(EVAL / f"{hub}_holdout_ev.parquet"), "pred", nmm)
    ev.write_parquet(EVAL / f"{out}_holdout_ev.parquet")
    print(f"holdout: {n:,} of {ev.height:,} rows from {hub}")
    (OOF / out).mkdir(parents=True, exist_ok=True)
    folds = sorted(p.name for p in (OOF / l2).glob("fold=*.parquet"))
    assert len(folds) == 5 and all((OOF / hub / f).exists() for f in folds)
    for f in folds:
        o, _ = pick(pl.read_parquet(OOF / l2 / f), pl.read_parquet(OOF / hub / f), "pred", nmm)
        o.write_parquet(OOF / out / f)
    if (SUB / f"{l2}.parquet").exists() and (SUB / f"{hub}.parquet").exists():
        rk, n = pick(pl.read_parquet(SUB / f"{l2}.parquet"), pl.read_parquet(SUB / f"{hub}.parquet"), "TAXITIME_SEC_mvt", nmm)
        assert rk["TAXITIME_SEC_mvt"].null_count() == 0
        rk.write_parquet(SUB / f"{out}.parquet")
        print(f"ranking: {n:,} of {rk.height:,} rows from {hub}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
