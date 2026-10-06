"""reports/ctx_adsb_stable_preregistration.md: the v40 corrector (ctxcorr2rb) at EHAM/LSZH/EDDF/LEBL, the v37
corrector (ctxcorr2r) elsewhere, through v37's chain, vs v37's chain.

Builds  cache/eval/ctxsel_holdout_ev.parquet, cache/oof/ctxsel/fold=*.parquet, data/submissions/ctxsel.parquet,
then runs the stage-2 comparison of oof_context_corrector_stage2.py.

Run:  PRC_LGB=ctxcorr2r .venv/Scripts/python.exe tests/ctx_adsb_stable_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

V37, V40 = "ctxcorr2r", "ctxcorr2rb"
STABLE = ["EHAM", "LSZH", "EDDF", "LEBL"]
NAME = "ctxsel"
EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"


def _airports() -> pl.DataFrame:
    srcs = [str(ROOT / "data" / "raw" / "training_*.parquet"), str(ROOT / "data" / "ranking" / "ranking.parquet")]
    return (pl.concat([pl.scan_parquet(s).filter(pl.col("PHASE_mvt") == "DEP")
                       .select(pl.col("MVT_ID_mvt").cast(pl.Float64), _apt=pl.col("ADEP_mvt")) for s in srcs])
            .collect().unique("MVT_ID_mvt"))


def _select(a: pl.DataFrame, b: pl.DataFrame, col: str, apt: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Frame a with `col` taken from b on STABLE-airport rows."""
    a2 = a.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id"))
    j = (a2.join(apt, left_on="_id", right_on="MVT_ID_mvt", how="left")
         .join(b.select(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id"), pl.col(col).alias("_b")), on="_id", how="left"))
    assert j.height == a.height and j["_apt"].null_count() == 0 and j["_b"].null_count() == 0
    sel = j["_apt"].is_in(STABLE)
    out = j.with_columns(pl.when(sel).then(pl.col("_b")).otherwise(pl.col(col)).cast(a.schema[col]).alias(col))
    return out.select(a.columns), int(sel.sum())


def build() -> None:
    apt = _airports()
    ev, n = _select(pl.read_parquet(EVAL / f"{V37}_holdout_ev.parquet"), pl.read_parquet(EVAL / f"{V40}_holdout_ev.parquet"), "pred", apt)
    ev.write_parquet(EVAL / f"{NAME}_holdout_ev.parquet")
    print(f"holdout: {n:,} of {ev.height:,} rows from {V40}")
    (OOF / NAME).mkdir(parents=True, exist_ok=True)
    folds = sorted(p.name for p in (OOF / V37).glob("fold=*.parquet"))
    assert len(folds) == 5
    for f in folds:
        o, n = _select(pl.read_parquet(OOF / V37 / f), pl.read_parquet(OOF / V40 / f), "pred", apt)
        o.write_parquet(OOF / NAME / f)
        print(f"oof {f}: {n:,} of {o.height:,} rows from {V40}")
    rk, n = _select(pl.read_parquet(SUB / f"{V37}.parquet"), pl.read_parquet(SUB / f"{V40}.parquet"), "TAXITIME_SEC_mvt", apt)
    rk.write_parquet(SUB / f"{NAME}.parquet")
    print(f"ranking: {n:,} of {rk.height:,} rows from {V40}")


def main() -> None:
    assert os.environ.get("PRC_LGB") == "ctxcorr2r", "the base must be v37's chain: PRC_LGB=ctxcorr2r"
    build()
    import oof_context_corrector_stage2 as S2
    S2.EV, S2.OOFD = EVAL / f"{NAME}_holdout_ev.parquet", OOF / NAME
    S2.main()


if __name__ == "__main__":
    main()
