"""Average corrector members fitted with different LightGBM seeds (reports/ctx_seedavg_rf_preregistration.md).

  member = (w_a * A + w_b * B) / (w_a + w_b)  for the holdout file, the five OOF folds and the ranking rows.

A, B are ctx_armc.py output names (e.g. ctxc1r_rf = v66's seed-42 fit, ctxc1r_rf_s2 = the in-process mean of seeds 43, 44).
The holdout file keeps A's chain columns with `pred` replaced; ranking rows are averaged from the unrounded armc_rank
output and rounded as ctx_armc.py prod does. Writes cache/eval/<OUT>_holdout_ev.parquet, cache/oof/<OUT>/fold=*.parquet,
data/submissions/<OUT>.parquet and cache/eval/armc_rank_<OUT>.parquet.

Run:  .venv/Scripts/python.exe src/post/ctx_seedavg_merge.py A W_A B W_B OUT
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"
A, WA, B, WB, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5]


def avg(fa: pl.DataFrame, fb: pl.DataFrame, col: str) -> tuple[pl.DataFrame, float]:
    """fa with `col` replaced by the weighted mean of fa and fb (joined on MVT_ID_mvt, same row set asserted)."""
    k = pl.col("MVT_ID_mvt").cast(pl.Int64)
    b = fb.select(k.alias("_id"), pl.col(col).cast(pl.Float64).alias("_b"))
    x = fa.with_columns(_id=k).join(b, on="_id", how="left")
    assert x.height == fa.height == fb.height and x["_b"].null_count() == 0, (A, B, col)
    a_, b_ = x[col].cast(pl.Float64).to_numpy(), x["_b"].to_numpy()
    m = (WA * a_ + WB * b_) / (WA + WB)
    return x.with_columns(pl.Series(col, m)).drop("_id", "_b"), float(np.sqrt(np.mean((a_ - b_) ** 2)))


h, r_h = avg(pl.read_parquet(EVAL / f"{A}_holdout_ev.parquet"), pl.read_parquet(EVAL / f"{B}_holdout_ev.parquet"), "pred")
h.write_parquet(EVAL / f"{OUT}_holdout_ev.parquet")
d = OOF / OUT
d.mkdir(parents=True, exist_ok=True)
for f in d.glob("*.parquet"):
    f.unlink()
folds = sorted(p.name for p in (OOF / A).glob("fold=*.parquet"))
assert len(folds) == 5 and folds == sorted(p.name for p in (OOF / B).glob("fold=*.parquet"))
r_o = []
for fn in folds:
    o, r = avg(pl.read_parquet(OOF / A / fn), pl.read_parquet(OOF / B / fn), "pred")
    o.write_parquet(d / fn)
    r_o.append(r)
rk, r_r = avg(pl.read_parquet(EVAL / f"armc_rank_{A}.parquet"), pl.read_parquet(EVAL / f"armc_rank_{B}.parquet"), "out")
rk.write_parquet(EVAL / f"armc_rank_{OUT}.parquet")
tpl = pl.read_parquet(SUB / f"{A}.parquet").select("MVT_ID_mvt")
sub = tpl.join(rk.select(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype),
                         TAXITIME_SEC_mvt=pl.col("out").round().cast(pl.Int32)), on="MVT_ID_mvt", how="left")
assert sub.height == tpl.height and sub["TAXITIME_SEC_mvt"].null_count() == 0
sub.write_parquet(SUB / f"{OUT}.parquet")
print(f"{OUT} = ({WA:g}*{A} + {WB:g}*{B}) / {WA + WB:g}; RMS {A} vs {B}: holdout {r_h:.2f} s, OOF folds "
      f"{[round(x, 2) for x in r_o]}, ranking {r_r:.2f} s")
