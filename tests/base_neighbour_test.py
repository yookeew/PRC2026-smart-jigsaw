"""Stage-B gate for reports/base_neighbour_preregistration.md.

LightGBM with the 13 model-free neighbour features (cache/eval/lgb_mixed_nb_holdout_ev.parquet)
vs the existing LightGBM, each stacked with catcorr via NNLS, cross-fit Jan <-> Jul.
Trimmed RMSE decides.

Run:  .venv/Scripts/python.exe tests/base_neighbour_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402

NEW_EV = ROOT / "cache" / "eval" / "lgb_mixed_nb_holdout_ev.parquet"


def crossfit(df: pl.DataFrame, eng: list[str]) -> np.ndarray:
    out = np.empty(df.height)
    for fm, am in ((A.JAN, A.JUL), (A.JUL, A.JAN)):
        f = df.filter(pl.col("ym") == fm)
        w, _ = nnls(f.select(eng).to_numpy(), f["taxi"].to_numpy())
        m = (df["ym"] == am).to_numpy()
        out[m] = df.filter(pl.Series(m)).select(eng).to_numpy() @ w
        print(f"  NNLS {eng} fit on {fm}: {np.round(w, 3)}")
    return out


def main() -> None:
    df = A.load()
    new = pl.read_parquet(NEW_EV).select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_nb=pl.col("pred"))
    df = df.join(new, on="MVT_ID_mvt", how="inner")
    print(f"holdout rows with both predictions: {df.height:,}")
    y = df["taxi"].to_numpy()
    for c in ("lgb", "lgb_nb"):
        tr = (df["taxi"] <= 18000).to_numpy()
        print(f"  {c} alone: trimmed RMSE {np.sqrt(np.mean((df[c].to_numpy() - y)[tr] ** 2)):.2f}")
    base = crossfit(df, ["lgb", "catcorr"])
    treat = crossfit(df, ["lgb_nb", "catcorr"])
    V.evaluate(df, base, treat, "STAGE B: stack(lgb_nb, catcorr) vs stack(lgb, catcorr)", decisive=True)


if __name__ == "__main__":
    main()
