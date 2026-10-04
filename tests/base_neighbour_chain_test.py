"""Stage-C full-chain gate for reports/base_neighbour_preregistration.md.

v30's method cross-fit (v23 ADS-B stages -> full-year combiner -> v2 within-day neighbour
corrector), computed twice on the Jan+Jul 2025 holdout: with the existing LightGBM, and with
the LightGBM that has the 13 model-free neighbour features (holdout predictions + OOF folds).
Trimmed RMSE decides.

Run:  .venv/Scripts/python.exe tests/base_neighbour_chain_test.py
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
import adsb_combiner_12m_test as T  # noqa: E402
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402
import neighbour_state_test as N  # noqa: E402
import neighbour_state_v2_test as N2  # noqa: E402

NEW_EV = ROOT / "cache" / "eval" / "lgb_mixed_nb_holdout_ev.parquet"
NEW_OOF = ROOT / "cache" / "oof" / "lgb_mixed_nb"
JAN, JUL = A.JAN, A.JUL


def v30_method(df: pl.DataFrame, ex: pl.DataFrame, lag: dict[str, float], arr: pl.DataFrame) -> np.ndarray:
    """v30's pipeline cross-fit on the holdout frame df (columns lgb, catcorr), with the
    OOF-month ADS-B frame ex (columns s, ...) as the extra training rows."""
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    base, _ = T.combiner_12m(d4, ex, v23, use_holdout=True)          # v27's method
    airports = sorted(d4["ADEP_mvt"].unique().to_list())
    nb_ex = N.nb_frame(ex, ex["s"].to_numpy(), lag, arr)
    ex_ok = (ex["taxi"] <= C.TRIM).to_numpy()
    yex = (ex["taxi"] - ex["s"]).to_numpy()
    iex = (ex["day"].dt.day() % 5 == 0).to_numpy()
    Xb_ex = N.x_b(ex, ex["s"].to_numpy(), nb_ex, airports)
    pop = d4.select(C2.has_adsb()).to_series().to_numpy()
    apt, day = d4["ADEP_mvt"].to_numpy(), d4["day"].to_numpy()
    new = base.copy()
    for fm, am in ((JAN, JUL), (JUL, JAN)):
        s = C.stack(d4, fm)
        Xb = N.x_b(d4, s, N.nb_frame(d4, s, lag, arr), airports)
        y = d4["taxi"].to_numpy() - s
        ih = (d4["day"].dt.day() % 5 == 0).to_numpy()
        fr = ((d4["ym"] == fm) & (d4["taxi"] <= C.TRIM)).to_numpy()
        mb, bb = N.fit(np.vstack([Xb_ex[ex_ok], Xb[fr]]), np.concatenate([yex[ex_ok], y[fr]]),
                       np.concatenate([iex[ex_ok], ih[fr]]), N.CAT_B)
        app = (d4["ym"] == am).to_numpy() & ~pop
        c = np.zeros(d4.height)
        c[app] = mb.predict(Xb[app])
        ct = N2.day_demean(c, apt, day, app)
        new[app] = np.clip(s[app] + ct[app], 0, N.CEIL)
        print(f"  corrector fit {fm}: best_iter {bb}", flush=True)
    return new


def with_new_lgb(df: pl.DataFrame, ex: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    ev = pl.read_parquet(NEW_EV).select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    df2 = df.join(ev, on="MVT_ID_mvt", how="left")
    assert df2["lgb_new"].null_count() == 0, "new holdout predictions missing rows"
    df2 = df2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new")
    folds = sorted(NEW_OOF.glob("fold=*.parquet"))
    assert len(folds) == 10, f"expected 10 OOF folds in {NEW_OOF}, found {len(folds)}"
    oof = pl.concat([pl.read_parquet(p) for p in folds]).select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    ex2 = ex.join(oof, on="MVT_ID_mvt", how="left")
    assert ex2["lgb_new"].null_count() == 0, "new OOF predictions missing rows"
    ex2 = ex2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new")
    w, _ = nnls(ex2.select("lgb", "cat").to_numpy(), ex2["taxi"].to_numpy())
    print(f"new OOF stack weights {np.round(w, 3)}")
    return df2, ex2.with_columns(s=pl.Series(ex2.select("lgb", "cat").to_numpy() @ w))


def main() -> None:
    df = A.load()
    ex, lag = N.train_extra()
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr = N.arrivals(paths, paths)
    print("== base: v30 method with the existing LightGBM")
    base = v30_method(df, ex, lag, arr)
    df2, ex2 = with_new_lgb(df, ex)
    print("== treatment: v30 method with the neighbour-feature LightGBM")
    new = v30_method(df2, ex2, lag, arr)
    d4 = V.with_det(df, V.read_det(C.V4))
    V.evaluate(d4, base, new, "STAGE C: v30 method, new LightGBM vs existing", decisive=True)


if __name__ == "__main__":
    main()
