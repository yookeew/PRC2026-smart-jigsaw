"""OOF context corrector on the base stack (reports/oof_context_corrector_preregistration.md).

Stack s = NNLS(lgb_mixed_nb, cat_mixed) + LOBT window, OOF on the 10 training months and the
fit-on-10-months predictions on Jan+Jul 2025. A LightGBM learns (y - s) on OOF-month rows only
and is applied to the holdout: control (s + basics) vs treatment (+ the six context blocks).

Run:  .venv/Scripts/python.exe tests/oof_context_corrector_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_corrector_test as CC  # noqa: E402
from post import nm_lines as NL  # noqa: E402

EVAL, OOF = ROOT / "cache" / "eval", ROOT / "cache" / "oof"
TRIM = 18000


def preds() -> pl.DataFrame:
    rd = lambda path, c: pl.read_parquet(path).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias(c))
    oof = rd(OOF / "lgb_mixed_nb" / "*.parquet", "lgb").join(rd(OOF / "cat_mixed" / "*.parquet", "cat"), on="MVT_ID_mvt")
    ho = rd(EVAL / "lgb_mixed_nb_holdout_ev.parquet", "lgb").join(rd(EVAL / "cat_mixed_holdout_ev.parquet", "cat"), on="MVT_ID_mvt")
    return pl.concat([oof.with_columns(split=pl.lit("oof")), ho.with_columns(split=pl.lit("ho"))])


def prepare() -> dict:
    dep_all, arr_all = CC.movements(str(CC.RAW / "training_*.parquet"))
    raw = NL.frame(str(CC.RAW / "training_*.parquet"))
    pr = preds().join(raw.select("MVT_ID_mvt", "lt", "d", "nm", "y"), on="MVT_ID_mvt", how="inner")
    o = pr.filter(pl.col("split") == "oof")
    w, _ = nnls(o.select("lgb", "cat").to_numpy(), o["y"].to_numpy())
    print("OOF stack weights:", np.round(w, 3))
    lt = pr["lt"].to_numpy()
    win = lambda q: np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606))
    s = win(pr.select("lgb", "cat").to_numpy() @ w)
    # fixed lines: OOF rows use the lines fitted on the other OOF months? keep it simple and leak-free:
    # rows the lines would serve (NM-unmatched, d > 6 h) are excluded from training and keep s.
    dd = np.nan_to_num(pr["d"].to_numpy(), nan=-np.inf)
    line_rows = pr["nm"].to_numpy() & (dd > NL.LINE_MIN_D)
    dep = pr.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="left")
    assert dep.height == pr.height and dep["apt"].null_count() == 0
    m = dep["m"].to_numpy()
    is_ho = (pr["split"] == "ho").to_numpy()
    # cell statistics: leave-one-month-out for OOF rows, all 10 training months for holdout rows
    train10 = dep_all.filter(~pl.col("m").is_in([1, 7]))
    cells_all = CC.cell_table(train10)
    X_parts = []
    arr = arr_all.filter(pl.col("L").is_not_null())
    X_full, ctrl, catn = CC.build(dep, arr, s, cells_all)            # blocks with all-month cells
    cel_cols = ["cel_p50", "cel_p90", "cel_sd", "cel_n"]
    cel = X_full.select(cel_cols).to_numpy().copy()
    for mm in sorted(set(m[~is_ho])):
        k = (~is_ho) & (m == mm)
        tab = CC.cell_table(train10.filter(pl.col("m") != mm))
        cel[k] = CC.apply_cells(dep.filter(pl.Series(k)), tab).to_numpy()
    X_full = X_full.with_columns([pl.Series(c, cel[:, i]) for i, c in enumerate(cel_cols)])
    X_full = X_full.with_columns(cel_rel=pl.Series(s) - pl.col("cel_p50"))
    print(f"rows {X_full.height:,} (OOF {(~is_ho).sum():,}, holdout {is_ho.sum():,}); features {X_full.width}")

    y = dep["y"].to_numpy(); nm = dep["nm"].to_numpy(); apt = dep["apt"].to_numpy()
    day = (dep["T"].to_numpy() // 86400).astype(int)
    return dict(dep=dep, pr=pr, X_full=X_full, ctrl=ctrl, catn=catn, s=s, win=win, line_rows=line_rows,
                is_ho=is_ho, m=m, y=y, nm=nm, apt=apt, inner=day % 5 == 0)


def main() -> None:
    P = prepare()
    dep, pr, X_full, ctrl, catn, s, win = (P[k] for k in ("dep", "pr", "X_full", "ctrl", "catn", "s", "win"))
    line_rows, is_ho, m, y, nm, apt, inner = (P[k] for k in ("line_rows", "is_ho", "m", "y", "nm", "apt", "inner"))
    fit = (~is_ho) & (~nm) & (y <= TRIM) & ~line_rows
    app = is_ho & ~line_rows
    res = {}
    for arm, cols in (("control", ctrl), ("treatment", X_full.columns)):
        Xa = X_full.select(cols).to_numpy().astype(np.float64)
        cats = [cols.index(c) for c in catn if c in cols]
        c, best = CC.fit_apply(Xa, y - s, fit, inner, app, cats)
        q = s.copy(); q[app] = s[app] + c
        res[arm] = win(np.clip(q, 0, None))
        print(f"  {arm}: best_iter {best}", flush=True)
    H = is_ho
    body = H & (~nm) & (y <= 3600)
    r = lambda q, k: float(np.sqrt(np.mean((y - q)[k] ** 2)))
    print(f"\n{'slice (holdout)':22s} {'stack s':>9s} {'control':>9s} {'treatment':>10s} {'t - s':>7s} {'t - c':>7s}")
    rows = [("body pooled", body), ("body Jan", body & (m == 1)), ("body Jul", body & (m == 7)),
            ("trimmed", H & (y <= TRIM)), ("full", H)]
    rows += [(f"  body {a}", body & (apt == a)) for a in sorted(set(apt))]
    for name, k in rows:
        a0, a1, a2 = r(s, k), r(res["control"], k), r(res["treatment"], k)
        print(f"{name:22s} {a0:9.2f} {a1:9.2f} {a2:10.2f} {a2 - a0:+7.2f} {a2 - a1:+7.2f}")
    gs = {mo: r(s, body & mk) - r(res["treatment"], body & mk) for mo, mk in (("pooled", True), ("Jan", m == 1), ("Jul", m == 7))}
    gc = {mo: r(res["control"], body & mk) - r(res["treatment"], body & mk) for mo, mk in (("pooled", True), ("Jan", m == 1), ("Jul", m == 7))}
    ok = (gs["pooled"] >= 2 and gc["pooled"] >= 1 and min(gs["Jan"], gs["Jul"], gc["Jan"], gc["Jul"]) > 0
          and r(res["treatment"], H & (y <= TRIM)) <= r(s, H & (y <= TRIM)))
    print(f"\nSTAGE 1: vs stack {gs}, vs control {gc} -> {'PROCEED to stage 2' if ok else 'STOP'}")
    pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], "split": pr["split"], "s": s, "control": res["control"],
                  "treatment": res["treatment"]}).write_parquet(EVAL / "oof_context_corrector.parquet")


if __name__ == "__main__":
    main()
