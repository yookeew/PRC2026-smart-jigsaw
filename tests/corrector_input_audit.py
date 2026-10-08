"""Input audit of v42's context corrector (Arm C c1, block-9 member) on 2026 rows, NM-matched and NM-unmatched
(PROGRESS.md §105; label-free, no fitting). §95 audited the combiner and the neighbour stage; v42 fixed the corrector's
never-seen null patterns on NM-unmatched rows only.

Builds the production training matrix (12 months, the arm's fit rows) and the 2026 matrix exactly as
src/post/ctx_armc.py prod() does, then per airport x NM group:
  - share of 2026 rows whose null pattern never occurs among the fit rows;
  - per feature, share of 2026 values outside the airport's fit-row [p0.5, p99.5] (cells > 3% listed);
  - categorical codes (apt_c, stand_pfx) never seen among the fit rows.

Run:  ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_ADSB=1 .venv/Scripts/python.exe -u tests/corrector_input_audit.py
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
sys.path.insert(0, str(ROOT / "tests"))
from post import ctx_armc as A  # noqa: E402


def matrices(with_stack: bool = False):
    """ctx_armc.prod() up to the fit: training X, fit mask, 2026 X, applied mask, frames (+ the stacks s, s_r)."""
    import context_corrector_test as CC
    from post import nm_lines as NL
    rd = lambda path, c, col="pred": pl.read_parquet(path).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col(col).cast(pl.Float64).alias(c))  # noqa: E731
    oof = rd(A.OOF / A.BASE / "*.parquet", "lgb").join(rd(A.OOF / "cat_mixed" / "*.parquet", "cat"), on="MVT_ID_mvt")
    ho = rd(A.EVAL / f"{A.BASE}_holdout_ev.parquet", "lgb").join(rd(A.EVAL / "cat_mixed_holdout_ev.parquet", "cat"), on="MVT_ID_mvt")
    lab = NL.frame(str(CC.RAW / "training_*.parquet")).select("MVT_ID_mvt", "y")
    oy = oof.join(lab, on="MVT_ID_mvt")
    w, _ = nnls(oy.select("lgb", "cat").to_numpy(), oy["y"].to_numpy())
    tr = pl.concat([oof, ho])
    dep_all, arr_all = CC.movements(str(CC.RAW / "training_*.parquet"))
    dep = tr.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="inner")
    tr = dep.select("MVT_ID_mvt").join(tr, on="MVT_ID_mvt", how="left")
    lt = (dep["T"] - dep["t_LOBT_flt"]).to_numpy()
    win = lambda q, lt=lt: np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606))  # noqa: E731
    s = win(tr.select("lgb", "cat").to_numpy() @ w)
    arr = arr_all.filter(pl.col("L").is_not_null())
    X, _, catn = CC.build(dep, arr, s, CC.cell_table(dep_all))
    m = dep["m"].to_numpy()
    cel_cols = ["cel_p50", "cel_p90", "cel_sd", "cel_n"]
    cel = X.select(cel_cols).to_numpy().copy()
    for mm in sorted(set(m)):
        k = m == mm
        cel[k] = CC.apply_cells(dep.filter(pl.Series(k)), CC.cell_table(dep_all.filter(pl.col("m") != mm))).to_numpy()
    X = X.with_columns([pl.Series(c, cel[:, i]) for i, c in enumerate(cel_cols)]).with_columns(cel_rel=pl.Series(s) - pl.col("cel_p50"))
    base_cols = X.columns
    X = A.add_arm(X, dep, None)
    y = dep["y"].to_numpy(); nm = dep["nm"].to_numpy()
    d = (dep["T"] - dep["t_SCHED_TIME_UTC_mvt"]).to_numpy()
    line_rows = nm & (np.nan_to_num(d, nan=-np.inf) > NL.LINE_MIN_D)
    fit = A.arm_fit((~nm) & (y <= CC.TRIM) & ~line_rows, nm, line_rows, s, y)
    rk_l = pl.read_parquet(A.SUB / f"{A.BASE}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    rk_c = pl.read_parquet(A.SUB / "cat_mixed.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), cat=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    dep_r, arr_r = CC.movements(str(A.RANKING))
    rk = dep_r.select("MVT_ID_mvt").join(rk_l, on="MVT_ID_mvt", how="left").join(rk_c, on="MVT_ID_mvt", how="left")
    lt_r = (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy()
    s_r = win(rk.select("lgb", "cat").to_numpy() @ w, lt_r)
    Xr, _, _ = CC.build(dep_r, arr_r.filter(pl.col("L").is_not_null()), s_r, CC.cell_table(dep_all))
    Xr = Xr.with_columns(cel_rel=pl.Series(s_r) - pl.col("cel_p50")).select(base_cols)
    Xr = A.add_arm(Xr, dep_r, None).select(X.columns)
    nm_r = dep_r["nm"].to_numpy(); d_r = (dep_r["T"] - dep_r["t_SCHED_TIME_UTC_mvt"]).to_numpy()
    app = ~(nm_r & (np.nan_to_num(d_r, nan=-np.inf) > NL.LINE_MIN_D))
    if with_stack:
        return X, fit, dep, Xr, app, dep_r, catn, s, s_r
    return X, fit, dep, Xr, app, dep_r, catn


def main() -> None:
    X, fit, dep, Xr, app, dep_r, catn = matrices()
    cols = X.columns
    Xa, Xra = X.to_numpy().astype(np.float64)[fit], Xr.to_numpy().astype(np.float64)
    apt_tr, apt_r = dep["apt"].to_numpy()[fit], dep_r["apt"].to_numpy()
    nm_r = dep_r["nm"].to_numpy()
    print(f"features {len(cols)}; fit rows {fit.sum():,}; 2026 rows {len(Xra):,} (applied {app.sum():,})")
    seen = set(A.null_patterns(Xa).tolist())
    unseen = np.array([q not in seen for q in A.null_patterns(Xra).tolist()])
    print("\nnever-seen null patterns, 2026 applied rows, by airport (NM-matched | NM-unmatched):")
    for a in sorted(set(apt_r)):
        km, ku = (apt_r == a) & app & ~nm_r, (apt_r == a) & app & nm_r
        print(f"  {a}  {unseen[km].mean() if km.sum() else 0:.4f} of {km.sum():6,} | {unseen[ku].mean() if ku.sum() else 0:.4f} of {ku.sum():5,}")
    if unseen[app].any():
        nulls = np.isnan(Xra[unseen & app])
        top = np.argsort(-nulls.mean(0))[:10]
        print("  columns most often null on those rows: " + ", ".join(f"{cols[j]} {nulls.mean(0)[j]:.2f}" for j in top))
    print("\ncategorical codes never seen among fit rows (2026 applied rows):")
    for c in catn:
        if c not in cols:
            continue
        j = cols.index(c)
        tr_codes = set(np.unique(Xa[:, j][~np.isnan(Xa[:, j])]).tolist())
        v = Xra[:, j]
        new = ~np.isnan(v) & ~np.isin(v, list(tr_codes))
        print(f"  {c}: {new[app].sum():,} rows; by airport " + ", ".join(f"{a} {int((new & app & (apt_r == a)).sum())}" for a in sorted(set(apt_r)) if (new & app & (apt_r == a)).sum()))
    print("\nshare of 2026 values outside the airport's fit-row [p0.5, p99.5] (NM-matched applied rows; cells > 3%, top 30):")
    hits = []
    for j, c in enumerate(cols):
        if c in catn:
            continue
        for a in sorted(set(apt_r)):
            xt = Xa[apt_tr == a, j]; xr = Xra[(apt_r == a) & app & ~nm_r, j]
            xt, xr = xt[~np.isnan(xt)], xr[~np.isnan(xr)]
            if len(xt) < 500 or len(xr) < 200:
                continue
            lo, hi = np.quantile(xt, [0.005, 0.995])
            sh = float(np.mean((xr < lo) | (xr > hi)))
            if sh > 0.03:
                hits.append((sh, c, a, len(xr)))
    for sh, c, a, n in sorted(hits, reverse=True)[:30]:
        print(f"  {c:22s} {a}  {sh:6.1%} of {n:,}")
    if not hits:
        print("  none")


if __name__ == "__main__":
    main()
