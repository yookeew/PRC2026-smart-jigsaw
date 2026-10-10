"""Day-shift allocation with within-day inputs, G3 / G3h vs v63's recipe G2x8 (reports/day_alloc_g3_preregistration.md).

Same frame, λ̂, periods, masks and hyper-parameters as tests/day_alloc_gbm.py; only the slope model's inputs change.

Run:  .venv/Scripts/python.exe tests/day_alloc_g3.py
"""
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
import v57_dsm as T  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402
import day_alloc_gbm as G  # noqa: E402
import hour_feats as H  # noqa: E402

FEAT_CACHE = ROOT / "cache" / "v57" / "hour_feats_2025.parquet"


def raw25() -> pl.DataFrame:
    return pl.concat([pl.read_parquet(f, columns=H.RAW_COLS) for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))],
                     how="vertical_relaxed")


def feats25(x: pl.DataFrame) -> pl.DataFrame:
    """Hour-window inputs for the 2025 frame's rows, neighbours' p = the frame's p (8 airports + LSZH)."""
    if FEAT_CACHE.exists():
        f = pl.read_parquet(FEAT_CACHE)
        if f.height == x.height and (f["MVT_ID_mvt"] == x["MVT_ID_mvt"]).all():
            return f
    dep, arr = H.movements(raw25())
    dep = dep.join(x.select("MVT_ID_mvt", "p"), on="MVT_ID_mvt", how="inner")
    f = H.build(dep, arr, H.tin_medians(arr))
    f = x.select("MVT_ID_mvt").join(f, on="MVT_ID_mvt", how="left")
    f.write_parquet(FEAT_CACHE)
    return f


def inputs3(x, f, lam0, own=True):
    cols = H.COLS if own else H.HOUR_COLS
    return np.column_stack([G.inputs(x, lam0, True)] + [f[c].to_numpy().astype(np.float64) for c in cols])


def main() -> None:
    x = T.frame25()
    lam, q0 = L.lam_rows(x)
    f = feats25(x)
    print("hour-feature coverage (non-null share, 8 airports):",
          {c: round(float(f.filter(pl.Series(np.isin(x["apt"].to_numpy(), T.A8)))[c].is_not_null().mean()), 3) for c in H.COLS}, flush=True)
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt, mo, per, nmu = x["apt"].to_numpy(), x["mo"].to_numpy(), x["period"].to_numpy(), x["nmu"].to_numpy()
    in8 = np.isin(apt, T.A8)
    ok = in8 & np.isfinite(lam)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    body, trim = ~nmu & (y <= 3600), y <= 18000
    r = p - y

    cands = {"G2x8 (v63)": G.inputs(x, lam0, True), "G3": inputs3(x, f, lam0, True), "G3h": inputs3(x, f, lam0, False)}
    qs = {}
    for name, X in cands.items():
        q = p.copy()
        sl = np.full(len(p), np.nan)
        for pk in T.PERIODS:
            mfit = ok & trim & (per != pk) & (np.abs(lam0) >= 1)
            bst = G.slope_fit(X, r, lam0, mfit)
            k = ok & (per == pk)
            sl[k] = np.clip(bst.predict(X[k], raw_score=True) + G.INIT, 0, 1)
            q[k] = np.maximum(p[k] - sl[k] * lam0[k], 0)
        qs[name] = q
        print(f"{name}: slope mean {np.nanmean(sl[ok]):.3f}, sd {np.nanstd(sl[ok]):.3f}", flush=True)
    qb = qs["G2x8 (v63)"]

    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    months = sorted(set(mo))
    rows = []
    for name, q in qs.items():
        res = {"cand": name, "full": rm(q, in8) - rm(qb, in8), "body": rm(q, in8 & body) - rm(qb, in8 & body),
               "trim": rm(q, in8 & trim) - rm(qb, in8 & trim),
               "mo_better": int(sum(rm(q, in8 & (mo == m)) < rm(qb, in8 & (mo == m)) for m in months)),
               "full_vs_R0": rm(q, in8) - rm(q0, in8)}
        for m in months:
            res[f"f{m[5:]}"] = round(rm(q, in8 & (mo == m)) - rm(qb, in8 & (mo == m)), 2)
        res["pass"] = (res["full"] <= -0.10 and res["body"] <= 0 and res["trim"] <= 0 and res["mo_better"] >= 8
                       and res["f02"] <= 0 and res["f06"] <= 0 and res["f01"] <= 0 and res["f07"] <= 0)
        rows.append(res)
    res = pl.DataFrame(rows)
    with pl.Config(tbl_rows=10, tbl_cols=40, float_precision=3, tbl_width_chars=300):
        print("vs v63's recipe (G2x8), 2025, 12 months, 8 airports' rows; fMM = full by month")
        print(res)
    passing = res.filter(pl.col("pass") & pl.col("cand").str.starts_with("G3")).sort("full")
    print("passing:", passing["cand"].to_list())
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = dict(rk.group_by("ADEP_mvt").agg(n=pl.len()).iter_rows())
    for name, q in qs.items():
        if name.startswith("G2"):
            continue
        print(f"{name} per airport full vs G2x8:", {a: round(rm(q, apt == a) - rm(qb, apt == a), 2) for a in T.A8})
        print(f"{name} per airport body vs G2x8:", {a: round(rm(q, (apt == a) & body) - rm(qb, (apt == a) & body), 2) for a in T.A8})
        dmse = sum(n26[a] * np.mean((y[apt == a] - q[apt == a]) ** 2 - (y[apt == a] - qb[apt == a]) ** 2) for a in T.A8) / sum(n26.values())
        print(f"{name} 2026 projection vs v63: {np.sqrt(238.32 ** 2 + dmse) - 238.32:+.3f} s")


if __name__ == "__main__":
    main()
