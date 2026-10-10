"""Learned day-shift allocation with a boosted slope model (reports/day_alloc_gbm_preregistration.md).

Shift = s(x)·λ̂ with s(x) from LightGBM (label (p − y)/λ̂, weight λ̂²), clipped to [0, 1]. Candidates G1/G2 at 8 or 7
airports vs v62's recipe (S3 at seven airports, R0 at LTFM), 2025, 12 months, leave-period-out as tests/v61_learned_dayshift.py.

Run:  .venv/Scripts/python.exe tests/day_alloc_gbm.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
import v57_dsm as T  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402

A7 = [a for a in T.A8 if a != "LTFM"]
PARAMS = dict(objective="regression", num_leaves=15, min_data_in_leaf=2000, learning_rate=0.05, lambda_l2=10.0,
              deterministic=True, force_row_wise=True, seed=0, verbose=-1, num_threads=8)
ROUNDS, INIT = 300, 0.5


def inputs(x, lam0, g2):
    a = np.searchsorted(np.array(T.A8), x["apt"].to_numpy())
    cols = [a, x["ts"].to_numpy() - x["p"].to_numpy(), x["hour"].to_numpy(), x["nmu"].to_numpy().astype(float),
            x["trk"].to_numpy().astype(float), x["opdi"].fill_null(False).to_numpy().astype(float), x["p"].to_numpy()]
    if g2:
        cols += [np.abs(lam0), np.sign(lam0)]
    return np.column_stack(cols).astype(np.float64)


def slope_fit(X, r, lam, m):
    ds = lgb.Dataset(X[m], label=r[m] / lam[m], weight=lam[m] ** 2, init_score=np.full(m.sum(), INIT),
                     categorical_feature=[0], free_raw_data=False)
    return lgb.train(PARAMS, ds, ROUNDS)


def production_model(g2=True):
    """Slope model fitted on all six 2025 periods (the build's model); also returns 2025 max |λ̂| per airport."""
    x = T.frame25()
    lam, _ = L.lam_rows(x)
    p, y, apt = x["p"].to_numpy().astype(float), x["y"].to_numpy(), x["apt"].to_numpy()
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    m = np.isin(apt, T.A8) & np.isfinite(lam) & (y <= 18000) & (np.abs(lam0) >= 1)
    lam25 = {a: float(np.nanmax(np.abs(lam[apt == a]))) for a in T.A8}
    return slope_fit(inputs(x, lam0, g2), p - y, lam0, m), lam25


def main() -> None:
    x = T.frame25()
    lam, q0 = L.lam_rows(x)
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt, mo, per, nmu = x["apt"].to_numpy(), x["mo"].to_numpy(), x["period"].to_numpy(), x["nmu"].to_numpy()
    b, td = L.band(x["ts"].to_numpy() - p), L.tod(x["hour"].to_numpy())
    dayk = (x["apt"] + "_" + x["day"].cast(pl.Utf8)).to_numpy()
    in8, in7 = np.isin(apt, T.A8), np.isin(apt, A7)
    ok = in8 & np.isfinite(lam)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    acell = np.where(in8, np.searchsorted(np.array(T.A8), apt), -1)
    body, trim = ~nmu & (y <= 3600), y <= 18000
    r = p - y

    # v62's recipe: S3 at seven airports, R0 at LTFM
    cell = np.where(in8, L.cells("S3", apt, b, td, lam0), -1)
    q8 = p.copy()
    for pk in T.PERIODS:
        s, _ = L.fit("S3", cell, acell, lam0, r, dayk, ok & trim & (per != pk))
        k = ok & (per == pk)
        q8[k] = np.maximum(p[k] - np.array([s.get(c, 0.0) for c in cell[k]]) * lam0[k], 0)
    qb = np.where(in7, q8, q0)

    qs = {}
    for g2 in (False, True):
        X = inputs(x, lam0, g2)
        q = p.copy()
        sl = np.full(len(p), np.nan)
        for pk in T.PERIODS:
            mfit = ok & trim & (per != pk) & (np.abs(lam0) >= 1)
            bst = slope_fit(X, r, lam0, mfit)
            k = ok & (per == pk)
            sl[k] = np.clip(bst.predict(X[k], raw_score=True) + INIT, 0, 1)
            q[k] = np.maximum(p[k] - sl[k] * lam0[k], 0)
        name = "G2" if g2 else "G1"
        qs[name + "x8"] = q
        qs[name + "x7"] = np.where(in7, q, q0)
        print(f"{name}: slope mean {np.nanmean(sl[ok]):.3f}, by tracked {np.nanmean(sl[ok & x['trk'].to_numpy()]):.3f} / "
              f"untracked {np.nanmean(sl[ok & ~x['trk'].to_numpy()]):.3f}; by band "
              f"{[round(float(np.nanmean(sl[ok & (b == i)])), 3) for i in range(4)]}", flush=True)

    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    months = sorted(set(mo))
    rows = []
    for name, q in [("S3x7 (v62)", qb), ("S3x8", q8)] + list(qs.items()):
        res = {"cand": name, "full": rm(q, in8) - rm(qb, in8), "body": rm(q, in8 & body) - rm(qb, in8 & body),
               "trim": rm(q, in8 & trim) - rm(qb, in8 & trim),
               "mo_better": int(sum(rm(q, in8 & (mo == m)) < rm(qb, in8 & (mo == m)) for m in months)),
               "full_vs_R0": rm(q, in8) - rm(q0, in8)}
        for m in months:
            res[f"f{m[5:]}"] = round(rm(q, in8 & (mo == m)) - rm(qb, in8 & (mo == m)), 2)
        res["pass"] = (res["full"] <= -0.10 and res["body"] <= 0 and res["trim"] <= 0 and res["mo_better"] >= 8
                       and res["f02"] <= 0 and res["f06"] <= 0)
        rows.append(res)
    res = pl.DataFrame(rows)
    with pl.Config(tbl_rows=10, tbl_cols=40, float_precision=3, tbl_width_chars=300):
        print("vs v62's recipe (S3 at 7 + R0 at LTFM), 2025, 12 months, 8 airports' rows; fMM = full by month")
        print(res)
    passing = res.filter(pl.col("pass") & (pl.col("cand").str.starts_with("G"))).sort("full")
    print("passing:", passing["cand"].to_list())
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = dict(rk.group_by("ADEP_mvt").agg(n=pl.len()).iter_rows())
    for name, q in qs.items():
        print(f"{name} per airport full vs v62 recipe:", {a: round(rm(q, apt == a) - rm(qb, apt == a), 2) for a in T.A8})
        print(f"{name} per airport body vs v62 recipe:", {a: round(rm(q, (apt == a) & body) - rm(qb, (apt == a) & body), 2) for a in T.A8})
        dmse = sum(n26[a] * np.mean((y[apt == a] - q[apt == a]) ** 2 - (y[apt == a] - qb[apt == a]) ** 2) for a in T.A8) / sum(n26.values())
        print(f"{name} 2026 projection vs v62: {np.sqrt(238.79 ** 2 + dmse) - 238.79:+.3f} s")


if __name__ == "__main__":
    main()
