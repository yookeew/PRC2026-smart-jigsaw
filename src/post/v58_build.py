"""v58 (reports/v57_daysum_model_preregistration.md, C3): v55 with only LSZH rows changed by the day-to-day variation
of the daily-sum level (expectation solve, month mean removed so v49's level stays), empirical-Bayes shrink
a = max(0, 1 - s2 / Var(u)), s2 = 2025 measurement noise (cache/v57/dsm_params.json). LOBT window last.

Run:  python src/post/v58_build.py data/submissions/smart-jigsaw_v58.parquet
"""
import os, sys, json
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
from post import pru_recon2 as R  # noqa: E402
from post import pru_dsm as M  # noqa: E402
import v57_dsm as T  # noqa: E402
import v57_build as B  # noqa: E402


def main(out_f):
    prm = json.load(open(ROOT / "cache/v57/dsm_params.json"))
    x = T.frame26()                         # p = v54; LSZH rows of v54 == v55 (asserted below)
    apt = x["apt"].to_numpy()
    v54, v55 = x["v54"].to_numpy(), x["v55"].to_numpy()
    lz = apt == "LSZH"
    assert (v54[lz] == v55[lz]).all()
    ho = T.frame25().filter(pl.col("period") == "P01")
    days = M.day_table(x, T.dists_from(ho), {"LSZH"})
    delta, summ = M.shifts(x, {"LSZH": days}, {"LSZH": []}, {"LSZH": prm["s2"]["LSZH"]}, level=False)
    print(f"LSZH 2026: {len(days)} days solved; {summ['LSZH']}")
    lam = np.array([d["lam"] for d in days]); mos = np.array([d["per"] for d in days])
    for m in (1, 7):
        print(f"  month {m}: mean lam (expectation) {lam[mos == m].mean():+.1f} s (not applied), within-month sd {lam[mos == m].std():.1f}")
    l0 = R.solve_days(x, None, skip=tuple(a for a in set(apt) if a != "LSZH"))
    for m in (1, 7):
        print(f"  month {m}: mean lam (point solve, v55 recipe) {np.mean([d['lam'] for d in l0 if d['per'] == m]):+.1f} s (not applied)")
    q = v55.astype(float).copy()
    q[lz] = np.maximum(v55[lz] - delta[lz], 0)
    q = B.finish(x, q)
    v55i = v55.astype(np.int32)
    assert (q[~lz] == v55i[~lz]).all(), "rows outside LSZH changed"
    dd = (q - v55i)[lz]
    print(f"vs v55: LSZH rows {lz.sum():,}, changed {(dd != 0).sum():,}, RMS {np.sqrt(np.mean(dd ** 2)):.2f}, mean {dd.mean():+.2f}, "
          f"max |d| {np.abs(dd).max()}; whole file RMS {np.sqrt(np.mean((q - v55i) ** 2)):.2f}")
    out = pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": q})
    tpl = pl.read_parquet(ROOT / "data/ranking/submitting.parquet")
    assert out.height == tpl.height == 344841 and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].cast(pl.Float64).to_list())
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(sys.argv[1])
