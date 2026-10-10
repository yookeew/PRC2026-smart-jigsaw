"""v62 with the learned day-shift allocation G2 at the eight airports (reports/day_alloc_gbm_preregistration.md).

Input p as v62: v54 (live board) or the four-month chain before its day stage (f4m_e), + v52's LSZH OPDI delta and
v60's EDDF demeaned OPDI delta (src/post/final4m_v62.py, step 1). Then:
  EDDF EDDM EGLL EHAM LEBL LEMD LFPG LTFM   delta = s(x) * lam_g, lam_g = expectation solve on the 2026 sums (eta from
              the Jan + Jul 2025 holdout chain), s(x) = G2 slope model fitted on all six 2025 periods
              (tests/day_alloc_gbm.py), clipped to [0, 1]
  LSZH, LIRF  rows of the v62 file
LOBT window last. Check printed first: the same inputs run through v62's recipe (S3 at seven airports, R0 at LTFM,
v58 at LSZH) reproduce the v62 file.

Run:  python src/post/day_alloc_build.py data/submissions/smart-jigsaw_v54.parquet data/submissions/smart-jigsaw_v57.parquet OUT
      PRC_SCOPE=final4m python src/post/day_alloc_build.py data/submissions/final4m/f4m_e.parquet \
          data/submissions/final4m/f4m_v62.parquet OUT
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
import scope  # noqa: E402
from post import pru_recon2 as R  # noqa: E402
from post import pru_dsm as M  # noqa: E402
import v57_dsm as T  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402
import final4m_v62 as F  # noqa: E402
import day_alloc_gbm as G  # noqa: E402
from v61_build import s3_slopes  # noqa: E402

A7 = [a for a in T.A8 if a != "LTFM"]


def main(p_file, v62_file, out_f):
    x = F.frame(p_file)
    apt, mo = x["apt"].to_numpy(), x["mo"].to_numpy()
    p0 = x["p0"].to_numpy()
    v62 = x.select("MVT_ID_mvt").join(pl.read_parquet(v62_file).select(F.F64, v="TAXITIME_SEC_mvt"),
                                      on="MVT_ID_mvt", how="left")["v"].to_numpy().astype(np.int32)

    # OPDI deltas exactly as v62 (final4m_v62 step 1)
    cd = pl.read_parquet(F.V52 / "comb_delta_opdi.parquet").select(F.F64, dc="delta")
    nd = pl.read_parquet(F.V52 / "nb_delta_opdi.parquet").select(F.F64, dn="delta")
    x = x.join(cd, on="MVT_ID_mvt", how="left").join(nd, on="MVT_ID_mvt", how="left").with_columns(
        raw=pl.col("dc").fill_null(0) + pl.col("dn").fill_null(0))
    x = x.with_columns(share=pl.col("cov").mean().over("apt", "tday"))
    keep = (pl.col("apt") == "EDDF") & pl.col("cov") & (pl.col("share") >= 0.2)
    x = x.with_columns(dm=pl.when(keep).then(pl.col("raw") - pl.col("raw").filter(keep).mean().over("apt", "tday")).otherwise(0.0))
    x = x.with_columns(raw=pl.when(pl.col("nmline")).then(0.0).otherwise(pl.col("raw")),
                       dm=pl.when(pl.col("nmline")).then(0.0).otherwise(pl.col("dm")))
    lz, ed = apt == "LSZH", apt == "EDDF"
    add = np.where(lz, x["raw"].to_numpy(), np.where(ed, x["dm"].to_numpy(), 0.0))
    x2 = F.with_p(x, p0 + add)
    p2 = x2["p"].to_numpy()
    dists = T.dists_from(T.frame25().filter(pl.col("period") == "P01"))
    lam = np.full(x2.height, np.nan)
    for d in M.day_table(x2, dists, set(T.A8)):
        lam[d["g"]] = d["lam"]

    # 0. check: v62's recipe on these inputs reproduces the v62 file
    s, _ = s3_slopes()
    b = L.band(x2["ts"].to_numpy() - p2)
    k7 = np.isin(apt, A7) & np.isfinite(lam)
    cell = np.searchsorted(np.array(T.A8), apt[k7]) * 100 + b[k7] * 10
    qc = p2.copy()
    qc[k7] = np.maximum(p2[k7] - np.array([s.get(int(c), s.get(int(c) // 100 * 100, 0.0)) for c in cell]) * lam[k7], 0.0)
    qr = R.apply(x2, R.solve_days(x2, None, skip=tuple(a for a in set(apt) if a != "LTFM")), 0.75)
    qc[apt == "LTFM"] = qr[apt == "LTFM"]
    prm = json.load(open(ROOT / "cache" / "v57" / "dsm_params.json"))
    delta, _ = M.shifts(x2, {"LSZH": M.day_table(x2, dists, {"LSZH"})}, {"LSZH": []}, {"LSZH": prm["s2"]["LSZH"]}, level=False)
    qc[lz] = np.maximum(p2[lz] - delta[lz], 0.0)
    qc[apt == "LIRF"] = p0[apt == "LIRF"]
    qc = F.finish(x2, qc)
    print(f"check, v62 recipe vs {Path(v62_file).name}: per airport rows differing / max |diff|")
    for a in sorted(set(apt)):
        m = apt == a
        print(f"  {a}: {(qc[m] != v62[m]).sum():6,} / {np.abs(qc[m] - v62[m]).max()}")

    # 1. G2 at the eight airports
    bst, lam25 = G.production_model(g2=True)
    k8 = np.isin(apt, T.A8) & np.isfinite(lam)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    X = G.inputs(x2.with_columns(opdi=pl.col("cov")), lam0, True)
    sl = np.clip(bst.predict(X[k8], raw_score=True) + G.INIT, 0, 1)
    q = p2.copy()
    q[k8] = np.maximum(p2[k8] - sl * lam[k8], 0.0)
    out_q = F.finish(x2, q)
    other = ~np.isin(apt, T.A8)
    out_q[other] = v62[other]
    bb = b[k8]
    print("G2 slopes in use, mean by band (< 0 / 0-15 / 15-60 / >= 60 min):")
    for a in T.A8:
        m = apt[k8] == a
        print(f"  {a}: " + " / ".join(f"{sl[m & (bb == i)].mean():.2f}" if (m & (bb == i)).any() else "  - " for i in range(4))
              + f" | 2026 max |lam| {np.nanmax(np.abs(lam[apt == a])):.0f} (2025 {lam25[a]:.0f})")
    dd = out_q.astype(np.int64) - v62
    print("vs v62: per airport x month RMS / mean; mean shift vs p (new / v62)")
    for a in sorted(set(apt)):
        m = apt == a
        print(f"  {a}: " + "  ".join(f"m{mm} {np.sqrt(np.mean(dd[m & (mo == mm)] ** 2.0)):6.1f} {dd[m & (mo == mm)].mean():+6.1f}"
                                    for mm in sorted(set(mo)))
              + f" | shift {np.mean(out_q[m] - p2[m]):+7.2f} / {np.mean(v62[m] - p2[m]):+7.2f}")
    print(f"all: rows changed {(dd != 0).sum():,}, RMS {np.sqrt(np.mean(dd ** 2.0)):.2f}, mean {dd.mean():+.2f}, max |d| {np.abs(dd).max()}")
    tpl = pl.read_parquet(scope.TEMPLATE)
    out = tpl.select("MVT_ID_mvt").join(
        pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": out_q}).with_columns(
            pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0 and (out["TAXITIME_SEC_mvt"] >= 0).all()
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
