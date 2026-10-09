"""v60 (reports/v60_eddf_opdi_demeaned_preregistration.md): v55 with EDDF rows replaced by v54 + the EDDF OPDI delta
(v52's production treatment - control) demeaned over each day's covered rows (days with covered share >= 0.2, covered =
OPDI ground event >= 60 s before T, uncovered rows 0, NM-line rows 0), then v55's day reconciliation (point solve,
0.75) re-solved on EDDF, LOBT window last. Every other row = v55.

Run:  python src/post/v60_build.py data/submissions/smart-jigsaw_v60.parquet
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
from post import pru_recon2 as R  # noqa: E402
import v57_build as B  # noqa: E402


def main(out_f):
    x = B.ranking_frame()
    of = pl.read_parquet(ROOT / "cache/opdi_feats/opdi_feats.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64), "o_lead_first")
    cd = pl.read_parquet(ROOT / "cache/v52/comb_delta_opdi.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64), dc="delta")
    nd = pl.read_parquet(ROOT / "cache/v52/nb_delta_opdi.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64), dn="delta")
    x = (x.join(of, on="MVT_ID_mvt", how="left").join(cd, on="MVT_ID_mvt", how="left").join(nd, on="MVT_ID_mvt", how="left")
         .with_columns(raw=pl.col("dc").fill_null(0) + pl.col("dn").fill_null(0), cov=(pl.col("o_lead_first").fill_null(0) >= 60),
                       tday=pl.col("MVT_TIME_UTC_mvt").dt.date(), nmline=pl.col("FLIGHT_ID_mvt").is_null() & (pl.col("ts") > 6 * 3600)))
    x = x.with_columns(share=pl.col("cov").mean().over("apt", "tday"))
    keep = (pl.col("apt") == "EDDF") & pl.col("cov") & (pl.col("share") >= 0.2)
    x = x.with_columns(dm=pl.when(keep).then(pl.col("raw") - pl.col("raw").filter(keep).mean().over("apt", "tday")).otherwise(0.0))
    x = x.with_columns(dm=pl.when(pl.col("nmline")).then(0.0).otherwise(pl.col("dm")))
    apt = x["apt"].to_numpy()
    ed = apt == "EDDF"
    dm = x["dm"].to_numpy()
    print(f"EDDF rows {ed.sum():,}; covered on kept days {(dm != 0).sum():,}; demeaned delta RMS {np.sqrt(np.mean(dm[ed] ** 2)):.2f}, "
          f"mean {dm[ed].mean():+.3f}; kept days {x.filter(keep)['tday'].n_unique()}")
    v54, v55 = x["v54"].to_numpy().astype(float), x["v55"].to_numpy().astype(np.int32)
    skip = tuple(a for a in set(apt) if a != "EDDF")
    # check: without the delta, the re-solve reproduces v55's EDDF rows
    q0 = B.finish(x, R.apply(x, R.solve_days(x, None, skip=skip), 0.75))
    assert (q0[ed] == v55[ed]).all(), "re-solve without delta does not reproduce v55 on EDDF"
    x2 = x.with_columns(p=pl.Series(np.maximum(v54 + dm, 0.0)))
    x2 = x2.with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
    x2 = x2.drop("D", "nD").join(B.DV.pub(2026), on=["apt", "day"], how="left")
    q1 = B.finish(x2, R.apply(x2, R.solve_days(x2, None, skip=skip), 0.75))
    q = np.where(ed, q1, v55).astype(np.int32)
    lam0 = {(d["apt"], str(x["day"][int(d["g"][0])])): d["lam"] for d in R.solve_days(x, None, skip=skip)}
    lam1 = {(d["apt"], str(x2["day"][int(d["g"][0])])): d["lam"] for d in R.solve_days(x2, None, skip=skip)}
    common = [k for k in lam0 if k in lam1]
    print(f"EDDF day lambda before / after the delta: {len(common)} days, mean change {np.mean([lam1[k] - lam0[k] for k in common]):+.2f} s, "
          f"max |change| {np.max([abs(lam1[k] - lam0[k]) for k in common]):.2f} s")
    dd = (q - v55)[ed]
    print(f"vs v55: EDDF rows changed {(dd != 0).sum():,}, RMS {np.sqrt(np.mean(dd ** 2)):.2f}, mean {dd.mean():+.2f}, max |d| {np.abs(dd).max()}; "
          f"whole file RMS {np.sqrt(np.mean((q - v55) ** 2)):.2f}; non-EDDF identical {bool((q[~ed] == v55[~ed]).all())}")
    print(f"rows < 120 s: {(q < 120).sum()} (v55 {(v55 < 120).sum()})")
    out = pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": q})
    tpl = pl.read_parquet(ROOT / "data/ranking/submitting.parquet")
    assert out.height == tpl.height == 344841 and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].cast(pl.Float64).to_list())
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(sys.argv[1])
