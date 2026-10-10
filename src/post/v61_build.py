"""v61 (reports/v61_learned_dayshift_preregistration.md; S3 failed its body condition by 0.025 s and was built on the
team's explicit override, PROGRESS §131): v60 + S3's learned day shift at the 8 airports + v58's LSZH rows.

  8 airports: p = v54 + v60's EDDF demeaned OPDI delta (EDDF) / v54; lam_g = expectation solve on the 2026 sums
              (eta from the 2025 holdout chain); delta = s_{airport, delay band} * lam_g, slopes clipped to [0, 1],
              learned on all six 2025 periods (thin cells -> airport slope); LOBT window last.
  LSZH rows = v58; LIRF rows = v55.

Run:  python src/post/v61_build.py data/submissions/smart-jigsaw_v61.parquet
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
from post import pru_dsm as M  # noqa: E402
import v57_build as B  # noqa: E402
import v57_dsm as T  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402


def s3_slopes():
    x = T.frame25()
    lam, _ = L.lam_rows(x)
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt = x["apt"].to_numpy()
    b, td = L.band(x["ts"].to_numpy() - p), L.tod(x["hour"].to_numpy())
    in8 = np.isin(apt, T.A8)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    cell = np.where(in8, L.cells("S3", apt, b, td, lam0), -1)
    acell = np.where(in8, np.searchsorted(np.array(T.A8), apt), -1)
    dayk = (x["apt"] + "_" + x["day"].cast(pl.Utf8)).to_numpy()
    s, base = L.fit("S3", cell, acell, lam0, p - y, dayk, in8 & np.isfinite(lam) & (y <= 18000))
    lam25 = {a: float(np.nanmax(np.abs(lam[apt == a]))) for a in T.A8}
    return s, lam25


def main(out_f):
    s, lam25 = s3_slopes()
    print("S3 production slopes (bands < 0 / 0-15 / 15-60 / >= 60 min):",
          {a: [round(s.get(i * 100 + k * 10, np.nan), 2) for k in range(4)] for i, a in enumerate(T.A8)})
    x = B.ranking_frame()
    # EDDF demeaned OPDI delta, exactly as src/post/v60_build.py
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
    v54 = x["v54"].to_numpy().astype(float)
    x2 = x.with_columns(p=pl.Series(np.maximum(v54 + x["dm"].to_numpy(), 0.0)))
    x2 = x2.with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
    x2 = x2.drop("D", "nD").join(B.DV.pub(2026), on=["apt", "day"], how="left")
    x2 = x2.with_columns(nmu=pl.col("FLIGHT_ID_mvt").is_null(), hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(), obs=pl.col("trk"))
    ho = T.frame25().filter(pl.col("period") == "P01")
    days = M.day_table(x2, T.dists_from(ho), set(T.A8))
    lam = np.full(x2.height, np.nan)
    for d in days:
        lam[d["g"]] = d["lam"]
    apt = x2["apt"].to_numpy()
    p = x2["p"].to_numpy().astype(float)
    b = L.band(x2["ts"].to_numpy().astype(float) - p)
    in8 = np.isin(apt, T.A8)
    k = in8 & np.isfinite(lam)
    cell = np.searchsorted(np.array(T.A8), apt[k]) * 100 + b[k] * 10
    sl = np.array([s.get(int(c), s.get(int(c) // 100 * 100, 0.0)) for c in cell])
    q = p.copy()
    q[k] = np.maximum(p[k] - sl * lam[k], 0.0)
    q = B.finish(x2, q)
    v55 = x["v55"].to_numpy().astype(np.int32)
    v58 = pl.read_parquet(ROOT / "data/submissions/smart-jigsaw_v58.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64), v58="TAXITIME_SEC_mvt")
    v60 = pl.read_parquet(ROOT / "data/submissions/smart-jigsaw_v60.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64), v60="TAXITIME_SEC_mvt")
    xx = x.select("MVT_ID_mvt").join(v58, on="MVT_ID_mvt", how="left").join(v60, on="MVT_ID_mvt", how="left")
    v58a, v60a = xx["v58"].to_numpy().astype(np.int32), xx["v60"].to_numpy().astype(np.int32)
    out_q = np.where(in8, q, np.where(apt == "LSZH", v58a, v55)).astype(np.int32)
    assert (out_q[apt == "LIRF"] == v55[apt == "LIRF"]).all() and (out_q[apt == "LSZH"] == v58a[apt == "LSZH"]).all()
    print(f"days solved {len(days)}; 2026 max |lam| vs 2025 max |lam| per airport:",
          {a: (round(float(np.nanmax(np.abs(lam[apt == a]))), 0), round(lam25[a], 0)) for a in T.A8})
    print("per airport: mean shift taxi (v61 - base p) vs v55's (v55 - v54); RMS vs v60")
    for a in sorted(set(apt)):
        m = apt == a
        print(f"  {a}: v61 shift mean {np.mean(out_q[m] - p[m]):+7.2f} | v55 shift mean {np.mean(v55[m] - v54[m]):+7.2f} | "
              f"vs v60 RMS {np.sqrt(np.mean((out_q[m] - v60a[m]) ** 2.0)):7.2f} mean {np.mean(out_q[m] - v60a[m]):+7.2f}")
    dd = out_q - v60a
    print(f"vs v60: rows changed {(dd != 0).sum():,}, RMS {np.sqrt(np.mean(dd ** 2.0)):.2f}, mean {dd.mean():+.2f}, max |d| {np.abs(dd).max()}")
    print(f"rows < 120 s: {(out_q < 120).sum()} (v60 {(v60a < 120).sum()})")
    out = pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": out_q})
    tpl = pl.read_parquet(ROOT / "data/ranking/submitting.parquet")
    assert out.height == tpl.height == 344841 and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].cast(pl.Float64).to_list())
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(sys.argv[1])
