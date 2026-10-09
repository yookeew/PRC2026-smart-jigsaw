"""v57 build (reports/v57_pru_recon2_preregistration.md): v54 + the daily reconciliation re-solved with V* at the
8 airports v55 shifted (not LIRF, not LSZH); every LIRF row = v56; LOBT window last.

eta for 2026 from the Jan + Jul 2025 holdout chain (cache/v50/chain_all.parquet), per airport, tracked / untracked
(tracked = adsb.lol pushback tier dwell / pass / fb_dwell in cache/adsb_pushback_v4). First checks that the engine's
R0 reproduces v55 from v54 exactly.

Run:  VSTAR_KIND=mult VSTAR_A=0.75 VSTAR_W=1 python src/post/v57_build.py OUT.parquet
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from post import pru_recon2 as R  # noqa: E402
import v57_recon2_dev as DV  # noqa: E402
import v57_recon2_gate as GT  # noqa: E402

SUB = ROOT / "data" / "submissions"
KIND = os.environ.get("VSTAR_KIND", "mult")
A = float(os.environ.get("VSTAR_A", "0.75"))
W = float(os.environ.get("VSTAR_W", "1"))


def sub(v):
    return pl.read_parquet(SUB / f"smart-jigsaw_{v}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64),
                                                                     pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(v))


def ranking_frame() -> pl.DataFrame:
    tpl = pl.read_parquet(ROOT / "data/ranking/submitting.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64))
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Float64))
    det = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_tier"]) for p in
                     sorted((ROOT / "cache/adsb_pushback_v4").glob("day=2026-*.parquet"))], how="diagonal").with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Float64))
    x = (tpl.join(rk, on="MVT_ID_mvt", how="left").join(sub("v54"), on="MVT_ID_mvt", how="left")
         .join(sub("v55"), on="MVT_ID_mvt", how="left").join(sub("v56"), on="MVT_ID_mvt", how="left")
         .join(det, on="MVT_ID_mvt", how="left"))
    assert x.height == 344841 and x["v54"].null_count() == 0 and x["ADEP_mvt"].null_count() == 0
    x = x.with_columns(p=pl.col("v54"), apt=pl.col("ADEP_mvt"), trk=pl.col("adsb_tier").is_in(list(R.TRACKED)).fill_null(False),
                       ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                       mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
    x = x.with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
    return x.join(DV.pub(2026), on=["apt", "day"], how="left")


def finish(x, q):
    lt = (x["MVT_TIME_UTC_mvt"] - x["LOBT_flt"]).dt.total_seconds().cast(pl.Float64).to_numpy()
    win = ~np.isnan(lt)
    q = q.copy()
    q[win] = np.clip(q[win], np.clip(lt[win] - 3606, 0, None), lt[win] + 3606)
    return np.clip(np.round(q), 0, None).astype(np.int32)


def main(out_f):
    x = ranking_frame()
    apt = x["apt"].to_numpy()
    # 1. engine check: R0 on v54 must reproduce v55
    q0 = finish(x, R.apply(x, R.solve_days(x, None), 0.75))
    v55 = x["v55"].to_numpy()
    print(f"engine R0 vs v55: max |diff| {np.abs(q0 - v55).max():.0f}, rows differing {(q0 != v55).sum()}")
    assert (q0 == v55).all(), "engine R0 does not reproduce v55"
    # 2. V* with eta from the holdout chain (both months)
    h = GT.holdout_frame()
    dists = DV.dists_from(h, split=True)
    days = R.solve_days(x, dists, KIND, W)
    qv = finish(x, R.apply(x, days, A))
    lirf = apt == "LIRF"
    q = np.where(lirf, x["v56"].to_numpy(), qv).astype(np.int32)
    out = pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": q})
    tpl = pl.read_parquet(ROOT / "data/ranking/submitting.parquet")
    assert out.height == tpl.height == 344841 and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].cast(pl.Float64).to_list())
    v56 = x["v56"].to_numpy()
    dd = q - v56
    print(f"V* = {KIND}, a {A}, w {W}: {len(days)} airport-days solved")
    print(f"vs v56: RMS {np.sqrt(np.mean(dd ** 2)):.2f}, mean {dd.mean():+.2f}, rows changed {(dd != 0).sum():,}, max |d| {np.abs(dd).max():.0f}")
    for a in sorted(set(apt)):
        k = apt == a
        print(f"  {a} vs v56 RMS {np.sqrt(np.mean(dd[k] ** 2)):7.2f} mean {dd[k].mean():+7.2f} | v55 shift mean {np.mean(v55[k] - x['v54'].to_numpy()[k]):+7.2f}"
              f" | V* shift mean {np.mean(q[k] - x['v54'].to_numpy()[k]):+7.2f}")
    lam = pl.DataFrame({"apt": [d["apt"] for d in days], "mo": [d["per"] for d in days], "lam": [d["lam"] for d in days]})
    l0 = R.solve_days(x, None)
    lam0 = pl.DataFrame({"apt": [d["apt"] for d in l0], "mo": [d["per"] for d in l0], "lam0": [d["lam"] for d in l0]})
    with pl.Config(tbl_rows=30, float_precision=1):
        print(lam.group_by("apt", "mo").agg(days=pl.len(), lam_med=pl.col("lam").median()).join(
            lam0.group_by("apt", "mo").agg(lam0_med=pl.col("lam0").median()), on=["apt", "mo"]).sort("apt", "mo"))
    rows_lt120 = int((q < 120).sum())
    print(f"rows < 120 s: {rows_lt120} (v56 {(v56 < 120).sum()}); LIRF identical to v56: {bool((q[lirf] == v56[lirf]).all())}; "
          f"LSZH identical to v54: {bool((q[apt == 'LSZH'] == x['v54'].to_numpy()[apt == 'LSZH']).all())}")
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(sys.argv[1])
