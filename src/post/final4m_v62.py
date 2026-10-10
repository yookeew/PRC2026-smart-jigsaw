"""v62's recipe on the four-month chain (PROGRESS.md §132; v62 = teammate build, §128-131).

Input: the four-month chain before its day-level stage (data/submissions/final4m/f4m_e.parquet: chain + identity deltas
+ LIRF hedges + v49 fixes), the counterpart of v54 (without v52). Per airport, as v62:
  LSZH        p + v52's LSZH-scoped OPDI delta (combiner + neighbour stage b, treatment - control; NM-line rows 0),
              then v58: the day-to-day variation of the daily-sum level (month mean removed), shrink from
              cache/v57/dsm_params.json
  EDDF        p + v60's EDDF OPDI delta demeaned over each day's covered rows (covered = OPDI ground event >= 60 s
              before T, days with covered share >= 0.2, NM-line rows 0), then S3
  EDDM EGLL EHAM LEBL LEMD LFPG   S3: delta = s_{airport, delay band} * lam_g, lam_g = expectation solve on the 2026
              sums with eta from the Jan + Jul 2025 holdout chain; slopes learned on all six 2025 periods
              (tests/v61_learned_dayshift.py, cached per-row 2025 lambdas), clipped to [0, 1]
  LTFM        v55's day shift (R0: point solve, 0.75)
  LIRF        unchanged
LOBT window last. The OPDI deltas come from src/post/v52_comb_delta.py / v52_nb_delta.py run on this chain
(PRC_LGB=ctxc1sel_ro, cache/v52/final4m/). Check printed first: the engine's R0 on p reproduces the four-month chain's
own day-level output (src/post/pru_daily_recon.py, f4m_full.parquet) at the eight shifted airports.

Run:  PRC_SCOPE=final4m python src/post/final4m_v62.py data/submissions/final4m/f4m_e.parquet \
          data/submissions/final4m/f4m_full.parquet data/submissions/final4m/f4m_v62.parquet
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
import v57_recon2_dev as DV  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402
from v61_build import s3_slopes  # noqa: E402

A7= [a for a in T.A8 if a != "LTFM"]
V52 = scope.out_dir(ROOT / "cache" / "v52")
OPDI = scope.out_dir(ROOT / "cache" / "opdi_feats") / "opdi_feats.parquet"
F64 = pl.col("MVT_ID_mvt").cast(pl.Float64)


def frame(p_file) -> pl.DataFrame:
    tpl = pl.read_parquet(scope.TEMPLATE).select(F64)
    rk = pl.read_parquet(scope.RANKING).filter(pl.col("PHASE_mvt") == "DEP").with_columns(F64)
    det = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "adsb_tier"])
                     for f in scope.day_files(ROOT / "cache" / "adsb_pushback_v4", "day=2026-*.parquet")], how="diagonal").with_columns(F64)
    of = pl.read_parquet(OPDI).select(F64, "o_lead_first")
    p = pl.read_parquet(p_file).select(F64, p0=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    x = (tpl.join(rk, on="MVT_ID_mvt", how="left").join(p, on="MVT_ID_mvt", how="left")
         .join(det, on="MVT_ID_mvt", how="left").join(of, on="MVT_ID_mvt", how="left"))
    assert x.height == tpl.height and x["p0"].null_count() == 0 and x["ADEP_mvt"].null_count() == 0
    return x.with_columns(
        apt=pl.col("ADEP_mvt"), trk=pl.col("adsb_tier").is_in(list(R.TRACKED)).fill_null(False),
        cov=pl.col("o_lead_first").fill_null(0) >= 60, nmu=pl.col("FLIGHT_ID_mvt").is_null(),
        ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
        tday=pl.col("MVT_TIME_UTC_mvt").dt.date()).with_columns(
        obs=pl.col("trk") | pl.col("cov"), nmline=pl.col("nmu") & (pl.col("ts") > 6 * 3600))


def with_p(x: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Set p, re-assign each row to its predicted off-block day and attach that day's published sums."""
    x = x.drop([c for c in ("p", "day", "D", "nD") if c in x.columns]).with_columns(p=pl.Series(np.maximum(p, 0.0)))
    x = x.with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
    return x.join(DV.pub(2026), on=["apt", "day"], how="left")


def finish(x, q):
    lt = (x["MVT_TIME_UTC_mvt"] - x["LOBT_flt"]).dt.total_seconds().cast(pl.Float64).to_numpy()
    win = ~np.isnan(lt)
    q = q.copy()
    q[win] = np.clip(q[win], np.clip(lt[win] - 3606, 0, None), lt[win] + 3606)
    return np.clip(np.round(q), 0, None).astype(np.int32)


def main(p_file, ref_file, out_f):
    assert scope.FINAL, "run with PRC_SCOPE=final4m"
    x = frame(p_file)
    apt, mo = x["apt"].to_numpy(), x["mo"].to_numpy()
    p0 = x["p0"].to_numpy()

    # 0. engine check: R0 on p0 reproduces the chain's own day-level stage at the eight airports
    x0 = with_p(x, p0)
    q0 = finish(x0, R.apply(x0, R.solve_days(x0, None), 0.75))
    ref = x.select("MVT_ID_mvt").join(pl.read_parquet(ref_file).select(F64, ref="TAXITIME_SEC_mvt"), on="MVT_ID_mvt", how="left")["ref"].to_numpy()
    in8 = np.isin(apt, T.A8)
    print(f"engine R0 vs {Path(ref_file).name} at the 8 airports: rows differing {(q0[in8] != ref[in8]).sum()}, "
          f"max |diff| {np.abs(q0[in8] - ref[in8]).max()}")

    # 1. OPDI deltas: v52 (LSZH, as built) and v60 (EDDF, demeaned per day over covered rows on kept days)
    cd = pl.read_parquet(V52 / "comb_delta_opdi.parquet").select(F64, dc="delta")
    nd = pl.read_parquet(V52 / "nb_delta_opdi.parquet").select(F64, dn="delta")
    x = x.join(cd, on="MVT_ID_mvt", how="left").join(nd, on="MVT_ID_mvt", how="left").with_columns(
        raw=pl.col("dc").fill_null(0) + pl.col("dn").fill_null(0))
    x = x.with_columns(share=pl.col("cov").mean().over("apt", "tday"))
    keep = (pl.col("apt") == "EDDF") & pl.col("cov") & (pl.col("share") >= 0.2)
    x = x.with_columns(dm=pl.when(keep).then(pl.col("raw") - pl.col("raw").filter(keep).mean().over("apt", "tday")).otherwise(0.0))
    # NM-line rows keep the input value (v52 froze them; v60 zeroes them after demeaning)
    x = x.with_columns(raw=pl.when(pl.col("nmline")).then(0.0).otherwise(pl.col("raw")),
                       dm=pl.when(pl.col("nmline")).then(0.0).otherwise(pl.col("dm")))
    raw, dm = x["raw"].to_numpy(), x["dm"].to_numpy()
    lz, ed = apt == "LSZH", apt == "EDDF"
    add = np.where(lz, raw, np.where(ed, dm, 0.0))
    for a, k in (("LSZH", lz), ("EDDF", ed)):
        for m in sorted(set(mo)):
            km = k & (mo == m)
            print(f"  OPDI delta {a} month {m}: rows {km.sum():,}, moved {(add[km] != 0).sum():,}, RMS {np.sqrt(np.mean(add[km] ** 2)):.1f}, "
                  f"mean {add[km].mean():+.2f}")
    x2 = with_p(x, p0 + add)
    p2 = x2["p"].to_numpy()

    # 2. S3 at seven airports (slopes as v61_build: all six 2025 periods)
    s, lam25 = s3_slopes()
    ho = T.frame25().filter(pl.col("period") == "P01")
    dists = T.dists_from(ho)
    lam = np.full(x2.height, np.nan)
    for d in M.day_table(x2, dists, set(A7)):
        lam[d["g"]] = d["lam"]
    b = L.band(x2["ts"].to_numpy() - p2)
    in7 = np.isin(apt, A7)
    k7 = in7 & np.isfinite(lam)
    cell = np.searchsorted(np.array(T.A8), apt[k7]) * 100 + b[k7] * 10
    sl = np.array([s.get(int(c), s.get(int(c) // 100 * 100, 0.0)) for c in cell])
    q = p2.copy()
    q[k7] = np.maximum(p2[k7] - sl * lam[k7], 0.0)
    print("S3 slopes (bands < 0 / 0-15 / 15-60 / >= 60 min):",
          {a: [round(s.get(i * 100 + kk * 10, np.nan), 2) for kk in range(4)] for i, a in enumerate(T.A8) if a in A7})
    print("2026 max |lam| vs 2025 max |lam|:", {a: (round(float(np.nanmax(np.abs(lam[apt == a]))), 0), round(lam25[a], 0)) for a in A7})

    # 3. LTFM: R0
    lt = apt == "LTFM"
    qr = R.apply(x2, R.solve_days(x2, None, skip=tuple(a for a in set(apt) if a != "LTFM")), 0.75)
    q[lt] = qr[lt]

    # 4. LSZH: v58's day variation on p + v52's delta
    prm = json.load(open(ROOT / "cache" / "v57" / "dsm_params.json"))
    days = M.day_table(x2, dists, {"LSZH"})
    delta, summ = M.shifts(x2, {"LSZH": days}, {"LSZH": []}, {"LSZH": prm["s2"]["LSZH"]}, level=False)
    q[lz] = np.maximum(p2[lz] - delta[lz], 0.0)
    print(f"LSZH: {len(days)} days solved; {summ.get('LSZH')}")

    # 5. LIRF unchanged; LOBT window last
    q[apt == "LIRF"] = p0[apt == "LIRF"]
    out_q = finish(x2, q)
    assert (out_q[apt == "LIRF"] == finish(x2, p0)[apt == "LIRF"]).all()

    dd = out_q - ref
    print("vs the chain's own day stage (f4m_full): per airport x month RMS / mean")
    for a in sorted(set(apt)):
        print(f"  {a}: " + "  ".join(f"m{m} {np.sqrt(np.mean(dd[(apt == a) & (mo == m)] ** 2.0)):6.1f} {dd[(apt == a) & (mo == m)].mean():+6.1f}"
                                    for m in sorted(set(mo))))
    print(f"all: RMS {np.sqrt(np.mean(dd ** 2.0)):.2f}, mean {dd.mean():+.2f}, rows changed {(dd != 0).sum():,}")
    tpl = pl.read_parquet(scope.TEMPLATE)
    out = tpl.select("MVT_ID_mvt").join(
        pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": out_q}).with_columns(
            pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0 and (out["TAXITIME_SEC_mvt"] >= 0).all()
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
