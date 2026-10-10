"""v49: three label-free 2026 fixes on v48 (reports/v49_bundle_preregistration.md).

  1  LSZH level: per month, shift NM-matched rows by o25 - o26 (clipped-mean offsets vs the anchor T - AOBT_3)
  2  EHAM storm-day NM-unmatched rows without ADS-B: add the corrector's bias seen on tracked storm rows
  3  LIRF NM-unmatched rows with off >= 87,000 s: P = min(P, off + e_T)

Run:  .venv/Scripts/python.exe -u src/post/v49_bundle.py IN.parquet OUT.parquet   (files in data/submissions)

PRC_SCOPE=final4m (src/scope.py, PROGRESS.md §132): final_ranking rows, files in data/submissions/final4m; fix 1 runs
for every month present; fix 2 reads the v4 detections (2026 days from cache/adsb_pushback_v4/final4m; v6 equals v4 on
adsb_tier / adsb_pushback_ts on all 245 days both hold) and uses the scope's corrector (ctxc1sel_ro) as its reference,
and is skipped when no storm day is found.
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import scope  # noqa: E402

RAW, SUB = ROOT / "data" / "raw", scope.out_dir(ROOT / "data" / "submissions")
RANKING, TEMPLATE = scope.RANKING, scope.TEMPLATE
ADSB = ROOT / "cache" / ("adsb_pushback_v4" if scope.FINAL else "adsb_pushback_v6")
CORR = "ctxc1sel_ro" if scope.FINAL else "ctxc1sel"
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)
CLIP = 300.0
STORM_MIN_NMU = 40
DOM_OFF = 87000


def frame(src) -> pl.DataFrame:
    ts = lambda c: pl.col(c).dt.epoch("s")  # noqa: E731
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(I64, apt="ADEP_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null(), T=ts("MVT_TIME_UTC_mvt"),
                    off=(ts("MVT_TIME_UTC_mvt") - ts("SCHED_TIME_UTC_mvt")).cast(pl.Float64),
                    a=(ts("MVT_TIME_UTC_mvt") - ts("AOBT_3_flt")).cast(pl.Float64),
                    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                    mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), day=pl.col("MVT_TIME_UTC_mvt").dt.date())
            .collect())


def cmean(s: pl.Series) -> float:
    return float(s.clip(-CLIP, CLIP).mean())


def main() -> None:
    src, dst = SUB / sys.argv[1], SUB / sys.argv[2]
    sub = pl.read_parquet(src)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    r = frame(str(RANKING)).join(sub.select(I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
    P0 = r["P"].to_numpy().copy()
    P = P0.copy()
    d25 = frame(str(RAW / "training_*.parquet"))

    # 1. LSZH level
    for m in sorted(r["mo"].unique().to_list()):
        k26 = (r["apt"] == "LSZH") & ~r["nmu"] & (r["mo"] == m)
        z26 = r.filter(k26 & r["a"].is_not_null())
        z25 = d25.filter((pl.col("apt") == "LSZH") & ~pl.col("nmu") & (pl.col("mo") == m) & pl.col("a").is_not_null())
        o26, o25 = cmean(z26["P"] - z26["a"]), cmean(z25["y"] - z25["a"])
        shift = o25 - o26
        kk = k26.to_numpy()
        P[kk] = np.maximum(P[kk] + shift, 0)
        print(f"fix 1 LSZH month {m}: o26 {o26:+.1f}, o25 {o25:+.1f} -> shift {shift:+.1f} s on {kk.sum()} rows")

    # 2. EHAM storm-day NM-unmatched rows without ADS-B
    ad = pl.concat([pl.read_parquet(f) for f in scope.day_files(ADSB, "day=2026-*.parquet") + scope.day_files(ADSB, "day=2025-01-*.parquet")],
                   how="diagonal_relaxed").filter(pl.col("airport") == "EHAM").select("MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts")
    good = pl.col("adsb_tier").is_in(["dwell", "pass"])
    t25 = (d25.filter((pl.col("apt") == "EHAM") & (pl.col("mo") == 1)).join(ad, on="MVT_ID_mvt").filter(good)
           .with_columns(at=pl.col("T") - pl.col("adsb_pushback_ts")).filter(pl.col("at").is_between(60, 7200)))
    lag = float((t25["y"] - t25["at"]).median())
    eh = r.filter(pl.col("apt") == "EHAM")
    storm = eh.filter(pl.col("nmu")).group_by("day").len().filter(pl.col("len") >= STORM_MIN_NMU)["day"].to_list()
    if not storm:
        print(f"fix 2 EHAM: no storm day (>= {STORM_MIN_NMU} NM-unmatched departures), skipped")
    else:
        C = pl.read_parquet(SUB / f"{CORR}.parquet").select(I64, C=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
        s = (eh.filter(pl.col("nmu") & pl.col("day").is_in(storm)).join(ad, on="MVT_ID_mvt", how="left").join(C, on="MVT_ID_mvt")
             .with_columns(at=pl.col("T") - pl.col("adsb_pushback_ts")))
        tr = s.filter(good & pl.col("at").is_between(60, 7200))
        dC = float((tr["at"] + lag - tr["C"]).median())
        un = s.filter(pl.col("adsb_tier").is_null())
        delta = dC - float((un["P"] - un["C"]).median())
        ids = set(un["MVT_ID_mvt"].to_list())
        k2 = np.array([i in ids for i in r["MVT_ID_mvt"].to_list()])
        P[k2] = np.maximum(P[k2] + delta, 0)
        print(f"fix 2 EHAM storm days {sorted(str(x) for x in storm)}; lag {lag:.0f}; tracked rows {tr.height}, corrector bias dC {dC:+.0f}; "
              f"untracked rows {un.height}, median(P-C) {float((un['P'] - un['C']).median()):+.0f} -> delta {delta:+.0f} s")

    # 3. dominated rows
    eT = ((r["T"] + 30) % 60 - 30).to_numpy().astype(float)
    k3 = ((r["apt"] == "LIRF") & r["nmu"] & (r["off"] >= DOM_OFF)).to_numpy()
    cap = r["off"].to_numpy() + eT
    before = P[k3].copy()
    P[k3] = np.minimum(P[k3], cap[k3])
    print(f"fix 3 dominated: {k3.sum()} rows in scope, {(P[k3] != before).sum()} moved, changes {np.round(P[k3] - before)}")

    dP = P - P0
    print(f"total: rows changed {(np.abs(dP) > 0.5).sum()}; sum dP^2 {np.sum(dP ** 2):.3g}; RMS over board {np.sqrt(np.mean(dP ** 2)):.2f} s")
    out = (sub.with_columns(_id=I64).join(r.select(pl.col("MVT_ID_mvt").alias("_id"), new=pl.Series(np.round(P))), on="_id", how="left")
           .with_columns(TAXITIME_SEC_mvt=pl.col("new").cast(dtype)).select(sub.columns))
    tpl = pl.read_parquet(TEMPLATE)
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list()) and float(out["TAXITIME_SEC_mvt"].min()) >= 0
    out.write_parquet(dst)
    print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
