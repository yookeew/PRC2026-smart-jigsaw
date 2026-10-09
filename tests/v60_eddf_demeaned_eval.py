"""v60 gate (reports/v60_eddf_opdi_demeaned_preregistration.md): EDDF OPDI delta demeaned within airport-day over
covered rows, on the leave-one-month-out run (cache/v57/lomo_2025-{09..12}.parquet).

Run:  python tests/v60_eddf_demeaned_eval.py
"""
import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[1]


def demean(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """df with apt, day, cov, share, col -> + dm (demeaned over covered rows of kept days; 0 elsewhere)."""
    return df.with_columns(dm=pl.when(pl.col("cov") & (pl.col("share") >= 0.2))
                           .then(pl.col(col) - pl.col(col).filter(pl.col("cov")).mean().over("ADEP_mvt", "day")).otherwise(0.0))


def main():
    lo = pl.concat([pl.read_parquet(ROOT / f"cache/v57/lomo_2025-{m}.parquet") for m in ("09", "10", "11", "12")])
    of = pl.read_parquet(ROOT / "cache/opdi_feats/opdi_feats.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "o_lead_first")
    lo = (lo.join(of, on="MVT_ID_mvt", how="left").with_columns(cov=(pl.col("o_lead_first").fill_null(0) >= 60))
          .with_columns(share=pl.col("cov").mean().over("ADEP_mvt", "day"), raw=pl.col("trt") - pl.col("ctl")))
    P = demean(lo.filter((pl.col("ADEP_mvt") == "EDDF") & (pl.col("day") >= pl.date(2025, 9, 16)) & (pl.col("share") >= 0.2)), "raw")
    P = P.with_columns(dmp=pl.col("ctl") + pl.col("dm"))
    y = P["taxi"].to_numpy()
    body = (~P["nmu"] & (P["taxi"] <= 3600)).to_numpy()
    cov = P["cov"].to_numpy()
    r = lambda c, m: float(np.sqrt(np.mean((y[m] - P[c].to_numpy()[m]) ** 2)))  # noqa: E731
    allr = np.ones(len(y), bool)
    print(f"P: {P.height:,} EDDF rows on {P['day'].n_unique()} days, covered {cov.mean():.3f}")
    for lab, m in (("full", allr), ("body", body), ("body covered", body & cov), ("body uncovered", body & ~cov)):
        print(f"  {lab:15s} ctl {r('ctl', m):7.2f}  demeaned {r('dmp', m):7.2f} ({r('dmp', m) - r('ctl', m):+6.2f})  raw {r('trt', m):7.2f} ({r('trt', m) - r('ctl', m):+6.2f})")
    ym = P["ym"].to_numpy()
    bym = {m: round(r("dmp", body & (ym == m)) - r("ctl", body & (ym == m)), 2) for m in sorted(set(ym))}
    print("  body by month (demeaned):", bym)
    days = P["day"].to_numpy()
    ud, di = np.unique(days[body], return_inverse=True)
    e0, e1 = (y[body] - P["ctl"].to_numpy()[body]) ** 2, (y[body] - P["dmp"].to_numpy()[body]) ** 2
    s0, s1, n = np.bincount(di, e0), np.bincount(di, e1), np.bincount(di)
    rng = np.random.default_rng(0)
    bs = [np.sqrt(s1[s].sum() / n[s].sum()) - np.sqrt(s0[s].sum() / n[s].sum()) for s in (rng.integers(0, len(ud), len(ud)) for _ in range(2000))]
    lo95, hi95 = np.percentile(bs, [2.5, 97.5])
    print(f"  body day-cluster bootstrap 95% CI [{lo95:+.2f}, {hi95:+.2f}]")
    db, df_ = r("dmp", body) - r("ctl", body), r("dmp", allr) - r("ctl", allr)
    gate = [db <= -2.0, sum(v < 0 for v in bym.values()) >= 3, hi95 < 0, df_ <= 0]
    print(f"  GATE (body<=-2.0, >=3/4 months, CI<0, full<=0): {gate} -> {'PASS' if all(gate) else 'FAIL'}")
    dm_cov = float(np.mean((y[cov] - P['ctl'].to_numpy()[cov]) ** 2 - (y[cov] - P['dmp'].to_numpy()[cov]) ** 2))
    raw_day = P.filter(pl.col("cov")).group_by("day").agg(m=pl.col("raw").mean())["m"]
    print(f"  per covered row dMSE (ctl - demeaned) {dm_cov:.0f} s^2; 2025 daily mean raw delta on covered rows: "
          f"median {raw_day.median():+.1f} s, IQR [{raw_day.quantile(0.25):+.1f}, {raw_day.quantile(0.75):+.1f}]")
    # 2026: raw delta on covered EDDF rows (v52 production deltas), daily means; projection
    cd = pl.read_parquet(ROOT / "cache/v52/comb_delta_opdi.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", dc="delta")
    nd = pl.read_parquet(ROOT / "cache/v52/nb_delta_opdi.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), dn="delta")
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]).filter(
        pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64), day=pl.col("MVT_TIME_UTC_mvt").dt.date())
    x = (cd.join(nd, on="MVT_ID_mvt", how="left").join(rk, on="MVT_ID_mvt").join(of, on="MVT_ID_mvt", how="left")
         .filter(pl.col("ADEP_mvt") == "EDDF").with_columns(raw=pl.col("dc") + pl.col("dn").fill_null(0),
                                                           cov=(pl.col("o_lead_first").fill_null(0) >= 60))
         .with_columns(share=pl.col("cov").mean().over("day")))
    k26 = x.filter(pl.col("cov") & (pl.col("share") >= 0.2))
    d26 = k26.group_by("day").agg(m=pl.col("raw").mean())["m"]
    print(f"  2026 EDDF covered rows on kept days {k26.height:,} ({k26['day'].n_unique()} days); daily mean raw delta median "
          f"{d26.median():+.1f} s, IQR [{d26.quantile(0.25):+.1f}, {d26.quantile(0.75):+.1f}] (removed by the demeaning)")
    proj = np.sqrt(239.40 ** 2 - dm_cov * k26.height / 344841) - 239.40
    print(f"  2026 projection (2025 covered-row dMSE x 2026 covered rows): board {proj:+.3f} s")


if __name__ == "__main__":
    main()
