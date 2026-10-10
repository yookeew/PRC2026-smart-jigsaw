"""v57 part B development (reports/v57_pru_recon2_preregistration.md): reconciliation variants on 2025 OOF data.

  DEV=D1  10 OOF months, every day, p = cache/oof/ctxc1sel (corrector OOF)
  DEV=D2  115 OOF ADS-B days, p = control arm of tests/v57_eddf_lomo.py (cache/v57/lomo_*.parquet), tracked split

eta calibration leave-one-month-out. Prints RMSE change vs the unshifted p (full, trimmed, body), per month.
No holdout (Jan/Jul 2025) row is read.

Run:  DEV=D1 python tests/v57_recon2_dev.py
"""
import os, sys, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from post import pru_recon2 as R  # noqa: E402

FULL = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
SKIP = tuple(a for a in os.environ.get("SKIP", "LIRF,LSZH").split(",") if a)


def pub(year):
    return pl.read_csv(ROOT / "external-data" / f"all_pre_departure_delays_{year}.csv", infer_schema_length=0).select(
        apt="APT_ICAO", day=pl.col("FLT_DATE").str.to_date(), D=pl.col("DLY_ALL_PRE_2").cast(pl.Float64),
        nD=pl.col("FLT_DEP_IFR_2").cast(pl.Float64))


def frame(pred: pl.DataFrame) -> pl.DataFrame:
    """pred: MVT_ID_mvt, p[, trk] -> labelled 2025 rows with ts, y, nmu, apt, ym, mo, day (predicted off-block), D, nD."""
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt"]
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))],
                    how="vertical_relaxed").filter((pl.col("PHASE_mvt") == "DEP") & pl.col("TAXITIME_SEC_mvt").is_not_null())
    x = raw.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(pred, on="MVT_ID_mvt", how="inner").with_columns(
        p=pl.col("p").cast(pl.Float64).clip(0, None), apt=pl.col("ADEP_mvt"), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
        nmu=pl.col("FLIGHT_ID_mvt").is_null(), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
        ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
    x = x.with_columns(mo=pl.col("ym"), day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
    return x.join(pub(2025), on=["apt", "day"], how="left").filter(pl.col("apt") != "LIRF")


def dists_from(c: pl.DataFrame, split: bool) -> dict:
    """Per airport (untracked, tracked) EtaDist from labelled calibration rows c."""
    out = {}
    key = (c["apt"] + "_" + c["day"].cast(pl.Utf8)).to_numpy()
    eta = R.eta_sample(c["y"].to_numpy(), c["p"].to_numpy(), key)
    apt = c["apt"].to_numpy()
    trk = c["trk"].to_numpy().astype(bool) if (split and "trk" in c.columns) else np.zeros(c.height, bool)
    for a in np.unique(apt):
        u = R.EtaDist(eta[(apt == a) & ~trk])
        t = R.EtaDist(eta[(apt == a) & trk]) if ((apt == a) & trk).sum() >= 200 else u
        out[a] = (u, t)
    return out


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def report(lab, x, q, out):
    y, p = x["y"].to_numpy(), x["p"].to_numpy()
    nmu, ym = x["nmu"].to_numpy(), x["ym"].to_numpy()
    body = ~nmu & (y <= 3600)
    trim = y <= 18000
    f = lambda m: rmse(y[m] - q[m]) - rmse(y[m] - p[m])  # noqa: E731
    per_m = {m: round(f(body & (ym == m)), 3) for m in sorted(set(ym))}
    out.append({"variant": lab, "full": f(np.ones(len(y), bool)), "trim": f(trim), "body": f(body), **{f"b{k[5:]}": v for k, v in per_m.items()}})


def run(x: pl.DataFrame, split: bool):
    months = sorted(x["ym"].unique().to_list())
    qs = {}
    AG = [round(a, 1) for a in np.arange(0.5, 1.01, 0.1)]
    WG = (0.0, 0.25, 0.5, 0.75)
    keys = ["oracle", "R0_0.75", "R0_1.0"] + [f"R1_{a}" for a in AG] + ["R2_eb"] + [f"R4_{a}" for a in AG]
    keys += [f"R3_{w}_{a}" for w in WG for a in (0.6, 0.75)] + [f"R34_{w}_{a}" for w in WG for a in (0.6, 0.75)]
    for k in keys:
        qs[k] = x["p"].to_numpy().astype(float).copy()
    for m in months:
        xm_mask = (x["ym"] == m).to_numpy()
        xm = x.filter(pl.Series(xm_mask))
        idx = np.flatnonzero(xm_mask)
        dists = dists_from(x.filter(~pl.Series(xm_mask)), split)
        # oracle: remove the mean residual per solved airport-day
        d0 = R.solve_days(xm, None, skip=SKIP)
        q = xm["p"].to_numpy().astype(float).copy()
        yy = xm["y"].to_numpy()
        for d in d0:
            g = d["g"]; q[g] = np.maximum(q[g] - np.mean(q[g] - yy[g]), 0)
        qs["oracle"][idx] = q
        for a in (0.75, 1.0):
            qs[f"R0_{a}"][idx] = R.apply(xm, d0, a)
        d1 = R.solve_days(xm, dists, skip=SKIP)
        for a in AG:
            qs[f"R1_{a}"][idx] = R.apply(xm, d1, a)
        qs["R2_eb"][idx] = R.apply(xm, d1, "eb")
        d4 = R.solve_days(xm, dists, "mult", skip=SKIP)
        for a in AG:
            qs[f"R4_{a}"][idx] = R.apply(xm, d4, a)
        if split:
            for w in WG:
                d3 = R.solve_days(xm, dists, "trk", w, skip=SKIP)
                d34 = R.solve_days(xm, dists, "trkmult", w, skip=SKIP)
                for a in (0.6, 0.75):
                    qs[f"R3_{w}_{a}"][idx] = R.apply(xm, d3, a)
                    qs[f"R34_{w}_{a}"][idx] = R.apply(xm, d34, a)
        sh = R.shrink_table(d1).group_by("apt").agg(a_med=pl.col("a").median(), lam=pl.col("lam").median()).sort("apt")
        lam0 = pl.DataFrame({"apt": [d["apt"] for d in d0], "lam0": [d["lam"] for d in d0]}).group_by("apt").agg(pl.col("lam0").median())
        print(f"{m}: days solved R0 {len(d0)}, R1 {len(d1)}; median lam R0 vs R1 and EB a:",
              {r["apt"]: (round(r["lam0"], 1), round(r["lam"], 1), round(r["a_med"], 2)) for r in sh.join(lam0, on="apt").iter_rows(named=True)},
              flush=True)
    out = []
    for k in keys:
        if k.startswith("R3") and not split:
            continue
        report(k, x, qs[k], out)
    res = pl.DataFrame(out)
    mcols = [c for c in res.columns if c.startswith("b") and c != "body"]
    r0 = res.filter(pl.col("variant") == "R0_0.75").row(0, named=True)
    res = res.with_columns(months_beat_R0=pl.sum_horizontal([(pl.col(c) < r0[c]).cast(pl.Int32) for c in mcols]))
    with pl.Config(tbl_rows=60, tbl_cols=30, tbl_width_chars=250, float_precision=3):
        print(res)
    # per airport body for the main ones
    y, nmu, apt = x["y"].to_numpy(), x["nmu"].to_numpy(), x["apt"].to_numpy()
    body = ~nmu & (y <= 3600)
    p = x["p"].to_numpy()
    for k in ("oracle", "R0_0.75", "R1_0.6", "R4_0.6", "R4_0.7"):
        print(k, {a: round(rmse(y[body & (apt == a)] - qs[k][body & (apt == a)]) - rmse(y[body & (apt == a)] - p[body & (apt == a)]), 2)
                  for a in sorted(set(apt))})
    return qs


def main():
    dev = os.environ.get("DEV", "D1")
    if dev == "D1":
        pred = pl.concat([pl.read_parquet(p) for p in sorted((ROOT / "cache/oof/ctxc1sel").glob("fold=*.parquet"))]).select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), p=pl.col("pred"))
        x = frame(pred)
        print(f"D1: {x.height:,} rows, months {sorted(x['ym'].unique().to_list())}; skip {SKIP}")
        run(x, split=False)
    else:
        lo = pl.concat([pl.read_parquet(p) for p in sorted((ROOT / "cache/v57").glob("lomo_*.parquet"))], how="diagonal").select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), p=pl.col("ctl"))
        det = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_tier"]) for p in
                         sorted((ROOT / "cache/adsb_pushback_v4").glob("day=2025-*.parquet"))], how="diagonal")
        lo = lo.join(det.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)), on="MVT_ID_mvt", how="left").with_columns(
            trk=pl.col("adsb_tier").is_in(list(R.TRACKED)).fill_null(False)).drop("adsb_tier")
        x = frame(lo)
        print(f"D2: {x.height:,} rows on {x['day'].n_unique()} predicted days, tracked share {x['trk'].mean():.3f}; skip {SKIP}")
        run(x, split=True)


if __name__ == "__main__":
    main()
