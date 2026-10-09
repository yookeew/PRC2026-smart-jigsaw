"""Day-level reconciliation with EUROCONTROL's daily all-causes pre-departure delay (PROGRESS §125).

Published per airport x UTC off-block date: D = sum over departures of max(0, AOBT - SOBT) [min, each flight rounded to the
minute]; FLT_DEP_IFR_2 = departures. 2025: exact at EDDM/EGLL/LEBL/LEMD/LIRF/LTFM, within minutes at EDDF/LSZH, ~35-40 min/day
noise at EHAM/LFPG (5 fewer / 2 more flights per day published). Admin-cleared EUROCONTROL data ("no change. still ok",
2026-10-09). Our d_hat = (T - SCHED) - p; rows are assigned to a day by the predicted off-block T - p.

  non-LIRF airports: per airport-day solve lambda with sum max(0, d_hat + lambda) = D; apply A_DAY * lambda (shift of d).
  LIRF: hedged rows (45 min <= T - S < 12 h, p strictly between TAU + 60 and T - S - 60) are two-point mixtures
        echo (taxi = T - S, adds 0 to D) / real (taxi = N, adds T - S - N). Prior pi from an isotonic map of
        (p - TAU)/(T - S - TAU) (+10 x NM-unmatched flag) fitted on the Jan+Jul 2025 holdout chain; N = (p - pi*E)/(1 - pi).
        Posterior from the day's residual (non-hedged rows' noise sd x NOISE_X, bias); q = post*E + (1 - post)*N.
        Then A_LIRF * the day shift on LIRF's non-hedged rows.
  LOBT window last (production).

Modes:
  .venv/Scripts/python.exe src/post/pru_daily_recon.py BASE.parquet OUT.parquet HOLDOUT_LIRF_CHAIN.parquet
  .venv/Scripts/python.exe src/post/pru_daily_recon.py --holdout HOLDOUT_NONLIRF_CHAIN.parquet HOLDOUT_LIRF_CHAIN.parquet
      (replay on Jan+Jul 2025 with the 2025 file; LIRF calibration cross-month; prints the gate metrics)
"""
import os, sys, glob, itertools
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "external-data"
TAU = 1100.0
N_MIN = float(os.environ.get("PRU_N_MIN", "650"))
A_DAY = float(os.environ.get("PRU_A_DAY", "0.75"))
A_LIRF = float(os.environ.get("PRU_A_LIRF", "0.5"))
NOISE_X = float(os.environ.get("PRU_NOISE_X", "1.5"))
DO_LIRF = os.environ.get("PRU_LIRF", "1") == "1"          # 0: LIRF rows untouched
DO_OTHER = os.environ.get("PRU_OTHER", "1") == "1"        # 0: non-LIRF rows untouched
# airports left untouched. LSZH by default: its 2026 base carries 2026-only level changes the holdout never saw (v49 LSZH fix
# -39/-28 s, teammate OPDI v52); the daily sum would undo v49's fix while the anchor and ADS-B (RMS) witnesses say v54 was right
SKIP = set(a for a in os.environ.get("PRU_SKIP", "LSZH").split(",") if a)


def pub(year):
    return pl.read_csv(EXT / f"all_pre_departure_delays_{year}.csv", infer_schema_length=0).select(
        apt="APT_ICAO", day=pl.col("FLT_DATE").str.to_date(), D=pl.col("DLY_ALL_PRE_2").cast(pl.Float64),
        nD=pl.col("FLT_DEP_IFR_2").cast(pl.Float64))


def frame(df, year):
    """df: departures with MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt, ADEP_mvt, FLIGHT_ID_mvt, p -> + ts, apt, nmu, mo, day, D, nD."""
    return (df.with_columns(ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                            apt=pl.col("ADEP_mvt"), nmu=pl.col("FLIGHT_ID_mvt").is_null(), mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
            .with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
            .join(pub(year), on=["apt", "day"], how="left"))


def hedged(p, ts):
    return (ts >= 2700) & (ts < 12 * 3600) & (p > TAU + 60) & (p < ts - 60)


def ratio(p, ts):
    return np.clip((p - TAU) / np.maximum(ts - TAU, 1.0), 0, 1)


def calibrate(h):
    """LIRF calibration from a labelled frame h (LIRF rows of a holdout chain): isotonic prior map, noise sd, bias."""
    y, p, ts, nmu = h["y"].to_numpy(), h["p"].to_numpy(), h["ts"].to_numpy(), h["nmu"].to_numpy()
    H = hedged(p, ts)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.998).fit(ratio(p, ts)[H] + 10 * nmu[H],
                                                                                 (np.abs(ts - y) <= 120)[H])
    key = h["day"].cast(pl.Utf8).to_numpy()
    e = np.maximum(0, ts - y) - np.maximum(0, ts - p)
    eN = np.array([e[(key == k) & ~H].sum() / 60 for k in sorted(set(key))])
    return iso, NOISE_X * eN.std(), eN.mean()


def lam_for(m, dhv, w, Dk):
    f = lambda lam: np.maximum(0.0, dhv[m] + lam * w).sum() / 60.0 - Dk  # noqa: E731
    if f(-5e4) * f(5e4) > 0:
        return np.nan
    return brentq(f, -5e4, 5e4, xtol=0.01)


def reconcile(x, cal):
    """x: frame(...) rows; cal: calibrate(...) or a callable month -> calibration. Returns new predictions and diagnostics."""
    p, ts = x["p"].to_numpy().astype(float), x["ts"].to_numpy()
    apt, nmu, mo = x["apt"].to_numpy(), x["nmu"].to_numpy(), x["mo"].to_numpy()
    Dv = x["D"].to_numpy()
    dh = ts - p
    key = (x["apt"] + "_" + x["day"].cast(pl.Utf8)).to_numpy()
    _, inv = np.unique(key, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    groups = np.split(order, bounds)
    # use only airport-days whose population matches the published one (our rows / FLT_DEP_IFR_2 within +-5%): drops days
    # outside the scored months that a few rows fall into by predicted off-block (e.g. 30 June: 1-2 of our rows vs the
    # whole day's published sum) and badly incomplete days
    nDv = x["nD"].to_numpy()
    cnt = np.bincount(inv)[inv]
    Dv = np.where(np.isfinite(nDv) & (np.abs(cnt / np.where(nDv > 0, nDv, np.nan) - 1) <= 0.05), Dv, np.nan)
    q = p.copy()
    H = (apt == "LIRF") & hedged(p, ts)
    post = np.full(len(p), np.nan)
    lams = []
    for m in groups:
        Dk = Dv[m[0]]
        if not np.isfinite(Dk):
            continue
        if apt[m[0]] in SKIP:
            continue
        if apt[m[0]] != "LIRF":
            if not DO_OTHER:
                continue
            lam = lam_for(m, dh, np.ones(len(m)), Dk)
            if np.isfinite(lam):
                q[m] = np.maximum(ts[m] - (dh[m] + A_DAY * lam), 0)
                lams.append((apt[m[0]], int(mo[m[0]]), lam))
            continue
        if not DO_LIRF:
            continue
        iso, sig, bias = cal(int(mo[m[0]])) if callable(cal) else cal
        hm = m[H[m]]
        if len(hm):
            pi = iso.predict(ratio(p[hm], ts[hm]) + 10 * nmu[hm]).clip(0.002, 0.998)
            N = (p[hm] - pi * ts[hm]) / (1 - pi)
            # physical floor for the real branch (p5 of LIRF non-echo taxi on >= 45 min delays, 2025); where it binds, re-derive
            # pi so that the prior mixture still equals p (a row only moves when the day sum says so)
            fl = N < N_MIN
            N = np.where(fl, N_MIN, N)
            pi = np.where(fl, (p[hm] - N_MIN) / np.maximum(ts[hm] - N_MIN, 1.0), pi).clip(0.002, 0.998)
            c = np.maximum(ts[hm] - N, 0) / 60
            R = Dk - np.maximum(0, dh[m[~H[m]]]).sum() / 60 - bias
            if len(hm) <= 16:
                cfg = np.array(list(itertools.product([0, 1], repeat=len(hm))))
                lp = (cfg * np.log(pi) + (1 - cfg) * np.log(1 - pi)).sum(1) - 0.5 * ((R - ((1 - cfg) * c).sum(1)) / sig) ** 2
                w = np.exp(lp - lp.max()); w /= w.sum()
                ps = (cfg * w[:, None]).sum(0)
            else:
                mu = ((1 - pi) * c).sum(); var = sig ** 2 + (pi * (1 - pi) * c ** 2).sum()
                mu_o = mu - (1 - pi) * c; var_o = var - pi * (1 - pi) * c ** 2
                l1 = np.log(pi) - 0.5 * (R - mu_o) ** 2 / var_o
                l0 = np.log(1 - pi) - 0.5 * (R - mu_o - c) ** 2 / var_o
                ps = 1 / (1 + np.exp(l0 - l1))
            post[hm] = ps
            q[hm] = ps * ts[hm] + (1 - ps) * N
        dh2 = ts - q
        wv = (~H[m]).astype(float)
        lam = lam_for(m, dh2, wv, Dk)
        if np.isfinite(lam):
            q[m] = np.maximum(ts[m] - (dh2[m] + A_LIRF * lam * wv), 0)
            lams.append(("LIRF", int(mo[m[0]]), lam))
    return q, H, post, pl.DataFrame(lams, schema=["apt", "mo", "lam"], orient="row")


def holdout_frame(nonlirf_f, lirf_f):
    files = [f for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "training_*.parquet"))) if "2025-01-01" in f or "2025-07-01" in f]
    raw = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").filter(
        (pl.col("PHASE_mvt") == "DEP") & pl.col("TAXITIME_SEC_mvt").is_not_null()).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    a = pl.read_parquet(nonlirf_f).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pa=pl.col("pred").cast(pl.Float64))
    b = pl.read_parquet(lirf_f).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pb=pl.col("v48").cast(pl.Float64))
    h = (raw.join(a, on="MVT_ID_mvt").join(b, on="MVT_ID_mvt")
         .with_columns(p=pl.when(pl.col("ADEP_mvt") == "LIRF").then(pl.col("pb")).otherwise(pl.col("pa")),
                       y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)))
    return frame(h, 2025)


def main_holdout(nonlirf_f, lirf_f):
    h = holdout_frame(nonlirf_f, lirf_f)
    lirf = h.filter(pl.col("apt") == "LIRF")
    cals = {m: calibrate(lirf.filter(pl.col("mo") != m)) for m in (1, 7)}      # cross-month
    q, H, post, lams = reconcile(h, lambda m: cals[m])
    y, p = h["y"].to_numpy(), h["p"].to_numpy()
    apt, mo, nmu = h["apt"].to_numpy(), h["mo"].to_numpy(), h["nmu"].to_numpy()
    r0, r1 = y - p, y - q
    body = ~nmu & (y <= 3600)
    f = lambda m: np.sqrt(np.mean(r1[m] ** 2)) - np.sqrt(np.mean(r0[m] ** 2))  # noqa: E731
    print(f"holdout rows {len(y):,}; base full {np.sqrt(np.mean(r0 ** 2)):.3f}")
    print("  ".join(f"{lab} {f(m):+.3f}" for lab, m in (("full", np.ones(len(y), bool)), ("body", body), ("bodyJan", body & (mo == 1)),
          ("bodyJul", body & (mo == 7)), ("fullJan", mo == 1), ("fullJul", mo == 7))))
    print("  per airport:", {a: round(f(apt == a), 2) for a in sorted(set(apt))})
    nb = 200
    days = (h["apt"] + "_" + h["day"].cast(pl.Utf8)).to_numpy()
    ud, di = np.unique(days, return_inverse=True)
    d_sse = np.bincount(di, (r1 ** 2 - r0 ** 2)); n_day = np.bincount(di)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(nb):
        s = rng.integers(0, len(ud), len(ud))
        bs.append(np.sqrt((np.sum(r0 ** 2) / len(y) * n_day[s].sum() + d_sse[s].sum()) / n_day[s].sum()) - np.sqrt(np.sum(r0 ** 2) / len(y)))
    print(f"  full delta, airport-day bootstrap: 95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], "
          f"P(worse) {np.mean(np.array(bs) > 0):.3f}")
    return h, q


def main_prod(base_f, out_f, ho_lirf_f, nonlirf_f):
    h = holdout_frame(nonlirf_f, ho_lirf_f)
    cal = calibrate(h.filter(pl.col("apt") == "LIRF"))
    print(f"LIRF calibration (Jan+Jul 2025 holdout): noise sd {cal[1]:.0f} min (x{NOISE_X}), bias {cal[2]:+.0f} min/day")
    tpl = pl.read_parquet(ROOT / "data" / "ranking" / "submitting.parquet")
    b = pl.read_parquet(base_f).select(pl.col("MVT_ID_mvt").cast(pl.Float64), p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    rk = pl.read_parquet(ROOT / "data" / "ranking" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Float64))
    x = tpl.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(rk, on="MVT_ID_mvt", how="left").join(b, on="MVT_ID_mvt", how="left")
    assert x["p"].null_count() == 0 and x["ADEP_mvt"].null_count() == 0
    x = frame(x, 2026)
    nD = x.group_by("apt", "day").agg(n=pl.len(), nD=pl.col("nD").first()).filter(pl.col("nD").is_not_null())
    print(f"airport-days {nD.height}; our rows / published flights median {(nD['n'] / nD['nD']).median():.3f}")
    q, H, post, lams = reconcile(x, cal)
    p = x["p"].to_numpy()
    lt = (x["MVT_TIME_UTC_mvt"] - x["LOBT_flt"]).dt.total_seconds().cast(pl.Float64).to_numpy()
    win = ~np.isnan(lt)
    q[win] = np.clip(q[win], np.clip(lt[win] - 3606, 0, None), lt[win] + 3606)
    q = np.clip(np.round(q), 0, None).astype(np.int32)
    out = pl.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"], "TAXITIME_SEC_mvt": q})
    assert out.height == 344841 and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
    dd = q - p
    apt = x["apt"].to_numpy()
    print(f"vs base: RMS {np.sqrt(np.mean(dd ** 2)):.2f}, mean {dd.mean():+.2f}, rows changed {(np.abs(dd) >= 1).sum():,}, max |d| {np.abs(dd).max():.0f}")
    print(lams.group_by("apt", "mo").agg(days=pl.len(), lam_med=pl.col("lam").median(), lam_p10=pl.col("lam").quantile(0.1),
                                          lam_p90=pl.col("lam").quantile(0.9)).sort("apt", "mo"))
    for a in sorted(set(apt)):
        k = apt == a
        print(f"  {a} RMS {np.sqrt(np.mean(dd[k] ** 2)):7.2f} mean {dd[k].mean():+7.2f}")
    if H.any():
        print(f"LIRF hedged rows {H.sum()}: posterior mean {np.nanmean(post[H]):.3f}; >= 0.8: {(post[H] >= 0.8).sum()}, <= 0.05: {(post[H] <= 0.05).sum()}")
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    if sys.argv[1] == "--holdout":
        main_holdout(sys.argv[2], sys.argv[3])
    else:
        main_prod(*sys.argv[1:5])
