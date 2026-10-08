"""Flight-number echo history for LIRF NM-unmatched departures (PROGRESS §111; analysis only -- the team rule (§87)
keeps these rows fixed). §103 used the airline prefix; this uses the full FLIGHT_mvt. Of 143 flight numbers with >= 3
late NM-unmatched rows in 2025, 69% are nearly pure (echo <= 0.1 or >= 0.9).

Per LIRF departure >= 1 h late (2025, all months; p = OOF echo probability as in tests/lirf_airline_day_echo.py), the
flight number's echo history from OTHER months only (late LIRF departures, NM-unmatched and matched counted separately):
  h_n, h_rate   NM-unmatched history: rows and smoothed echo rate (prior 0.49, weight 2)
  m_n, m_rate   NM-matched history
Leave-one-month-out logistic of echo on logit(p) [+ history], as §103: log-loss, Brier, offset-weighted squared error.
"far": history from months at least 3 away (2026 rows are 6-18 months from 2025).
2026: how many LIRF NM-unmatched late rows have a flight-number history, and the p shift the model would make.

Run:  .venv/Scripts/python.exe -u tests/lirf_flightno_echo.py [surgical]
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from lirf_airline_day_echo import EVAL, OOF, p_frame  # noqa: E402

I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)
PRIOR, W = 0.49, 2.0


def deps(src) -> pl.DataFrame:
    return (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")).select(
        I64, fn="FLIGHT_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null(), mi=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        has_blk=pl.col("BLOCK_TIME_UTC_mvt").is_not_null(),
        echo=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() < 30)
        .filter(pl.col("off") >= 3600).collect())


def history(target: pl.DataFrame, hist: pl.DataFrame, gap: int | None) -> np.ndarray:
    """[h_n, h_rate, m_n, m_rate] per target row from hist rows of the same flight number; gap=None -> all hist rows,
    gap=k -> only hist rows whose month differs by >= k (circular distance is not used: 2025 Jan and Dec are 11 apart)."""
    h = hist.filter(pl.col("fn").is_not_null() & pl.col("has_blk")).select("fn", "nmu", hm="mi", he=pl.col("echo").cast(pl.Float64))
    t = target.select("MVT_ID_mvt", "fn", "mi").with_row_index("_r")
    j = t.join(h, on="fn", how="inner")
    if gap is not None:
        j = j.filter((pl.col("mi") - pl.col("hm")).abs() >= gap)
    g = j.group_by("_r", "nmu").agg(n=pl.len(), e=pl.col("he").sum())
    out = np.zeros((t.height, 4))
    out[:, 1] = out[:, 3] = PRIOR
    for nmu, c in ((True, 0), (False, 2)):
        s = g.filter(pl.col("nmu") == nmu)
        r = s["_r"].to_numpy()
        out[r, c] = s["n"].to_numpy()
        out[r, c + 1] = (s["e"].to_numpy() + W * PRIOR) / (s["n"].to_numpy() + W)
    return out


def feats(lp, H):
    lo = lambda q: np.log(np.clip(q, .02, .98) / (1 - np.clip(q, .02, .98)))  # noqa: E731
    return np.column_stack([lp, lo(H[:, 1]) * (H[:, 0] > 0), np.log1p(H[:, 0]), lo(H[:, 3]) * (H[:, 2] > 0), np.log1p(H[:, 2])])


def main() -> None:
    p25 = p_frame(sorted((OOF / "lgb_mixed_nb").glob("*.parquet")) + [EVAL / "lgb_mixed_nb_holdout_ev.parquet"],
                  sorted((OOF / "cat_mixed").glob("*.parquet")) + [EVAL / "cat_mixed_holdout_ev.parquet"])
    d25 = deps(str(ROOT / "data" / "raw" / "training_*.parquet"))
    x = d25.join(p25, on="MVT_ID_mvt", how="left").filter(pl.col("nmu") & pl.col("has_blk") & pl.col("p").is_not_null())
    y = x["echo"].to_numpy().astype(int)
    p = np.clip(x["p"].to_numpy(), 1e-4, 1 - 1e-4)
    lp = np.log(p / (1 - p))
    mo = x["mi"].to_numpy()
    off2 = np.clip(x["off"].to_numpy(), 0, 140000) ** 2
    print(f"2025 LIRF NM-unmatched >= 1 h late with p: {len(y):,}, echo rate {y.mean():.3f}")
    ll = lambda q: float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))  # noqa: E731
    ws = lambda q, k=slice(None): float(np.sum(((q - y) ** 2 * off2)[k]))  # noqa: E731
    res = {}
    for tag, gap in (("LOMO", 1), ("far >= 3 months", 3)):
        q0, q1 = np.zeros(len(y)), np.zeros(len(y))
        Hs = np.zeros((len(y), 4))
        for m in sorted(set(mo)):
            k = mo == m
            hist = d25.filter(pl.col("mi") != m)
            Hk = history(x.filter(pl.col("mi") == m), hist, gap)
            Hs[k] = Hk
            # training rows' features also from out-of-their-month history (within the training months)
            tr = x.filter(pl.col("mi") != m)
            Htr = np.zeros((tr.height, 4))
            idx_tr = np.where(~k)[0]
            for m2 in sorted(set(mo[~k])):
                kk = mo[idx_tr] == m2
                Htr[kk] = history(tr.filter(pl.col("mi") == m2), d25.filter(~pl.col("mi").is_in([m, m2])), gap)
            q0[k] = LogisticRegression(C=1.0, max_iter=2000).fit(lp[~k, None], y[~k]).predict_proba(lp[k, None])[:, 1]
            q1[k] = LogisticRegression(C=1.0, max_iter=2000).fit(feats(lp[~k], Htr), y[~k]).predict_proba(feats(lp[k], Hk))[:, 1]
        cov = (Hs[:, 0] + Hs[:, 2]) > 0
        print(f"\n[{tag}] rows with any flight-number history: {cov.mean():.1%} (NM-unmatched history {np.mean(Hs[:, 0] > 0):.1%})")
        print(f"   raw p           log-loss {ll(p):.4f}  offset-weighted {ws(p):.4g}")
        print(f"   recalibrated p  log-loss {ll(q0):.4f}  offset-weighted {ws(q0):.4g}")
        print(f"   p + flight no.  log-loss {ll(q1):.4f}  offset-weighted {ws(q1):.4g}  (vs raw p {ws(q1) / ws(p) - 1:+.1%})")
        print("   per month offset-weighted (p + flight no.) / raw p: "
              + "  ".join(f"{m}:{ws(q1, mo == m) / max(ws(p, mo == m), 1):.2f}" for m in sorted(set(mo))))
        hb = Hs[:, 0] > 0
        for lo_, hi_ in ((0, .2), (.2, .8), (.8, 1.01)):
            kk = hb & (Hs[:, 1] >= lo_) & (Hs[:, 1] < hi_)
            if kk.sum():
                print(f"   NM-unmatched history rate [{lo_:.1f},{hi_:.1f}): n {kk.sum():4d}  echo {y[kk].mean():.3f}  mean p {p[kk].mean():.3f}  mean q1 {q1[kk].mean():.3f}")
        res[tag] = q1
    # 2026 reach
    d26 = deps(str(ROOT / "data" / "ranking" / "ranking.parquet")).filter(pl.col("nmu"))
    H26 = history(d26, d25, None)
    print(f"\n2026 LIRF NM-unmatched >= 1 h late: {d26.height}; with NM-unmatched flight-number history {np.mean(H26[:, 0] > 0):.1%}, "
          f"any history {np.mean((H26[:, 0] + H26[:, 2]) > 0):.1%}")
    hb = H26[:, 0] > 0
    for lo_, hi_ in ((0, .2), (.2, .8), (.8, 1.01)):
        kk = hb & (H26[:, 1] >= lo_) & (H26[:, 1] < hi_)
        print(f"   history rate [{lo_:.1f},{hi_:.1f}): {kk.sum()} rows, offset sum {d26.filter(pl.Series(kk))['off'].sum() / 3600:.0f} h")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def surgical() -> None:
    """Change p only where the flight number's NM-unmatched history (>= GAP months away) is pure: rate <= LO -> P_LO,
    rate >= HI -> P_HI, with at least NMIN history rows. Offset-weighted SE vs raw p, per band and month."""
    p25 = p_frame(sorted((OOF / "lgb_mixed_nb").glob("*.parquet")) + [EVAL / "lgb_mixed_nb_holdout_ev.parquet"],
                  sorted((OOF / "cat_mixed").glob("*.parquet")) + [EVAL / "cat_mixed_holdout_ev.parquet"])
    d25 = deps(str(ROOT / "data" / "raw" / "training_*.parquet"))
    x = d25.join(p25, on="MVT_ID_mvt", how="left").filter(pl.col("nmu") & pl.col("has_blk") & pl.col("p").is_not_null())
    y = x["echo"].to_numpy().astype(float)
    p = x["p"].to_numpy()
    mo = x["mi"].to_numpy()
    off = np.clip(x["off"].to_numpy(), 0, 140000)
    print("\n=== surgical rule (history >= GAP months away; raw p elsewhere)")
    for gap in (1, 3, 6):
        H = np.zeros((len(y), 4))
        for m in sorted(set(mo)):
            k = mo == m
            H[k] = history(x.filter(pl.col("mi") == m), d25.filter(pl.col("mi") != m), gap)
        for nmin in (1, 2, 3):
            for lo_, plo, hi_, phi in ((0.2, 0.07, 0.8, 0.97), (0.15, 0.10, 0.85, 0.95)):
                q = p.copy()
                klo = (H[:, 0] >= nmin) & (H[:, 1] <= lo_)
                khi = (H[:, 0] >= nmin) & (H[:, 1] >= hi_)
                q[klo], q[khi] = plo, phi
                se0, se1 = (p - y) ** 2 * off ** 2, (q - y) ** 2 * off ** 2
                mm = "  ".join(f"{m}:{se1[mo == m].sum() / max(se0[mo == m].sum(), 1):.2f}" for m in sorted(set(mo)))
                worse = sum(se1[mo == m].sum() > se0[mo == m].sum() * 1.001 for m in sorted(set(mo)))
                print(f"gap {gap} nmin {nmin} lo<={lo_}->{plo} hi>={hi_}->{phi}: rows lo {klo.sum()} (echo {y[klo].mean() if klo.any() else float('nan'):.3f}) "
                      f"hi {khi.sum()} (echo {y[khi].mean() if khi.any() else float('nan'):.3f}); offset-weighted SE {se1.sum() / se0.sum() - 1:+.1%} "
                      f"[lo part {(se1 - se0)[klo].sum() / se0.sum():+.1%}, hi part {(se1 - se0)[khi].sum() / se0.sum():+.1%}]; months worse {worse}/12")
                if gap == 3 and nmin == 1 and lo_ == 0.2:
                    print("   per month new/raw: " + mm)
                    bad = np.argsort(-(se1 - se0))[:5]
                    print("   largest per-row cost increases (offset h, echo, p -> q):",
                          [(round(off[i] / 3600, 1), int(y[i]), round(p[i], 2), round(q[i], 2)) for i in bad if se1[i] > se0[i]])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "surgical":
    surgical()
