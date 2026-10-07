"""Same-day airline arrival echo as a signal for LIRF NM-unmatched departures (PROGRESS.md §103; analysis only -- the team
rule (§87) keeps these rows fixed).

§63 found LIRF echo is a per-operator, per-day recording failure visible on the arrival side, but every use keyed it on
AIRCRAFT_OPERATOR_flt, null on NM-unmatched rows. Here the airline is the FLIGHT_mvt prefix (first 3 characters), present
on arrivals and on 96% of NM-unmatched departures. Arrivals keep BLOCK_TIME in 2026, so the signal is readable on 2026
days; no departure label is used to build it.

Per LIRF departure (offset = T - SCHED >= 1 h), same UTC day, same airline prefix, arrivals at LIRF:
  a_n / a_echo        all arrivals: count, share with |BLOCK - SCHED| < 30 s
  al_n / al_echo      arrivals > 30 min off schedule (where an echo is clearly a recording artefact)
2025 (all 12 months, p = OOF echo probability, 0.577 lgb + 0.428 cat):
  1  echo rate and mean p by al_echo band (NM-unmatched, and NM-matched for reference), month stability of the top band
  2  does it add to p? leave-one-month-out logistic regression of echo on logit(p) with / without the signal:
     out-of-month log-loss and Brier, and the Brier-implied SSE share on the offset (sum (p - echo)^2 offset^2)
2026: rows per band and mean p.

Run:  .venv/Scripts/python.exe -u tests/lirf_airline_day_echo.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EVAL, OOF = ROOT / "cache" / "eval", ROOT / "cache" / "oof"
W_LGB, W_CAT = 0.577, 0.428
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)
BANDS = [("no late arr", None, None), ("0", 0.0, 0.0), ("(0,.3]", 1e-9, 0.3), ("(.3,.6]", 0.3 + 1e-9, 0.6), ("(.6,1]", 0.6 + 1e-9, 1.0)]


def frame(src) -> pl.DataFrame:
    m = pl.scan_parquet(src).filter(((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
                                    | ((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))).with_columns(
        al=pl.col("FLIGHT_mvt").str.slice(0, 3), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        late=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() > 1800,
        echo=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() < 30).collect()
    arr = (m.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("al").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
           .group_by("al", "day").agg(a_n=pl.len(), a_echo=pl.col("echo").mean(),
                                      al_n=pl.col("late").sum(), al_echo=pl.col("echo").filter(pl.col("late")).mean()))
    dep = m.filter(pl.col("PHASE_mvt") == "DEP").select(
        I64, "al", "day", nmu=pl.col("FLIGHT_ID_mvt").is_null(), mo=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
        off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        echo=pl.col("echo"), has_blk=pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    return dep.filter(pl.col("off") >= 3600).join(arr, on=["al", "day"], how="left").with_columns(
        pl.col("a_n").fill_null(0), pl.col("al_n").fill_null(0))


def p_frame(paths_l, paths_c) -> pl.DataFrame:
    rd = lambda ps, c: pl.concat([pl.read_parquet(p).select(I64, pl.col("echo_prob").alias(c)) for p in ps])  # noqa: E731
    return rd(paths_l, "pl").join(rd(paths_c, "pc"), on="MVT_ID_mvt").select("MVT_ID_mvt", p=W_LGB * pl.col("pl") + W_CAT * pl.col("pc"))


def band_of(x: pl.DataFrame) -> np.ndarray:
    al_n, al_e = x["al_n"].to_numpy(), x["al_echo"].fill_null(np.nan).to_numpy()
    out = np.full(len(x), "no late arr", dtype=object)
    for name, lo, hi in BANDS[1:]:
        out[(al_n > 0) & (al_e >= lo) & (al_e <= hi)] = name
    return out


def main() -> None:
    p25 = p_frame(sorted((OOF / "lgb_mixed_nb").glob("*.parquet")) + [EVAL / "lgb_mixed_nb_holdout_ev.parquet"],
                  sorted((OOF / "cat_mixed").glob("*.parquet")) + [EVAL / "cat_mixed_holdout_ev.parquet"])
    p26 = p_frame([EVAL / "lgb_direct_nolirf_nb_rank.parquet"], [EVAL / "cat_mixed_rank.parquet"])
    x25 = frame(str(ROOT / "data" / "raw" / "training_*.parquet")).join(p25, on="MVT_ID_mvt", how="left")
    x26 = frame(str(ROOT / "data" / "ranking" / "ranking.parquet")).join(p26, on="MVT_ID_mvt", how="left")
    x25 = x25.with_columns(band=pl.Series(band_of(x25)))
    x26 = x26.with_columns(band=pl.Series(band_of(x26)))
    print(f"LIRF departures >= 1 h late: 2025 {x25.height:,} (NM-unmatched {x25['nmu'].sum():,}; p missing {x25['p'].null_count()}; "
          f"airline prefix missing {x25['al'].is_null().sum()}), 2026 {x26.height:,} (NM-unmatched {x26['nmu'].sum():,})")

    print("\n1. echo rate vs same-day same-airline late-arrival echo rate (al_echo)")
    for nmu in (True, False):
        print(f"  {'NM-unmatched' if nmu else 'NM-matched'}:  band: 2025 n / echo rate / mean p   |  2026 n / mean p")
        for name, _, _ in BANDS:
            a, b = x25.filter((pl.col("nmu") == nmu) & (pl.col("band") == name)), x26.filter((pl.col("nmu") == nmu) & (pl.col("band") == name))
            if a.height == 0 and b.height == 0:
                continue
            ea = f"{a['echo'].mean():.3f}" if a.height else "  -  "
            pa = f"{a['p'].mean():.3f}" if a.height else "  -  "
            pb = f"{b['p'].mean():.3f}" if b.height else "  -  "
            print(f"    {name:12s} {a.height:6,} / {ea} / {pa}   |  {b.height:5,} / {pb}")
    print("  NM-unmatched by month, top band (.6,1] and band 0: n / echo rate / mean p")
    for name in ("(.6,1]", "0"):
        g = (x25.filter(pl.col("nmu") & (pl.col("band") == name)).group_by("mo")
             .agg(n=pl.len(), e=pl.col("echo").mean(), p=pl.col("p").mean()).sort("mo"))
        print(f"    {name:7s} " + "  ".join(f"{r['mo'][5:]}:{r['n']}/{r['e']:.2f}/{r['p']:.2f}" for r in g.iter_rows(named=True)))

    # 2. leave-one-month-out logistic: echo ~ logit(p) [+ signal]
    from sklearn.linear_model import LogisticRegression
    d = x25.filter(pl.col("nmu") & pl.col("has_blk") & pl.col("p").is_not_null())
    y = d["echo"].to_numpy().astype(int)
    p = np.clip(d["p"].to_numpy(), 1e-4, 1 - 1e-4)
    lp = np.log(p / (1 - p))
    al_n, al_e = d["al_n"].to_numpy().astype(float), d["al_echo"].fill_null(np.nan).to_numpy()
    a_n, a_e = d["a_n"].to_numpy().astype(float), d["a_echo"].fill_null(np.nan).to_numpy()
    has = (al_n > 0).astype(float)
    sig = np.column_stack([has, np.nan_to_num(al_e) * has, np.log1p(al_n), (a_n > 0) * np.nan_to_num(a_e), np.log1p(a_n)])
    X0, X1 = lp[:, None], np.column_stack([lp, sig])
    mo, off2 = d["mo"].to_numpy(), np.clip(d["off"].to_numpy(), 0, 140000) ** 2
    q0, q1 = np.zeros(len(y)), np.zeros(len(y))
    for m in sorted(set(mo)):
        k = mo == m
        q0[k] = LogisticRegression(C=1.0, max_iter=1000).fit(X0[~k], y[~k]).predict_proba(X0[k])[:, 1]
        q1[k] = LogisticRegression(C=1.0, max_iter=1000).fit(X1[~k], y[~k]).predict_proba(X1[k])[:, 1]
    ll = lambda q: float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))  # noqa: E731
    br = lambda q: float(np.mean((q - y) ** 2))  # noqa: E731
    ws = lambda q: float(np.sum((q - y) ** 2 * off2))  # noqa: E731
    print(f"\n2. NM-unmatched, leave-one-month-out (n {len(y):,}, echo rate {y.mean():.3f}):")
    print(f"   raw p            log-loss {ll(p):.4f}  Brier {br(p):.4f}  offset-weighted {ws(p):.4g}")
    print(f"   recalibrated p   log-loss {ll(q0):.4f}  Brier {br(q0):.4f}  offset-weighted {ws(q0):.4g}")
    print(f"   p + airline-day  log-loss {ll(q1):.4f}  Brier {br(q1):.4f}  offset-weighted {ws(q1):.4g}")
    print("   per month offset-weighted (p + airline-day) / (recalibrated p):")
    print("   " + "  ".join(f"{m[5:]}:{np.sum(((q1 - y) ** 2 * off2)[mo == m]) / max(np.sum(((q0 - y) ** 2 * off2)[mo == m]), 1):.2f}" for m in sorted(set(mo))))
    m1 = LogisticRegression(C=1.0, max_iter=1000).fit(X1, y)
    print("   full-data coefficients [logit p, has late arr, late-arr echo, log1p late n, all-arr echo, log1p all n]:",
          np.round(m1.coef_[0], 3))


if __name__ == "__main__":
    main()
