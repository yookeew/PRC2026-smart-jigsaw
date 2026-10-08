"""Scoping: can stand occupancy settle the echo hedge on LIRF NM-unmatched rows? (PROGRESS.md §102; analysis only --
the team rule (§87) forbids changing these rows, so nothing here feeds a submission.)

src/link/stand_link.py searches for the next arrival only after T - 12 min, so its L bound is <= 12 min and cannot
see a multi-hour row. Here, per LIRF departure with offset = T - SCHED >= 1 h, from arrivals at the same stand:
  pre    last arrival with in-block <= SCHED (time, type)
  btw    arrivals with in-block in (SCHED, T - 10 min] (count; first one's time and type)
Identity: aircraft type where the departure has one; else (all LIRF NM-unmatched rows: type is null) the airline
prefix of FLIGHT_mvt, compared with the arrival's flight-number prefix.
Configurations (label-free, inputs exist in 2026):
  LATE   no arrival in (SCHED, T-10m] other than one same-type arrival -> the aircraft probably arrived after SCHED:
         an echo label (= offset) would put the off-block before the aircraft's arrival.
  REOCC  the pre arrival is same-type and a different-type arrival takes the stand in (SCHED, T - 30 min]: our aircraft
         had left by then, so a normal-taxi label is impossible (label >= T - X unless the echo / stand is wrong).
  QUIET  pre arrival same-type, nothing in between (uninformative).
  OTHER  everything else (no stand, no pre arrival, mixed).
For 2025 (labels, all 12 months; p from the OOF folds + holdout, mixed 0.577 lgb + 0.428 cat): per configuration and
month, n, echo rate (|BLOCK - SCHED| < 30 s), share of labels <= 1 h ("normal"), share >= 0.5 offset ("long"), mean p.
For 2026: counts per configuration and mean p.

Run:  .venv/Scripts/python.exe -u tests/lirf_stand_bounds.py
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


def movements(src) -> tuple[pl.DataFrame, pl.DataFrame]:
    m = pl.scan_parquet(src).filter(((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
                                    | ((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))).collect()
    dep = m.filter(pl.col("PHASE_mvt") == "DEP").select(
        I64, stand="STAND_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null(),
        typ=pl.when(pl.col("AIRCRAFT_TYPE_mvt").is_not_null()).then(pl.col("AIRCRAFT_TYPE_mvt"))
        .otherwise(pl.lit("@") + pl.col("FLIGHT_mvt").str.slice(0, 3)),
        T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), S=pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s"),
        y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
        d=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        mo=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"))
    arr = m.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()).select(
        stand="STAND_mvt", t_ac="AIRCRAFT_TYPE_mvt", t_al=pl.lit("@") + pl.col("FLIGHT_mvt").str.slice(0, 3),
        A=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s")).sort("A")
    return dep, arr


def configure(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    d = dep.filter(((pl.col("T") - pl.col("S")) >= 3600) & pl.col("stand").is_not_null()).with_row_index("_i")
    # pre: last arrival at the stand with in-block <= SCHED
    pre = (d.sort("S").join_asof(arr.rename({"A": "A_pre", "t_ac": "pre_ac", "t_al": "pre_al"}), left_on="S", right_on="A_pre",
                                 by="stand", strategy="backward")
           .with_columns(t_pre=pl.when(pl.col("typ").str.starts_with("@")).then("pre_al").otherwise("pre_ac")))
    # arrivals in (SCHED, T - 600]: all pairs via a window join on the stand
    w = d.select("_i", "stand", "S", "T", "typ").join(arr, on="stand", how="inner").filter(
        (pl.col("A") > pl.col("S")) & (pl.col("A") <= pl.col("T") - 600)).sort("_i", "A").with_columns(
        atyp=pl.when(pl.col("typ").str.starts_with("@")).then("t_al").otherwise("t_ac"))
    btw = w.group_by("_i").agg(n_btw=pl.len(), A1=pl.col("A").first(), t1=pl.col("atyp").first(),
                               n_same=(pl.col("atyp") == pl.col("typ")).sum(), n_diff=(pl.col("atyp") != pl.col("typ")).sum(),
                               A_diff=pl.col("A").filter(pl.col("atyp") != pl.col("typ")).first())
    x = pre.join(btw, on="_i", how="left").with_columns(pl.col("n_btw").fill_null(0), pl.col("n_same").fill_null(0), pl.col("n_diff").fill_null(0))
    same_pre = pl.col("t_pre") == pl.col("typ")
    return x.with_columns(cfg=pl.when((pl.col("n_btw") == 1) & (pl.col("n_same") == 1)).then(pl.lit("LATE"))
                          .when(same_pre & (pl.col("n_diff") >= 1) & (pl.col("A_diff") <= pl.col("T") - 1800)).then(pl.lit("REOCC"))
                          .when(same_pre & (pl.col("n_btw") == 0)).then(pl.lit("QUIET"))
                          .otherwise(pl.lit("OTHER")),
                          off=(pl.col("T") - pl.col("S")).cast(pl.Float64),
                          late_by=(pl.col("A1") - pl.col("S")).cast(pl.Float64))


def p_2025() -> pl.DataFrame:
    rd = lambda path, c: pl.read_parquet(path).select(I64, pl.col("echo_prob").alias(c))  # noqa: E731
    lg = pl.concat([rd(OOF / "lgb_mixed_nb" / "*.parquet", "pl"), rd(EVAL / "lgb_mixed_nb_holdout_ev.parquet", "pl")])
    ct = pl.concat([rd(OOF / "cat_mixed" / "*.parquet", "pc"), rd(EVAL / "cat_mixed_holdout_ev.parquet", "pc")])
    return lg.join(ct, on="MVT_ID_mvt").select("MVT_ID_mvt", p=W_LGB * pl.col("pl") + W_CAT * pl.col("pc"))


def p_2026() -> pl.DataFrame:
    a = pl.read_parquet(EVAL / "lgb_direct_nolirf_nb_rank.parquet").select(I64, pl=pl.col("echo_prob"))
    b = pl.read_parquet(EVAL / "cat_mixed_rank.parquet").select(I64, pc=pl.col("echo_prob"))
    return a.join(b, on="MVT_ID_mvt").select("MVT_ID_mvt", p=W_LGB * pl.col("pl") + W_CAT * pl.col("pc"))


def summary(x: pl.DataFrame, labelled: bool, title: str) -> None:
    print(f"\n{title}")
    aggs = [pl.len().alias("n"), pl.col("p").mean().alias("mean_p")]
    if labelled:
        aggs += [(pl.col("d").abs() < 30).mean().alias("echo"), (pl.col("y") <= 3600).mean().alias("normal"),
                 (pl.col("y") >= 0.5 * pl.col("off")).mean().alias("long")]
    for nmu in (True, False):
        g = x.filter(pl.col("nmu") == nmu).group_by("cfg").agg(aggs).sort("cfg")
        print(f"  {'NM-unmatched' if nmu else 'NM-matched'}")
        for r in g.iter_rows(named=True):
            s = f"    {r['cfg']:6s} n {r['n']:6,}  mean p {r['mean_p']:.3f}" if r["mean_p"] is not None else f"    {r['cfg']:6s} n {r['n']:6,}"
            if labelled:
                s += f"  echo {r['echo']:.3f}  normal(<=1h) {r['normal']:.3f}  long(>=0.5 off) {r['long']:.3f}"
            print(s)


def main() -> None:
    dep25, arr25 = movements(str(ROOT / "data" / "raw" / "training_*.parquet"))
    dep26, arr26 = movements(str(ROOT / "data" / "ranking" / "ranking.parquet"))
    # 2026 arrivals at the stand include late-Dec 2025 arrivals for early-Jan departures: add the 2025 arrivals
    arr26 = pl.concat([arr25, arr26]).unique().sort("A")
    x25 = configure(dep25, arr25).join(p_2025(), on="MVT_ID_mvt", how="left")
    x26 = configure(dep26, arr26).join(p_2026(), on="MVT_ID_mvt", how="left")
    print(f"LIRF departures with offset >= 1 h and a stand: 2025 {x25.height:,} (p missing {x25['p'].null_count()}), 2026 {x26.height:,}")
    print(f"  stand null among LIRF NM-unmatched departures: 2025 {dep25.filter(pl.col('nmu'))['stand'].is_null().mean():.3f}, "
          f"2026 {dep26.filter(pl.col('nmu'))['stand'].is_null().mean():.3f}")
    summary(x25, True, "2025, all 12 months")
    summary(x26, False, "2026 ranking")

    print("\n2025 NM-unmatched, by month: n / echo rate / normal share, for LATE and REOCC")
    g = (x25.filter(pl.col("nmu") & pl.col("cfg").is_in(["LATE", "REOCC"])).group_by("cfg", "mo")
         .agg(n=pl.len(), echo=(pl.col("d").abs() < 30).mean(), normal=(pl.col("y") <= 3600).mean(), p=pl.col("p").mean()).sort("cfg", "mo"))
    for cfg in ("LATE", "REOCC"):
        s = g.filter(pl.col("cfg") == cfg)
        print(f"  {cfg}: " + "  ".join(f"{r['mo'][5:]}:{r['n']}/{r['echo']:.2f}/{r['normal']:.2f}" for r in s.iter_rows(named=True)))

    # LATE: echo vs how late the same-type arrival came, and the U bound check for echo rows
    late = x25.filter(pl.col("cfg") == "LATE")
    print("\n2025 LATE (both NM groups): echo rate by how long after SCHED the same-type arrival came")
    for lo, hi in ((0, 1800), (1800, 3600), (3600, 7200), (7200, 1e9)):
        s = late.filter((pl.col("late_by") > lo) & (pl.col("late_by") <= hi))
        if s.height:
            print(f"  {lo / 60:5.0f}-{min(hi, 99999) / 60:<5.0f} min: n {s.height:5,}  echo {(s['d'].abs() < 30).mean():.3f}  "
                  f"normal {(s['y'] <= 3600).mean():.3f}  mean p {s['p'].mean():.3f}  "
                  f"label <= T - A1 (U holds) {(s['y'] <= (s['T'] - s['A1']) + 60).mean():.3f}")


if __name__ == "__main__":
    main()
