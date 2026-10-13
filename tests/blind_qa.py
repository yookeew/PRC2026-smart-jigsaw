"""Label-free QA of a blind-board file (four months: Jan + Feb + Jun + Jul 2026), PROGRESS.md §162.

The final phase ranks on the four months, and a team whose Feb / Jun rank is far from its four-month rank may be looked at.
So Feb / Jun must carry the same stages as Jan / Jul. This prints, per airport x month:
  - the mean prediction vs the 2025 same-month mean label (Feb / Jun offsets should look like Jan / Jul's);
  - NM-unmatched rows: count and mean prediction; LIRF NM-unmatched >= 1 h: share at the 600 s floor, share near the echo;
  - LOBT-window violations on NM-matched rows (|T - pred - LOBT| > 3,606 s);
and checks IDs / nulls / negatives and that Jan / Jul equal the live file.

Run:  .venv/Scripts/python.exe tests/blind_qa.py BLIND.parquet [LIVE.parquet]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW, RK = ROOT / "data" / "raw", ROOT / "data" / "ranking"
F64 = pl.col("MVT_ID_mvt").cast(pl.Float64)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)


def deps(src):
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(F64, apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), nmu=pl.col("FLIGHT_ID_mvt").is_null(),
                    off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                    tl=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64),
                    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).collect())


def main(blind_f, live_f):
    tpl = pl.read_parquet(RK / "final_submitting.parquet").select(F64)
    b = pl.read_parquet(blind_f).select(F64, p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    assert b.height == tpl.height and set(b["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list()), "IDs differ from final_submitting"
    print(f"{blind_f}: {b.height:,} rows, nulls {b['p'].null_count()}, negatives {(b['p'] < 0).sum()}, min {b['p'].min():.0f}, max {b['p'].max():.0f}")
    r = deps(str(RK / "final_ranking.parquet")).join(b, on="MVT_ID_mvt", how="inner")
    if live_f:
        lv = pl.read_parquet(live_f).select(F64, live=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
        j = r.join(lv, on="MVT_ID_mvt", how="inner")
        print(f"Jan / Jul rows equal to {Path(live_f).name}: {(j['p'] == j['live']).sum():,} of {j.height:,} "
              f"(blind Jan / Jul rows {r.filter(pl.col('mo').is_in([1, 7])).height:,})")
    t = deps(str(RAW / "training_*.parquet")).filter(pl.col("mo").is_in([1, 2, 6, 7]) & (pl.col("y") <= 5 * 3600))
    lab = t.group_by("apt", "mo").agg(y25=pl.col("y").mean())
    tr = r.filter(pl.col("p") <= 5 * 3600)
    lvl = (tr.group_by("apt", "mo").agg(p26=pl.col("p").mean()).join(lab, on=["apt", "mo"])
           .with_columns(d=(pl.col("p26") - pl.col("y25")).round(1)).pivot(on="mo", index="apt", values="d").sort("apt"))
    print("mean prediction 2026 - mean label 2025, same month, rows <= 5 h (s):")
    print(lvl.select("apt", *[c for c in ("1", "2", "6", "7") if c in lvl.columns]))
    nm = (r.filter(pl.col("nmu")).group_by("apt", "mo").agg(n=pl.len(), p=pl.col("p").mean().round(0))
          .pivot(on="mo", index="apt", values=["n", "p"]).sort("apt"))
    print("NM-unmatched departures: count and mean prediction by month:")
    print(nm)
    li = r.filter((pl.col("apt") == "LIRF") & pl.col("nmu") & (pl.col("off") >= 3600))
    print("LIRF NM-unmatched >= 1 h late, by month (share at the 600 s floor, share within 5 % of the echo, mean p / off):")
    print(li.group_by("mo").agg(n=pl.len(), floor600=((pl.col("p") - 600).abs() < 1).mean().round(3),
                                near_echo=((pl.col("p") - pl.col("off")).abs() < 0.05 * pl.col("off")).mean().round(3),
                                p_over_off=(pl.col("p") / pl.col("off")).mean().round(3)).sort("mo"))
    w = r.filter(~pl.col("nmu") & pl.col("tl").is_not_null())
    viol = (w["p"] < w["tl"] - 3606 - 1) | (w["p"] > w["tl"] + 3606 + 1)
    print("LOBT-window violations (NM-matched):", w.with_columns(v=viol).group_by("mo").agg(pl.col("v").sum()).sort("mo").to_dicts())
    q = r.group_by("mo").agg(*[pl.col("p").quantile(x).alias(f"q{int(x * 100)}") for x in (0.1, 0.5, 0.9, 0.99)], mean=pl.col("p").mean().round(1)).sort("mo")
    print("prediction quantiles by month:"); print(q)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
