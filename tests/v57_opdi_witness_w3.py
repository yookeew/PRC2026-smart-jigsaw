"""v57 part A witness W3 (reports/v57_eddf_opdi_preregistration.md, amendment W-A1): which clock moved at EDDF?

Anchors on EDDF departures with o_lead_exitpark: x_hat = o_lead_exitpark + c (c from Sep 16 - Dec 2025 labels),
a_hat = (T - AOBT_3) + c3. 2025: truth and the ctxc1sel OOF prediction minus the anchors; 2026: v54 and v55 (v54 with
the daily-sum level shift) minus the anchors. Daily means, then the median over days, per month.

Run:  python tests/v57_opdi_witness_w3.py
"""
import os, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "data" / "submissions"


def main():
    of = pl.read_parquet(ROOT / "cache/opdi_feats/opdi_feats.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"]
    fs = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))[8:] + [str(ROOT / "data/ranking/ranking.parquet")]
    d = (pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")
         .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == os.environ.get("W3_APT", "EDDF"))).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
         .join(of.select("MVT_ID_mvt", "o_lead_exitpark"), on="MVT_ID_mvt")
         .filter(pl.col("o_lead_exitpark").is_not_null() & (pl.col("MVT_TIME_UTC_mvt").dt.date() >= pl.date(2025, 9, 16)))
         .with_columns(ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
                       a3=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
                       y=(pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64)))
    oof = pl.concat([pl.read_parquet(p) for p in sorted((ROOT / "cache/oof/ctxc1sel").glob("fold=*.parquet"))]).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), oof=pl.col("pred").cast(pl.Float64))
    sub = None
    for v in ("v54", "v55"):
        s = pl.read_parquet(SUB / f"smart-jigsaw_{v}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                                                                    pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(v))
        sub = s if sub is None else sub.join(s, on="MVT_ID_mvt")
    d = d.join(oof, on="MVT_ID_mvt", how="left").join(sub, on="MVT_ID_mvt", how="left")
    p25 = d.filter(pl.col("ym") < "2026")
    c = p25.select((pl.col("y") - pl.col("o_lead_exitpark")).median()).item()
    c3 = p25.filter(pl.col("a3").is_between(60, 5400)).select((pl.col("y") - pl.col("a3")).median()).item()
    print(f"c (truth - exit-park lead, P25 median) {c:.0f} s; c3 (truth - AOBT_3 taxi) {c3:.0f} s")
    d = d.with_columns(xh=pl.col("o_lead_exitpark") + c, ah=pl.when(pl.col("a3").is_between(60, 5400)).then(pl.col("a3") + c3))
    pred = pl.when(pl.col("ym") < "2026").then(pl.col("y")).otherwise(pl.col("v55"))
    day = d.group_by("ym", "day").agg(
        n=pl.len(),
        truth_or_v55_x=(pred - pl.col("xh")).mean(), truth_or_v55_a=(pred - pl.col("ah")).mean(),
        oof_or_v54_x=(pl.when(pl.col("ym") < "2026").then(pl.col("oof")).otherwise(pl.col("v54")) - pl.col("xh")).mean(),
        oof_or_v54_a=(pl.when(pl.col("ym") < "2026").then(pl.col("oof")).otherwise(pl.col("v54")) - pl.col("ah")).mean())
    out = day.filter(pl.col("n") >= 20).group_by("ym").agg(days=pl.len(), rows=pl.col("n").sum(),
                                                           **{k: pl.col(k).median() for k in ["truth_or_v55_x", "truth_or_v55_a", "oof_or_v54_x", "oof_or_v54_a"]}).sort("ym")
    with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=1):
        print("median over days of the daily mean; 2025 rows: truth / OOF corrector, 2026 rows: v55 / v54")
        print(out)


if __name__ == "__main__":
    main()
