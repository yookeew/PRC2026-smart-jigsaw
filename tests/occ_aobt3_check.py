"""AOBT_3 vs takeoff at LFPG / LTFM (and all airports), 2025 months vs Jan/Jul 2026 -- label-free shape check,
plus 2025-only accuracy of AOBT_3 as an off-block (label used only for the 2025 accuracy columns)."""
import os, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
fs = sorted(glob.glob("data/raw/training_2025-*.parquet")) + ["data/ranking/ranking.parquet"]
d = pl.concat([pl.read_parquet(f).filter(pl.col("PHASE_mvt") == "DEP").select(
    "ADEP_mvt", "RUNWAY_mvt", ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    a=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)) for f in fs], how="vertical_relaxed")
d = d.filter(pl.col("a").is_not_null()).with_columns(am=(pl.col("a") / 60).round(0))
# per (airport, runway) modal minute share = "default spike" proxy; and the top-minute itself
g = d.group_by("ADEP_mvt", "ym", "am").len().with_columns(sh=pl.col("len") / pl.col("len").sum().over("ADEP_mvt", "ym"))
top = g.sort("sh", descending=True).group_by("ADEP_mvt", "ym", maintain_order=True).first()
s = d.group_by("ADEP_mvt", "ym").agg(n=pl.len(), a_med=pl.col("a").median(), a_p90=pl.col("a").quantile(0.9),
                                     acc120=((pl.col("a") - pl.col("y")).abs() <= 120).mean(),
                                     bias=(pl.col("a") - pl.col("y")).filter(pl.col("y") <= 3600).median())
t = s.join(top.select("ADEP_mvt", "ym", top_min="am", top_share="sh"), on=["ADEP_mvt", "ym"]).sort("ADEP_mvt", "ym")
with pl.Config(tbl_rows=200, float_precision=3, tbl_formatting="ASCII_MARKDOWN"):
    print("T - AOBT_3 per airport x month. top_min/top_share = most common whole minute and its share (default spike);"
          "\nacc120 = share |AOBT_3 - true off-block| <= 120 s (2025 only); bias = median (T-AOBT_3) - y on y<=1h")
    print(t.filter(pl.col("ADEP_mvt").is_in(["LFPG", "LTFM"])))
    rw = d.filter(pl.col("ADEP_mvt").is_in(["LFPG", "LTFM"])).with_columns(yr=pl.col("ym").str.slice(0, 4))
    g2 = rw.group_by("ADEP_mvt", "RUNWAY_mvt", "yr", "am").len().with_columns(sh=pl.col("len") / pl.col("len").sum().over("ADEP_mvt", "RUNWAY_mvt", "yr"))
    tp = g2.sort("sh", descending=True).group_by("ADEP_mvt", "RUNWAY_mvt", "yr", maintain_order=True).head(2)
    n = rw.group_by("ADEP_mvt", "RUNWAY_mvt", "yr").agg(n=pl.len(), acc120=((pl.col("a") - pl.col("y")).abs() <= 120).mean())
    print(tp.join(n, on=["ADEP_mvt", "RUNWAY_mvt", "yr"]).filter(pl.col("n") > 2000).sort("ADEP_mvt", "RUNWAY_mvt", "yr", "sh", descending=[False, False, False, True]))
