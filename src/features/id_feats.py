"""Identity of the FIRST arrival at our stand after SOBT + 5 min (and before T): own late inbound or re-occupier?
Label-free cues (2025 check: type differs & operator matches & origin differs -> 77% re-occupier; type match & origin
match -> 0.1%): appended to reocc_feats.parquet (v50's 9 columns) -> reocc_feats6.parquet.
Columns: fa_tmatch, fa_omatch, fa_origin (arrival ADEP == our ADES), fa_flnear (same 2-letter prefix, numbers +-1),
fa_tin_x (its taxi-in minus the 2025 median for its stand/runway); null when there is no such arrival.
PRC_SCOPE=final4m (src/scope.py, PROGRESS.md §132): 2026 rows from final_ranking.parquet; reads and writes
cache/v50/final4m/ (run src/features/reocc_feats.py in that scope first)."""
import os, glob, sys
from pathlib import Path
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scope  # noqa: E402
SCR = os.environ.get("SCR", str(scope.out_dir(Path("cache/v50"))))


def fl_num(c):
    return pl.col(c).str.extract(r"(\d+)\D*$", 1).cast(pl.Int64, strict=False)


def build(df: pl.DataFrame, med: pl.DataFrame) -> pl.DataFrame:
    dep = (df.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("STAND_mvt").is_not_null() & pl.col("MVT_TIME_UTC_mvt").is_not_null())
           .select(pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt", stand="STAND_mvt", T="MVT_TIME_UTC_mvt",
                   S="SCHED_TIME_UTC_mvt", ades="ADES_mvt", dtype=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"),
                   dop="AIRCRAFT_OPERATOR_flt", dfl="FLIGHT_mvt")
           .with_columns(t_lo=pl.col("S") + pl.duration(minutes=5)))
    arr = (df.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
           .select(apt="ADES_mvt", stand="STAND_mvt", a_in="BLOCK_TIME_UTC_mvt", a_adep="ADEP_mvt", a_rwy="RUNWAY_mvt",
                   atype=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"), aop="AIRCRAFT_OPERATOR_flt", afl="FLIGHT_mvt",
                   a_tin=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
           .join(med, on=["apt", "stand", "a_rwy"], how="left").with_columns(a_tin_x=pl.col("a_tin") - pl.col("a_tin_med"))
           .sort("a_in"))
    j = dep.filter(pl.col("t_lo").is_not_null()).sort("t_lo").join_asof(
        arr.select("apt", "stand", "a_in", "a_adep", "atype", "aop", "afl", "a_tin_x"),
        left_on="t_lo", right_on="a_in", by=["apt", "stand"], strategy="forward")
    ok = pl.col("a_in").is_not_null() & (pl.col("a_in") < pl.col("T"))
    out = j.select(
        "MVT_ID_mvt",
        fa_tmatch=pl.when(ok).then((pl.col("atype") == pl.col("dtype")).cast(pl.Float64)),
        fa_omatch=pl.when(ok).then((pl.col("aop") == pl.col("dop")).cast(pl.Float64)),
        fa_origin=pl.when(ok).then((pl.col("a_adep") == pl.col("ades")).cast(pl.Float64)),
        fa_flnear=pl.when(ok).then(((pl.col("afl").str.slice(0, 2) == pl.col("dfl").str.slice(0, 2))
                                    & ((fl_num("afl") - fl_num("dfl")).abs() <= 1)).fill_null(False).cast(pl.Float64)),
        fa_tin_x=pl.when(ok).then(pl.col("a_tin_x")),
    )
    return out


if __name__ == "__main__":
    tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="vertical_relaxed")
    med = (tr.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null())
           .select(apt="ADES_mvt", stand="STAND_mvt", a_rwy="RUNWAY_mvt",
                   a_tin=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
           .filter(pl.col("a_tin").is_between(0, 3600)).group_by("apt", "stand", "a_rwy").agg(a_tin_med=pl.col("a_tin").median()))
    a = build(tr, med)
    b = build(pl.read_parquet(scope.RANKING), med)
    m = pl.concat([a, b]).unique("MVT_ID_mvt", keep="first")
    base = pl.read_parquet(f"{SCR}/reocc_feats.parquet")
    out = base.join(m, on="MVT_ID_mvt", how="left")
    assert out.height == base.height
    out.write_parquet(f"{SCR}/reocc_feats6.parquet")
    for c in ["fa_tmatch", "fa_omatch", "fa_origin", "fa_flnear", "fa_tin_x"]:
        print(f"{c:10s} 2025 nonnull {a[c].is_not_null().mean():.4f} mean {a[c].mean():8.3f} | 2026 nonnull {b[c].is_not_null().mean():.4f} mean {b[c].mean():8.3f}")
