"""Label-free stand re-occupation features for every departure (2025 training + 2026 ranking).

Uses only arrival in-block times, departure takeoff times, stands, aircraft types and operators
(none blanked in the ranking file). Keyed on MVT_ID_mvt (Int64). Columns:
  ro_gap_last   T - in-block of the last arrival at the stand before T (s)
  ro_tm_last    type(last arrival) == type(departure)
  ro_om_last    operator(last arrival) == operator(departure)
  ro_two_dep    next departure from the stand (after T) comes before the next arrival after T
  ro_tm_next    type(last arrival) == type(next departure)
  ro_gap_prev   in-block(last) - in-block(previous arrival) (s)
  ro_tm_prev    type(previous arrival) == type(departure)
  ro_n_since_s  arrivals at the stand with in-block in (SOBT + 5 min, T)
  ro_first_since_s  T - earliest such arrival (s)

PRC_SCOPE=final4m (src/scope.py, PROGRESS.md §132): 2026 rows from final_ranking.parquet, output in
cache/v50/final4m/, plus reocc_feats.parquet there (the first nine columns, which is what the jj
cache/v50/reocc_feats.parquet holds; checked equal 2026-10-09) for src/features/id_feats.py.
"""
import os, glob, sys
from pathlib import Path
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scope  # noqa: E402
SCR = os.environ.get("SCR", str(scope.out_dir(Path("cache/v50"))))
V50_COLS = ["MVT_ID_mvt", "ro_gap_last", "ro_tm_last", "ro_om_last", "ro_two_dep", "ro_tm_next", "ro_gap_prev",
            "ro_tm_prev", "ro_n_since_s", "ro_first_since_s"]
MED = None


def build(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    dep = df.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", apt="ADEP_mvt", stand="STAND_mvt", T="MVT_TIME_UTC_mvt", S="SCHED_TIME_UTC_mvt",
        dtype=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"), dop="AIRCRAFT_OPERATOR_flt",
    ).filter(pl.col("stand").is_not_null() & pl.col("T").is_not_null())
    arr = df.filter(pl.col("PHASE_mvt") == "ARR").select(
        apt="ADES_mvt", stand="STAND_mvt", a_in="BLOCK_TIME_UTC_mvt",
        atype=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"), aop="AIRCRAFT_OPERATOR_flt",
        a_land="MVT_TIME_UTC_mvt", a_rwy="RUNWAY_mvt",
    ).filter(pl.col("stand").is_not_null() & pl.col("a_in").is_not_null()).sort("a_in")
    arr = arr.with_columns(a_tin=(pl.col("a_in") - pl.col("a_land")).dt.total_seconds().cast(pl.Float64))
    # taxi-in excess over the (airport, stand, runway) median: a long one = the arrival waited for its stand
    med = MED if MED is not None else arr.filter(pl.col("a_tin").is_between(0, 3600)).group_by(
        "apt", "stand", "a_rwy").agg(a_tin_med=pl.col("a_tin").median())
    arr = arr.join(med, on=["apt", "stand", "a_rwy"], how="left").with_columns(a_tin_x=pl.col("a_tin") - pl.col("a_tin_med"))
    j = dep.sort("T").join_asof(arr.select("apt", "stand", "a_in", "atype", "aop", "a_land", "a_tin_x"),
                                left_on="T", right_on="a_in", by=["apt", "stand"], strategy="backward")
    nd = dep.select("apt", "stand", Tn="T", ntype="dtype").sort("Tn")
    j = j.with_columns(Tq=pl.col("T") + pl.duration(microseconds=1), Tb=pl.col("T") - pl.duration(microseconds=1)).sort("Tb")
    pdp = dep.select("apt", "stand", Tp="T", pdtype="dtype").sort("Tp")
    j = j.join_asof(pdp, left_on="Tb", right_on="Tp", by=["apt", "stand"], strategy="backward").sort("Tq")
    j = j.join_asof(nd, left_on="Tq", right_on="Tn", by=["apt", "stand"], strategy="forward")
    na = arr.select("apt", "stand", a_after="a_in").sort("a_after")
    j = j.sort("T").join_asof(na, left_on="T", right_on="a_after", by=["apt", "stand"], strategy="forward")
    pa = arr.select("apt", "stand", p_in="a_in", ptype="atype").sort("p_in")
    j = j.with_columns(aq=pl.col("a_in") - pl.duration(microseconds=1))
    j = pl.concat([j.filter(pl.col("aq").is_null()).with_columns(p_in=pl.lit(None, j.schema["a_in"]), ptype=pl.lit(None, pl.Utf8)),
                   j.filter(pl.col("aq").is_not_null()).sort("aq").join_asof(pa, left_on="aq", right_on="p_in",
                                                                               by=["apt", "stand"], strategy="backward")],
                  how="diagonal")
    # arrivals with in-block in (SOBT + 5 min, T): cumulative rank per stand, two as-of lookups
    rk = arr.select("apt", "stand", "a_in").sort("apt", "stand", "a_in").with_columns(
        rank=pl.int_range(1, pl.len() + 1).over(["apt", "stand"])).sort("a_in")
    q = j.select("MVT_ID_mvt", "apt", "stand", "S", "T").filter(pl.col("S").is_not_null()).with_columns(
        t_hi=pl.col("T") - pl.duration(microseconds=1), t_lo=pl.col("S") + pl.duration(minutes=5))
    q = q.sort("t_hi").join_asof(rk.select("apt", "stand", r_hi="rank", a_hi="a_in"), left_on="t_hi", right_on="a_hi",
                                 by=["apt", "stand"], strategy="backward")
    q = q.sort("t_lo").join_asof(rk.select("apt", "stand", r_lo="rank", a_lo="a_in"), left_on="t_lo", right_on="a_lo",
                                 by=["apt", "stand"], strategy="backward")
    q = q.with_columns(t_lo1=pl.col("t_lo") + pl.duration(microseconds=1)).sort("t_lo1").join_asof(
        rk.select("apt", "stand", first_in="a_in"), left_on="t_lo1", right_on="first_in", by=["apt", "stand"], strategy="forward")
    q = q.with_columns(ro_n_since_s=(pl.col("r_hi").fill_null(0) - pl.col("r_lo").fill_null(0)).clip(lower_bound=0),
                       first_in=pl.when(pl.col("first_in") < pl.col("T")).then(pl.col("first_in")))
    j = j.join(q.select("MVT_ID_mvt", "ro_n_since_s", "first_in"), on="MVT_ID_mvt", how="left")
    sec = lambda e: e.dt.total_seconds().cast(pl.Float64)  # noqa: E731
    # seam guard (ranking frame jumps 31 Jan -> 1 Jul): links further than 24 h away are treated as absent
    day = pl.duration(hours=24)
    j = j.with_columns(
        a_in=pl.when(pl.col("T") - pl.col("a_in") <= day).then(pl.col("a_in")),
        Tn=pl.when(pl.col("Tn") - pl.col("T") <= day).then(pl.col("Tn")),
        Tp=pl.when(pl.col("T") - pl.col("Tp") <= day).then(pl.col("Tp")),
        a_after=pl.when(pl.col("a_after") - pl.col("T") <= day).then(pl.col("a_after")),
    ).with_columns(
        p_in=pl.when(pl.col("a_in").is_not_null() & (pl.col("a_in") - pl.col("p_in") <= day)).then(pl.col("p_in")),
        atype=pl.when(pl.col("a_in").is_not_null()).then(pl.col("atype")),
        aop=pl.when(pl.col("a_in").is_not_null()).then(pl.col("aop")),
        a_tin_x=pl.when(pl.col("a_in").is_not_null()).then(pl.col("a_tin_x")),
        a_land=pl.when(pl.col("a_in").is_not_null()).then(pl.col("a_land")),
        ntype=pl.when(pl.col("Tn").is_not_null()).then(pl.col("ntype")),
    ).with_columns(ptype=pl.when(pl.col("p_in").is_not_null()).then(pl.col("ptype")))
    out = j.select(
        "MVT_ID_mvt",
        ro_gap_last=sec(pl.col("T") - pl.col("a_in")),
        ro_tm_last=(pl.col("atype") == pl.col("dtype")).cast(pl.Float64),
        ro_om_last=(pl.col("aop") == pl.col("dop")).cast(pl.Float64),
        ro_two_dep=(pl.col("Tn").is_not_null() & (pl.col("a_after").is_null() | (pl.col("Tn") < pl.col("a_after")))
                    & pl.col("a_in").is_not_null()).cast(pl.Float64),
        ro_tm_next=(pl.col("atype") == pl.col("ntype")).cast(pl.Float64),
        ro_gap_prev=sec(pl.col("a_in") - pl.col("p_in")),
        ro_tm_prev=(pl.col("ptype") == pl.col("dtype")).cast(pl.Float64),
        ro_n_since_s=pl.col("ro_n_since_s").fill_null(0).cast(pl.Float64),
        ro_first_since_s=sec(pl.col("T") - pl.col("first_in")),
        ro_last_tin_x=pl.col("a_tin_x"),
        ro_last_land_gap=sec(pl.col("T") - pl.col("a_land")),
        ro_prevdep_gap=sec(pl.col("T") - pl.col("Tp")),
        ro_prevdep_after_last=(pl.col("Tp").is_not_null() & pl.col("a_in").is_not_null() & (pl.col("Tp") > pl.col("a_in"))).cast(pl.Float64),
        ro_tm_prevdep_last=pl.when(pl.col("Tp").is_not_null() & pl.col("a_in").is_not_null()).then((pl.col("pdtype") == pl.col("atype")).cast(pl.Float64)),
        ro_tm_prevdep_self=pl.when(pl.col("Tp").is_not_null()).then((pl.col("pdtype") == pl.col("dtype")).cast(pl.Float64)),
    )
    return out


if __name__ == "__main__":
    tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="vertical_relaxed")
    arr25 = tr.filter(pl.col("PHASE_mvt") == "ARR").select(apt="ADES_mvt", stand="STAND_mvt", a_rwy="RUNWAY_mvt", a_tin=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
    MED = arr25.filter(pl.col("a_tin").is_between(0, 3600) & pl.col("stand").is_not_null()).group_by("apt", "stand", "a_rwy").agg(a_tin_med=pl.col("a_tin").median())
    a = build(tr)
    rk = pl.read_parquet(scope.RANKING)
    b = build(rk)
    out = pl.concat([a, b])
    assert out["MVT_ID_mvt"].n_unique() == out.height
    out.write_parquet(f"{SCR}/reocc_feats3.parquet")
    if scope.FINAL:
        out.select(V50_COLS).write_parquet(f"{SCR}/reocc_feats.parquet")
    print("rows", a.height, b.height)
    print(out.select([pl.col(c).is_not_null().mean().alias(c) for c in out.columns if c != "MVT_ID_mvt"]))
    # drift check 2025 vs 2026 (label-free)
    for c in [c for c in out.columns if c != "MVT_ID_mvt"]:
        print(f"{c:18s} 2025 mean {a[c].mean():10.2f} med {a[c].median():10.2f} | 2026 mean {b[c].mean():10.2f} med {b[c].median():10.2f}")
