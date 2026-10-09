"""v57 part A witness (reports/v57_eddf_opdi_preregistration.md, section W): is the OPDI ground-event clock stable
from late 2025 to 2026 at EDDF? Label-free in 2026 (arrival in-block times are kept in the ranking file).

W1: arrivals matched to OPDI flights (callsign, ADES, nearest exit-runway within 300 s of landing);
    lag = first entry-parking_position after landing - BLOCK_TIME (in-block). Median / IQR per period.
W2: departures with o_lead_exitpark: (T - AOBT_3) - o_lead_exitpark, and adsb.lol dwell/pass adsb_taxi - o_lead_exitpark.

Run:  python tests/v57_opdi_witness.py
"""
import os, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
APTS = ["EDDF", "LSZH"]
TOL = 300.0


def period(c: str) -> pl.Expr:
    d = pl.col(c).dt.date()
    return (pl.when(d.is_between(pl.date(2025, 9, 16), pl.date(2025, 12, 31)) & ~d.is_between(pl.date(2025, 10, 2), pl.date(2025, 10, 11)))
            .then(pl.lit("P25")).when(d.is_between(pl.date(2026, 1, 11), pl.date(2026, 1, 31))).then(pl.lit("J26"))
            .when(d.is_between(pl.date(2026, 7, 1), pl.date(2026, 7, 31))).then(pl.lit("U26")).otherwise(None))


def opdi_arrival_flights() -> pl.DataFrame:
    out = []
    for f in sorted(glob.glob(str(ROOT / "cache" / "opdi_events" / "flight_events_*.parquet"))):
        e = pl.read_parquet(f).filter(pl.col("airport").is_in(APTS)).with_columns(src=pl.lit(os.path.basename(f)))
        g = e.group_by("src", "flight_id", "airport").agg(
            cs=pl.col("callsign").drop_nulls().mode().first(),
            t_rwy=pl.col("t").filter(pl.col("type") == "exit-runway"),
            t_park=pl.col("t").filter(pl.col("type") == "entry-parking_position"))
        out.append(g.explode("t_rwy").filter(pl.col("t_rwy").is_not_null()))
    return pl.concat(out)


def main():
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "CALLSIGN_flt",
            "FLIGHT_mvt", "AOBT_3_flt"]
    fs = sorted(glob.glob(str(ROOT / "data" / "raw" / "training_2025-*.parquet")))[8:] + [str(ROOT / "data" / "ranking" / "ranking.parquet")]
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed").with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64), per=period("MVT_TIME_UTC_mvt"),
        cs=pl.coalesce(pl.col("CALLSIGN_flt").str.strip_chars(), pl.col("FLIGHT_mvt").str.strip_chars()),
        T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms").cast(pl.Float64) / 1000.0).filter(pl.col("per").is_not_null())
    # ---- W1 arrivals
    arr = raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("ADES_mvt").is_in(APTS) & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()) \
             .with_columns(B=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("ms").cast(pl.Float64) / 1000.0, apt=pl.col("ADES_mvt"))
    of = opdi_arrival_flights().sort("t_rwy")
    m = arr.sort("T").join_asof(of.rename({"airport": "apt"}), left_on="T", right_on="t_rwy", by=["apt", "cs"],
                                strategy="nearest", tolerance=TOL, check_sortedness=False)
    m = m.with_columns(matched=pl.col("flight_id").is_not_null())
    x = (m.filter(pl.col("matched")).select("MVT_ID_mvt", "apt", "per", "T", "B", "t_rwy", "t_park").explode("t_park")
         .filter((pl.col("t_park") >= pl.col("t_rwy")) & (pl.col("t_park") <= pl.col("t_rwy") + 7200))
         .group_by("MVT_ID_mvt", "apt", "per", "B").agg(t_park=pl.col("t_park").min())
         .with_columns(lag=pl.col("t_park") - pl.col("B")))
    q = lambda c, p: pl.col(c).quantile(p)  # noqa: E731
    w1 = (m.group_by("apt", "per").agg(n_arr=pl.len(), match=pl.col("matched").mean())
          .join(x.filter(pl.col("lag").abs() <= 1800).group_by("apt", "per").agg(
              n_park=pl.len(), lag_p25=q("lag", 0.25), lag_med=q("lag", 0.5), lag_p75=q("lag", 0.75)), on=["apt", "per"])
          .with_columns(park_share=pl.col("n_park") / pl.col("n_arr")).sort("apt", "per"))
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print("W1 arrivals: lag = first OPDI entry-parking_position after landing - in-block [s]")
        print(w1)
        # ---- W2 departures
        of2 = pl.read_parquet(ROOT / "cache" / "opdi_feats" / "opdi_feats.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
        det = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts"]) for p in
                         sorted((ROOT / "cache" / "adsb_pushback_v4").glob("day=202[56]-*.parquet"))], how="diagonal")
        dep = (raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(APTS))
               .join(of2.select("MVT_ID_mvt", "o_lead_exitpark", "o_lead_first"), on="MVT_ID_mvt", how="left")
               .join(det, on="MVT_ID_mvt", how="left")
               .with_columns(a3=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
                             ad=pl.col("T") - pl.col("adsb_pushback_ts")))
        w2 = dep.group_by("ADEP_mvt", "per").agg(
            n=pl.len(), share_g60=(pl.col("o_lead_first").fill_null(0) >= 60).mean(),
            n_xp=pl.col("o_lead_exitpark").is_not_null().sum(),
            a3_minus_xp_med=(pl.col("a3") - pl.col("o_lead_exitpark")).filter(pl.col("a3").is_between(60, 5400)).median(),
            n_adsb=(pl.col("adsb_tier").is_in(["dwell", "pass"]) & pl.col("o_lead_exitpark").is_not_null()).sum(),
            adsb_minus_xp_med=(pl.col("ad") - pl.col("o_lead_exitpark")).filter(pl.col("adsb_tier").is_in(["dwell", "pass"])).median(),
            lead_first_med=pl.col("o_lead_first").filter(pl.col("o_lead_first") >= 60).median(),
            lead_xp_med=pl.col("o_lead_exitpark").median()).sort("ADEP_mvt", "per")
        print("\nW2 departures (seconds)")
        print(w2)


if __name__ == "__main__":
    main()
