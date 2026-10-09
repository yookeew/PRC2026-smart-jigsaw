"""Step 1 of reports/adsb_queue_scoping.md: how well adsb.lol sees departing aircraft near the runway, per airport
and month, 2025 vs 2026. Label-free. Needs cache/adsb_rwyruns/ (tests/adsb_queue_runs.py)."""
import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "cache" / "adsb_rwyruns"
V4 = ROOT / "cache" / "adsb_pushback_v4"
ENTRY_GAP = 60.0          # a 1 km entry counts as measured only if the samples around it are <= 60 s apart


def load_runs() -> pl.DataFrame:
    r = pl.concat([pl.read_parquet(p) for p in sorted(RUNS.glob("day=*.parquet"))], how="diagonal_relaxed")
    return r.filter(pl.col("runway").is_not_null()).with_columns(
        ent=pl.col("t_1km").is_not_null() & (pl.col("t_1km_gap") <= ENTRY_GAP))


def departures(days: list) -> pl.DataFrame:
    src = [str(p) for p in sorted((ROOT / "data" / "raw").glob("training_2025-*.parquet"))]
    tr = pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
    rk = pl.scan_parquet(str(ROOT / "data" / "ranking" / "ranking.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
    cols = [pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "RUNWAY_mvt",
            (pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0).alias("mts"), pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day")]
    d = pl.concat([tr.select(*cols).collect(), rk.select(*cols).collect()])
    return d.filter(pl.col("day").is_in(days))


def neighbour_reach(dep: pl.DataFrame, runs: pl.DataFrame, lo=-900.0, hi=600.0) -> np.ndarray:
    """Per departure: number of OTHER runs with a measured entry on the same runway ending in [T+lo, T+hi]."""
    out = np.zeros(dep.height, dtype=np.int32)
    dep = dep.with_row_index("_i")
    rr = runs.filter("ent")
    for (ap, rw), g in dep.group_by("ADEP_mvt", "RUNWAY_mvt"):
        rg = rr.filter((pl.col("airport") == ap) & (pl.col("runway") == rw)).sort("t_end")
        if rg.height == 0:
            continue
        te, rm = rg["t_end"].to_numpy(), rg["MVT_ID_mvt"].to_numpy()
        T, mid, idx = g["mts"].to_numpy(), g["MVT_ID_mvt"].to_numpy(), g["_i"].to_numpy()
        a, b = np.searchsorted(te, T + lo), np.searchsorted(te, T + hi)
        n = b - a
        own = np.array([np.any(rm[x:y] == m) for x, y, m in zip(a, b, mid)])
        out[idx] = n - own
    return out


def main():
    runs = load_runs()
    days = sorted(runs["day"].unique().to_list())
    dep = departures(days)
    v4 = pl.concat([pl.read_parquet(V4 / f"day={d.isoformat()}.parquet") for d in days if (V4 / f"day={d.isoformat()}.parquet").exists()])
    dep = (dep.join(runs.filter(pl.col("MVT_ID_mvt").is_not_null()).select("MVT_ID_mvt", "ent", "n_mid", "slow_1km", "t_1km"),
                    on="MVT_ID_mvt", how="left")
           .join(v4.select("MVT_ID_mvt", "adsb_tier"), on="MVT_ID_mvt", how="left"))
    dep = dep.with_columns(nb=pl.Series(neighbour_reach(dep, runs)),
                           per=pl.when(pl.col("day").dt.month().is_in([1, 7])).then(pl.col("day").dt.strftime("%Y-%m"))
                           .otherwise(pl.lit("2025-09/11 (2 days)")))
    tab = (dep.group_by("ADEP_mvt", "per").agg(
        n=pl.len(), matched=pl.col("ent").is_not_null().mean(), entry=pl.col("ent").fill_null(False).mean(),
        reach1=(pl.col("nb") >= 1).mean(), reach2=(pl.col("nb") >= 2).mean(),
        pushback=pl.col("adsb_tier").is_in(["dwell", "pass", "appear"]).mean(),
        deice=(pl.col("n_mid").fill_null(0) > 0).sum() / pl.col("ent").is_not_null().sum())
        .sort("ADEP_mvt", "per"))
    with pl.Config(tbl_rows=100, tbl_formatting="ASCII_MARKDOWN", float_precision=3, tbl_width_chars=200):
        print("Per airport x month. matched = own run matched (same runway, +-120 s); entry = own 1 km entry measured;"
              "\nreach1/2 = >= 1/2 other same-runway runs with a measured entry in [T-15, T+10] min; pushback = v4 tier;"
              "\ndeice = matched runs with a long stop > 1 km from the roll start (not the first stop)\n")
        print(tab)
        nl = dep.filter(pl.col("ADEP_mvt") != "LIRF")
        print("\nNon-LIRF pooled by period:")
        print(nl.group_by("per").agg(n=pl.len(), matched=pl.col("ent").is_not_null().mean(),
                                     entry=pl.col("ent").fill_null(False).mean(), reach1=(pl.col("nb") >= 1).mean(),
                                     reach2=(pl.col("nb") >= 2).mean(),
                                     pushback=pl.col("adsb_tier").is_in(["dwell", "pass", "appear"]).mean()).sort("per"))
    dep.select("MVT_ID_mvt", "ADEP_mvt", "per", "nb", "ent", "adsb_tier").write_parquet(ROOT / "cache" / "analysis" / "adsb_queue_cov.parquet")


if __name__ == "__main__":
    main()
