"""Follow-up to tests/base_drift_map.py (PROGRESS §110): are the flagged structural shifts outside 2025's month-to-month
range? Monthly means of each flagged feature at its airport, all 12 months of 2025 vs Jan / Jul 2026, plus
dep_rwy_config share tables where configuration carried the adversarial gain.

Run:  .venv/Scripts/python.exe -u tests/base_drift_followup.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "cache" / "features_nb"
CHECK = {"EDDF": ["nbnm_apt_900", "nbnm_anom_900", "nboff_apt_900", "arrx_1800", "n_dep_60m_prev"],
         "LTFM": ["nbnm_apt_900", "nbnm_anom_900", "nboff_apt_900", "n_dep_60m_prev"],
         "LEMD": ["arrx_900", "arrx_1800", "nboff_apt_900"],
         "EGLL": ["n_dep_15m_next", "n_deprwy_15m_next", "n_deprwy_30m_prev", "n_dep_60m_prev", "nbnm_apt_900"],
         "LFPG": ["nbnm_apt_900", "n_dep_60m_prev"],
         "EDDM": ["nbnm_apt_900", "nbnm_anom_900"]}


def main() -> None:
    cols = sorted({c for v in CHECK.values() for c in v} | {"ADEP_mvt", "T", "dep_rwy_config"})
    f25 = pl.concat([pl.read_parquet(FEAT / f"{n}.parquet", columns=cols) for n in ("train2025", "holdout_gap2025")])
    f26 = pl.read_parquet(FEAT / "ranking.parquet", columns=cols)
    f = pl.concat([f25, f26]).with_columns(ym=pl.col("T").dt.strftime("%Y-%m"))
    for a, cs in CHECK.items():
        g = (f.filter(pl.col("ADEP_mvt") == a).group_by("ym")
             .agg(pl.len().alias("n"), *[pl.col(c).cast(pl.Float64).fill_nan(None).mean().round(1).alias(c) for c in cs]).sort("ym"))
        print(f"\n=== {a}")
        with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200):
            print(g)
        m25 = g.filter(pl.col("ym").str.starts_with("2025"))
        for c in cs:
            lo, hi = m25[c].min(), m25[c].max()
            for ym in ("2026-01", "2026-07"):
                v = g.filter(pl.col("ym") == ym)[c].item()
                if v is not None and (v < lo or v > hi):
                    print(f"  OUTSIDE 2025 monthly range: {c} {ym} {v} (2025 {lo}..{hi})")
    for a in ("EDDF", "LFPG", "EGLL", "EHAM"):
        s = (f.filter(pl.col("ADEP_mvt") == a).with_columns(yr=pl.col("ym").str.slice(0, 4), mo=pl.col("ym").str.slice(5, 2))
             .filter(pl.col("mo").is_in(["01", "07"])).group_by("yr", "mo", "dep_rwy_config").len()
             .with_columns(share=(pl.col("len") / pl.col("len").sum().over("yr", "mo")).round(3)))
        p = s.pivot(on=["yr", "mo"], index="dep_rwy_config", values="share").fill_null(0)
        p = p.with_columns(mx=pl.max_horizontal(pl.exclude("dep_rwy_config"))).filter(pl.col("mx") >= 0.02).sort("mx", descending=True).drop("mx")
        print(f"\n=== {a} dep_rwy_config share (Jan/Jul 2025 vs 2026, configs >= 2% anywhere)")
        with pl.Config(tbl_rows=30, tbl_width_chars=200):
            print(p)


if __name__ == "__main__":
    main()
