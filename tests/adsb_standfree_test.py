"""Stand-free ADS-B fields in the combiner (reports/adsb_standfree_preregistration.md).

v27's full-year combiner method cross-fit on the Jan+Jul 2025 holdout, twice: v4 inputs
(control) and v4 inputs + the v6 stand-free fields (treatment). Same rows and parameters.
Primary: body RMSE (NM-matched, y <= 1 h) on the rows the combiner serves.

Run:  .venv/Scripts/python.exe tests/adsb_standfree_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_combiner_12m_test as T  # noqa: E402
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402
import neighbour_state_test as N  # noqa: E402

V6 = ROOT / "cache" / "adsb_pushback_v6"
EXTRA = ["adsb_move0_ts", "adsb_stop_end_ts", "adsb_stop_dur_s", "adsb_stop_own_m",
         "adsb_n_stops_after", "adsb_gap_max_s"]
NEW = ["sf_stop", "sf_stop_rel", "sf_move0", "sf_move0_rel", "adsb_stop_dur_s", "adsb_stop_own_m",
       "adsb_n_stops_after", "adsb_gap_max_s"]


def v6_extra() -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", *EXTRA]) for p in sorted(V6.glob("day=2025-*.parquet"))])


_orig, _orig_feats = C.features, list(C.FEATS)


def features_v6(df: pl.DataFrame, s: np.ndarray, airports: list[str]) -> pl.DataFrame:
    cur = C.FEATS
    C.FEATS = _orig_feats
    try:
        base = _orig(df, s, airports)
    finally:
        C.FEATS = cur
    t = df["mvt_ts"]
    extra = df.select(
        sf_stop=t - pl.col("adsb_stop_end_ts"), sf_move0=t - pl.col("adsb_move0_ts"),
        adsb_stop_dur_s="adsb_stop_dur_s", adsb_stop_own_m="adsb_stop_own_m",
        adsb_n_stops_after=pl.col("adsb_n_stops_after").cast(pl.Float64), adsb_gap_max_s="adsb_gap_max_s",
    ).with_columns(sf_stop_rel=pl.col("sf_stop") - pl.Series(s), sf_move0_rel=pl.col("sf_move0") - pl.Series(s))
    return pl.concat([base, extra.select(NEW)], how="horizontal")


def main() -> None:
    df = A.load()
    d3 = V.with_det(df, V.read_det(V.V3))
    d4 = V.with_det(df, V.read_det(C.V4))
    v23, _ = V.pipeline(d3, fb_stage=True, gate=True)
    ex, _ = N.train_extra()
    x6 = v6_extra()
    d6 = d4.join(x6, on="MVT_ID_mvt", how="left")
    e6 = ex.join(x6, on="MVT_ID_mvt", how="left")
    assert d6.height == d4.height and e6.height == ex.height
    print(f"v6 fields present: holdout {d6['adsb_move0_ts'].is_not_null().mean():.3f}, "
          f"stop {d6['adsb_stop_end_ts'].is_not_null().mean():.3f}; OOF days {e6['adsb_move0_ts'].is_not_null().mean():.3f}")

    p0, i0 = T.combiner_12m(d4, ex, v23, use_holdout=True)
    C.features, C.FEATS = features_v6, _orig_feats + NEW
    p1, i1 = T.combiner_12m(d6, e6, v23, use_holdout=True)
    C.features, C.FEATS = _orig, _orig_feats
    print("control fits:", i0)
    print("treatment fits:", i1)

    y, ym, apt = d4["taxi"].to_numpy(), d4["ym"].to_numpy(), d4["ADEP_mvt"].to_numpy()
    body = d4["FLIGHT_ID_mvt"].is_null().not_().to_numpy() & (y <= 3600)
    served = d4.select(C2.has_adsb()).to_series().to_numpy() & (apt != "LIRF")
    r = lambda p, k: float(np.sqrt(np.mean((y - p)[k] ** 2)))
    rows = [("body served, pooled", body & served), ("body served, Jan", body & served & (ym == "2025-01")),
            ("body served, Jul", body & served & (ym == "2025-07")), ("body all rows", body),
            ("trimmed all rows", y <= 18000), ("full all rows", np.ones_like(body))]
    rows += [(f"  body served {a}", body & served & (apt == a)) for a in sorted(set(apt[served]))]
    print(f"\n{'slice':26s} {'control':>9s} {'+stand-free':>12s} {'delta':>7s}")
    for name, k in rows:
        print(f"{name:26s} {r(p0, k):9.2f} {r(p1, k):12.2f} {r(p1, k) - r(p0, k):+7.2f}")
    g = r(p0, body & served) - r(p1, body & served)
    gj = r(p0, body & served & (ym == "2025-01")) - r(p1, body & served & (ym == "2025-01"))
    gl = r(p0, body & served & (ym == "2025-07")) - r(p1, body & served & (ym == "2025-07"))
    ok = g >= 2 and gj > 0 and gl > 0 and r(p1, y <= 18000) <= r(p0, y <= 18000)
    print(f"\nDECISION: served body gain {g:.2f} (Jan {gj:.2f}, Jul {gl:.2f}) -> {'ADOPT' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
