"""v43 candidate of reports/celn_scale_preregistration.md (A1/A3): v42 plus the cel_n-fix effect.

v43 = v42 + (chain on the cel_n-scaled corrector - chain on the control corrector), both chains built on this machine
from one corrector fit per member (tests/celn_one_fit.py), so only the input fix moves the scored v42 and refit noise does
not. LIRF NM-unmatched rows stay byte-identical to v42 (team rule, PROGRESS §87). Rounded to int, floored at 0.

Also builds v44 of reports/ctx_seedavg_preregistration.md (post-hoc exception, PROGRESS §109) with argv
CONTROL TREATMENT OUT: v44 = v42 + (chain_ctxc1sel_1fits3 - chain_ctxc1sel_1fits42).

Run:  .venv/Scripts/python.exe src/post/celn_build.py [CONTROL TREATMENT OUT]
      .venv/Scripts/python.exe src/post/celn_build.py ctxc1sel_1fits42 ctxc1sel_1fits3 smart-jigsaw_v44
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "data" / "submissions"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
N_ROWS = 344_841


def main() -> None:
    ctl_n, trt_n, out_n = sys.argv[1:4] if len(sys.argv) == 4 else ("ctxc1sel_1fitctl", "ctxc1sel_1fitceln", "smart-jigsaw_v43")
    tpl = pl.read_parquet(TEMPLATE)
    r = lambda f, c: pl.read_parquet(SUB / f).select(pl.col("MVT_ID_mvt"), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
    info = (pl.scan_parquet(ROOT / "data" / "ranking" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
            .select("MVT_ID_mvt", apt=pl.col("ADEP_mvt"), nm=pl.col("FLIGHT_ID_mvt").is_null()).collect()
            .with_columns(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)))
    x = (tpl.select("MVT_ID_mvt").join(r("smart-jigsaw_v42.parquet", "v42"), on="MVT_ID_mvt", how="left")
         .join(r(f"chain_{ctl_n}_final.parquet", "ctl"), on="MVT_ID_mvt", how="left")
         .join(r(f"chain_{trt_n}_final.parquet", "cel"), on="MVT_ID_mvt", how="left")
         .join(info, on="MVT_ID_mvt", how="left"))
    assert x.null_count().sum_horizontal().item() == 0
    lirf_nmu = ((x["apt"] == "LIRF") & x["nm"]).to_numpy()
    v42 = x["v42"].to_numpy()
    delta = np.where(lirf_nmu, 0.0, x["cel"].to_numpy() - x["ctl"].to_numpy())
    v43 = np.clip(np.round(v42 + delta), 0, None).astype(np.int32)
    out = pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": v43})
    # checks
    assert out.height == N_ROWS == tpl.height
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
    assert out["TAXITIME_SEC_mvt"].null_count() == 0 and (out["TAXITIME_SEC_mvt"] >= 0).all()
    assert (v43[lirf_nmu] == v42[lirf_nmu]).all()
    d = v43 - v42
    print(f"checks: {out.height:,} rows, ID set = template, no nulls, no negatives; LIRF NM-unmatched rows {lirf_nmu.sum()} identical to v42")
    print(f"vs v42: RMS {np.sqrt(np.mean(d ** 2)):.2f} s, mean {d.mean():+.3f} s, rows changed {(d != 0).sum():,}, max |d| {np.abs(d).max():.0f}")
    for a in sorted(x["apt"].unique()):
        k = (x["apt"] == a).to_numpy()
        print(f"  {a}  RMS {np.sqrt(np.mean(d[k] ** 2)):6.2f}  mean {d[k].mean():+6.2f}")
    out.write_parquet(SUB / f"{out_n}.parquet")
    print(f"wrote {out_n}.parquet")


if __name__ == "__main__":
    main()
