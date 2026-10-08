"""Production side of reports/ctx_seedavg_preregistration.md (post-hoc exception, PROGRESS §109): in ONE process, the
corrector fitted with seeds 42, 43, 44 on the production rows; writes the seed-42 correction (the single-fit control)
and the 3-seed mean, so v42 + (chain(mean) - chain(seed 42)) moves v42 by the averaging effect alone.

Writes data/submissions/ctxc1{r,rb}_1fits42.parquet and ctxc1{r,rb}_1fits3.parquet.
Run:  ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 [CTX_ADSB=1] .venv/Scripts/python.exe -u tests/seedavg_one_fit.py
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
sys.path.insert(0, str(ROOT / "tests"))
import corrector_input_audit as AU  # noqa: E402
from post import ctx_armc as A  # noqa: E402

SEEDS = [42, 43, 44]


def main() -> None:
    import context_corrector_test as CC
    X, fit, dep, Xr, app, dep_r, catn, s_tr, s_r = AU.matrices(with_stack=True)
    cols = X.columns
    Xb = np.vstack([X.to_numpy().astype(np.float64), Xr.to_numpy().astype(np.float64)])
    y = dep["y"].to_numpy()
    n_tr, n_r = len(y), Xr.height
    z = np.zeros(n_r, bool)
    inner = (dep["T"].to_numpy() // 86400).astype(int) % 5 == 0
    apt = np.concatenate([dep["apt"].to_numpy(), dep_r["apt"].to_numpy()])
    cats = [cols.index(c) for c in catn]
    p0, pa0 = dict(CC.PARAMS), dict(CC.PARAMS_APT)
    cs = []
    for sd in SEEDS:
        CC.PARAMS, CC.PARAMS_APT = {**p0, "seed": sd}, {**pa0, "seed": sd}
        c, b = CC.fit_apply_ens(Xb, np.concatenate([y - s_tr, np.zeros(n_r)]), np.concatenate([fit, z]),
                                np.concatenate([inner, z]), np.concatenate([np.zeros(n_tr, bool), app]), cats, apt)
        cs.append(c)
        print(f"seed {sd}: global best_iter {b}", flush=True)
    CC.PARAMS, CC.PARAMS_APT = p0, pa0
    print(f"RMS(seed 42 - mean) {np.sqrt(np.mean((cs[0] - np.mean(cs, 0)) ** 2)):.2f} s", flush=True)
    lt_r = (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy()
    win = lambda q: np.where(np.isnan(lt_r), q, np.clip(q, np.clip(lt_r - 3606, 0, None), lt_r + 3606))  # noqa: E731
    tpl = pl.read_parquet(A.TEMPLATE)
    for tag, c in (("s42", cs[0]), ("s3", np.mean(cs, 0))):
        out = s_r.copy(); out[app] = s_r[app] + c
        out = win(np.clip(out, 0, None))
        sub = pl.DataFrame({"MVT_ID_mvt": dep_r["MVT_ID_mvt"], "TAXITIME_SEC_mvt": np.round(out).astype(np.int32)})
        sub = tpl.select("MVT_ID_mvt").join(sub.with_columns(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
        assert sub.height == tpl.height and sub["TAXITIME_SEC_mvt"].null_count() == 0
        sub.write_parquet(A.SUB / f"ctxc1{A.MEMBER}_1fit{tag}.parquet")
    print(f"wrote ctxc1{A.MEMBER}_1fit{{s42,s3}}.parquet")


if __name__ == "__main__":
    main()
