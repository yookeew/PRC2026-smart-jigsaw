"""Measurement of reports/celn_scale_preregistration.md, amendment A1: one corrector fit per member, two 2026 predictions
(cel_n as built / scaled by 11/12), so the difference is the input effect alone (refit noise excluded).

Writes data/submissions/ctxc1{r,rb}_1fit{ctl,celn}.parquet (corrector outputs, prod's window and template order).
Run:  ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 [CTX_ADSB=1] .venv/Scripts/python.exe -u tests/celn_one_fit.py
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


def main() -> None:
    import context_corrector_test as CC
    X, fit, dep, Xr, app, dep_r, catn, s_tr, s_r = AU.matrices(with_stack=True)
    cols = X.columns
    Xa, Xr0 = X.to_numpy().astype(np.float64), Xr.to_numpy().astype(np.float64)
    Xr1 = Xr0.copy()
    Xr1[:, cols.index("cel_n")] *= 11 / 12
    y = dep["y"].to_numpy()
    n_tr, n_r = len(y), len(Xr0)
    inner = (dep["T"].to_numpy() // 86400).astype(int) % 5 == 0
    Xb = np.vstack([Xa, Xr0, Xr1])
    z = np.zeros(n_r, bool)
    apt = np.concatenate([dep["apt"].to_numpy(), dep_r["apt"].to_numpy(), dep_r["apt"].to_numpy()])
    cats = [cols.index(c) for c in catn]
    target = np.concatenate([y - s_tr, np.zeros(2 * n_r)])
    corr, best = CC.fit_apply_ens(Xb, target, np.concatenate([fit, z, z]), np.concatenate([inner, z, z]),
                                  np.concatenate([np.zeros(n_tr, bool), app, app]), cats, apt)
    k = int(app.sum())
    c0, c1 = corr[:k], corr[k:]
    lt_r = (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy()
    win = lambda q: np.where(np.isnan(lt_r), q, np.clip(q, np.clip(lt_r - 3606, 0, None), lt_r + 3606))  # noqa: E731
    tpl = pl.read_parquet(A.TEMPLATE)
    print(f"global best_iter {best}; RMS(c_celn - c_ctl) over applied rows {np.sqrt(np.mean((c1 - c0) ** 2)):.2f} s")
    for tag, c in (("ctl", c0), ("celn", c1)):
        out = s_r.copy(); out[app] = s_r[app] + c
        out = win(np.clip(out, 0, None))
        sub = pl.DataFrame({"MVT_ID_mvt": dep_r["MVT_ID_mvt"], "TAXITIME_SEC_mvt": np.round(out).astype(np.int32)})
        sub = tpl.select("MVT_ID_mvt").join(sub.with_columns(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
        assert sub.height == tpl.height and sub["TAXITIME_SEC_mvt"].null_count() == 0
        sub.write_parquet(A.SUB / f"ctxc1{A.MEMBER}_1fit{tag}.parquet")
    print(f"wrote ctxc1{A.MEMBER}_1fit{{ctl,celn}}.parquet")


if __name__ == "__main__":
    main()
