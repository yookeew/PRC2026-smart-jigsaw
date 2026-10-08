"""Echo-hedge cost audit (PROGRESS.md §101). No training.

Every base prediction is p * clip(offset, 0, 140000) + (1 - p) * R, with offset = T - SCHED, p the echo
classifier's P(|BLOCK - SCHED| < 30 s) and R the non-echo estimate (clipped raw regressor, or the median prior
on use_prior rows). With a multi-hour offset and p away from 0/1 the prediction is wrong by hours either way.
Question: how much of v42's squared error is this hedge, where does it sit, and does it look the same in 2026?

Holdout (labels):
  A  v42 SSE by mechanism: echo rows / non-echo rows, by offset band; LIRF NM-unmatched separate.
  B  the 40 largest v42 squared errors, with p, offset, R, P, y.
  C  calibration of p (stack mix 0.58 lgb + 0.42 cat) by p band x offset band: mean p vs echo rate.
  D  label-free expected squared error per row. P_ne = (P - p off) / (1 - p) is the non-echo branch implied by v42's
     prediction; H = p (1 - p) (off - P_ne)^2 is the echo-uncertainty (hedge) cost, removed if the echo status were
     known; E = H + (1 - p) s2, s2 = residual variance of (y - P_ne) on non-echo rows per segment x offset band
     (holdout, in-sample). E is checked against the realised SSE per segment.
2026 (no labels):
  E  the same E and H on the ranking rows with P = v42's submission, against the board's total squared error
     (249.31^2 x 344,841), by segment.
R for 2026 is recovered by inverting each base submission, R = (pred - p clip(off)) / (1 - p); the inversion is
checked on the holdout against the stored raw values first.

Run:  .venv/Scripts/python.exe -u tests/hedge_cost_audit.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EVAL, SUB = ROOT / "cache" / "eval", ROOT / "data" / "submissions"
W_LGB, W_CAT = 0.577, 0.428          # OOF stack weights of the base members (logs, O.prepare)
WIDE = 140000.0
BOARD = 249.31
OFF_BANDS = [(-1e9, 3600, "<1h"), (3600, 3 * 3600, "1-3h"), (3 * 3600, 6 * 3600, "3-6h"), (6 * 3600, 12 * 3600, "6-12h"), (12 * 3600, 1e12, ">12h")]
P_BANDS = [(0, .02, "<.02"), (.02, .1, ".02-.1"), (.1, .3, ".1-.3"), (.3, .7, ".3-.7"), (.7, .9, ".7-.9"), (.9, 1.01, ">.9")]
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)


def band(x: np.ndarray, bands) -> np.ndarray:
    out = np.empty(len(x), dtype=object)
    for lo, hi, n in bands:
        out[(x >= lo) & (x < hi)] = n
    return out


def invert(pred, p, off):
    """Non-echo component R from a reconstructed prediction (exact up to the int rounding of submissions)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (pred - p * np.clip(off, 0, WIDE)) / (1 - p)


def implied(P, pm, offc, R):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (P - pm * offc) / (1 - pm)
    return np.where(pm < 0.99, x, R)


def mvt_info(src) -> pl.DataFrame:
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(I64, apt="ADEP_mvt", nm=pl.col("FLIGHT_ID_mvt").is_null(),
                    off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                    d=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
            .collect())


def seg(df: pl.DataFrame) -> np.ndarray:
    lirf_nmu = (df["apt"] == "LIRF").to_numpy() & df["nm"].to_numpy()
    return np.where(lirf_nmu, "LIRF NM-unm", np.where(df["apt"].to_numpy() == "LIRF", "LIRF NM-m", "non-LIRF"))


def main() -> None:
    # ------------------------------------------------------------------ holdout frame
    raw = mvt_info([str(ROOT / "data" / "raw" / f"training_2025-{a}-01_2025-{b}-01.parquet") for a, b in (("01", "02"), ("07", "08"))])
    lg = pl.read_parquet(EVAL / "lgb_mixed_nb_holdout_ev.parquet").select(I64, p_l="echo_prob", raw_l="taxi_model_raw", up="use_prior", pred_l="pred", off_ev="sched_takeoff_offset")
    ct = pl.read_parquet(EVAL / "cat_mixed_holdout_ev.parquet").select(I64, p_c="echo_prob", raw_c="taxi_model_raw", pred_c="pred")
    v42 = pl.read_parquet(EVAL / "v42_chain_holdout.parquet").select(I64, P="v42")
    h = v42.join(raw, on="MVT_ID_mvt").join(lg, on="MVT_ID_mvt").join(ct, on="MVT_ID_mvt")
    assert h.height == v42.height, (h.height, v42.height)
    assert float((h["off"] - h["off_ev"].cast(pl.Float64)).abs().max()) == 0.0
    off, y, P = h["off"].to_numpy(), h["y"].to_numpy(), h["P"].to_numpy()
    pl_, pc = h["p_l"].to_numpy(), h["p_c"].to_numpy()
    # inversion check: R from pred vs the stored clipped raw (non-prior rows)
    ceil = np.where(h["apt"].to_numpy() == "LIRF", WIDE, 10800.0)
    up = h["up"].to_numpy()
    for nme, pr, pp, rw in (("lgb", h["pred_l"].to_numpy(), pl_, h["raw_l"].to_numpy()), ("cat", h["pred_c"].to_numpy(), pc, h["raw_c"].to_numpy())):
        k = ~up & (pp < 0.99)
        err = np.abs(invert(pr, pp, off)[k] - np.clip(rw, 0, ceil)[k])
        print(f"inversion check {nme}: rows {k.sum():,}; |R_inv - clip(raw)| p50 {np.median(err):.3g}, p99.9 {np.quantile(err, .999):.3g}, max {err.max():.3g}")
    R_l = invert(h["pred_l"].to_numpy(), pl_, off)
    R_c = invert(h["pred_c"].to_numpy(), pc, off)
    pm = W_LGB * pl_ + W_CAT * pc
    R = np.where(pl_ >= 0.99, np.clip(h["raw_l"].to_numpy(), 0, ceil), W_LGB * R_l + W_CAT * np.nan_to_num(R_c, nan=0)) / np.where(pl_ >= 0.99, 1, W_LGB + W_CAT * (~np.isnan(R_c)))
    echo = np.abs(h["d"].to_numpy()) < 30
    sg = seg(h)
    ob = band(off, OFF_BANDS)
    se = (y - P) ** 2
    SSE = se.sum()
    N = len(y)
    print(f"\nholdout rows {N:,}; v42 RMSE {np.sqrt(SSE / N):.1f}; echo rate {echo.mean():.4f}")

    # ------------------------------------------------------------------ A
    print("\nA. v42 holdout SSE by mechanism (labelled)")
    print(f"  {'segment':12s} {'offset':6s} {'echo':5s} {'n':>7s} {'%SSE':>6s} {'rmse':>7s} {'mean p':>7s} {'mean off':>9s} {'mean y':>8s} {'mean P':>8s}")
    for s in ("non-LIRF", "LIRF NM-m", "LIRF NM-unm"):
        for _, _, o in OFF_BANDS:
            for e in (True, False):
                k = (sg == s) & (ob == o) & (echo == e)
                if k.sum() == 0 or se[k].sum() / SSE < 0.001:
                    continue
                print(f"  {s:12s} {o:6s} {str(e):5s} {k.sum():7,} {100 * se[k].sum() / SSE:6.1f} {np.sqrt(se[k].mean()):7.0f} "
                      f"{pm[k].mean():7.3f} {off[k].mean():9.0f} {y[k].mean():8.0f} {P[k].mean():8.0f}")
    for s in ("non-LIRF", "LIRF NM-m", "LIRF NM-unm"):
        k = sg == s
        print(f"  total {s:12s} n {k.sum():7,}  %SSE {100 * se[k].sum() / SSE:5.1f}")

    # ------------------------------------------------------------------ B
    print("\nB. 40 largest v42 squared errors (holdout)")
    print(f"  {'apt':4s} {'nm-unm':6s} {'%SSE':>5s} {'y':>7s} {'P':>7s} {'off':>7s} {'d':>7s} {'echo':5s} {'p_l':>5s} {'p_c':>5s} {'R':>7s}")
    for i in map(int, np.argsort(-se)[:40]):
        print(f"  {h['apt'][i]:4s} {str(h['nm'][i]):6s} {100 * se[i] / SSE:5.1f} {y[i]:7.0f} {P[i]:7.0f} {off[i]:7.0f} {h['d'][i]:7.0f} "
              f"{str(echo[i]):5s} {pl_[i]:5.2f} {pc[i]:5.2f} {R[i]:7.0f}")
    top = np.argsort(-se)[:40]
    print(f"  top 40 = {100 * se[top].sum() / SSE:.1f}% of SSE; of these echo {echo[top].sum()}, LIRF NM-unmatched {(sg[top] == 'LIRF NM-unm').sum()}")

    # ------------------------------------------------------------------ C
    print("\nC. calibration of the stack-mixed echo probability (offset >= 1 h rows), by segment")
    for s in ("non-LIRF", "LIRF NM-m", "LIRF NM-unm"):
        print(f"  {s}")
        for _, _, o in OFF_BANDS[1:]:
            for lo, hi, pn in P_BANDS:
                k = (sg == s) & (ob == o) & (pm >= lo) & (pm < hi)
                if k.sum() < 10:
                    continue
                print(f"    off {o:6s} p {pn:7s} n {k.sum():6,}  mean p {pm[k].mean():.3f}  echo rate {echo[k].mean():.3f}  %SSE {100 * se[k].sum() / SSE:5.1f}")

    # ------------------------------------------------------------------ D
    ne = ~echo
    offc = np.clip(off, 0, WIDE)
    # the non-echo branch implied by v42's own prediction, P = pm offc + (1 - pm) P_ne (base R where pm >= 0.99):
    # chain corrections stay inside P_ne, so H below counts only the echo uncertainty
    Pne = implied(P, pm, offc, R)
    s2 = {}
    for _, _, o in OFF_BANDS:
        for s in ("non-LIRF", "LIRF NM-m", "LIRF NM-unm"):
            k = ne & (ob == o) & (sg == s)
            s2[(s, o)] = float(np.mean((y[k] - Pne[k]) ** 2)) if k.sum() > 30 else float(np.mean((y[ne & (ob == o)] - Pne[ne & (ob == o)]) ** 2))
    S2 = np.array([s2[(a, b)] for a, b in zip(sg, ob)])
    H = pm * (1 - pm) * (offc - Pne) ** 2
    E = H + (1 - pm) * S2
    print("\nD. expected (label-free) vs realised squared error on the holdout, by segment x offset band")
    print(f"  {'segment':12s} {'offset':6s} {'n':>7s} {'realised %':>10s} {'expected %':>10s} {'ratio':>6s} {'hedge H %':>9s}")
    for s in ("non-LIRF", "LIRF NM-m", "LIRF NM-unm"):
        for _, _, o in OFF_BANDS:
            k = (sg == s) & (ob == o)
            if k.sum() == 0:
                continue
            print(f"  {s:12s} {o:6s} {k.sum():7,} {100 * se[k].sum() / SSE:10.1f} {100 * E[k].sum() / SSE:10.1f} "
                  f"{E[k].sum() / max(se[k].sum(), 1):6.2f} {100 * H[k].sum() / SSE:9.1f}")
    print(f"  all: realised {SSE:.4g}, expected {E.sum():.4g} (ratio {E.sum() / SSE:.2f}); hedge H {100 * H.sum() / SSE:.1f}% of realised SSE")
    # realised hedge cost: SSE on rows where echo status matters, vs if P were set to the true branch's estimate
    oracle = np.where(echo, offc, R)
    k1 = ob != "<1h"
    print(f"  oracle check (offset >= 1 h rows): realised SSE {100 * se[k1].sum() / SSE:.1f}% of total; with P = offset on echo rows and R on "
          f"the rest it would be {100 * ((y - oracle) ** 2)[k1].sum() / SSE:.1f}%")

    # ------------------------------------------------------------------ E (2026)
    rk = mvt_info(str(ROOT / "data" / "ranking" / "ranking.parquet")).drop("y", "d")
    sub = lambda f, c: pl.read_parquet(SUB / f).select(I64, pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
    r = (rk.join(sub("smart-jigsaw_v42.parquet", "P"), on="MVT_ID_mvt").join(sub("lgb_mixed_nb.parquet", "pred_l"), on="MVT_ID_mvt")
         .join(sub("cat_mixed.parquet", "pred_c"), on="MVT_ID_mvt")
         .join(pl.read_parquet(EVAL / "lgb_direct_nolirf_nb_rank.parquet").select(I64, p_l="echo_prob"), on="MVT_ID_mvt")
         .join(pl.read_parquet(EVAL / "cat_mixed_rank.parquet").select(I64, p_c="echo_prob", raw_c="taxi_model_raw", predc_f="pred"), on="MVT_ID_mvt"))
    assert r.height == 344_841, r.height
    off2, P2 = r["off"].to_numpy(), r["P"].to_numpy()
    pl2, pc2 = r["p_l"].to_numpy(), r["p_c"].to_numpy()
    # the cat submission file vs its rank cache (same model?)
    print(f"\nE. 2026. cat_mixed.parquet vs cat_mixed_rank pred: max |diff| {np.max(np.abs(r['pred_c'].to_numpy() - r['predc_f'].to_numpy())):.2f}")
    Rl2, Rc2 = invert(r["pred_l"].to_numpy(), pl2, off2), invert(r["predc_f"].to_numpy(), pc2, off2)
    ok_l, ok_c = pl2 < 0.99, pc2 < 0.99
    wl, wc = W_LGB * ok_l, W_CAT * ok_c
    R2 = np.where(wl + wc > 0, (wl * np.nan_to_num(Rl2) + wc * np.nan_to_num(Rc2)) / np.maximum(wl + wc, 1e-9), np.nan)
    print(f"  rows with both p >= 0.99 (R unknown, weight (1-p) negligible): {(~ok_l & ~ok_c).sum()}")
    R2 = np.where(np.isnan(R2), P2, R2)
    pm2 = W_LGB * pl2 + W_CAT * pc2
    sg2, ob2 = seg(r), band(off2, OFF_BANDS)
    S22 = np.array([s2[(a, b)] for a, b in zip(sg2, ob2)])
    offc2 = np.clip(off2, 0, WIDE)
    Pne2 = implied(P2, pm2, offc2, R2)
    H2 = pm2 * (1 - pm2) * (offc2 - Pne2) ** 2
    E2 = H2 + (1 - pm2) * S22
    TOT = BOARD ** 2 * r.height
    print(f"  board total SSE {TOT:.4g}; expected total {E2.sum():.4g} (= RMSE {np.sqrt(E2.sum() / r.height):.1f}); "
          f"hedge H {H2.sum():.4g} = {100 * H2.sum() / TOT:.1f}% of the board's SSE")
    print(f"  {'segment':12s} {'offset':6s} {'n26':>7s} {'n25':>7s} {'mean p 26':>9s} {'25':>6s} {'E % board':>9s} {'H % board':>9s} {'H % 2025 SSE':>12s}")
    for s in ("non-LIRF", "LIRF NM-m", "LIRF NM-unm"):
        for _, _, o in OFF_BANDS:
            k2, k = (sg2 == s) & (ob2 == o), (sg == s) & (ob == o)
            if k2.sum() == 0:
                continue
            print(f"  {s:12s} {o:6s} {k2.sum():7,} {k.sum():7,} {pm2[k2].mean():9.3f} {pm[k].mean() if k.sum() else np.nan:6.3f} "
                  f"{100 * E2[k2].sum() / TOT:9.1f} {100 * H2[k2].sum() / TOT:9.1f} {100 * H[k].sum() / SSE:12.1f}")
    print("\n  2026 rows with the largest hedge cost H (top 25):")
    print(f"  {'apt':4s} {'nm-unm':6s} {'H % board':>9s} {'off':>7s} {'P':>7s} {'R':>7s} {'p_l':>5s} {'p_c':>5s}")
    for i in map(int, np.argsort(-H2)[:25]):
        print(f"  {r['apt'][i]:4s} {str(r['nm'][i]):6s} {100 * H2[i] / TOT:9.2f} {off2[i]:7.0f} {P2[i]:7.0f} {R2[i]:7.0f} {pl2[i]:5.2f} {pc2[i]:5.2f}")
    t = np.argsort(-H2)
    for n in (25, 100, 500):
        print(f"  top {n} hedge rows: H = {100 * H2[t[:n]].sum() / TOT:.1f}% of the board's SSE "
              f"(LIRF NM-unmatched among them: {(sg2[t[:n]] == 'LIRF NM-unm').sum()})")


if __name__ == "__main__":
    main()
