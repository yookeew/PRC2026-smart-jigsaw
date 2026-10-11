"""G2 refit on the current corrector (reports/day_alloc_refit_preregistration.md).

Production G2 (v63-v65) was fitted on the 2025 frame of the old corrector `ctxc1sel` (v50's holdout chain + its OOF folds):
both the slope model s(x) and the residual distributions (eta) of the daily lambda solve. Production applies it on top of a
newer corrector. Frame B = the same 2025 rows with p from the production corrector (PRC_G2_HO / PRC_G2_OOF given here as
B_HO / B_OOF). Both arms are evaluated on frame B, leave-period-out (six periods, as tests/day_alloc_gbm.py):
  OLD : eta from frame A's calibration rows, slope model fitted on frame A (its lambda, inputs, residuals) -- production now
  NEW : eta from frame B's calibration rows, slope model fitted on frame B
Rows: the eight shifted airports (LSZH / LIRF are not shifted by G2).

Run:  B_HO=cache/v65/gate_rh2/chain_all.parquet B_OOF=cache/oof/ctxc1sel_rh2 B_LAM=cache/v65/lam_row_2025_rh2.parquet \
      .venv/Scripts/python.exe tests/day_alloc_refit.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
assert "PRC_G2_HO" not in os.environ and "PRC_G2_OOF" not in os.environ and "PRC_G2_LAM" not in os.environ
from post import pru_dsm as M  # noqa: E402
import v57_dsm as T  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402
import day_alloc_gbm as G  # noqa: E402

B_HO, B_OOF, B_LAM = os.environ["B_HO"], os.environ["B_OOF"], ROOT / os.environ["B_LAM"]
BOARD = float(os.environ.get("BOARD", "237.4538"))


def lam_with(xt: pl.DataFrame, xc: pl.DataFrame) -> np.ndarray:
    """L.lam_rows on target frame xt with the calibration rows (eta) taken from frame xc (same rows, other p)."""
    mo = xt["mo"].to_numpy()
    lam = np.full(xt.height, np.nan)
    for pk, (m1, m2) in T.PERIODS.items():
        for m in (m1, m2):
            mk = mo == m
            cal = xc.filter(pl.col("mo").is_in([m1, m2]) & (pl.col("mo") != m)) if pk == "P01" else xc.filter(~pl.col("period").is_in(["P01", pk]))
            xm = xt.filter(pl.Series(mk)); idx = np.flatnonzero(mk)
            for d in M.day_table(xm, T.dists_from(cal), set(T.A8)):
                lam[idx[d["g"]]] = d["lam"]
    return lam


def main() -> None:
    xa = T.frame25()
    xb = T.frame25(ho_path=B_HO, oof_dir=B_OOF)
    assert xa.height == xb.height and (xa["MVT_ID_mvt"] == xb["MVT_ID_mvt"]).all(), "frames must hold the same rows"
    for c in ("y", "apt", "mo", "period", "trk", "nmu", "ts"):
        assert (xa[c] == xb[c]).all(), c
    nd = int((xa["day"] != xb["day"]).sum())
    print(f"rows {xa.height:,}; p change A -> B: RMS {np.sqrt(np.mean((xa['p'] - xb['p']).to_numpy() ** 2)):.1f} s; "
          f"rows whose predicted off-block day moves {nd:,}", flush=True)
    lam_a, _ = L.lam_rows(xa)                      # production cache (frame A, eta from A)
    if B_LAM.exists() and pl.read_parquet(B_LAM).height == xb.height:
        c = pl.read_parquet(B_LAM)
        assert (c["MVT_ID_mvt"] == xb["MVT_ID_mvt"]).all()
        lam_bn, lam_bo = c["lam_new"].to_numpy(), c["lam_old"].to_numpy()
    else:
        lam_bn, lam_bo = lam_with(xb, xb), lam_with(xb, xa)
        pl.DataFrame({"MVT_ID_mvt": xb["MVT_ID_mvt"], "lam_new": lam_bn, "lam_old": lam_bo}).write_parquet(B_LAM)
    print("lambda done", flush=True)

    y, pa, pb = xb["y"].to_numpy(), xa["p"].to_numpy().astype(float), xb["p"].to_numpy().astype(float)
    apt, mo, per, nmu = xb["apt"].to_numpy(), xb["mo"].to_numpy(), xb["period"].to_numpy(), xb["nmu"].to_numpy()
    in8 = np.isin(apt, T.A8)
    trim, body = y <= 18000, ~nmu & (y <= 3600)
    z = lambda v: np.where(np.isfinite(v), v, 0.0)  # noqa: E731
    la, lbn, lbo = z(lam_a), z(lam_bn), z(lam_bo)
    Xa, Xbn, Xbo = G.inputs(xa, la, True), G.inputs(xb, lbn, True), G.inputs(xb, lbo, True)
    q = {"OLD": pb.copy(), "NEW": pb.copy()}
    for pk in T.PERIODS:
        k = in8 & (per == pk)
        ma = in8 & np.isfinite(lam_a) & trim & (per != pk) & (np.abs(la) >= 1)
        mb = in8 & np.isfinite(lam_bn) & trim & (per != pk) & (np.abs(lbn) >= 1)
        ba, bb = G.slope_fit(Xa, pa - y, la, ma), G.slope_fit(Xbn, pb - y, lbn, mb)
        ko, kn = k & np.isfinite(lam_bo), k & np.isfinite(lam_bn)
        q["OLD"][ko] = np.maximum(pb[ko] - np.clip(ba.predict(Xbo[ko], raw_score=True) + G.INIT, 0, 1) * lbo[ko], 0)
        q["NEW"][kn] = np.maximum(pb[kn] - np.clip(bb.predict(Xbn[kn], raw_score=True) + G.INIT, 0, 1) * lbn[kn], 0)
        print(f"  {pk} done", flush=True)

    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    o, n = q["OLD"], q["NEW"]
    print(f"\nframe B, 8 airports: no day stage {rm(pb, in8):.3f}, OLD {rm(o, in8):.3f}, NEW {rm(n, in8):.3f}")
    res = {"full": rm(n, in8) - rm(o, in8), "body": rm(n, in8 & body) - rm(o, in8 & body), "trim": rm(n, in8 & trim) - rm(o, in8 & trim)}
    months = sorted(set(mo))
    mm = {m: rm(n, in8 & (mo == m)) - rm(o, in8 & (mo == m)) for m in months}
    print("NEW - OLD:", {k: round(v, 3) for k, v in res.items()})
    print("by month (full):", {m[5:]: round(v, 2) for m, v in mm.items()})
    print("per airport (full):", {a: round(rm(n, apt == a) - rm(o, apt == a), 2) for a in T.A8})
    print("per airport (body):", {a: round(rm(n, (apt == a) & body) - rm(o, (apt == a) & body), 2) for a in T.A8})
    better = sum(v < 0 for v in mm.values())
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = dict(rk.group_by("ADEP_mvt").agg(n=pl.len()).iter_rows())
    dmse = sum(n26[a] * np.mean((y[apt == a] - n[apt == a]) ** 2 - (y[apt == a] - o[apt == a]) ** 2) for a in T.A8) / sum(n26.values())
    print(f"2026 projection: {np.sqrt(BOARD ** 2 + dmse) - BOARD:+.3f} s")
    ok = {"full <= -0.05": res["full"] <= -0.05, "body <= 0": res["body"] <= 0, "trimmed <= 0": res["trim"] <= 0,
          ">= 8 of 12 months better": better >= 8, "Feb <= 0 and Jun <= 0": mm["2025-02"] <= 0 and mm["2025-06"] <= 0}
    print("conditions:", ok, f"(months better {better})")
    print("DECISION:", "PASS" if all(ok.values()) else "FAIL")


if __name__ == "__main__":
    main()
