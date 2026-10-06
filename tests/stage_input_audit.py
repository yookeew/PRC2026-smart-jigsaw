"""Input-mismatch audit of the trained post-stages on 2026 rows (PROGRESS.md §94, step 1). No training.

For the ADS-B combiner (`src/post/adsb_combiner.py --full`) and the neighbour-state part (b) corrector
(`src/post/neighbour_state.py --v2`), the training matrix and the 2026 ranking matrix are built with the
stages' own code, then compared:
  - rows whose missing-value pattern never occurs in the training rows (by airport x population);
  - per feature and airport, the share of 2026 values outside that airport's training [p0.5, p99.5];
  - training support of airport x ADS-B tier cells against their 2026 counts.
The stack input is v41's corrector (PRC_LGB=ctxsel); v42's Arm C corrector only changes the stack value
on NM-unmatched rows, which affects the `s`-derived columns there and nothing structural.

Run:  PRC_LGB=ctxsel .venv/Scripts/python.exe -u tests/stage_input_audit.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0], "--full", "--v2"]          # the production flags of both stages
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402
import neighbour_state_test as N  # noqa: E402
from post import adsb_combiner as PC  # noqa: E402

RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
ENG = ["lgb", "catcorr"]


def novelty(name, Xtr, Xrk, cols, apt_rk, groups):
    pt = np.packbits(np.isnan(Xtr), axis=1); pr = np.packbits(np.isnan(Xrk), axis=1)
    seen = {r.tobytes() for r in pt}
    new = np.array([r.tobytes() not in seen for r in pr])
    print(f"\n[{name}] 2026 rows with a never-seen missing pattern: {new.sum():,} of {len(new):,} ({new.mean():.2%})")
    for gname, g in groups.items():
        for a in sorted(set(apt_rk)):
            m = (apt_rk == a) & g
            if m.sum() and new[m].mean() > 0.005:
                print(f"   {a} {gname:14s} {new[m].sum():6d} of {m.sum():6d} ({new[m].mean():6.2%})")
    return new


def out_of_range(name, Xtr, Xrk, cols, apt_tr, apt_rk, skip=()):
    print(f"[{name}] share of 2026 values outside the airport's 2025 training [p0.5, p99.5] (cells > 3%):")
    hits = []
    for j, c in enumerate(cols):
        if c in skip:
            continue
        for a in sorted(set(apt_rk)):
            xt = Xtr[apt_tr == a, j]; xr = Xrk[apt_rk == a, j]
            xt, xr = xt[~np.isnan(xt)], xr[~np.isnan(xr)]
            if len(xt) < 200 or len(xr) < 50:
                continue
            lo, hi = np.quantile(xt, [0.005, 0.995])
            share = np.mean((xr < lo) | (xr > hi))
            if share > 0.03:
                hits.append((share, c, a, len(xr)))
    for share, c, a, n in sorted(hits, reverse=True)[:25]:
        print(f"   {c:20s} {a}  {share:6.1%} of {n:,}  (expected ~1%)")
    if not hits:
        print("   none")


def main() -> None:
    ho = V.with_det(A.load(), V.read_det(C.V4))
    w, _ = nnls(ho.select(ENG).to_numpy(), ho["taxi"].to_numpy())
    airports = sorted(ho["ADEP_mvt"].unique().to_list())
    import adsb_combiner_12m_test as T12
    import adsb_combiner_fullyear_test as TF
    T12.MONTHS = TF.FULL
    ex = T12.oof_frame()
    s_ho = ho.select(ENG).to_numpy() @ w
    rk = PC.ranking_frame()
    s_rk = rk.select(ENG).to_numpy() @ w

    # ---- combiner: training = holdout non-LIRF rows <= TRIM + OOF-month rows <= TRIM non-LIRF
    f = lambda d, s: C.features(d, s, airports).to_numpy().astype(np.float64)
    fit_ho = ((ho["taxi"] <= C.TRIM) & (ho["ADEP_mvt"] != "LIRF")).to_numpy()
    fit_ex = ((ex["taxi"] <= C.TRIM) & (ex["ADEP_mvt"] != "LIRF")).to_numpy()
    Xtr = np.vstack([f(ho, s_ho)[fit_ho], f(ex, ex["s"].to_numpy())[fit_ex]])
    apt_tr = np.concatenate([ho["ADEP_mvt"].to_numpy()[fit_ho], ex["ADEP_mvt"].to_numpy()[fit_ex]])
    Xrk = f(rk, s_rk)
    apt_rk = rk["ADEP_mvt"].to_numpy()
    has = rk.select(C2.has_adsb()).to_series().to_numpy()
    applied = (apt_rk != "LIRF")        # the combiner serves every non-LIRF row (LIRF keeps v23)
    cols = C.FEATS
    novelty("combiner", Xtr, Xrk[applied], cols, apt_rk[applied], {"with ADS-B": has[applied], "no ADS-B": ~has[applied]})
    out_of_range("combiner", Xtr, Xrk[applied], cols, apt_tr, apt_rk[applied], skip=("airport", "tier"))
    tier_i = cols.index("tier")
    print("[combiner] airport x tier: training rows -> 2026 rows (cells where 2026 >= 500):")
    for a in airports:
        if a == "LIRF":
            continue
        for t, tn in enumerate(C.TIERS):
            ntr = int(((apt_tr == a) & (Xtr[:, tier_i] == t)).sum()); nrk = int(((apt_rk == a) & (Xrk[:, tier_i] == t)).sum())
            if nrk >= 500:
                flag = "  <-- thin" if ntr < nrk / 5 else ""
                print(f"   {a} {tn:10s} {ntr:7,} -> {nrk:7,}{flag}")

    # ---- neighbour state part (b): training = holdout + OOF rows <= TRIM (all rows); applied = 2026 rows without ADS-B info
    ex, lag = N.train_extra()          # the neighbour stage's own OOF-month frame
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr25, arr26 = N.arrivals(paths, paths), N.arrivals([RANKING], paths)
    raw = pl.read_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "RUNWAY_mvt",
        aobt3_taxi=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
        offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
    rk2 = rk.join(raw, on="MVT_ID_mvt", how="left")
    ok_ho, ok_ex = (ho["taxi"] <= C.TRIM).to_numpy(), (ex["taxi"] <= C.TRIM).to_numpy()
    Xb_tr = np.vstack([N.x_b(ho, s_ho, N.nb_frame(ho, s_ho, lag, arr25), airports)[ok_ho],
                       N.x_b(ex, ex["s"].to_numpy(), N.nb_frame(ex, ex["s"].to_numpy(), lag, arr25), airports)[ok_ex]])
    has_tr = np.concatenate([ho.select(C2.has_adsb()).to_series().to_numpy()[ok_ho], ex.select(C2.has_adsb()).to_series().to_numpy()[ok_ex]])
    aptb_tr = np.concatenate([ho["ADEP_mvt"].to_numpy()[ok_ho], ex["ADEP_mvt"].to_numpy()[ok_ex]])
    Xb_rk = N.x_b(rk2, s_rk, N.nb_frame(rk2, s_rk, lag, arr26), airports)
    pop = rk2.select(C2.has_adsb()).to_series().to_numpy()
    bcols = N.B_FEATS + N.NB_FEATS
    nm_rk = rk2["aobt3_taxi"].is_null().to_numpy()
    app = ~pop
    print(f"\n[neighbour (b)] training rows {len(Xb_tr):,} (of which without ADS-B info {(~has_tr).sum():,}); "
          f"applied to 2026 rows without ADS-B info: {app.sum():,}")
    print("[neighbour (b)] rows without ADS-B info, training -> 2026, by airport:")
    for a in airports:
        print(f"   {a}  {int(((aptb_tr == a) & ~has_tr).sum()):7,} -> {int(((rk2['ADEP_mvt'].to_numpy() == a) & app).sum()):7,}")
    novelty("neighbour (b)", Xb_tr, Xb_rk[app], bcols, rk2["ADEP_mvt"].to_numpy()[app],
            {"NM-matched": ~nm_rk[app], "NM-unmatched": nm_rk[app]})
    out_of_range("neighbour (b)", Xb_tr, Xb_rk[app], bcols, aptb_tr, rk2["ADEP_mvt"].to_numpy()[app], skip=("airport",))


if __name__ == "__main__":
    main()
