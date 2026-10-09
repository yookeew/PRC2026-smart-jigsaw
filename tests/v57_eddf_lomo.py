"""v57 leave-one-month-out run of v52's production post-stage recipe (reports/v57_eddf_opdi_preregistration.md, G).

Training rows as src/post/v52_comb_delta.py / v52_nb_delta.py (Jan+Jul 2025 holdout + the 115 OOF ADS-B days), minus
every row of the held-out month m. Arms: ctl = v50's inputs (combiner POS + RO, stage b RO); trt = ctl + OPDI (O + REL).
Held-out rows with ADS-B information (non-LIRF) get s + combiner, the others s + stage b (day-demeaned over non-ADS-B
rows per airport-day). Writes SCR/lomo_<m>.parquet: MVT_ID_mvt, ADEP_mvt, ym, day, taxi, nmu, s, served, ctl[, trt].

The EDDF build was stopped by witness W3 (amendment W-A2); this run supplies part B's chain-like OOF predictions
(ctl) and reports the EDDF treatment as information only.

Run:  PRC_LGB=ctxc1sel SCR=cache/v57 MONTHS=all TRT_MONTHS=2025-09,2025-10,2025-11,2025-12 python tests/v57_eddf_lomo.py
"""
import os, sys, time
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import nnls
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from post import v52_comb_delta as CD  # noqa: E402  (sets C.V4 = position cache, V.DET_COLS += POS; asserts PRC_LGB)
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402
import neighbour_state_test as N  # noqa: E402
import neighbour_state_v2_test as N2  # noqa: E402

SCR = Path(os.environ.get("SCR", "cache/v57"))
OPDI = CD.O + list(CD.REL)
ARMS = {"ctl": (CD.POS + CD.RO, CD.RO), "trt": (CD.POS + CD.RO + OPDI, CD.RO + OPDI)}
FULL = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]


def xb(df, s, nb, airports, extra):
    X = N.x_b(df, s, nb, airports)
    d = df.with_columns(pl.Series("_s", np.asarray(s, dtype=np.float64)))
    E = d.select([(pl.col(CD.REL[c]) - pl.col("_s")).alias(c) if c in CD.REL else pl.col(c).cast(pl.Float64) for c in extra])
    return np.column_stack([X, E.to_numpy()])


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    ro = pl.read_parquet(CD.V50 / "reocc_feats.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    of = pl.read_parquet(CD.OPDI_FEATS).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).select("MVT_ID_mvt", *CD.O)
    ro = ro.join(of, on="MVT_ID_mvt", how="full", coalesce=True)
    ho = CD.with_det(A.load(), CD.read_det(CD.POSDIR)).join(ro, on="MVT_ID_mvt", how="left")
    w, _ = nnls(ho.select(CD.ENG).to_numpy(), ho["taxi"].to_numpy())
    s_ho = ho.select(CD.ENG).to_numpy() @ w
    airports = sorted(ho["ADEP_mvt"].unique().to_list())
    ex, lag = N.train_extra()                       # OOF ADS-B days (T12.MONTHS = FULL), s = OOF stack
    ex = ex.join(ro, on="MVT_ID_mvt", how="left")
    s_ex = ex["s"].to_numpy()
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr = N.arrivals(paths, paths)
    nb_ho, nb_ex = N.nb_frame(ho, s_ho, lag, arr), N.nb_frame(ex, s_ex, lag, arr)
    nmu = pl.scan_parquet(str(N.RAW / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), nmu=pl.col("FLIGHT_ID_mvt").is_null()).collect()
    ex = ex.join(nmu, on="MVT_ID_mvt", how="left")
    print(f"loaded in {time.time() - t0:.0f}s: holdout {ho.height:,} rows (stack w {np.round(w, 3)}), OOF ADS-B days "
          f"{ex.height:,} rows on {ex['day'].n_unique()} days", flush=True)

    # fixed pieces
    ho_ok_c = ((ho["taxi"] <= C.TRIM) & (ho["ADEP_mvt"] != "LIRF")).to_numpy()
    ex_ok_c = ((ex["taxi"] <= C.TRIM) & (ex["ADEP_mvt"] != "LIRF")).to_numpy()
    ho_ok_b, ex_ok_b = (ho["taxi"] <= C.TRIM).to_numpy(), (ex["taxi"] <= C.TRIM).to_numpy()
    y_ho, y_ex = ho["taxi"].to_numpy() - s_ho, (ex["taxi"] - ex["s"]).to_numpy()
    i_ho, i_ex = (ho["day"].dt.day() % 5 == 0).to_numpy(), (ex["day"].dt.day() % 5 == 0).to_numpy()
    pop = ex.select(C2.has_adsb()).to_series().to_numpy()
    nonl = (ex["ADEP_mvt"] != "LIRF").to_numpy()
    apt, day, ym = ex["ADEP_mvt"].to_numpy(), ex["day"].to_numpy(), ex["ym"].to_numpy()
    months = FULL if os.environ.get("MONTHS", "all") == "all" else os.environ["MONTHS"].split(",")
    trt_months = set(os.environ.get("TRT_MONTHS", "2025-09,2025-10,2025-11,2025-12").split(","))
    X = {}
    for arm, (cx, bx) in ARMS.items():
        C.FEATS = CD.FEATS0 + cx
        X[arm] = (CD.feats(ho, s_ho, airports, cx), CD.feats(ex, s_ex, airports, cx),
                  xb(ho, s_ho, nb_ho, airports, bx), xb(ex, s_ex, nb_ex, airports, bx))
    C.FEATS = CD.FEATS0
    for m in months:
        t1 = time.time()
        inm = ym == m
        serve_c, serve_b = inm & pop & nonl, inm & ~pop
        res = ex.filter(pl.Series(inm)).select("MVT_ID_mvt", "ADEP_mvt", "ym", "day", "taxi", "nmu", "s").with_columns(
            served=pl.Series(np.where(serve_c, "comb", np.where(serve_b, "b", "none"))[inm]))
        for arm in (["ctl", "trt"] if m in trt_months else ["ctl"]):
            Xc_ho, Xc_ex, Xb_ho, Xb_ex = X[arm]
            C.FEATS = CD.FEATS0 + ARMS[arm][0]
            tr_c = ex_ok_c & ~inm
            mc, bc = CD._fit0(np.vstack([Xc_ho[ho_ok_c], Xc_ex[tr_c]]), np.concatenate([y_ho[ho_ok_c], y_ex[tr_c]]),
                              np.concatenate([i_ho[ho_ok_c], i_ex[tr_c]]))
            C.FEATS = CD.FEATS0
            tr_b = ex_ok_b & ~inm
            mb, bb = N.fit(np.vstack([Xb_ho[ho_ok_b], Xb_ex[tr_b]]), np.concatenate([y_ho[ho_ok_b], y_ex[tr_b]]),
                           np.concatenate([i_ho[ho_ok_b], i_ex[tr_b]]), N.CAT_B)
            pred = s_ex.copy()
            pred[serve_c] = s_ex[serve_c] + mc.predict(Xc_ex[serve_c])
            cb = np.zeros(len(pred))
            cb[serve_b] = mb.predict(Xb_ex[serve_b])
            cb = N2.day_demean(cb, apt, day, serve_b)
            pred[serve_b] = s_ex[serve_b] + cb[serve_b]
            pred = np.clip(pred, 0, N.CEIL)
            res = res.with_columns(pl.Series(arm, pred[inm]))
            print(f"  {m} {arm}: combiner best_iter {bc}, stage b best_iter {bb}", flush=True)
        res.write_parquet(SCR / f"lomo_{m}.parquet")
        print(f"month {m}: {inm.sum():,} rows ({serve_c.sum():,} combiner, {serve_b.sum():,} stage b), {time.time() - t1:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
