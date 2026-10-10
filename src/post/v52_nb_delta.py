"""v52 production neighbour-stage (part b, --v2) delta (reports/v52_opdi_gate_preregistration.md): control =
v50's inputs (+ stand re-occupation), treatment = + OPDI ground-event inputs, both fit in one process on the
production rows (Jan+Jul 2025 holdout with the production stack + 115 OOF ADS-B days). Writes
SCR/nb_delta_opdi.parquet (MVT_ID_mvt, delta) for ranking rows without ADS-B information, zero elsewhere.
Derived from src/post/v50_nb_delta.py.

Run:  PRC_LGB=ctxc1sel SCR=cache/v52 python src/post/v52_nb_delta.py
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import nnls
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
assert os.environ.get("PRC_LGB", "").startswith("ctxc1sel")   # ctxc1sel (v52), or ctxc1sel_ro (four-month chain, §132)
import scope  # noqa: E402  (PRC_SCOPE=final4m: final_ranking rows, final4m caches/outputs; PROGRESS.md §132)
SCR = Path(os.environ.get("SCR", str(scope.out_dir(Path("cache/v52")))))
V50 = scope.out_dir(ROOT / "cache" / "v50")
sys.argv = [sys.argv[0]]                       # post.adsb_combiner reads flags at import: none -> v4 detections
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402
import neighbour_state_test as N  # noqa: E402
import neighbour_state_v2_test as N2  # noqa: E402
from post import adsb_combiner as PC  # noqa: E402

RO = ["ro_gap_last", "ro_tm_last", "ro_om_last", "ro_two_dep", "ro_tm_next", "ro_gap_prev", "ro_tm_prev",
      "ro_n_since_s", "ro_first_since_s"]
O = ["o_matched", "o_lead_first", "o_lead_exitpark", "o_lead_entrypark", "o_lead_taxi", "o_lead_apron",
     "o_lead_rwy", "o_deice", "o_n_ground"]
REL = {"o_rel_first": "o_lead_first", "o_rel_exitpark": "o_lead_exitpark", "o_rel_taxi": "o_lead_taxi"}
OPDI_FEATS = scope.out_dir(ROOT / "cache" / "opdi_feats") / "opdi_feats.parquet"
RANKING = scope.RANKING
ENG = ["lgb", "catcorr"]


def xb(df, s, nb, airports, extra):
    X = N.x_b(df, s, nb, airports)
    if not extra:
        return X
    d = df.with_columns(pl.Series("_s", np.asarray(s, dtype=np.float64)))
    E = d.select([(pl.col(REL[c]) - pl.col("_s")).alias(c) if c in REL else pl.col(c).cast(pl.Float64) for c in extra])
    return np.column_stack([X, E.to_numpy()])


def main():
    ro = pl.read_parquet(V50 / "reocc_feats.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    of = pl.read_parquet(OPDI_FEATS).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).select("MVT_ID_mvt", *O)
    ro = ro.join(of, on="MVT_ID_mvt", how="full", coalesce=True)
    SCR.mkdir(parents=True, exist_ok=True)
    ho = V.with_det(A.load(), V.read_det(C.V4)).join(ro, on="MVT_ID_mvt", how="left")
    w, _ = nnls(ho.select(ENG).to_numpy(), ho["taxi"].to_numpy())
    print("stack weights:", dict(zip(ENG, np.round(w, 4))))
    airports = sorted(ho["ADEP_mvt"].unique().to_list())
    ex, lag = N.train_extra()
    ex = ex.join(ro, on="MVT_ID_mvt", how="left")
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr25 = N.arrivals(paths, paths)
    s_ho, s_ex = ho.select(ENG).to_numpy() @ w, ex["s"].to_numpy()
    nb_ho, nb_ex = N.nb_frame(ho, s_ho, lag, arr25), N.nb_frame(ex, s_ex, lag, arr25)
    ok = np.concatenate([(ho["taxi"] <= C.TRIM).to_numpy(), (ex["taxi"] <= C.TRIM).to_numpy()])
    y = np.concatenate([ho["taxi"].to_numpy() - s_ho, (ex["taxi"] - ex["s"]).to_numpy()])
    inner = np.concatenate([(ho["day"].dt.day() % 5 == 0).to_numpy(), (ex["day"].dt.day() % 5 == 0).to_numpy()])
    rk = PC.ranking_frame()
    raw = pl.read_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "RUNWAY_mvt",
        aobt3_taxi=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
        offset=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
    rk = rk.join(raw, on="MVT_ID_mvt", how="left").join(ro, on="MVT_ID_mvt", how="left")
    arr26 = N.arrivals([RANKING], paths)
    s_rk = rk.select(ENG).to_numpy() @ w
    nb_rk = N.nb_frame(rk, s_rk, lag, arr26)
    pop = rk.select(C2.has_adsb()).to_series().to_numpy()
    day = pl.from_epoch(rk["mvt_ts"].cast(pl.Int64), "s").dt.date().to_numpy()
    out = {}
    for arm, extra in (("ctl", RO), ("opdi", RO + O + list(REL))):
        X = np.vstack([xb(ho, s_ho, nb_ho, airports, extra), xb(ex, s_ex, nb_ex, airports, extra)])
        mb, bb = N.fit(X[ok], y[ok], inner[ok], N.CAT_B)
        c = mb.predict(xb(rk, s_rk, nb_rk, airports, extra))
        c = N2.day_demean(c, rk["ADEP_mvt"].to_numpy(), day, ~pop)
        out[arm] = np.clip(s_rk + c, 0, N.CEIL)
        print(f"{arm}: part (b) best_iter {bb}", flush=True)
    delta = np.where(~pop, out["opdi"] - out["ctl"], 0.0)
    res = rk.select("MVT_ID_mvt", "ADEP_mvt").with_columns(ctl=pl.Series(out["ctl"]), trt=pl.Series(out["opdi"]),
                                                           served=pl.Series(~pop), delta=pl.Series(delta))
    res.write_parquet(SCR / "nb_delta_opdi.parquet")
    print(res.group_by("ADEP_mvt").agg(n=pl.col("served").sum(), rms=(pl.col("delta") ** 2).mean().sqrt(),
                                       mean=pl.col("delta").mean()).sort("ADEP_mvt"))
    print(f"delta RMS over all rows {np.sqrt(np.mean(delta ** 2)):.2f}, served {(~pop).sum()}")


if __name__ == "__main__":
    main()
