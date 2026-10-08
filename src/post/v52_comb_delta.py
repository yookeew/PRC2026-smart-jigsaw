"""v52 production combiner delta (reports/v52_opdi_gate_preregistration.md): v50's full-year combiner (control:
v50's inputs POS + RO) and the same + OPDI ground-event inputs (treatment), fit in one process on the same rows;
writes SCR/comb_delta_opdi.parquet with MVT_ID_mvt, ctl, trt, delta (trt - ctl) for ranking rows the combiner
serves (non-LIRF, ADS-B information), zero elsewhere. Derived from src/post/v50_comb_delta.py.

Run:  PRC_LGB=ctxc1sel SCR=cache/v52 python src/post/v52_comb_delta.py
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import nnls
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
assert os.environ.get("PRC_LGB") == "ctxc1sel"
SCR = Path(os.environ.get("SCR", "cache/v52"))
V50 = ROOT / "cache" / "v50"
import adsb_combiner_test as C  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import ltfm_egll_anatomy as A  # noqa: E402
import adsb_combiner_12m_test as T12  # noqa: E402
import adsb_combiner_fullyear_test as TF  # noqa: E402

POS = ["adsb_first_x", "adsb_first_y", "adsb_first_dx", "adsb_first_dy"]
RO = ["ro_gap_last", "ro_tm_last", "ro_om_last", "ro_two_dep", "ro_tm_next", "ro_gap_prev", "ro_tm_prev",
      "ro_n_since_s", "ro_first_since_s"]
O = ["o_matched", "o_lead_first", "o_lead_exitpark", "o_lead_entrypark", "o_lead_taxi", "o_lead_apron",
     "o_lead_rwy", "o_deice", "o_n_ground"]
REL = {"o_rel_first": "o_lead_first", "o_rel_exitpark": "o_lead_exitpark", "o_rel_taxi": "o_lead_taxi"}
CTL_EXTRA = POS + RO                                  # v50's combiner inputs
ARM = "opdi"
EXTRA = POS + RO + O + list(REL)
ENS_ARMS = set()
OPDI_FEATS = ROOT / "cache" / "opdi_feats" / "opdi_feats.parquet"
_fit0 = C.fit_combiner


class Ens2:
    def __init__(self, g, per):
        self.g, self.per = g, per

    def predict(self, X):
        out = self.g.predict(X)
        for code, m in self.per.items():
            k = X[:, 0] == code
            if k.any():
                out[k] = 0.5 * out[k] + 0.5 * m.predict(X[k])
        return out


def fit_ens2(X, y, inner):
    g, best = _fit0(X, y, inner)
    per = {}
    for code in np.unique(X[:, 0]):
        k = X[:, 0] == code
        if k.sum() >= 5000 and inner[k].any() and (~inner[k]).any():
            per[code], _ = _fit0(X[k], y[k], inner[k])
    return Ens2(g, per), best
DET0, FEATS0 = list(V.DET_COLS), list(C.FEATS)
POSDIR = ROOT / "cache" / "adsb_pos_v4"
C.V4 = POSDIR
V.DET_COLS = DET0 + POS
SUB = ROOT / "data" / "submissions"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
ENG = ["lgb", "catcorr"]
CEIL, LIRF_CEIL = 10800, 140000


def read_det(d, pat="day=2025-0[17]*.parquet"):
    det = pl.concat([pl.read_parquet(p) for p in sorted(d.glob(pat))], how="diagonal")
    for c in V.DET_COLS:
        if c not in det.columns:
            det = det.with_columns(pl.lit(None, pl.Float64).alias(c))
    return det.select("MVT_ID_mvt", *V.DET_COLS, "adsb_fallback")


def with_det(base, det):
    drop = [c for c in V.DET_COLS + ["adsb_taxi"] if c in base.columns]
    return (base.drop(drop).join(det, on="MVT_ID_mvt", how="left")
            .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")))


def feats(df, s, airports, extra):
    saved, C.FEATS = C.FEATS, FEATS0          # the original builder selects C.FEATS at its end
    try:
        X = C.features(df, s, airports)
    finally:
        C.FEATS = saved
    if extra:
        d = df.with_columns(pl.Series("_s", np.asarray(s, dtype=np.float64)))
        X = pl.concat([X, d.select([(pl.col(REL[c]) - pl.col("_s")).alias(c) if c in REL else pl.col(c).cast(pl.Float64)
                                    for c in extra])], how="horizontal")
    return X.to_numpy().astype(np.float64)


def main():
    ro = pl.read_parquet(V50 / "reocc_feats.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    of = pl.read_parquet(OPDI_FEATS).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).select("MVT_ID_mvt", *O)
    ro = ro.join(of, on="MVT_ID_mvt", how="full", coalesce=True)
    SCR.mkdir(parents=True, exist_ok=True)
    ho = with_det(A.load(), read_det(POSDIR)).join(ro, on="MVT_ID_mvt", how="left")
    w, _ = nnls(ho.select(ENG).to_numpy(), ho["taxi"].to_numpy())
    print("stack weights:", dict(zip(ENG, np.round(w, 4))))
    airports = sorted(ho["ADEP_mvt"].unique().to_list())
    s_ho = ho.select(ENG).to_numpy() @ w
    fit = ((ho["taxi"] <= C.TRIM) & (ho["ADEP_mvt"] != "LIRF")).to_numpy()
    T12.MONTHS = TF.FULL
    ex = T12.oof_frame().join(ro, on="MVT_ID_mvt", how="left")
    ex_rows = ((ex["taxi"] <= C.TRIM) & (ex["ADEP_mvt"] != "LIRF")).to_numpy()
    y = np.concatenate([(ho["taxi"].to_numpy() - s_ho)[fit], (ex["taxi"] - ex["s"]).to_numpy()[ex_rows]])
    inner = np.concatenate([(ho["day"].dt.day() % 5 == 0).to_numpy()[fit], (ex["day"].dt.day() % 5 == 0).to_numpy()[ex_rows]])
    # ranking frame (as adsb_combiner.ranking_frame, detector = position cache)
    rk = None
    for e in ENG:
        s = pl.read_parquet(SUB / f"{os.environ['PRC_LGB'] if e == 'lgb' else e + '_mixed'}.parquet").select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(e))
        rk = s if rk is None else rk.join(s, on="MVT_ID_mvt", how="inner")
    mv = (pl.read_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
                  mvt_ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0))
    det = read_det(POSDIR, "day=2026-*.parquet")
    assert det.height > 340000, det.height
    rk = (rk.join(mv, on="MVT_ID_mvt", how="left").join(det, on="MVT_ID_mvt", how="left")
          .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")).join(ro, on="MVT_ID_mvt", how="left"))
    s_rk = rk.select(ENG).to_numpy() @ w
    out = {}
    for arm, extra in (("ctl", CTL_EXTRA), (ARM, EXTRA)):
        C.FEATS = FEATS0 + extra
        X = np.vstack([feats(ho, s_ho, airports, extra)[fit], feats(ex, ex["s"].to_numpy(), airports, extra)[ex_rows]])
        m, best = (fit_ens2 if arm in ENS_ARMS else _fit0)(X, y, inner)
        hi = np.where(rk["ADEP_mvt"].to_numpy() == "LIRF", LIRF_CEIL, CEIL)
        out[arm] = np.clip(s_rk + m.predict(feats(rk, s_rk, airports, extra)), 0, hi)
        print(f"{arm}: best_iter {best}, rows {X.shape}", flush=True)
    C.FEATS = FEATS0
    has = (rk["adsb_matched"].fill_null(False) | rk["adsb_tier"].is_in(["fb_dwell", "fb_appear"]).fill_null(False)).to_numpy()
    serve = has & (rk["ADEP_mvt"] != "LIRF").to_numpy()
    delta = np.where(serve, out[ARM] - out["ctl"], 0.0)
    res = rk.select("MVT_ID_mvt", "ADEP_mvt").with_columns(ctl=pl.Series(out["ctl"]), trt=pl.Series(out[ARM]),
                                                           serve=pl.Series(serve), delta=pl.Series(delta))
    res.write_parquet(SCR / f"comb_delta_{ARM}.parquet")
    print(res.group_by("ADEP_mvt").agg(n=pl.col("serve").sum(), rms=(pl.col("delta") ** 2).mean().sqrt(),
                                       mean=pl.col("delta").mean()).sort("ADEP_mvt"))
    print(f"delta RMS over all rows {np.sqrt(np.mean(delta ** 2)):.2f}, served {serve.sum()}")


if __name__ == "__main__":
    main()
