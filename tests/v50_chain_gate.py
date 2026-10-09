"""v42's holdout chain (ctxc1sel through v30's method + LOBT window + NM lines), several arms in ONE process.

Arms (env ARMS, comma list; 'ctl' is always run first):
  ctl    : production combiner + neighbour corrector inputs (should reproduce cache/eval/v42_chain_holdout)
  pos    : combiner + first-sighting position (adsb_first_x/y/dx/dy)
  ro_c   : combiner + stand re-occupation features
  ro_b   : neighbour corrector (part b) + stand re-occupation features
  all    : pos + ro_c + ro_b
Writes SCR/chain_<arm>.parquet (MVT_ID_mvt, pred) with LIRF NM-unmatched rows from v41's chain (as v42).

Run:  PRC_LGB=ctxcorr2r ARMS=ctl,pos python chain_gate.py
"""
import os, sys, time
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import nnls
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
assert os.environ.get("PRC_LGB") == "ctxcorr2r"
SCR = Path(os.environ.get("SCR", "cache/v50"))
EVAL, OOF = ROOT / "cache" / "eval", ROOT / "cache" / "oof"
CORR = os.environ.get("CORR", "ctxc1sel")
import adsb_combiner_test as C  # noqa: E402
import adsb_v3_test as V  # noqa: E402
import base_neighbour_chain_test as B  # noqa: E402
import neighbour_state_test as N  # noqa: E402
from post import nm_lines as NL  # noqa: E402

POS = ["adsb_first_x", "adsb_first_y", "adsb_first_dx", "adsb_first_dy"]
RO = ["ro_gap_last", "ro_tm_last", "ro_om_last", "ro_two_dep", "ro_tm_next", "ro_gap_prev", "ro_tm_prev",
      "ro_n_since_s", "ro_first_since_s"]
DET0, FEATS0 = list(V.DET_COLS), list(C.FEATS)
C.V4 = ROOT / "cache" / "adsb_pos_v4"
V.DET_COLS = DET0 + POS


def read_det(d):
    det = pl.concat([pl.read_parquet(p) for p in sorted(d.glob("day=2025-0[17]*.parquet"))], how="diagonal")
    if "adsb_fallback" not in det.columns:
        det = det.with_columns(adsb_fallback=pl.lit(False))
    for c in V.DET_COLS:
        if c not in det.columns:
            det = det.with_columns(pl.lit(None, pl.Float64).alias(c))
    return det.select("MVT_ID_mvt", *V.DET_COLS, "adsb_fallback")


def with_det(base, det):
    drop = [c for c in V.DET_COLS + ["adsb_taxi"] if c in base.columns]
    return (base.drop(drop).join(det, on="MVT_ID_mvt", how="left")
            .with_columns(adsb_taxi=pl.col("mvt_ts") - pl.col("adsb_pushback_ts")))


V.read_det, V.with_det = read_det, with_det
_features0, _x_b0 = C.features, N.x_b
STATE = {"comb_extra": [], "b_extra": []}


def features(df, s, airports):
    saved, C.FEATS = C.FEATS, FEATS0          # the original builder selects C.FEATS at its end
    try:
        base = _features0(df, s, airports)
    finally:
        C.FEATS = saved
    ex = STATE["comb_extra"]
    if not ex:
        return base
    return pl.concat([base, df.select([pl.col(c).cast(pl.Float64) for c in ex])], how="horizontal")


def x_b(df, s, nb, airports):
    X = _x_b0(df, s, nb, airports)
    ex = STATE["b_extra"]
    if not ex:
        return X
    return np.column_stack([X, df.select([pl.col(c).cast(pl.Float64) for c in ex]).to_numpy()])


C.features, N.x_b = features, x_b
RO2 = ["ro_last_tin_x", "ro_last_land_gap"]
NBR = [f"nbro_rate_{s}_{w}" for s in ("apt", "rwy") for w in (900, 1800, 3600)]
NBG = [f"nbro_gap_{s}_{w}" for s in ("apt", "rwy") for w in (900, 1800, 3600)]
FA = ["fa_tmatch", "fa_omatch", "fa_origin", "fa_flnear", "fa_tin_x"]
RO3X = ["ro_last_tin_x", "ro_last_land_gap", "ro_prevdep_gap", "ro_prevdep_after_last", "ro_tm_prevdep_last", "ro_tm_prevdep_self"]
ARM_DEF = {"ctl": ([], []), "pos": (POS, []), "ro_c": (RO, []), "ro_b": ([], RO), "all": (POS + RO, RO),
           "pos_rob": (POS, RO), "e2": ([], []), "pos_e2": (POS, []), "all_e2": (POS + RO, RO),
           "all2": (POS + RO + ["ro_last_tin_x", "ro_last_land_gap"], RO + ["ro_last_tin_x", "ro_last_land_gap"]),
           "all3": (POS + RO + RO3X, RO + RO3X),
           "all2_e2": (POS + RO + ["ro_last_tin_x", "ro_last_land_gap"], RO + ["ro_last_tin_x", "ro_last_land_gap"]),
           "a2e2_nbr": (POS + RO + RO2 + NBR, RO + RO2 + NBR), "a2e2_nbrg": (POS + RO + RO2 + NBR + NBG, RO + RO2 + NBR + NBG),
           "a2e2_nbe2": (POS + RO + RO2, RO + RO2),
           "all_id": (POS + RO + FA, RO + FA)}
ENS_ARMS = {"e2", "pos_e2", "all_e2", "all2_e2", "a2e2_nbr", "a2e2_nbrg", "a2e2_nbe2"}
_fit_combiner0 = C.fit_combiner
_nfit0 = N.fit
NB_ENS_ARMS = {"a2e2_nbe2"}


class Ens2:
    """Equal-weight mean of the global combiner and a per-airport combiner (column 0 = airport code)."""
    def __init__(self, g, per):
        self.g, self.per = g, per

    def predict(self, X):
        out = self.g.predict(X)
        a = X[:, 0]
        for code, m in self.per.items():
            k = a == code
            if k.any():
                out[k] = 0.5 * out[k] + 0.5 * m.predict(X[k])
        return out


def nfit_ens2(X, y, inner, cat_idx):
    """Neighbour stage (b) as the equal-weight mean of the global and a per-airport fit (column 0 = airport code)."""
    g, best = _nfit0(X, y, inner, cat_idx)
    per = {}
    for code in np.unique(X[:, 0]):
        k = X[:, 0] == code
        if k.sum() >= 5000 and inner[k].any() and (~inner[k]).any():
            per[code], _ = _nfit0(X[k], y[k], inner[k], cat_idx)
    return Ens2(g, per), best


def fit_combiner_ens2(X, y, inner):
    g, best = _fit_combiner0(X, y, inner)
    per = {}
    for code in np.unique(X[:, 0]):
        k = X[:, 0] == code
        if k.sum() >= 5000 and inner[k].any() and (~inner[k]).any():
            per[code], _ = _fit_combiner0(X[k], y[k], inner[k])
    return Ens2(g, per), best


def main():
    arms = ["ctl"] + [a for a in os.environ.get("ARMS", "pos").split(",") if a and a != "ctl"]
    t0 = time.time()
    df = B.A.load()
    ex, lag = N.train_extra()
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr = N.arrivals(paths, paths)
    ro = pl.read_parquet(SCR / os.environ.get("RO_FILE", "reocc_feats.parquet")).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    ev = pl.read_parquet(EVAL / f"{CORR}_holdout_ev.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    df2 = df.join(ev, on="MVT_ID_mvt", how="left"); assert df2["lgb_new"].null_count() == 0
    df2 = df2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new").join(ro, on="MVT_ID_mvt", how="left")
    oof = pl.concat([pl.read_parquet(p) for p in sorted((OOF / CORR).glob("fold=*.parquet"))]).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    ex2 = ex.join(oof, on="MVT_ID_mvt", how="left"); assert ex2["lgb_new"].null_count() == 0
    ex2 = ex2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new")
    w, _ = nnls(ex2.select("lgb", "cat").to_numpy(), ex2["taxi"].to_numpy())
    ex2 = ex2.with_columns(s=pl.Series(ex2.select("lgb", "cat").to_numpy() @ w)).join(ro, on="MVT_ID_mvt", how="left")
    print(f"loaded in {time.time() - t0:.0f}s; OOF stack w {np.round(w, 3)}; ex2 position coverage "
          f"{ex2['adsb_first_x'].is_not_null().mean():.3f}", flush=True)
    raw = NL.frame(str(ROOT / "data" / "raw" / "training_*.parquet"))
    prm = NL.fit(raw.filter(~pl.col("m").is_in([1, 7])))
    h = df.select("MVT_ID_mvt").join(raw, on="MVT_ID_mvt", how="left")
    lt = h["lt"].to_numpy()
    fin = lambda q: NL.apply(h, np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606)), prm)[0]  # noqa: E731
    v41 = pl.read_parquet(EVAL / "v41_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), v41="new")
    info = df.select("MVT_ID_mvt", "ADEP_mvt").join(v41, on="MVT_ID_mvt", how="left").join(
        raw.select("MVT_ID_mvt", "nm"), on="MVT_ID_mvt", how="left")
    lirf_nmu = ((info["ADEP_mvt"] == "LIRF") & info["nm"]).to_numpy()
    for arm in arms:
        t1 = time.time()
        STATE["comb_extra"], STATE["b_extra"] = ARM_DEF[arm]
        C.FEATS = FEATS0 + ARM_DEF[arm][0]
        C.fit_combiner = fit_combiner_ens2 if arm in ENS_ARMS else _fit_combiner0
        N.fit = nfit_ens2 if arm in NB_ENS_ARMS else _nfit0
        chain = fin(B.v30_method(df2, ex2, lag, arr))
        pred = np.where(lirf_nmu, info["v41"].to_numpy(), chain)
        pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], "pred": pred}).write_parquet(SCR / f"chain_{arm}.parquet")
        print(f"arm {arm}: {time.time() - t1:.0f}s", flush=True)
    C.FEATS = FEATS0


if __name__ == "__main__":
    main()
