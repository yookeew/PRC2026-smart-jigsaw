"""OPDI scoping S2 (reports/opdi_scoping.md): does OPDI ground-event timing carry information beyond v50's chain?

Residual model on v50's holdout chain (cache/v50/chain_all; LIRF rows excluded), cross-fitted Jan <-> Jul 2025,
LightGBM Huber. Rows are selected on the prediction (<= 3,600 s), never on the label. Control = chain inputs
already known (airport, hour, pred, adsb.lol tier / leads); treatment = control + OPDI features. Decision on
treatment - control (body = NM-matched and y <= 1 h; trimmed y <= 5 h; full), both months, per airport.
"""
import os, glob, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
import opdi_s1_coverage as S  # noqa: E402

GROUND = S.GROUND + ["entry-deicing_pad", "exit-deicing_pad"]
PARAMS = dict(objective="huber", alpha=300.0, num_leaves=31, learning_rate=0.03, min_data_in_leaf=300,
              lambda_l2=10.0, feature_fraction=0.9, max_bin=127, seed=42, deterministic=True, force_row_wise=True,
              verbose=-1, num_threads=16)
ROUNDS = 600
CTL = ["apt", "hour", "pred", "tier", "adsb_rel", "L_rel"]


def opdi_feats(dep: pl.DataFrame, files: list[str]) -> pl.DataFrame:
    """dep: MVT_ID_mvt, apt, T, cs. Label-free OPDI features per departure."""
    od = S.opdi_departures(files)
    m = S.match(dep, od)
    x = (m.filter(pl.col("flight_id").is_not_null()).select("MVT_ID_mvt", "T", "ev_t", "ev_type")
         .explode("ev_t", "ev_type").filter((pl.col("ev_t") >= pl.col("T") - 3 * 3600) & (pl.col("ev_t") <= pl.col("T") + 60)))
    f = x.group_by("MVT_ID_mvt").agg(
        o_lead_first=pl.col("T").first() - pl.col("ev_t").filter(pl.col("ev_type").is_in(GROUND)).min(),
        o_lead_exitpark=(pl.col("T") - pl.col("ev_t")).filter(pl.col("ev_type") == "exit-parking_position").min(),
        o_lead_entrypark=(pl.col("T") - pl.col("ev_t")).filter(pl.col("ev_type") == "entry-parking_position").max(),
        o_lead_taxi=(pl.col("T") - pl.col("ev_t")).filter(pl.col("ev_type") == "entry-taxiway").max(),
        o_lead_apron=(pl.col("T") - pl.col("ev_t")).filter(pl.col("ev_type").is_in(["entry-apron", "exit-apron"])).max(),
        o_lead_rwy=(pl.col("T") - pl.col("ev_t")).filter(pl.col("ev_type") == "entry-runway").min(),
        o_deice=(pl.col("ev_type") == "entry-deicing_pad").any().cast(pl.Float64),
        o_n_ground=pl.col("ev_type").is_in(GROUND).sum().cast(pl.Float64))
    out = m.select("MVT_ID_mvt", o_matched=pl.col("flight_id").is_not_null().cast(pl.Float64)).join(f, on="MVT_ID_mvt", how="left")
    return out


OPDI = ["o_matched", "o_lead_first", "o_lead_exitpark", "o_lead_entrypark", "o_lead_taxi", "o_lead_apron",
        "o_lead_rwy", "o_deice", "o_n_ground", "o_rel_first", "o_rel_exitpark", "o_rel_taxi"]


def frame() -> pl.DataFrame:
    raw = S.departures_2025()
    k = S.dep_keys(raw).join(raw.select(pl.col("MVT_ID_mvt").cast(pl.Int64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                                        hour=pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64),
                                        mon=pl.col("MVT_TIME_UTC_mvt").dt.month()), on="MVT_ID_mvt")
    files = [f for f in sorted(glob.glob("cache/opdi_events/*.parquet"))
             if any(t in f for t in ("20241226", "20250105", "20250115", "20250125", "20250624", "20250704", "20250714", "20250724"))]
    assert len(files) == 8, files
    of = opdi_feats(k.select("MVT_ID_mvt", "apt", "T", "cs"), files)
    a = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("cache/adsb_pushback_v4/day=2025-0[17]*.parquet"))],
                  how="vertical_relaxed").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_tier", "adsb_pushback_ts", "adsb_first_ts")
    c = pl.read_parquet("cache/v50/chain_all.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "pred")
    d = k.join(c, on="MVT_ID_mvt").join(a, on="MVT_ID_mvt", how="left").join(of, on="MVT_ID_mvt", how="left")
    apts = sorted(d["apt"].unique().to_list())
    tiers = ["dwell", "pass", "appear", "fb_dwell", "fb_appear"]
    d = d.with_columns(
        apt_c=pl.col("apt").replace_strict({x: i for i, x in enumerate(apts)}, return_dtype=pl.Float64),
        tier=pl.col("adsb_tier").replace_strict({t: i + 1 for i, t in enumerate(tiers)}, default=0, return_dtype=pl.Float64),
        adsb_rel=(pl.col("T") - pl.col("adsb_pushback_ts")) - pl.col("pred"),
        L_rel=(pl.col("T") - pl.col("adsb_first_ts")) - pl.col("pred"),
        o_rel_first=pl.col("o_lead_first") - pl.col("pred"), o_rel_exitpark=pl.col("o_lead_exitpark") - pl.col("pred"),
        o_rel_taxi=pl.col("o_lead_taxi") - pl.col("pred"))
    return d


def run(d: pl.DataFrame, feats: list[str]) -> np.ndarray:
    cols = ["apt_c" if f == "apt" else f for f in feats]
    X = d.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    y = (d["y"].clip(upper_bound=18000) - d["pred"]).to_numpy()
    sel = ((d["pred"] <= 3600) & (d["apt"] != "LIRF")).to_numpy()
    mon = d["mon"].to_numpy()
    out = d["pred"].to_numpy().copy()
    for fm, am in ((1, 7), (7, 1)):
        tr = sel & (mon == fm)
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], categorical_feature=[0]), ROUNDS)
        ap = sel & (mon == am)
        out[ap] = out[ap] + m.predict(X[ap])
    return out


def report(d: pl.DataFrame, a: np.ndarray, b: np.ndarray, title: str) -> None:
    e = d.select("apt", "mon", "y", "nmu").with_columns(ea=(pl.col("y") - pl.Series(a)) ** 2, eb=(pl.col("y") - pl.Series(b)) ** 2)
    e = e.filter(pl.col("apt") != "LIRF")
    sl = {"body": (~pl.col("nmu")) & (pl.col("y") <= 3600), "trimmed": pl.col("y") <= 18000, "full": pl.lit(True)}
    print(f"\n== {title} (b - a, RMSE s; non-LIRF)")
    for name, f in sl.items():
        s = e.filter(f)
        parts = [f"{name}: {np.sqrt(s['eb'].mean()) - np.sqrt(s['ea'].mean()):+.3f}"]
        for mo in (1, 7):
            q = s.filter(pl.col("mon") == mo)
            parts.append(f"m{mo} {np.sqrt(q['eb'].mean()) - np.sqrt(q['ea'].mean()):+.3f}")
        print("  " + "  ".join(parts))
    s = e.filter(sl["body"])
    tot = s["ea"].sum(); n = s.height
    t = s.group_by("apt").agg(n=pl.len(), rmse_a=pl.col("ea").mean().sqrt(), rmse_b=pl.col("eb").mean().sqrt(),
                              dsse=(pl.col("eb") - pl.col("ea")).sum()).with_columns(
        d_rmse=pl.col("rmse_b") - pl.col("rmse_a"), pooled_contrib=pl.col("dsse") / (2 * np.sqrt(tot / n) * n)).sort("apt")
    with pl.Config(tbl_rows=20, float_precision=3, tbl_width_chars=200):
        print(t.drop("dsse"))


if __name__ == "__main__":
    os.chdir(r"C:\Users\zhang\smart-jigsaw")
    d = frame()
    cov = d.group_by("apt", "mon").agg(o=pl.col("o_lead_first").fill_null(0).ge(60).mean()).sort("apt", "mon")
    print("OPDI ground-event share (lead >= 60 s):", {f"{r['apt']}-{r['mon']}": round(r["o"], 3) for r in cov.iter_rows(named=True)})
    ctl = run(d, CTL)
    trt = run(d, CTL + OPDI)
    report(d, d["pred"].to_numpy(), ctl, "control vs chain (generic residual model)")
    report(d, ctl, trt, "DECISION: treatment (+OPDI) vs control")
    lz = d.filter((pl.col("apt") == "LSZH") & ~pl.col("nmu") & (pl.col("y") <= 3600) & pl.col("o_lead_exitpark").is_not_null())
    r = lz.select(dt=pl.col("o_lead_exitpark") - pl.col("y"))
    print("\nLSZH body rows with exit-parking: n", lz.height, "exit-park lag vs true off-block (s): median",
          r["dt"].median(), "IQR", r["dt"].quantile(0.25), r["dt"].quantile(0.75))
