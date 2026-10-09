"""v52 gate (reports/v52_opdi_gate_preregistration.md): v50's holdout chain with OPDI ground-event inputs.

Arms, one process: 'all' = v50's chain (control; must reproduce cache/v50/chain_all.parquet), 'opdi' = all + OPDI
features in the combiner and in neighbour stage b. Writes SCR/chain_<arm>.parquet (MVT_ID_mvt, pred).

Run:  PRC_LGB=ctxcorr2r ARMC_BASE=ctxc1sel SCR=cache/v52 python tests/v52_opdi_gate.py
"""
import os, sys, time
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("ARMC_BASE", "ctxc1sel")
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import nnls
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
SCR = Path(os.environ.get("SCR", "cache/v52"))
os.environ["SCR"] = "cache/v50"                 # v50_chain_gate reads its RO table from SCR at import-free main only
import v50_chain_gate as G  # noqa: E402
os.environ["SCR"] = str(SCR)
C, N, B, NL = G.C, G.N, G.B, G.NL

O = ["o_matched", "o_lead_first", "o_lead_exitpark", "o_lead_entrypark", "o_lead_taxi", "o_lead_apron",
     "o_lead_rwy", "o_deice", "o_n_ground"]
REL = {"o_rel_first": "o_lead_first", "o_rel_exitpark": "o_lead_exitpark", "o_rel_taxi": "o_lead_taxi"}
OALL = O + list(REL)
ARMS = {"all": (G.POS + G.RO, G.RO), "opdi": (G.POS + G.RO + OALL, G.RO + OALL)}
OPDI_FEATS = ROOT / "cache" / "opdi_feats" / "opdi_feats.parquet"


def extra_cols(df: pl.DataFrame, s: np.ndarray, names: list[str]) -> pl.DataFrame:
    d = df.with_columns(pl.Series("_s", np.asarray(s, dtype=np.float64)))
    return d.select([(pl.col(REL[c]) - pl.col("_s")).alias(c) if c in REL else pl.col(c).cast(pl.Float64) for c in names])


def features(df, s, airports):
    saved, C.FEATS = C.FEATS, G.FEATS0          # the original builder selects C.FEATS at its end
    try:
        base = G._features0(df, s, airports)
    finally:
        C.FEATS = saved
    ex = G.STATE["comb_extra"]
    return pl.concat([base, extra_cols(df, s, ex)], how="horizontal") if ex else base


def x_b(df, s, nb, airports):
    X = G._x_b0(df, s, nb, airports)
    ex = G.STATE["b_extra"]
    return np.column_stack([X, extra_cols(df, s, ex).to_numpy()]) if ex else X


C.features, N.x_b = features, x_b


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    df = B.A.load()
    ex, lag = N.train_extra()
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr = N.arrivals(paths, paths)
    ro = pl.read_parquet(ROOT / "cache" / "v50" / "reocc_feats.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    of = pl.read_parquet(OPDI_FEATS).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).select("MVT_ID_mvt", *O)
    ev = pl.read_parquet(G.EVAL / "ctxc1sel_holdout_ev.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    df2 = df.join(ev, on="MVT_ID_mvt", how="left"); assert df2["lgb_new"].null_count() == 0
    df2 = (df2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new").join(ro, on="MVT_ID_mvt", how="left")
           .join(of, on="MVT_ID_mvt", how="left"))
    oof = pl.concat([pl.read_parquet(p) for p in sorted((G.OOF / "ctxc1sel").glob("fold=*.parquet"))]).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    ex2 = ex.join(oof, on="MVT_ID_mvt", how="left"); assert ex2["lgb_new"].null_count() == 0
    ex2 = ex2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new")
    w, _ = nnls(ex2.select("lgb", "cat").to_numpy(), ex2["taxi"].to_numpy())
    ex2 = (ex2.with_columns(s=pl.Series(ex2.select("lgb", "cat").to_numpy() @ w)).join(ro, on="MVT_ID_mvt", how="left")
           .join(of, on="MVT_ID_mvt", how="left"))
    cov = lambda d: (d["o_lead_first"].fill_null(0) >= 60).mean()  # noqa: E731
    print(f"loaded in {time.time() - t0:.0f}s; OOF stack w {np.round(w, 3)}; OPDI ground-event share: holdout "
          f"{cov(df2):.3f}, OOF ADS-B days {cov(ex2):.3f}; OPDI day coverage holdout "
          f"{df2.join(pl.read_parquet(OPDI_FEATS).select(pl.col('MVT_ID_mvt').cast(pl.Int64), 'o_day'), on='MVT_ID_mvt', how='left')['o_day'].mean():.3f}", flush=True)
    raw = NL.frame(str(ROOT / "data" / "raw" / "training_*.parquet"))
    prm = NL.fit(raw.filter(~pl.col("m").is_in([1, 7])))
    h = df.select("MVT_ID_mvt").join(raw, on="MVT_ID_mvt", how="left")
    lt = h["lt"].to_numpy()
    fin = lambda q: NL.apply(h, np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606)), prm)[0]  # noqa: E731
    v41 = pl.read_parquet(G.EVAL / "v41_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), v41="new")
    info = df.select("MVT_ID_mvt", "ADEP_mvt").join(v41, on="MVT_ID_mvt", how="left").join(
        raw.select("MVT_ID_mvt", "nm"), on="MVT_ID_mvt", how="left")
    lirf_nmu = ((info["ADEP_mvt"] == "LIRF") & info["nm"]).to_numpy()
    run = [a for a in os.environ.get("ARMS", "all,opdi").split(",") if a]
    for arm, (cx, bx) in ((a, ARMS[a]) for a in run):
        t1 = time.time()
        G.STATE["comb_extra"], G.STATE["b_extra"] = cx, bx
        C.FEATS = G.FEATS0 + cx
        C.fit_combiner = G._fit_combiner0
        chain = fin(B.v30_method(df2, ex2, lag, arr))
        pred = np.where(lirf_nmu, info["v41"].to_numpy(), chain)
        out = pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], "pred": pred})
        out.write_parquet(SCR / f"chain_{arm}.parquet")
        if arm == "all":
            ref = pl.read_parquet(ROOT / "cache" / "v50" / "chain_all.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), ref="pred")
            dd = out.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(ref, on="MVT_ID_mvt")
            md = float(np.abs(dd["pred"] - dd["ref"]).max())
            print(f"control vs cache/v50/chain_all: rows {dd.height}, max |diff| {md:.2e}", flush=True)
            assert dd.height == df.height and md < 1e-6, "control does not reproduce v50's chain"
        print(f"arm {arm}: {time.time() - t1:.0f}s", flush=True)
    C.FEATS = G.FEATS0


if __name__ == "__main__":
    main()
