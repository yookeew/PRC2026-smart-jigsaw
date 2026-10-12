"""Scoping: do the flow inputs (block 13) predict echo among LIRF NM-unmatched departures 1-6 h late (v45's scope)
beyond logit p (production OOF echo probability) and v45's signals (nF, nF7, nD, nA, fnull)? Leave-one-month-out
logistic, log-loss and offset-weighted squared cost of the hedge E[(y - P)^2] proxy. 2025 only, analysis only."""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
ROOT = "C:/Users/zhang/smart-jigsaw"
sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/src/post")
import lirf_signals as LS
S = ROOT + "/cache/analysis/flow_screen"
sig = LS.frame(ROOT + "/data/raw/training_*.parquet")
eo = pl.read_parquet(ROOT + "/cache/lirf_echo_sig_oof.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "p_prod")
fl = pl.read_parquet(S + "/dep25_flow2.parquet").with_columns(
    sq_net=(pl.col("sq_over").cast(pl.Float64) - pl.col("sq_under").cast(pl.Float64)) / pl.col("sq_n"))
fl = fl.select("MVT_ID_mvt", *[c for c in fl.columns if c.startswith(("xo60_", "so90_", "rso_", "rxo_", "sq_"))])
d = (sig.join(eo, on="MVT_ID_mvt", how="inner").join(fl, on="MVT_ID_mvt", how="left")
     .filter(pl.col("nmu") & (pl.col("off") >= LS.LO) & (pl.col("off") < LS.HI) & pl.col("echo").is_not_null() & (pl.col("p_prod") < 0.99)))
print("rows", d.height, "echo rate", d["echo"].mean(), "months", d["mt"].n_unique())
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
y = d["echo"].cast(pl.Int8).to_numpy(); mt = d["mt"].to_numpy(); off = d["off"].to_numpy()
base = np.column_stack([lg(d["p_prod"].to_numpy())] + [d[c].cast(pl.Float64).to_numpy() for c in ("nF", "nF7", "nD", "nA", "fnull")])
def flowX(cols):
    X = d.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    miss = np.isnan(X).any(axis=1, keepdims=True).astype(float)
    return np.column_stack([np.nan_to_num(X, nan=0.0), miss])
def lomo(X):
    q = np.zeros(len(y))
    for m in np.unique(mt):
        tr = mt != m
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(C=0.5, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        q[~tr] = clf.predict_proba((X[~tr] - mu) / sd)[:, 1]
    return q
def report(name, q):
    ll = -np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1)))
    cost = np.sum(off ** 2 * (y - q) ** 2)    # squared error of a two-point hedge with branches 0 / off (non-echo taxi ~0)
    worse = sum(np.sum((off ** 2 * (y - q) ** 2)[mt == m]) > np.sum((off ** 2 * (y - qb) ** 2)[mt == m]) for m in np.unique(mt)) if name not in ("base",) else 0
    print(f"{name:26s} log-loss {ll:.4f}  offset-weighted cost {cost / 1e9:8.3f}e9  months worse vs base {worse}/{len(np.unique(mt))}")
qb = lomo(base); qp = d["p_prod"].to_numpy(); report("p_prod alone (no refit)", qp)
report("base", qb)
sets = {"dest flow": ["xo60_n", "xo60_moff", "xo60_f30", "so90_moff"], "region flow": ["rxo_moff", "rxo_f30", "rso_moff"],
        "schedule displ.": ["sq_n", "sq_net", "sq_mdT"]}
for k, v in sets.items():
    report("base + " + k, lomo(np.column_stack([base, flowX(v)])))
report("base + all three", lomo(np.column_stack([base, flowX(sum(sets.values(), []))])))
