"""Block-15 candidate without the external-data family: L (LOBT / AOBT_3 order) + C (configuration change ahead), screened
on top of v66's corrector (ctxc1sel_rf), PROGRESS.md §140. Features from dir_screen.py's cached frame (= src/post/order_feats.py).
Runs at 4 LightGBM threads (a corrector fit was running alongside)."""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen
_train = screen.lgb.train
screen.lgb.train = lambda params, *a, **k: _train({**params, "num_threads": 4}, *a, **k)
S = "C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"
f = pl.read_parquet(S + "/dep25_dir_rf.parquet")
L = [c for c in f.columns if c.startswith(("lq_", "aq_"))]
C = [c for c in f.columns if c.startswith("cf_")]
screen.run(f, L + C, label="L+C (on rf)")
