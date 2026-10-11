"""Screen the airline-network inputs (airline_net.py) on top of v66's corrector (ctxc1sel_rf OOF + holdout)."""
import glob, os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen import run
ROOT = "C:/Users/zhang/smart-jigsaw"; S = ROOT + "/cache/analysis/flow_screen"
rd = lambda f: pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").cast(pl.Float64))
p = pl.concat([rd(f) for f in sorted(glob.glob(ROOT + "/cache/oof/ctxc1sel_rf/fold=*.parquet"))] + [rd(ROOT + "/cache/eval/ctxc1sel_rf_holdout_ev.parquet")]).unique("MVT_ID_mvt")
f = pl.read_parquet(S + "/dep25_air.parquet").drop("pred").join(p, on="MVT_ID_mvt", how="left")
f.write_parquet(S + "/dep25_air_rf.parquet")
run(f, [c for c in f.columns if c.startswith("anx_")], label="airline network, other airports' departures (on rf)")
run(f, [c for c in f.columns if c.startswith(("anx_", "ana_"))], label="airline network, departures + arrivals (on rf)")
