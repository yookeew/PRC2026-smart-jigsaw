"""Hybrid corrector selection for an early read: R (new, with the stand re-occupation block) at the non-STABLE airports,
RB at the STABLE ones (EHAM, LSZH, EDDF, LEBL), exactly as ctx_armc.py sel does. Writes cache/eval/<OUT>_holdout_ev
and cache/oof/<OUT>/fold=*.parquet.  Usage: hybrid_sel.py R RB OUT   e.g. ctxc1r_ro ctxc1rb ctxc1selh_ro"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import polars as pl
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import ctx_adsb_stable_test as ST  # noqa: E402
r, rb, out = sys.argv[1:4]
EVAL, OOF = ROOT / "cache" / "eval", ROOT / "cache" / "oof"
assert not (EVAL / f"{out}_holdout_ev.parquet").exists() and not (OOF / out).exists(), "refusing to overwrite"
apt = ST._airports()
ev, n = ST._select(pl.read_parquet(EVAL / f"{r}_holdout_ev.parquet"), pl.read_parquet(EVAL / f"{rb}_holdout_ev.parquet"), "pred", apt)
ev.write_parquet(EVAL / f"{out}_holdout_ev.parquet")
print(f"holdout: {n:,} of {ev.height:,} rows from {rb}")
(OOF / out).mkdir(parents=True)
folds = sorted(p.name for p in (OOF / r).glob("fold=*.parquet"))
assert len(folds) == 5, folds
for f in folds:
    o, n = ST._select(pl.read_parquet(OOF / r / f), pl.read_parquet(OOF / rb / f), "pred", apt)
    o.write_parquet(OOF / out / f)
print("wrote", out)
