#!/usr/bin/env bash
# Local rebuild of the v64 hour members' holdout + OOF inputs (scope-independent), as a fallback for the files on the other machine.
set -euo pipefail
export PRC_SCOPE=jj PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe; L=logs/v64/local
echo "start $(date '+%F %H:%M')"
ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 PRC_ARMC_TAG=_roh $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_ctxc1r_roh.log 2>&1
echo "r_roh done $(date '+%F %H:%M')"
ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_ADSB=1 CTX_HOUR=1 PRC_ARMC_TAG=_rohb $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_ctxc1rb_rohb.log 2>&1
echo "rb_rohb done $(date '+%F %H:%M')"
cp -n cache/eval/ctxc1r_roh_holdout_ev.parquet cache/eval/ctxc1r_rohb_holdout_ev.parquet
mkdir -p cache/oof/ctxc1r_rohb && cp -n cache/oof/ctxc1r_roh/*.parquet cache/oof/ctxc1r_rohb/
ARMC_ARM=c1 PRC_ARMC_TAG=_rohb $PY -u src/post/ctx_armc.py sel > $L/armc_sel_rohb.log 2>&1
echo "sel done $(date '+%F %H:%M')"
