#!/usr/bin/env bash
# CTX_HOUR2 candidate (reports/ctx_hour2_preregistration.md): holdout + OOF inputs of the two block-12 members, then sel.
set -euo pipefail
export PRC_SCOPE=jj PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe; L=logs/ctx_hour2
echo "start $(date '+%F %H:%M')"
ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 PRC_ARMC_TAG=_roh2 $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_ctxc1r_roh2.log 2>&1
echo "r_roh2 done $(date '+%F %H:%M')"
ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_ADSB=1 CTX_HOUR=1 CTX_HOUR2=1 PRC_ARMC_TAG=_rohb2 $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_ctxc1rb_rohb2.log 2>&1
echo "rb_rohb2 done $(date '+%F %H:%M')"
cp -n cache/eval/ctxc1r_roh2_holdout_ev.parquet cache/eval/ctxc1r_rohb2_holdout_ev.parquet
mkdir -p cache/oof/ctxc1r_rohb2 && cp -n cache/oof/ctxc1r_roh2/*.parquet cache/oof/ctxc1r_rohb2/
ARMC_ARM=c1 PRC_ARMC_TAG=_rohb2 $PY -u src/post/ctx_armc.py sel > $L/armc_sel_rohb2.log 2>&1
echo "sel done $(date '+%F %H:%M')"
