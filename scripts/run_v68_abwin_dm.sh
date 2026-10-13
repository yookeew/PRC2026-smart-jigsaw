#!/usr/bin/env bash
# Gate of reports/ctx_abwin_dm_preregistration.md:  bash scripts/run_v68_abwin_dm.sh hy|l2
#   hy: Huber gate passed -> members CTX_HUBER=1000 CTX_ABWIN_DM=1 (_rh2hud), hybrid ctxc1sel_rh2hdy vs control ctxc1sel_rh2hy
#   l2: otherwise          -> members CTX_ABWIN_DM=1 (_rh2d), ctxc1sel_rh2d vs control ctxc1sel_rh2
set -euo pipefail
B=$1
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PRC_SCOPE=jj
PY=.venv/Scripts/python.exe; L=logs/v68
step() { echo "=== $1  $(date '+%F %H:%M:%S')"; }
if [ "$B" = hy ]; then
  TG=_rh2hud; M="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_HUBER=1000 CTX_ABWIN_DM=1 PRC_ARMC_TAG=$TG"
  TREAT=rh2hdy; CTL=rh2hy; CDIR=cache/v68
else
  TG=_rh2d; M="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_ABWIN_DM=1 PRC_ARMC_TAG=$TG"
  TREAT=rh2d; CTL=rh2; CDIR=cache/v66
fi
step "1. holdout + OOF inputs, member r ($TG)";  env $M $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_r$TG.log 2>&1
step "2. holdout + OOF inputs, member rb ($TG)"; env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_rb$TG.log 2>&1
step "3. sel";                                   ARMC_ARM=c1 PRC_ARMC_TAG=$TG $PY -u src/post/ctx_armc.py sel > $L/armc_sel$TG.log 2>&1
if [ "$B" = hy ]; then
  step "4. hybrid"; $PY -u src/post/ctx_huber_hybrid.py ctxc1sel_rh2 ctxc1sel$TG ctxc1sel_$TREAT > $L/hybrid_dm_inputs.log 2>&1
fi
step "5. holdout chain $TREAT"
mkdir -p cache/v68/gate_$TREAT && cp -n cache/v50/reocc_feats.parquet cache/v68/gate_$TREAT/
PRC_LGB=ctxcorr2r CORR=ctxc1sel_$TREAT SCR=cache/v68/gate_$TREAT ARMS=all RO_FILE=reocc_feats.parquet $PY -u tests/v50_chain_gate.py > $L/gate_$TREAT.log 2>&1
step "6. gate"; CTL=$CTL CTL_DIR=$CDIR TAG=$TREAT GATE_DIR=cache/v68 BOARD=236.8 $PY -u tests/ctx_hour_gate.py > $L/ctx_abwin_dm_gate.log 2>&1
step "done"
