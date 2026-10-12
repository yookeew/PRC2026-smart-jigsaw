#!/usr/bin/env bash
# Gate of reports/ctx_huber_preregistration.md: Huber members (holdout + OOF), sel, hybrid, holdout chain, OOF gate;
# then the production members, sel, hybrid and the output-spread stop (tests/ctx_huber_drift.py).
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PRC_SCOPE=jj
PY=.venv/Scripts/python.exe; L=logs/v68
step() { echo "=== $1  $(date '+%F %H:%M:%S')"; }
M="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_HUBER=1000 PRC_ARMC_TAG=_rh2hu"
step "1. holdout + OOF inputs, member r";  env $M $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_r_rh2hu.log 2>&1
step "2. holdout + OOF inputs, member rb"; env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_rb_rh2hu.log 2>&1
step "3. sel";                             ARMC_ARM=c1 PRC_ARMC_TAG=_rh2hu $PY -u src/post/ctx_armc.py sel > $L/armc_sel_rh2hu.log 2>&1
step "4. hybrid";                          $PY -u src/post/ctx_huber_hybrid.py ctxc1sel_rh2 ctxc1sel_rh2hu ctxc1sel_rh2hy > $L/hybrid_inputs.log 2>&1
step "5. holdout chain rh2hy"
mkdir -p cache/v68/gate_rh2hy && cp -n cache/v50/reocc_feats.parquet cache/v68/gate_rh2hy/
PRC_LGB=ctxcorr2r CORR=ctxc1sel_rh2hy SCR=cache/v68/gate_rh2hy ARMS=all RO_FILE=reocc_feats.parquet $PY -u tests/v50_chain_gate.py > $L/gate_rh2hy.log 2>&1
step "6. gate"; CTL=rh2 CTL_DIR=cache/v66 TAG=rh2hy GATE_DIR=cache/v68 BOARD=236.8 $PY -u tests/ctx_hour_gate.py > $L/ctx_huber_gate.log 2>&1
step "7. production member r";  env $M $PY -u src/post/ctx_armc.py prod > $L/armc_prod_r_rh2hu.log 2>&1
step "8. production member rb"; env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py prod > $L/armc_prod_rb_rh2hu.log 2>&1
step "9. sel + hybrid (prod)"
ARMC_ARM=c1 PRC_ARMC_TAG=_rh2hu $PY -u src/post/ctx_armc.py sel > $L/armc_sel_rh2hu_prod.log 2>&1
$PY -u src/post/ctx_huber_hybrid.py ctxc1sel_rh2 ctxc1sel_rh2hu ctxc1sel_rh2hy > $L/hybrid_prod.log 2>&1
step "10. output-spread stop"; $PY -u tests/ctx_huber_drift.py > $L/ctx_huber_drift.log 2>&1
step "done"
