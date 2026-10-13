#!/usr/bin/env bash
# v66 (reports/ctx_flow_preregistration.md): block 13 (flow state) in both corrector members on top of blocks 11 + 12,
# gate vs v65's ctxc1sel_rh2, production fits, production chain, delta vs chain_ctxc1sel_ro, assembly on v54, v63's day
# recipe -> smart-jigsaw_v66. The build steps run regardless of the gate outcome; the file is used only if the gate passes.
# Run from the repo root:  bash scripts/run_v66.sh
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe; L=logs/v66; S=data/submissions
mkdir -p $L cache/v66/gate_rf
step() { echo "=== $1  $(date '+%F %H:%M:%S')"; }
M="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_FLOW=1 PRC_ARMC_TAG=_rf"

step "0. drift (condition f)";             $PY -u tests/ctx_flow_drift.py > $L/ctx_flow_drift.log 2>&1
step "1. holdout + OOF inputs, member r";  env $M $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_r.log 2>&1
step "2. holdout + OOF inputs, member rb"; env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_rb.log 2>&1
step "3. production fit, member r";        env $M $PY -u src/post/ctx_armc.py prod > $L/armc_prod_r.log 2>&1
step "4. production fit, member rb";       env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py prod > $L/armc_prod_rb.log 2>&1
step "5. sel";                             ARMC_ARM=c1 PRC_ARMC_TAG=_rf $PY -u src/post/ctx_armc.py sel > $L/armc_sel.log 2>&1
step "6. holdout chain gate run"
cp -n cache/v64/gate_rohb/reocc_feats.parquet cache/v66/gate_rf/
PRC_LGB=ctxcorr2r CORR=ctxc1sel_rf SCR=cache/v66/gate_rf ARMS=all RO_FILE=reocc_feats.parquet $PY -u tests/v50_chain_gate.py > $L/gate_rf.log 2>&1
step "7. gate";  CTL=rh2 CTL_DIR=cache/v65 TAG=rf GATE_DIR=cache/v66 BOARD=237.0694 $PY -u tests/ctx_hour_gate.py > $L/ctx_flow_gate.log 2>&1
cat $L/ctx_flow_gate.log; tail -n 1 $L/ctx_flow_drift.log
step "8. production chain"; bash scripts/run_v66_chain.sh ctxc1sel_rf
step "9. delta, assembly, day recipe"
$PY -u src/post/v64_delta.py cache/v66/delta_rf.parquet ctxc1sel_rf cache/v66 > $L/delta.log 2>&1
FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py $S/smart-jigsaw_v54.parquet cache/v66/p66.parquet cache/v66/delta_rf.parquet > $L/assemble.log 2>&1
$PY -u src/post/day_alloc_build.py cache/v66/p66.parquet $S/smart-jigsaw_v62.parquet $S/smart-jigsaw_v66.parquet > $L/v66_build.log 2>&1
step "done"
