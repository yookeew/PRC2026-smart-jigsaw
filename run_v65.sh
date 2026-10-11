#!/usr/bin/env bash
# v65 (reports/ctx_hour2_preregistration.md): block 12 in both corrector members, gate vs v64's ctxc1sel_rohb,
# production fits, production chain, delta vs chain_ctxc1sel_ro, assembly on v54, v63's day recipe -> smart-jigsaw_v65.
# The build steps run regardless of the gate outcome; the file is used only if the gate passes.
# Run from the repo root:  bash run_v65.sh
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe; L=logs/v65; S=data/submissions
mkdir -p $L cache/v65/gate_rh2
step() { echo "=== $1  $(date '+%F %H:%M:%S')"; }
M="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 PRC_ARMC_TAG=_rh2"

step "1. holdout + OOF inputs, member r";  env $M $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_r.log 2>&1
step "2. holdout + OOF inputs, member rb"; env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_rb.log 2>&1
step "3. production fit, member r";        env $M $PY -u src/post/ctx_armc.py prod > $L/armc_prod_r.log 2>&1
step "4. production fit, member rb";       env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py prod > $L/armc_prod_rb.log 2>&1
step "5. sel";                             ARMC_ARM=c1 PRC_ARMC_TAG=_rh2 $PY -u src/post/ctx_armc.py sel > $L/armc_sel.log 2>&1
step "6. holdout chain gate run"
cp -n cache/v64/gate_rohb/reocc_feats.parquet cache/v65/gate_rh2/
PRC_LGB=ctxcorr2r CORR=ctxc1sel_rh2 SCR=cache/v65/gate_rh2 ARMS=all RO_FILE=reocc_feats.parquet $PY -u tests/v50_chain_gate.py > $L/gate_rh2.log 2>&1
step "7. gate";  CTL=rohb TAG=rh2 GATE_DIR=cache/v65 BOARD=237.4538 $PY -u tests/ctx_hour_gate.py > $L/ctx_hour2_gate.log 2>&1
step "8. drift"; $PY -u tests/ctx_hour2_drift.py > $L/ctx_hour2_drift.log 2>&1
step "9. production chain"; bash run_v65_chain.sh ctxc1sel_rh2
step "10. delta, assembly, day recipe"
$PY -u src/post/v64_delta.py cache/v65/delta_rh2.parquet ctxc1sel_rh2 cache/v65 > $L/delta.log 2>&1
FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py $S/smart-jigsaw_v54.parquet cache/v65/p65.parquet cache/v65/delta_rh2.parquet > $L/assemble.log 2>&1
$PY -u src/post/day_alloc_build.py cache/v65/p65.parquet $S/smart-jigsaw_v62.parquet $S/smart-jigsaw_v65.parquet > $L/v65_build.log 2>&1
step "done"
