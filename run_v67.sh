#!/usr/bin/env bash
# Block 14 candidate (reports/ctx_stop_preregistration.md): runway stoppage / discharge in both corrector members on top
# of blocks 11-13, gate vs v66's ctxc1sel_rf, production fits, chain, delta vs chain_ctxc1sel_ro, assembly on v54, v63's
# day recipe -> cache/v67/v67_candidate.parquet (promoted to smart-jigsaw_v67 only if the gate passes).
# Run from the repo root:  bash run_v67.sh
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe; L=logs/v67; S=data/submissions
mkdir -p $L cache/v67/gate_rs
step() { echo "=== $1  $(date '+%F %H:%M:%S')"; }
M="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_FLOW=1 CTX_STOP=1 PRC_ARMC_TAG=_rs"

step "0. drift (condition f)";             $PY -u tests/ctx_stop_drift.py > $L/ctx_stop_drift.log 2>&1
step "1. holdout + OOF inputs, member r";  env $M $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_r.log 2>&1
step "2. holdout + OOF inputs, member rb"; env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_rb.log 2>&1
step "3. production fit, member r";        env $M $PY -u src/post/ctx_armc.py prod > $L/armc_prod_r.log 2>&1
step "4. production fit, member rb";       env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py prod > $L/armc_prod_rb.log 2>&1
step "5. sel";                             ARMC_ARM=c1 PRC_ARMC_TAG=_rs $PY -u src/post/ctx_armc.py sel > $L/armc_sel.log 2>&1
step "6. holdout chain gate run"
cp -n cache/v64/gate_rohb/reocc_feats.parquet cache/v67/gate_rs/
PRC_LGB=ctxcorr2r CORR=ctxc1sel_rs SCR=cache/v67/gate_rs ARMS=all RO_FILE=reocc_feats.parquet $PY -u tests/v50_chain_gate.py > $L/gate_rs.log 2>&1
step "7. gate";  CTL=rf CTL_DIR=cache/v66 TAG=rs GATE_DIR=cache/v67 BOARD=237.0694 $PY -u tests/ctx_hour_gate.py > $L/ctx_stop_gate.log 2>&1
cat $L/ctx_stop_gate.log; tail -n 1 $L/ctx_stop_drift.log
step "8. production chain"; bash run_v67_chain.sh ctxc1sel_rs
step "9. delta, assembly, day recipe"
$PY -u src/post/v64_delta.py cache/v67/delta_rs.parquet ctxc1sel_rs cache/v67 > $L/delta.log 2>&1
FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py $S/smart-jigsaw_v54.parquet cache/v67/p67.parquet cache/v67/delta_rs.parquet > $L/assemble.log 2>&1
$PY -u src/post/day_alloc_build.py cache/v67/p67.parquet $S/smart-jigsaw_v62.parquet cache/v67/v67_candidate.parquet > $L/v67_build.log 2>&1
step "done"
