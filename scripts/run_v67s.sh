#!/usr/bin/env bash
# v67 candidate (reports/ctx_seedavg_rf_preregistration.md): v66's two corrector members refitted with LightGBM seeds 43, 44,
# averaged 1:2 with v66's own seed-42 fits -> ctxc1sel_rf3; gate vs v66's ctxc1sel_rf; if PASS, production chain, delta vs
# chain_ctxc1sel_ro, assembly on v54, v63's day recipe -> smart-jigsaw_v67. Nothing is built if the gate fails.
# Run from the repo root:  bash scripts/run_v67s.sh
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe; L=logs/v67s; S=data/submissions; C=cache/v67s
mkdir -p $L $C/gate_rf3
step() { echo "=== $1  $(date '+%F %H:%M:%S')"; }
M="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_FLOW=1 PRC_ARMC_TAG=_rf PRC_CTX_SEEDS=43,44"

step "0. build-path check: v66 rebuilt from its own chain must equal the uploaded v66"
$PY -u src/post/v64_delta.py $C/delta_rf_check.parquet ctxc1sel_rf cache/v66 > $L/check_delta.log 2>&1
FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py $S/smart-jigsaw_v54.parquet $C/p66_check.parquet $C/delta_rf_check.parquet > $L/check_assemble.log 2>&1
$PY -u src/post/day_alloc_build.py $C/p66_check.parquet $S/smart-jigsaw_v62.parquet $C/v66_check.parquet > $L/check_build.log 2>&1
$PY -c "import polars as pl; a=pl.read_parquet('$S/smart-jigsaw_v66.parquet'); b=pl.read_parquet('$C/v66_check.parquet'); print('v66 rebuild identical:', a.equals(b))"

step "1. holdout + OOF inputs, member r, seeds 43,44";  env $M $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_r.log 2>&1
step "2. holdout + OOF inputs, member rb, seeds 43,44"; env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py inputs > $L/armc_inputs_rb.log 2>&1
step "3. production fit, member r, seeds 43,44";        env $M $PY -u src/post/ctx_armc.py prod > $L/armc_prod_r.log 2>&1
step "4. production fit, member rb, seeds 43,44";       env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py prod > $L/armc_prod_rb.log 2>&1
step "5. merge 1:2 with v66's seed-42 fits, sel"
$PY -u src/post/ctx_seedavg_merge.py ctxc1r_rf 1 ctxc1r_rf_s2 2 ctxc1r_rf3 > $L/merge_r.log 2>&1
$PY -u src/post/ctx_seedavg_merge.py ctxc1rb_rf 1 ctxc1rb_rf_s2 2 ctxc1rb_rf3 > $L/merge_rb.log 2>&1
cat $L/merge_r.log $L/merge_rb.log
ARMC_ARM=c1 PRC_ARMC_TAG=_rf3 $PY -u src/post/ctx_armc.py sel > $L/armc_sel.log 2>&1
step "6. holdout chain gate run"
cp -n cache/v64/gate_rohb/reocc_feats.parquet $C/gate_rf3/
PRC_LGB=ctxcorr2r CORR=ctxc1sel_rf3 SCR=$C/gate_rf3 ARMS=all RO_FILE=reocc_feats.parquet $PY -u tests/v50_chain_gate.py > $L/gate_rf3.log 2>&1
step "7. gate"
CTL=rf CTL_DIR=cache/v66 TAG=rf3 GATE_DIR=$C BOARD=236.8136 $PY -u tests/ctx_hour_gate.py > $L/ctx_seedavg_gate.log 2>&1
cat $L/ctx_seedavg_gate.log
if ! grep -q "DECISION: PASS" $L/ctx_seedavg_gate.log; then step "gate FAIL: nothing built"; exit 0; fi
step "8. production chain"; bash scripts/run_v67s_chain.sh ctxc1sel_rf3
step "9. delta, assembly, day recipe"
$PY -u src/post/v64_delta.py $C/delta_rf3.parquet ctxc1sel_rf3 $C > $L/delta.log 2>&1
FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py $S/smart-jigsaw_v54.parquet $C/p67.parquet $C/delta_rf3.parquet > $L/assemble.log 2>&1
$PY -u src/post/day_alloc_build.py $C/p67.parquet $S/smart-jigsaw_v62.parquet $S/smart-jigsaw_v67.parquet > $L/v67_build.log 2>&1
cat $L/delta.log
step "done"
