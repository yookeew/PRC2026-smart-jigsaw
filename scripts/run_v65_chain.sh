#!/usr/bin/env bash
# Production chain (scripts/run_final4m.sh step 5-6, live-board scope) for PRC_LGB=$1; logs to logs/v65/chain_$1_*.log; identity deltas in cache/v65/id_$1 (used by scripts/run_v65.sh)
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PRC_LGB=$1
PY=.venv/Scripts/python.exe; S=data/submissions; L=logs/v65; N=$1
$PY -u src/post/stack_submit.py --cat-corrected --adsb-v3 > $L/chain_${N}_stack.log 2>&1
$PY -u src/post/adsb_combiner.py --full > $L/chain_${N}_combiner.log 2>&1
$PY -u src/post/neighbour_state.py --v2 > $L/chain_${N}_neighbour.log 2>&1
$PY -u src/post/lobt_window.py $S/chain_${N}.parquet $S/chain_${N}_win.parquet > $L/chain_${N}_lobt.log 2>&1
$PY -u src/post/nm_lines_nocheck.py $S/chain_${N}_win.parquet $S/chain_${N}_final.parquet > $L/chain_${N}_nmlines.log 2>&1
mkdir -p cache/v65/id_$N && cp cache/v50/reocc_feats6.parquet cache/v65/id_$N/
SCR=cache/v65/id_$N ARM=all_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_comb_delta.py > $L/chain_${N}_comb_delta.log 2>&1
SCR=cache/v65/id_$N NB_ARM=ro_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_nb_delta.py > $L/chain_${N}_nb_delta.log 2>&1
echo "chain $N done $(date)"
