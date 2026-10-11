#!/usr/bin/env bash
# Production chain (as run_v66_chain.sh) for PRC_LGB=$1; logs to logs/v67/chain_$1_*.log; identity deltas in cache/v67/id_$1 (used by run_v67.sh)
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PRC_LGB=$1
PY=.venv/Scripts/python.exe; S=data/submissions; L=logs/v67; N=$1
mkdir -p $L
$PY -u src/post/stack_submit.py --cat-corrected --adsb-v3 > $L/chain_${N}_stack.log 2>&1
$PY -u src/post/adsb_combiner.py --full > $L/chain_${N}_combiner.log 2>&1
$PY -u src/post/neighbour_state.py --v2 > $L/chain_${N}_neighbour.log 2>&1
$PY -u src/post/lobt_window.py $S/chain_${N}.parquet $S/chain_${N}_win.parquet > $L/chain_${N}_lobt.log 2>&1
$PY -u src/post/nm_lines_nocheck.py $S/chain_${N}_win.parquet $S/chain_${N}_final.parquet > $L/chain_${N}_nmlines.log 2>&1
mkdir -p cache/v67/id_$N && cp cache/v50/reocc_feats6.parquet cache/v67/id_$N/
SCR=cache/v67/id_$N ARM=all_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_comb_delta.py > $L/chain_${N}_comb_delta.log 2>&1
SCR=cache/v67/id_$N NB_ARM=ro_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_nb_delta.py > $L/chain_${N}_nb_delta.log 2>&1
echo "chain $N done $(date)"
