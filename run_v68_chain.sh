#!/usr/bin/env bash
# Production chain (run_v65_chain.sh steps; logs in logs/v68, identity deltas in cache/v68/id_$1) for PRC_LGB=$1.
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PRC_LGB=$1 PRC_SCOPE=jj
PY=.venv/Scripts/python.exe; S=data/submissions; L=logs/v68; N=$1
$PY -u src/post/stack_submit.py --cat-corrected --adsb-v3 > $L/chain_${N}_stack.log 2>&1
$PY -u src/post/adsb_combiner.py --full > $L/chain_${N}_combiner.log 2>&1
$PY -u src/post/neighbour_state.py --v2 > $L/chain_${N}_neighbour.log 2>&1
$PY -u src/post/lobt_window.py $S/chain_${N}.parquet $S/chain_${N}_win.parquet > $L/chain_${N}_lobt.log 2>&1
$PY -u src/post/nm_lines_nocheck.py $S/chain_${N}_win.parquet $S/chain_${N}_final.parquet > $L/chain_${N}_nmlines.log 2>&1
mkdir -p cache/v68/id_$N && cp cache/v50/reocc_feats6.parquet cache/v68/id_$N/
SCR=cache/v68/id_$N ARM=all_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_comb_delta.py > $L/chain_${N}_comb_delta.log 2>&1
SCR=cache/v68/id_$N NB_ARM=ro_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_nb_delta.py > $L/chain_${N}_nb_delta.log 2>&1
echo "chain $N done $(date)"
