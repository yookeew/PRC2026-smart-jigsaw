#!/usr/bin/env bash
# v68 build (reports/ctx_huber_preregistration.md), only after the gate passes and the output-spread stop does not fire.
# Needs: run_v68_huber_gate.sh done (production ctxc1sel_rh2hy). Control build B(rh2) = cache/v66/B_rh2.parquet (same machine).
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PRC_SCOPE=jj
PY=.venv/Scripts/python.exe; L=logs/v68; S=data/submissions; X=rh2hy
step() { echo "=== $1  $(date '+%F %H:%M:%S')"; }
step "1. production chain ctxc1sel_$X"; bash run_v68_chain.sh ctxc1sel_$X
step "2. delta, assembly, day stage"
$PY -u src/post/v64_delta.py cache/v68/delta_$X.parquet ctxc1sel_$X cache/v68 > $L/delta_$X.log 2>&1
FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py $S/smart-jigsaw_v54.parquet cache/v68/p_$X.parquet cache/v68/delta_$X.parquet > $L/assemble_$X.log 2>&1
$PY -u src/post/day_alloc_build.py cache/v68/p_$X.parquet $S/smart-jigsaw_v65.parquet cache/v68/B_$X.parquet > $L/build_$X.log 2>&1
step "3. v68"; $PY -u src/post/v68_build.py $S/smart-jigsaw_v66.parquet cache/v68/B_$X.parquet cache/v66/B_rh2.parquet $S/smart-jigsaw_v68.parquet > $L/v68_build.log 2>&1
cat $L/v68_build.log; md5sum $S/smart-jigsaw_v68.parquet
step "done"
