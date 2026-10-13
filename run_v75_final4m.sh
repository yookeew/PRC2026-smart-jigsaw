#!/usr/bin/env bash
# Blind board (four months: Jan + Feb + Jun + Jul 2026) carrying everything in the live v75 (PROGRESS.md §162).
#
# The live chain since blind_v5 (v64):
#   v68  corrector ctxc1sel_rf3 (blocks 11-13, members r / rb averaged over LightGBM seeds 42 | 43,44 at 1:2; §152, §154)
#        = v54 + delta(chain rf3 + id deltas - chain ro + id deltas) -> FREEZE_LIRF assembly -> v63's day recipe
#        (v65 rh2 and v66 rf are superseded: every corrector build is a delta against the same control chain ro)
#   v69  lszh_dayclose.py a = 0.75                      v70  lirf_count_dates.py, then nm_lines_remove.py
#   v71  lirf_hedge_repair.py                           v72  lirf_airline_record.py      v75  same, ALR_L2LOOSE=1
#
# Part A (live scope) rebuilds v68 from fresh production fits and compares it with the uploaded file: the reproduction
# check for the corrector half. The post-stages v68 -> v75 were replayed byte for byte (§162).
# Part B runs the same steps on the four-month frame; blind_v7 = Jan/Jul from smart-jigsaw_v75, Feb/Jun from f4m_v75.
#
# Run from the repo root:  bash run_v75_final4m.sh [A|B|B1|all]   (B1: four-month production fits only)   (default all; every step is skipped when its output exists)
set -euo pipefail
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe
PART=${1:-all}
step() { echo "=== $1  $(date '+%F %H:%M:%S')"; }
done_() { [ -e "$1" ] && echo "    (exists: $1)"; }
M="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_FLOW=1 PRC_ARMC_TAG=_rf"

inputs() {    # holdout + OOF inputs of both members, seed 42 and seeds 43,44 (scope-independent; the chain's stack, combiner and
              # neighbour stages fit on them). Run once, in the live scope.
    local L=$1 E=cache/eval
    done_ "$E/ctxc1r_rf_holdout_ev.parquet"     || env $M $PY -u src/post/ctx_armc.py inputs > "$L/armc_inputs_r.log" 2>&1
    done_ "$E/ctxc1rb_rf_holdout_ev.parquet"    || env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py inputs > "$L/armc_inputs_rb.log" 2>&1
    done_ "$E/ctxc1r_rf_s2_holdout_ev.parquet"  || env $M PRC_CTX_SEEDS=43,44 $PY -u src/post/ctx_armc.py inputs > "$L/armc_inputs_r_s2.log" 2>&1
    done_ "$E/ctxc1rb_rf_s2_holdout_ev.parquet" || env $M PRC_CTX_SEEDS=43,44 CTX_ADSB=1 $PY -u src/post/ctx_armc.py inputs > "$L/armc_inputs_rb_s2.log" 2>&1
}

members() {   # production fits of both members, seed 42 and seeds 43,44; merge 1:2; sel -> ctxc1sel_rf3 (in scope $1)
    local S=$1 L=$2
    done_ "$S/ctxc1r_rf.parquet"     || env $M $PY -u src/post/ctx_armc.py prod > "$L/armc_prod_r.log" 2>&1
    done_ "$S/ctxc1rb_rf.parquet"    || env $M CTX_ADSB=1 $PY -u src/post/ctx_armc.py prod > "$L/armc_prod_rb.log" 2>&1
    done_ "$S/ctxc1r_rf_s2.parquet"  || env $M PRC_CTX_SEEDS=43,44 $PY -u src/post/ctx_armc.py prod > "$L/armc_prod_r_s2.log" 2>&1
    done_ "$S/ctxc1rb_rf_s2.parquet" || env $M PRC_CTX_SEEDS=43,44 CTX_ADSB=1 $PY -u src/post/ctx_armc.py prod > "$L/armc_prod_rb_s2.log" 2>&1
    done_ "$S/ctxc1r_rf3.parquet"    || $PY -u src/post/ctx_seedavg_merge.py ctxc1r_rf 1 ctxc1r_rf_s2 2 ctxc1r_rf3 > "$L/merge_r.log" 2>&1
    done_ "$S/ctxc1rb_rf3.parquet"   || $PY -u src/post/ctx_seedavg_merge.py ctxc1rb_rf 1 ctxc1rb_rf_s2 2 ctxc1rb_rf3 > "$L/merge_rb.log" 2>&1
    done_ "$S/ctxc1sel_rf3.parquet"  || ARMC_ARM=c1 PRC_ARMC_TAG=_rf3 $PY -u src/post/ctx_armc.py sel > "$L/armc_sel.log" 2>&1
    tail -n 1 "$L/merge_r.log" "$L/merge_rb.log" "$L/armc_sel.log"
}

chain() {     # production chain + identity deltas for PRC_LGB=$1 (scope set by the caller); id deltas into $4
    local N=$1 S=$2 L=$3 D=$4 RO=$5
    export PRC_LGB=$N
    done_ "$S/stack_lgb_catcorr_v3_${N}_adsbq_partial_fb.parquet" || $PY -u src/post/stack_submit.py --cat-corrected --adsb-v3 > "$L/chain_${N}_stack.log" 2>&1
    done_ "$S/smart-jigsaw_v27${N}.parquet" || $PY -u src/post/adsb_combiner.py --full > "$L/chain_${N}_combiner.log" 2>&1
    done_ "$S/chain_${N}.parquet" || $PY -u src/post/neighbour_state.py --v2 > "$L/chain_${N}_neighbour.log" 2>&1
    done_ "$S/chain_${N}_win.parquet" || $PY -u src/post/lobt_window.py "$S/chain_${N}.parquet" "$S/chain_${N}_win.parquet" > "$L/chain_${N}_lobt.log" 2>&1
    done_ "$S/chain_${N}_final.parquet" || $PY -u src/post/nm_lines_nocheck.py "$S/chain_${N}_win.parquet" "$S/chain_${N}_final.parquet" > "$L/chain_${N}_nmlines.log" 2>&1
    mkdir -p "$D" && cp -n "$RO" "$D/"
    done_ "$D/comb_delta_all_id.parquet" || SCR=$D ARM=all_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_comb_delta.py > "$L/chain_${N}_comb_delta.log" 2>&1
    done_ "$D/nb_delta_ro_id.parquet" || SCR=$D NB_ARM=ro_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_nb_delta.py > "$L/chain_${N}_nb_delta.log" 2>&1
    unset PRC_LGB
}

if [ "$PART" = A ] || [ "$PART" = all ]; then
    export PRC_SCOPE=jj
    S=data/submissions L=logs/v75/rebuild_v68 C=cache/v75/jj
    mkdir -p "$L" "$C"
    if [ -e cache/eval/ctxc1sel_rf3_holdout_ev.parquet ] && [ -d cache/oof/ctxc1sel_rf3 ] && [ -e "$S/ctxc1sel_rf3.parquet" ]; then
        step "A0/A1 skipped: ctxc1sel_rf3 holdout, OOF and ranking files present (copied from the machine that built v68)"
    else
        step "A0. holdout + OOF inputs of the members (seed 42 and 43,44)";  inputs "$L"
        step "A1. live-scope members (seed 42 and 43,44), merge, sel";  members "$S" "$L"
    fi
    step "A2. live-scope chain on ctxc1sel_rf3";  chain ctxc1sel_rf3 "$S" "$L" "$C/id_ctxc1sel_rf3" cache/v50/reocc_feats6.parquet
    step "A3. delta vs chain ro, assembly on v54, day recipe -> v68 rebuild"
    $PY -u src/post/v64_delta.py "$C/delta_rf3.parquet" ctxc1sel_rf3 "$C" > "$L/delta.log" 2>&1
    FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py $S/smart-jigsaw_v54.parquet "$C/p68.parquet" "$C/delta_rf3.parquet" > "$L/assemble.log" 2>&1
    # the "v62" input supplies LSZH / LIRF rows only: upload v57 is team v62 there (§142 numbering; identical to v68 on both)
    $PY -u src/post/day_alloc_build.py "$C/p68.parquet" $S/smart-jigsaw_v57.parquet "$C/v68_rebuild.parquet" > "$L/day_alloc_build.log" 2>&1
    $PY - "$C/v68_rebuild.parquet" $S/smart-jigsaw_v68.parquet <<'EOF' | tee "$L/compare_v68.log"
import sys, numpy as np, polars as pl
a, b = (pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)) for f in sys.argv[1:3])
d = a.join(b, on="MVT_ID_mvt", suffix="_u")
x = (d["TAXITIME_SEC_mvt"] - d["TAXITIME_SEC_mvt_u"]).to_numpy()
print(f"v68 rebuild vs uploaded v68: identical {np.all(x == 0)}; rows differing {(x != 0).sum():,}; RMS {np.sqrt(np.mean(x ** 2)):.3f} s; "
      f"max |d| {np.abs(x).max():.0f}; RMS over the file as a board effect ~{np.sqrt(np.mean(x ** 2)) ** 2 / (2 * 234.5):.4f} s (upper bound)")
EOF
    unset PRC_SCOPE
fi

if [ "$PART" = B ] || [ "$PART" = B1 ] || [ "$PART" = all ]; then
    export PRC_SCOPE=final4m
    S=data/submissions/final4m L=logs/v75/final4m C=cache/v75/final4m
    mkdir -p "$L" "$C"
    step "B1. four-month members, merge, sel";  members "$S" "$L"
    [ "$PART" = B1 ] && { step "B1 only: done"; exit 0; }
    for f in cache/eval/ctxc1sel_rf3_holdout_ev.parquet cache/oof/ctxc1sel_rf3; do [ -e "$f" ] || { echo "missing $f (run part A, or copy it)"; exit 1; }; done
    step "B2. four-month chain on ctxc1sel_rf3 (control chain ro and its id deltas: run_v64_final4m.sh)"
    chain ctxc1sel_rf3 "$S" "$L" "$C/id_ctxc1sel_rf3" cache/v50/final4m/reocc_feats6.parquet
    step "B3. delta vs chain ro, assembly on f4m_e (LIRF frozen), day recipe -> f4m_v68"
    $PY -u - "$S" "$C" <<'EOF' > "$L/delta.log" 2>&1
import sys, numpy as np, polars as pl
S, C, V = sys.argv[1], sys.argv[2], "cache/v64/final4m"
tpl = pl.read_parquet("data/ranking/final_submitting.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64))
col = lambda p, c: tpl.join(pl.read_parquet(p).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col(c).cast(pl.Float64).alias("x")),
                            on="MVT_ID_mvt", how="left")["x"].fill_null(0.0).to_numpy()
ids = lambda D: col(f"{D}/comb_delta_all_id.parquet", "delta") + col(f"{D}/nb_delta_ro_id.parquet", "delta")
new, old = col(f"{S}/chain_ctxc1sel_rf3_final.parquet", "TAXITIME_SEC_mvt"), col(f"{S}/chain_ctxc1sel_ro_final.parquet", "TAXITIME_SEC_mvt")
d = (new + ids(f"{C}/id_ctxc1sel_rf3")) - (old + ids(f"{V}/id_ctxc1sel_ro"))
print(f"delta RMS {np.sqrt(np.mean(d ** 2)):.2f}, mean {d.mean():+.3f}")
pl.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"], "delta": d}).write_parquet(f"{C}/delta_rf3.parquet")
EOF
    cat "$L/delta.log"
    FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py "$S/f4m_e.parquet" "$C/p68.parquet" "$C/delta_rf3.parquet" > "$L/assemble.log" 2>&1
    $PY -u src/post/day_alloc_build.py "$C/p68.parquet" "$S/f4m_v62.parquet" "$S/f4m_v68.parquet" > "$L/day_alloc_build.log" 2>&1
    step "B4. post-stages as v69 -> v75"
    $PY -u src/post/lszh_dayclose.py "$S/f4m_v68.parquet" "$C/f4m_v69.parquet" 0.75 > "$L/v69_lszh.log" 2>&1
    $PY -u src/post/lirf_count_dates.py "$C/f4m_v69.parquet" "$C/f4m_v70a.parquet" > "$L/v70_count_dates.log" 2>&1
    $PY -u src/post/nm_lines_remove.py "$C/f4m_v70a.parquet" "$C/f4m_v70.parquet" "$S/chain_ctxc1sel_rf3_win.parquet" "$S/chain_ctxc1sel_rf3_final.parquet" > "$L/v70_nm_lines_remove.log" 2>&1
    $PY -u src/post/lirf_hedge_repair.py "$C/f4m_v70.parquet" "$C/f4m_v71.parquet" > "$L/v71_hedge_repair.log" 2>&1
    $PY -u src/post/lirf_airline_record.py "$C/f4m_v71.parquet" "$C/f4m_v72.parquet" > "$L/v72_alr.log" 2>&1
    ALR_L2LOOSE=1 $PY -u src/post/lirf_airline_record.py "$C/f4m_v72.parquet" "$S/f4m_v75.parquet" > "$L/v75_l2loose.log" 2>&1
    step "B5. blind_v7 = Jan/Jul from smart-jigsaw_v75, Feb/Jun from f4m_v75"
    $PY -u src/post/final4m_compose.py data/submissions/smart-jigsaw_v75.parquet "$S/f4m_v75.parquet" "$S/blind_v7.parquet" > "$L/compose_blind_v7.log" 2>&1
    tail -n 25 "$L/compose_blind_v7.log"; md5sum "$S/blind_v7.parquet"
    unset PRC_SCOPE
fi
step "done"
