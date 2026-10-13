#!/usr/bin/env bash
# v64's hour-window corrector (reports/ctx_hour_preregistration.md, PROGRESS.md §135) on the four-month blind board.
# Same delta method as the live board: delta = (chain on ctxc1sel_rohb + identity deltas) - (chain on ctxc1sel_ro + identity
# deltas), both chains run here; p = f4m_e + delta (LIRF and NM-line rows frozen); then v63's recipe (OPDI deltas + G2) via
# src/post/day_alloc_build.py; blind_v5 = Jan/Jul from smart-jigsaw_v64, Feb/Jun from this build.
# Holdout / OOF inputs of the new members are scope-independent and already exist (cache/eval, cache/oof).
#
# Run from the repo root:  bash scripts/run_v64_final4m.sh
set -euo pipefail
export PRC_SCOPE=final4m PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe
S=data/submissions/final4m
L=logs/v64/final4m
mkdir -p "$L" cache/v64/final4m
step() { echo "=== $1  $(date '+%F %H:%M')"; }
need() { for f in "$@"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done; }
done_() { [ -f "$1" ] && echo "    (exists: $1)"; }

need data/ranking/final_ranking.parquet data/ranking/final_submitting.parquet data/submissions/smart-jigsaw_v64.parquet \
     "$S/lgb_mixed_nb.parquet" "$S/cat_mixed_f4m.parquet" "$S/catcorr_mixed.parquet" \
     "$S/ctxc1r_ro.parquet" "$S/ctxc1rb_ro.parquet" "$S/ctxc1sel_ro.parquet" "$S/f4m_e.parquet" "$S/f4m_v62.parquet" \
     cache/adsb_pushback/final4m cache/adsb_pushback_v3/final4m cache/adsb_pushback_v4/final4m cache/adsb_pos_v4/final4m \
     cache/v52/final4m/comb_delta_opdi.parquet cache/v52/final4m/nb_delta_opdi.parquet cache/opdi_feats/final4m/opdi_feats.parquet \
     cache/eval/ctxc1sel_rohb_holdout_ev.parquet cache/oof/ctxc1sel_rohb

step "1. Stand re-occupation and identity features (four-month frame)"
done_ cache/v50/final4m/reocc_feats6.parquet || { $PY -u src/features/reocc_feats.py > "$L/reocc_feats.log" 2>&1 &&
                                                  $PY -u src/features/id_feats.py > "$L/id_feats.log" 2>&1; }

step "2. Hour-block corrector members on the four-month ranking rows, then sel"
done_ "$S/ctxc1r_roh.parquet" || ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 PRC_ARMC_TAG=_roh \
    $PY -u src/post/ctx_armc.py prod > "$L/armc_prod_ctxc1r_roh.log" 2>&1
done_ "$S/ctxc1rb_rohb.parquet" || ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_ADSB=1 CTX_HOUR=1 PRC_ARMC_TAG=_rohb \
    $PY -u src/post/ctx_armc.py prod > "$L/armc_prod_ctxc1rb_rohb.log" 2>&1
cp -n "$S/ctxc1r_roh.parquet" "$S/ctxc1r_rohb.parquet"
done_ "$S/ctxc1sel_rohb.parquet" || ARMC_ARM=c1 PRC_ARMC_TAG=_rohb $PY -u src/post/ctx_armc.py sel > "$L/armc_sel_rohb.log" 2>&1

chain() {   # production chain + identity deltas for PRC_LGB=$1, identity deltas into cache/v64/final4m/id_$1
    local N=$1
    export PRC_LGB=$N
    done_ "$S/stack_lgb_catcorr_v3_${N}_adsbq_partial_fb.parquet" || $PY -u src/post/stack_submit.py --cat-corrected --adsb-v3 > "$L/chain_${N}_stack.log" 2>&1
    done_ "$S/smart-jigsaw_v27${N}.parquet" || $PY -u src/post/adsb_combiner.py --full > "$L/chain_${N}_combiner.log" 2>&1
    done_ "$S/chain_${N}.parquet" || $PY -u src/post/neighbour_state.py --v2 > "$L/chain_${N}_neighbour.log" 2>&1
    done_ "$S/chain_${N}_win.parquet" || $PY -u src/post/lobt_window.py "$S/chain_${N}.parquet" "$S/chain_${N}_win.parquet" > "$L/chain_${N}_lobt.log" 2>&1
    done_ "$S/chain_${N}_final.parquet" || $PY -u src/post/nm_lines_nocheck.py "$S/chain_${N}_win.parquet" "$S/chain_${N}_final.parquet" > "$L/chain_${N}_nmlines.log" 2>&1
    local D=cache/v64/final4m/id_$N
    mkdir -p "$D" && cp -n cache/v50/final4m/reocc_feats6.parquet "$D/"
    done_ "$D/comb_delta_all_id.parquet" || SCR=$D ARM=all_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_comb_delta.py > "$L/chain_${N}_comb_delta.log" 2>&1
    done_ "$D/nb_delta_ro_id.parquet" || SCR=$D NB_ARM=ro_id RO_FILE=reocc_feats6.parquet $PY -u src/post/v50_nb_delta.py > "$L/chain_${N}_nb_delta.log" 2>&1
    unset PRC_LGB
}

step "3. Control chain (ctxc1sel_ro). If the other machine's chain_ctxc1sel_ro_final is present it is kept as _ORIG and compared"
[ -f "$S/chain_ctxc1sel_ro_final.parquet" ] && [ ! -f "$S/chain_ctxc1sel_ro_final_ORIG.parquet" ] && \
    cp "$S/chain_ctxc1sel_ro_final.parquet" "$S/chain_ctxc1sel_ro_final_ORIG.parquet" && rm -f "$S/chain_ctxc1sel_ro_final.parquet" "$S/chain_ctxc1sel_ro_win.parquet"
chain ctxc1sel_ro
[ -f "$S/chain_ctxc1sel_ro_final_ORIG.parquet" ] && md5sum "$S/chain_ctxc1sel_ro_final.parquet" "$S/chain_ctxc1sel_ro_final_ORIG.parquet"

step "4. Treatment chain (ctxc1sel_rohb)"
chain ctxc1sel_rohb

step "5. Delta, assembly on f4m_e (LIRF frozen), v63's recipe (OPDI deltas + G2)"
$PY -u - <<'EOF' > "$L/delta.log" 2>&1
import numpy as np, polars as pl
S, V = "data/submissions/final4m", "cache/v64/final4m"
tpl = pl.read_parquet("data/ranking/final_submitting.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64))
col = lambda p, c: tpl.join(pl.read_parquet(p).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col(c).cast(pl.Float64).alias("x")),
                            on="MVT_ID_mvt", how="left")["x"].fill_null(0.0).to_numpy()
ids = lambda n: col(f"{V}/id_{n}/comb_delta_all_id.parquet", "delta") + col(f"{V}/id_{n}/nb_delta_ro_id.parquet", "delta")
new, old = col(f"{S}/chain_ctxc1sel_rohb_final.parquet", "TAXITIME_SEC_mvt"), col(f"{S}/chain_ctxc1sel_ro_final.parquet", "TAXITIME_SEC_mvt")
d = (new + ids("ctxc1sel_rohb")) - (old + ids("ctxc1sel_ro"))
print(f"delta RMS {np.sqrt(np.mean(d ** 2)):.2f}, mean {d.mean():+.3f}")
pl.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"], "delta": d}).write_parquet(f"{V}/delta_rohb.parquet")
EOF
cat "$L/delta.log"
FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py "$S/f4m_e.parquet" "$S/f4m_e_v64.parquet" cache/v64/final4m/delta_rohb.parquet > "$L/assemble.log" 2>&1
$PY -u src/post/day_alloc_build.py "$S/f4m_e_v64.parquet" "$S/f4m_v62.parquet" "$S/f4m_v64.parquet" > "$L/day_alloc_build.log" 2>&1
tail -n 15 "$L/day_alloc_build.log"

step "6. blind_v5 = Jan/Jul from smart-jigsaw_v64, Feb/Jun from f4m_v64"
$PY -u src/post/final4m_compose.py data/submissions/smart-jigsaw_v64.parquet "$S/f4m_v64.parquet" "$S/blind_v5.parquet" > "$L/compose_blind_v5.log" 2>&1
tail -n 25 "$L/compose_blind_v5.log"
step "done"
