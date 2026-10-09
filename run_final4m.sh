#!/usr/bin/env bash
# Four-month blind-board chain (Jan + Feb + Jun + Jul 2026), v55's recipe on final_ranking.parquet. PROGRESS.md §126.
# Every step writes under final4m/ subfolders (src/scope.py); finished steps are skipped, so the script can be re-run.
#
# Prerequisites (each also PRC_SCOPE=final4m):
#   features   src/features/export_model_inputs.py, src/features/neighbour_feats.py
#   ADS-B      src/link/adsb_pushback.py [--v3|--v4] DAYS, src/link/adsb_pos.py DAYS  (2026 days -> <cache>/final4m/)
#   LightGBM   src/models/run_neighbour_base.py prod  -> models/lgb_mixed_nb/, data/submissions/final4m/lgb_mixed_nb.parquet
#   CatBoost   Colab cell "CatBoost for the four-month pipeline" -> data/submissions/final4m/cat_mixed_f4m.parquet,
#              cache/eval/final4m/cat_mixed_f4m_rank.parquet
# Output: data/submissions/final4m/f4m_full.parquet (all four months), then src/post/final4m_compose.py builds the upload.
#
# Run from the repo root:  bash run_final4m.sh
set -euo pipefail
export PRC_SCOPE=final4m PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/Scripts/python.exe
S=data/submissions/final4m
E=cache/eval/final4m
V=cache/v50/final4m
L=logs/final4m
mkdir -p "$L"
step() { echo "=== $1  $(date '+%F %H:%M')"; }
need() { for f in "$@"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done; }
done_() { [ -f "$1" ] && echo "    (exists: $1)"; }

need "$S/lgb_mixed_nb.parquet" "$S/cat_mixed_f4m.parquet" "$E/cat_mixed_f4m_rank.parquet" models/lgb_mixed_nb/meta.json

step "1. LightGBM ranking echo probabilities from the saved model (LIRF hedges)"
done_ "$E/lgb_mixed_nb_rank.parquet" || $PY -u -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from models.fit import predict_saved
predict_saved(Path('models/lgb_mixed_nb'), Path('cache/features_nb/final4m'), name='lgb_mixed_nb_saved',
              rank_out='$E/lgb_mixed_nb_rank.parquet')" > "$L/lgb_predict_saved.log" 2>&1

step "2. OOF corrector on CatBoost (12-month corrector)"
done_ "$S/catcorr_mixed.parquet" || $PY -u src/post/corrector_v2.py > "$L/corrector_v2.log" 2>&1

step "3. Stand re-occupation and identity features"
done_ "$V/reocc_feats6.parquet" || { $PY -u src/features/reocc_feats.py > "$L/reocc_feats.log" 2>&1 &&
                                    $PY -u src/features/id_feats.py > "$L/id_feats.log" 2>&1; }

step "4. Context corrector, stand-aware members r_ro / rb_ro, then sel"
done_ "$S/ctxc1r_ro.parquet" || ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 PRC_ARMC_TAG=_ro \
    $PY -u src/post/ctx_armc.py prod > "$L/armc_prod_ctxc1r_ro.log" 2>&1
done_ "$S/ctxc1rb_ro.parquet" || ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_ADSB=1 PRC_ARMC_TAG=_ro \
    $PY -u src/post/ctx_armc.py prod > "$L/armc_prod_ctxc1rb_ro.log" 2>&1
done_ "$S/ctxc1sel_ro.parquet" || ARMC_ARM=c1 PRC_ARMC_TAG=_ro $PY -u src/post/ctx_armc.py sel > "$L/armc_sel_ro.log" 2>&1

step "5. Chain: stack + ADS-B stages -> combiner -> neighbour stage -> LOBT window -> NM lines"
export PRC_LGB=ctxc1sel_ro
done_ "$S/stack_lgb_catcorr_v3_ctxc1sel_ro_adsbq_partial_fb.parquet" || \
    $PY -u src/post/stack_submit.py --cat-corrected --adsb-v3 > "$L/chain_stack.log" 2>&1
done_ "$S/smart-jigsaw_v27ctxc1sel_ro.parquet" || $PY -u src/post/adsb_combiner.py --full > "$L/chain_combiner.log" 2>&1
done_ "$S/chain_ctxc1sel_ro.parquet" || $PY -u src/post/neighbour_state.py --v2 > "$L/chain_neighbour.log" 2>&1
done_ "$S/chain_ctxc1sel_ro_win.parquet" || \
    $PY -u src/post/lobt_window.py "$S/chain_ctxc1sel_ro.parquet" "$S/chain_ctxc1sel_ro_win.parquet" > "$L/chain_lobt.log" 2>&1
done_ "$S/chain_ctxc1sel_ro_final.parquet" || \
    $PY -u src/post/nm_lines_nocheck.py "$S/chain_ctxc1sel_ro_win.parquet" "$S/chain_ctxc1sel_ro_final.parquet" > "$L/chain_nmlines.log" 2>&1

step "6. Identity-arm deltas (combiner all_id, neighbour stage ro_id)"
done_ "$V/comb_delta_all_id.parquet" || ARM=all_id RO_FILE=reocc_feats6.parquet \
    $PY -u src/post/v50_comb_delta.py > "$L/comb_delta.log" 2>&1
done_ "$V/nb_delta_ro_id.parquet" || NB_ARM=ro_id RO_FILE=reocc_feats6.parquet \
    $PY -u src/post/v50_nb_delta.py > "$L/nb_delta.log" 2>&1
unset PRC_LGB

step "7. Assemble chain + deltas (LIRF and NM-line rows frozen, LOBT window last)"
done_ "$S/f4m_a.parquet" || FREEZE_LIRF=1 $PY -u src/post/v50_assemble.py "$S/chain_ctxc1sel_ro_final.parquet" "$S/f4m_a.parquet" \
    "$V/comb_delta_all_id.parquet" "$V/nb_delta_ro_id.parquet" > "$L/assemble.log" 2>&1

step "8. LIRF NM-unmatched hedges: signals (v45), twin bounds (v47), round 2 (v48)"
done_ "$S/f4m_b.parquet" || $PY -u src/post/lirf_signals.py "$S/f4m_a.parquet" "$S/f4m_b.parquet" > "$L/lirf_signals.log" 2>&1
done_ "$S/f4m_c.parquet" || $PY -u src/post/lirf_twin_bounds.py f4m_b.parquet f4m_c.parquet > "$L/lirf_twin.log" 2>&1
done_ "$S/f4m_d.parquet" || $PY -u src/post/lirf_hedge2.py f4m_c.parquet f4m_d.parquet > "$L/lirf_hedge2.log" 2>&1

step "9. v49 fixes (LSZH level per month, EHAM storm days, dominated LIRF rows)"
done_ "$S/f4m_e.parquet" || $PY -u src/post/v49_bundle.py f4m_d.parquet f4m_e.parquet > "$L/v49_bundle.log" 2>&1

step "10. Day-level reconciliation (v55 settings: LIRF untouched, LSZH skipped)"
done_ "$S/f4m_full.parquet" || PRU_LIRF=0 $PY -u src/post/pru_daily_recon.py "$S/f4m_e.parquet" "$S/f4m_full.parquet" \
    cache/v54/v48m_chain_holdout.parquet cache/v54/ctxc1sel_ro_chain_all_holdout.parquet > "$L/day_recon.log" 2>&1

step "done"
md5sum "$S/f4m_full.parquet"
