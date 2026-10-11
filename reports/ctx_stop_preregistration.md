# Runway stoppage / discharge block (block 14) in the context corrector (CTX_STOP) -- pre-registration

Written 2026-10-11 **before any gate number below was computed**. Base: v66 (block 13, reports/ctx_flow_preregistration.md,
gate PASS, board pending). Focus: the Jan/Jul holdout and the live board; a general model improvement, no board fitting.

## Why

In an A-CDM airport a late flight normally waits at the gate, so a long taxi needs something that happened after
pushback: a takeoff stoppage (storm, runway closure, ground stop) followed by a discharge burst. Blocks 7 / 11 / 12 see
cadence and neighbour residuals, not the stoppage itself.

**Disclosed screens on the 2025 data** (PROGRESS §138, `tests/flow_screen/stop_screen.py`, before this note): paired
residual LightGBM on top of v66's `ctxc1sel_rf` OOF + holdout output (7 periods, leave-period-out):
- the block below: trimmed **−0.317**, body −0.064; by period Jan +0.01, Jul −0.43, Feb–Mar −0.96, Apr–May −0.16,
  Jun+Aug −0.16, Sep–Oct −0.30, Nov–Dec −0.12; LTFM −1.04, LIRF −0.82, the other airports −0.00 to −0.10.
- dropped (not part of this test): airline network −0.08 / −0.14 (Jan worse), neighbour rotation pressure −0.15 (most
  airports worse), rotation pressure + stoppage −0.36.
- For scale: block 13 screened −0.60 and gave gate −0.35 (non-LIRF −0.71). This block is expected to be weaker and may
  fail (b) on January.

## Arm

Block 14 (`CTX_STOP=1`, `tests/context_corrector_test.py` → `src/post/stop_feats.build`), 9 label-free inputs from the
frame's own movements. For each departure at takeoff T, over [T − 60 min, T] (the window start counts as a boundary):
- same departure runway's takeoffs: longest gap between consecutive takeoffs `st_rw_max`, time from that gap's end to T
  `st_rw_since`, takeoffs since it ended `st_rw_cnt`;
- all the airport's takeoffs: `st_ap_max`, `st_ap_since`, `st_ap_cnt`;
- the airport's landings: `st_ar_max`, `st_ar_since`, `st_ar_cnt`.
Both members refitted with blocks 11 + 12 + 13 + 14 (`PRC_ARMC_TAG=_rs`): `ctxc1r_rs`, `ctxc1rb_rs` (+ `CTX_ADSB=1`),
sel **`ctxc1sel_rs`**. Nothing else changes.

## Gate

Control = v66's production corrector `ctxc1sel_rf` (holdout chain `cache/v66/gate_rf/chain_all.parquet`, OOF
`cache/oof/ctxc1sel_rf`). Run: `CTL=rf CTL_DIR=cache/v66 TAG=rs GATE_DIR=cache/v67 BOARD=<best at the time> tests/ctx_hour_gate.py`.
**Pass iff** (as §135 / §136 / §138): (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul;
(c) holdout chain full ≤ 0; (d) holdout chain body ≤ +0.10; (e) OOF trimmed better in February and in June and in ≥ 7 of
10 OOF months; (f) no block-14 input with pooled |SMD| > 0.25, 2025 training rows vs 2026 ranking rows
(`tests/ctx_stop_drift.py`).

## Build if it passes

As v66: production fits, `sel`, production chain (`run_v67_chain.sh`), delta = (chain_rs + id deltas) − (chain_ctxc1sel_ro
+ id deltas), p = v54 + delta (`FREEZE_LIRF=1`), v63's day recipe → **v67**. Keep iff board < the best at that time
(v66's score if v66 is kept, else 237.0694).

## Amendments

**A1 (2026-10-11, before any gate number).** The first launch stopped in the drift script: rows with no landing in the past
hour carried NaN rather than null in `st_ar_*`, which the script's `drop_nulls` kept (empty per-airport list). `stop_feats.build`
now returns null there (as `hour_feats` does); the corrector sees NaN either way, so the model inputs are unchanged. The
run was relaunched from the start; nothing had been fitted.
