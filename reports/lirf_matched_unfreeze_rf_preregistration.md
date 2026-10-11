# LIRF NM-matched rows from the flow-block corrector chain (v66) -- pre-registration (third attempt)

Written 2026-10-11 **before any LIRF number for `ctxc1sel_rf` on the holdout chain or the `ctxc1sel` OOF comparison was
computed**. Base: v66 (reports/ctx_flow_preregistration.md, gate PASS).

**Third attempt, disclosed.** Attempt 1 (`ctxc1sel_roh`, reports/lirf_matched_unfreeze_preregistration.md): pooled −0.080
vs bar −0.10, OOF June +0.07 → FAIL. Attempt 2 (`ctxc1sel_rh2`, reports/lirf_matched_unfreeze_rh2_preregistration.md):
pooled −0.216 vs bar −0.15, OOF June +0.87 (May +0.62, Sep +2.10) → FAIL on (e).
**Already seen before this note (disclosed):** v66's gate log has LIRF on the holdout chain, all LIRF rows: trimmed +0.98
vs `ctxc1sel_rh2` (NM-unmatched rows in that chain come from v41's chain, so this is mostly NM-matched rows); and member r
at corrector level vs `ctxc1r_rh2`, LIRF NM-matched 12 months 343.20 → 342.48 (9/12 months, June −1.59). Neither is the
gate's comparison (control below is the frozen v42-era recipe, OOF control is `ctxc1sel`).
Because this is a second retry, the pooled bar is **tightened again to −0.20**; everything else is unchanged.

## Candidate

v66 with LIRF **NM-matched** rows set to `chain_ctxc1sel_rf_final` (v66's production chain, LOBT window and NM lines
included). LIRF NM-unmatched rows (hedges v45 / v47 / v48, v49 caps) unchanged; no identity-arm deltas and no day stage
at LIRF (as now).

## Gate (`GATE=cache/v66/gate_rf TAG=rf BAR=-0.20 BOARD=237.0694 tests/lirf_matched_gate.py`)

Control = `cache/eval/v42_chain_holdout.parquet` (the recipe behind the frozen LIRF rows); treatment = v66's holdout chain.
**Pass iff:** (a) pooled trimmed ≤ **−0.20** s; (b) trimmed better in Jan and in Jul; (c) full ≤ 0; (d) body ≤ 0;
(e) corrector level, OOF months, LIRF NM-matched trimmed: `ctxc1sel_rf` better than `ctxc1sel` in February, in June and
in ≥ 7 of 10 months.

## Upload rule

Separate file (v67) after v66 is read, so the LIRF part is read cleanly: keep iff board < the best at that time.

## Outcome (2026-10-11; `logs/v66/lirf_matched_gate_rf.log`)

LIRF NM-matched holdout rows 26,131 (change RMS 79.6 s, mean −2.6): trimmed 383.71 → 383.32, body 313.89 → 312.77.
Pooled: trimmed **−0.047** (a fails: bar −0.20), Jan −0.156 / **Jul +0.015** (b fails), full −0.036 (c ok), body −0.139 (d ok).
OOF corrector level by month: Feb −3.10, Mar −2.93, Apr −6.29, May −0.27, Jun −0.72, Aug −4.43, Sep +0.68, Oct −4.99,
Nov −2.21, Dec −4.87 (e ok, 9/10). 2026 projection −0.05 s.
**FAIL; v67 not built. LIRF stays frozen.** The corrector-level LIRF gains do not survive the holdout chain's post-stages.
Three attempts on this idea are now spent; it is closed.
