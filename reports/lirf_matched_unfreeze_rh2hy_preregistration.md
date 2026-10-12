# LIRF NM-matched rows from the Huber-hybrid corrector chain -- pre-registration (third attempt)

Written 2026-10-11 **before any LIRF holdout or OOF number of `ctxc1sel_rh2hy` was computed** (its gate,
reports/ctx_huber_preregistration.md, was running). Runs only if that gate passes and its output-spread stop does not fire.

**Earlier attempts, disclosed.** Every LIRF row has been frozen since v48 (v63's 26,516 LIRF NM-matched rows equal
`chain_ctxc1sel_final`).
- First (`ctxc1sel_roh`, bar −0.10): pooled trimmed −0.080, OOF June +0.07: FAIL.
- Second (`ctxc1sel_rh2`, bar −0.15): pooled trimmed −0.216, Jan −0.123 / Jul −0.275, full −0.178, body −0.233; OOF
  June +0.87 (7 of 10 months better): FAIL on (e) only.
- Corrector-level screens of the Huber hybrid vs L2 (reports/ctx_huber_preregistration.md, member `r`, 8 OOF months):
  LIRF NM-matched trimmed −1.06 (Feb + Aug: Feb −1.48, Aug −0.85) and −3.63 (May + Nov: May −3.45, Nov −3.89). June was
  not in either screen.

Because this is a third attempt on the same idea, the pooled bar is **tightened to −0.20**; everything else is unchanged.

## Candidate

The v68 build (if built) with LIRF **NM-matched** rows set to `chain_ctxc1sel_rh2hy_final` (production chain of the
hybrid corrector, LOBT window and NM lines included). LIRF NM-unmatched rows (hedges v45/v47/v48, v49 caps) unchanged; no
identity-arm deltas and no day stage at LIRF (as now).

## Gate (`GATE=cache/v68/gate_rh2hy TAG=rh2hy BAR=-0.20 BOARD=236.8 tests/lirf_matched_gate.py`)

Control = `cache/eval/v42_chain_holdout.parquet` (the recipe behind the frozen LIRF rows); treatment = the hybrid's
holdout chain. **Pass iff:** (a) pooled trimmed ≤ **−0.20** s; (b) trimmed better in Jan and in Jul; (c) full ≤ 0;
(d) body ≤ 0; (e) corrector level, OOF months, LIRF NM-matched trimmed: `ctxc1sel_rh2hy` better than `ctxc1sel` in
February, in June and in ≥ 7 of 10 months.

## Upload rule

Separate file after v68 is read, so the LIRF part is read cleanly: keep iff board < the best at that time.

## Outcome (2026-10-11): not run -- its precondition (the Huber gate) failed. On the Huber gate's holdout chain LIRF trimmed
was +1.91 worse than `ctxc1sel_rh2`'s, so this candidate would not have helped either.
