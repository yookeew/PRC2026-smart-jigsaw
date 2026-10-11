# LIRF NM-matched rows from the block-12 corrector chain -- pre-registration (second attempt)

Written 2026-10-11 **before any v65 / rh2 number was computed** (the rh2 fits are running). Base: v65 if its gate passes
(reports/ctx_hour2_preregistration.md); otherwise this test is not run.

**Second attempt, disclosed.** The first attempt (reports/lirf_matched_unfreeze_preregistration.md, hour-block corrector
`ctxc1sel_roh`) failed: pooled trimmed −0.080 against a −0.10 bar, OOF June +0.07. Because this is a retry of the same idea
on a new corrector, the pooled bar is **tightened to −0.15**; everything else is unchanged.

## Candidate

v65 with LIRF **NM-matched** rows set to `chain_ctxc1sel_rh2_final` (the production chain of v65's corrector, LOBT window
and NM lines included). LIRF NM-unmatched rows (hedges v45/v47/v48, v49 caps) unchanged; no identity-arm deltas and no day
stage at LIRF (as now).

## Gate (`GATE=cache/v65/gate_rh2 TAG=rh2 BAR=-0.15 BOARD=237.4538 tests/lirf_matched_gate.py`)

Control = `cache/eval/v42_chain_holdout.parquet` (the recipe behind the frozen LIRF rows); treatment = v65's holdout chain.
**Pass iff:** (a) pooled trimmed ≤ **−0.15** s; (b) trimmed better in Jan and in Jul; (c) full ≤ 0; (d) body ≤ 0;
(e) corrector level, OOF months, LIRF NM-matched trimmed: `ctxc1sel_rh2` better than `ctxc1sel` in February, in June and
in ≥ 7 of 10 months.

## Upload rule

Separate file after v65 is read, so the LIRF part is read cleanly: keep iff board < the best at that time.

## Outcome (2026-10-11; `logs/v65/lirf_matched_gate_rh2.log`)

LIRF NM-matched holdout rows 26,131 (change RMS 71.7 s, mean −2.2): trimmed 383.71 → 381.89, body 313.89 → 312.00.
Pooled: trimmed **−0.216** (a ok, bar −0.15), Jan −0.123 / Jul −0.275 (b ok), full −0.178 (c ok), body −0.233 (d ok).
OOF corrector level by month: Feb −3.74, Mar −3.68, Apr −4.83, May +0.62, **Jun +0.87**, Aug −2.92, Sep +2.10, Oct −2.55,
Nov −0.87, Dec −3.32 (e fails: June; 7 of 10 better). 2026 projection −0.23 s.
**FAIL; not built. LIRF stays frozen.** June was also the failing month of the first attempt (+0.07).
