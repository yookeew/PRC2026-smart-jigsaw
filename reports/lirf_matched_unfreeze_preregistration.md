# LIRF NM-matched rows from the current corrector chain (v65) -- pre-registration

Written 2026-10-11 **before any LIRF holdout number below was computed**. Base: v64 candidate (CTX_HOUR, §135).
Every LIRF row has been frozen since v48 (`FREEZE_LIRF=1` in v50–v54's assembly; v55+ leave LIRF untouched): v63's 26,516
LIRF NM-matched rows equal `chain_ctxc1sel_final` (the v42-era chain) exactly. The freeze was set because LIRF changes
kept missing on the board (§87, §121), all of them on **NM-unmatched / echo** rows, and because `ctxc1sel_ro` was +0.3
worse at LIRF on the holdout (§124). The CTX_HOUR gate (§135) shows the hour-block chain better at LIRF.

## Candidate

**v65** = v64 with LIRF **NM-matched** rows set to `chain_ctxc1sel_roh_final` (production chain of the hour-block corrector,
LOBT window and NM lines included). LIRF NM-unmatched rows (hedges v45/v47/v48, v49 caps) unchanged. No identity-arm
deltas at LIRF (never applied there). No day-level stage at LIRF (as now).

## Gate (holdout, Jan + Jul 2025, LIRF NM-matched rows; the rest of the file is v64)

Control = `cache/eval/v42_chain_holdout.parquet` (the recipe behind v63's LIRF rows); treatment = `cache/v64/gate_roh/chain_all`
(hour-block corrector through v50's post-stages). Effect expressed on the whole holdout (other rows identical).
**Pass iff:** (a) pooled trimmed ≤ −0.10 s; (b) trimmed better in Jan and in Jul; (c) full ≤ 0; (d) body ≤ 0;
(e) corrector level, OOF months, LIRF NM-matched trimmed: `ctxc1sel_roh` better than `ctxc1sel` in February, in June and
in ≥ 7 of 10 months. 2026 projection reported.

## Upload rule

Separate file, so the LIRF part is read cleanly: **keep iff board < v64's score** (upload after v64 is read).

## Outcome (2026-10-11; `logs/v64/lirf_matched_gate.log`)

LIRF NM-matched holdout rows 26,131 (change RMS 62.9 s): on those rows trimmed 383.71 → 383.04, body 313.89 → 313.46.
Pooled: trimmed −0.080 (a fails: bar −0.10), Jan −0.140 / Jul −0.047 (b ok), full −0.063, body −0.053. OOF corrector level
by month: Feb −4.84, Mar −4.62, Apr −2.63, May +2.77, Jun +0.07, Aug −1.69, Sep +2.31, Oct −2.80, Nov −1.27, Dec −0.72
(e fails: June). 2026 projection −0.08 s. **FAIL; v65 not built. LIRF stays frozen.**
