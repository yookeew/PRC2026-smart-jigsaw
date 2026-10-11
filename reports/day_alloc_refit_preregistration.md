# G2 refit on the current corrector -- pre-registration

Written 2026-10-11 **before any number below was computed**. Base: v64 = 237.4538 (v65 pending, §136).

## Why

G2 (§133, v63) learns the day-shift allocation s(x)·λ̂. Both of its 2025 ingredients come from the frame
`tests/v57_dsm.frame25()`: v50's holdout chain on the **old** corrector `ctxc1sel` plus `ctxc1sel`'s OOF folds.
- the slope model s(x) is fitted on that frame's residuals and λ̂;
- the daily λ̂ solve uses residual distributions (eta) from that frame's calibration rows (the 2026 λ̂ in
  `day_alloc_build.py` uses its P01 = Jan + Jul 2025 holdout chain).
Production has since applied G2 on top of newer correctors (`ctxc1sel_ro` in v54 → `_rohb` in v64 → `_rh2` if v65 passes),
whose 2025 residuals are smaller and partly de-correlated within the day. The ingredients are stale; this test refits them.

## Arms (evaluated on frame B, leave-period-out over the six 2025 periods, eight shifted airports)

Frame B = `frame25` with p from the production corrector of the file the refit would be built on:
**`ctxc1sel_rh2`** (holdout chain `cache/v65/gate_rh2/chain_all.parquet`, OOF `cache/oof/ctxc1sel_rh2`) if v65's gate
passes; **`ctxc1sel_rohb`** (`cache/v64/gate_rohb/chain_all.parquet`, `cache/oof/ctxc1sel_rohb`) otherwise. The rohb frame
may be run first while v65's fits run; it decides only if v65 fails. Both results will be reported.
- **OLD** (production now): eta from frame A's calibration rows, slope model fitted on frame A.
- **NEW**: eta and slope model from frame B.
Same G2 inputs, parameters, clipping, periods, masks and λ rules as `tests/day_alloc_gbm.py`.

**Pass iff** NEW − OLD on frame B: full ≤ −0.05; body ≤ 0; trimmed ≤ 0; ≥ 8 of 12 months better (full);
February ≤ 0 and June ≤ 0 (the blind board's extra months).

## Build if it passes

`PRC_G2_HO`, `PRC_G2_OOF`, `PRC_G2_LAM` set to frame B's files for `day_alloc_build.py` (so both the slope model and the
2026 λ̂ eta come from frame B), on the same p as the base file (v65's or v64's). Next free version number.
Keep iff board < the best at upload time. Blind board: the same env on the four-month build.

## Outcome (2026-10-11; `logs/v65/day_alloc_refit_rohb.log`)

Frame B = `ctxc1sel_rohb` (run first, while v65's fits ran; 1,924,263 rows, p change A → B RMS 40.0 s, 52 rows change
predicted off-block day). On frame B, 8 airports: no day stage 203.863, OLD 201.353, NEW 201.347.
NEW − OLD: full −0.006, body +0.045, trimmed −0.009; months better 8 of 12 (Feb −0.47, Jun −0.01, Jul +0.19, Dec +0.18);
2026 projection −0.005 s. **FAIL** (full and body). G2's 2025 ingredients do not go stale with the corrector: the refit
changes nothing measurable. Since v65's gate passed, the rh2 frame is the deciding one by the rule above; it is run for
the record (`logs/v65/day_alloc_refit_rh2.log`).

**Deciding frame (`ctxc1sel_rh2`, `logs/v65/day_alloc_refit_rh2.log`):** no day stage 202.693, OLD 200.250, NEW 200.202.
NEW − OLD: full −0.048, body +0.043, trimmed −0.052; 8 of 12 months better (Feb −0.81, Jun −0.01, Jul +0.12, Dec +0.14);
per airport full LTFM −0.20 (body +0.34), others within ±0.04; 2026 projection −0.036 s.
**FAIL** (full misses −0.05 by 0.002; body +0.043). Not built; production keeps G2 as fitted in §133.
