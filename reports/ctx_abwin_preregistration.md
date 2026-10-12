# ADS-B neighbour window (block 13) in the context corrector (CTX_ABWIN) -- pre-registration

Written 2026-10-11 **before any gate number below was computed**. Base: v65 = 237.0694 (§142/§143).

**Disclosed (2025 data already used):**
- §140 (withdrawn CTX_HOUR2): the same window as a paired residual screen on top of `ctxc1sel_rohb`: −0.30 pooled / −0.37
  holdout, 6/6 periods.
- Today's scratch screens (session scratchpad, `scr_run.py`): member `r` with blocks 11 + 12 refitted on 8 OOF months,
  validated on two held-out pairs, all airports in scope, block 13 vs no block 13:
  - Feb + Aug: body −0.59 (both months better), trimmed −1.00, non-LIRF trimmed −0.96;
  - May + Nov: body −0.43 (both months better), trimmed −0.61, non-LIRF trimmed −0.38.
  Airports with no ADS-B (LTFM, LFPG) moved ±0.3 in those screens, which is the refit-noise floor.

## Arm

**Block 13** (`CTX_ABWIN=1`, `abwin_block` in `tests/context_corrector_test.py`): over the other departures of the same
airport with takeoff within ±30 min, the mean of clip(T − adsb_pushback_ts − p, ±600 s) for v4-detector tier `appear` and,
separately, for tier `dwell` (at least one neighbour each), plus their combined count. p = the frame's input prediction
(the stack), as blocks 11 and 12. Label-free.

**Scope, fixed before any number:** the inputs are computed at **EDDF, EDDM, EHAM, LEBL, LSZH** and are null at every other
airport, in training and ranking frames alike. Reason: 2025 → 2026 tracked-neighbour coverage (share of departures with
≥ 1 appear/dwell neighbour within ±30 min; mean count), measured today without labels:

| airport | 2025 fetched days | Jan 2026 | Jul 2026 | in scope |
|---|---|---|---|---|
| EDDF | 0.97 / 7.5 | 0.77 / 2.1 | 0.70 / 1.6 | yes (coverage falls: rows look like unfetched 2025 days) |
| EDDM | 0.82 / 15.9 | 1.00 / 18.6 | 1.00 / 22.2 | yes |
| EHAM | 1.00 / 26.3 | 1.00 / 26.2 | 1.00 / 33.6 | yes |
| LEBL | 0.96 / 9.3 | 0.95 / 8.4 | 0.99 / 13.6 | yes |
| LSZH | 0.96 / 6.7 | 1.00 / 8.9 | 1.00 / 7.1 | yes |
| EGLL | 0.34 / 1.1 | 0.96 / 7.9 | 1.00 / 28.9 | **no** (§86 / v40: ADS-B corrector inputs fail where coverage shifts) |
| LEMD | 0.49 / 3.8 | 0.98 / 8.3 | 0.77 / 4.3 | **no** |
| LIRF | 0.37 / 1.0 | 0.85 / 2.8 | 0 / 0 | **no** |
| LFPG, LTFM | ≈ 0 | ≈ 0 | ≈ 0 | no data |

**Members.** Both members refitted with blocks 11 + 12 + 13 (`ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1
CTX_HOUR2=1 CTX_ABWIN=1`, `rb` adds `CTX_ADSB=1`): `ctxc1r_rh2ab`, `ctxc1rb_rh2ab`, `sel` → **ctxc1sel_rh2ab**. Nothing else
changes (rows, parameters, early stopping, ensemble2).

## Control

v65's `ctxc1sel_rh2`. The byte-identical files are on the other machine, so the control is **refitted on this machine**
with the same scripts (`PRC_ARMC_TAG=_rh2`). Disclosed: control and treatment are separate processes, so the comparison
carries refit noise (§105e); today's screens put it near ±0.3 s per airport, smaller pooled.

## Gate (same conditions as reports/ctx_hour_preregistration.md / v65)

1. **Holdout chain** (`tests/v50_chain_gate.py`, `ARMS=all`, `RO_FILE=reocc_feats.parquet`): chain(ctxc1sel_rh2ab) vs
   chain(ctxc1sel_rh2), LIRF NM-unmatched rows as v41 in both.
2. **OOF months** (corrector level): ctxc1sel_rh2ab vs ctxc1sel_rh2 OOF folds, labelled rows, y ≤ 5 h for trimmed.

**Pass iff:** (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul; (c) holdout chain full ≤ 0;
(d) holdout chain body ≤ +0.10; (e) OOF trimmed better in February and in June and in ≥ 7 of 10 OOF months.
**Drift stop (label-free):** if any in-scope airport has |SMD| > 0.25 between 2025 (fetched days) and 2026 for a block-13
input, the build stops. Reported: per airport, body by month, 2026 projection on the rows production changes.

## Build if it passes

Production fits of both members, `sel`; production chain for ctxc1sel_rh2ab and for ctxc1sel_ro (the control chain must
reproduce `chain_ctxc1sel_ro_final`, md5 98099262…), identity-arm deltas on both; delta = (new + id) − (ro + id)
(`src/post/v64_delta.py`); p = v54 + delta (`FREEZE_LIRF=1 v50_assemble.py`); v63's day recipe (`day_alloc_build.py`).
The day recipe takes LSZH / LIRF rows from its second argument; v62's file is on the other machine, so v65 is passed
(its LSZH / LIRF rows are v62's, §135 / §143). **v66. Keep iff board < 237.0694.**

## Outcome (2026-10-11; `tests/ctx_abwin_drift.py`, `logs/ctx_abwin_drift.log`) -- STOPPED at the drift check, no gate run

Run first (label-free), before any corrector fit. Max in-scope |SMD| 1.057 > 0.25: **drift stop triggered.**
- The count `h3_ab_n` moves with coverage at every in-scope airport (EDDF −1.06, EDDM +0.44, LEBL +0.41, EHAM +0.37,
  LSZH +0.30).
- The residual means move too: LSZH dwell −0.70 (−75 → −145 s, §125 / §137's LSZH level conflict again), EHAM appear +0.36,
  EDDF dwell −0.34. Only EDDM and LEBL pass on both means (|SMD| ≤ 0.13).
- Not amended: an EDDM + LEBL scope without the count is worth ≈ 0.07 s on the board at today's screen sizes, which does
  not pay for a rebuild. Block 13 stays in `tests/context_corrector_test.py`, off by default (`CTX_ABWIN`).
