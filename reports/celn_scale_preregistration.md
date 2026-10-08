# Corrector `cel_n` train/production scale mismatch — pre-registration

Written 2026-10-07, before any result. Changes made after results exist go
under Amendments, dated.

## Finding (label-free audit, PROGRESS §105, `tests/corrector_input_audit.py`)

- The context corrector's `cel_n` is the number of 2025 departures in the row's (airport, stand, runway) cell, with fallbacks to coarser cells.
- **Training rows** (`src/post/ctx_armc.py` prod, as in `ctx_corrector.py`) take their cell table from the other 11 months.
- **2026 rows** take it from all 12 months.
- So 2026 `cel_n` runs about 12/11 above the training scale. 5–7% of NM-matched 2026 rows at EDDF / EGLL / EDDM fall above the training p99.5.
- **The holdout can't see this:** the holdout and OOF rows are built with the same leave-out convention as training.
- It is the same class of production-only mismatch as v37 (encoding) and v42 (null patterns).

## Change

- `PRC_CELN_SCALE=1` in `src/post/ctx_armc.py` prod multiplies the 2026 rows' `cel_n` by 11/12. Nothing else changes.
- **Not addressed:** the fallback level. A cell with 30–32 rows in 12 months may pass `CELL_MIN = 30` where its 11-month count would not.

## Measurement

Both Arm C members (`r`, `rb`) and their ctxsel selection are rebuilt on this machine twice, from identical inputs:
- **control:** `PRC_ARMC_TAG=_ctl`;
- **treatment:** `PRC_ARMC_TAG=_celn PRC_CELN_SCALE=1`.

Compared on the 2026 ranking rows, corrector output (`ctxc1sel`):
- RMS and mean change, treatment vs control;
- by airport;
- for the rows whose `cel_n` was above the training p99.5.

## Decision

- **RMS change < 3 s over all rows: close.** Not worth a submission against the board's noise.
- **Otherwise:** build v43 through v42's recipe on the treatment correctors (LIRF NM-unmatched rows copied from v41, team rule). The upload is the team's call. Keep iff the board score is below v42's 249.31.
  - The case rests on correctness by construction. No holdout gate exists for it.

## Amendments

**A1 (2026-10-07, after the first measurement, before any board or holdout use).** The control and treatment correctors
trained on identical rows but early-stopped at different rounds (r 747 vs 834, rb 803 vs 1,196). Training is not
reproducible run to run on this machine, so the measured 17.32 s RMS mixes the `cel_n` effect with refit noise.
Rebuilding the unchanged corrector already differs from v42's own (teammate's machine) by 18.81 s RMS.
- **New measurement:** each member is trained once, and the same model predicts the 2026 rows twice, with `cel_n` as
  built and scaled by 11/12 (`tests/celn_one_fit.py`). The difference is then the input effect alone.
- The decision rule (3 s RMS) is unchanged and is applied to this measurement. The first measurement is kept in the Results as void.
- **If the rule is met,** the candidate is v42 + (treatment chain − control chain), both chains built on this machine
  from the one-fit outputs. That puts only the `cel_n` effect on top of the scored v42, not the refit noise.

**A2 (2026-10-07, after the A1 result, before any build).** The candidate is **not built**, although the A1 measurement
(3.94 s) clears the 3 s rule. The rule was set without the bound that matters: a prediction change Δ can lower the
board's MSE by at most mean(Δ²) = 3.94² ≈ 15.5 s², i.e. 249.31 → 249.28 (−0.03 s), even if every row moved the right way.
That is below the board's resolution (one 3 h miss ≈ 0.6 s). Future thresholds on label-free changes should be set
on this bound.

**A3 (2026-10-07, same day, before any build or upload).** A2 is retracted: its bound was wrong. A prediction change Δ
moves RMSE by up to ±RMS(Δ) (triangle inequality), so a 3.94 s RMS change can move the board by up to ~±3.9 s, not
0.03 s. Where inside that range it lands depends on how the change aligns with the 2026 errors, which no holdout can
measure for a 2026-only fix (v42: holdout −0.9, board −4.6). The A1 decision stands: build the candidate
v42 + (treatment chain − control chain); post-stage refit noise is measured as RMS(chain diff) against the corrector-level
3.94 s before anything is built.

## Results

- **Original measurement (void, A1):** separate control / treatment refits, RMS 17.32 s. Mostly refit noise (r 747 vs 834 rounds, rb 803 vs 1,196).
- **A1 measurement** (`tests/celn_one_fit.py`, `logs/celn_one_fit.log`; one fit per member, two predictions):
  - ctxsel output RMS **3.94 s**, mean −0.09 s;
  - 13.9% of rows move > 1 s, 0.18% > 30 s;
  - NM-matched 2.88 s, NM-unmatched 21.90 s;
  - by airport 1.6–2.4 s, LFPG 5.9, LTFM 4.1, LIRF 9.0 (mostly NM-unmatched rows, which v42's recipe copies from v41 anyway).
- **Refit noise on this machine** (control vs v42's own corrector, teammate's machine): 18.70 s RMS.
- **Chain level** (`logs/celn_chain.log`): v42's chain on the control and treatment correctors, built on this machine with v42's own corrector holdout / OOF inputs.
  - RMS(treatment − control) 3.85 s (3.37 s excluding LIRF NM-unmatched), against 3.94 s at the corrector level, so no post-stage refit noise.
  - The control chain differs from v42's own chain by 16.45 s RMS (machine refit noise, which the delta construction removes).
- **v43** = `data/submissions/smart-jigsaw_v43.parquet` (`src/post/celn_build.py`, `logs/v43_build.log`): v42 + (treatment chain − control chain); LIRF NM-unmatched rows (383) identical to v42.
  - vs v42: RMS 3.37 s, mean −0.09 s, 101,938 rows changed, max |d| 285 s.
- **Rule:** the upload is the team's call. Keep iff the board score is below 249.31. Possible range per the triangle inequality: ±3.4 s.
- **Board (2026-10-07): v43 = 249.3322 vs v42 249.31 (+0.02). REJECTED; v42 stays.** The fix is neutral on the 2026 rows: the scale mismatch was real but the corrector doesn't lean on `cel_n` where it matters.
  - `PRC_CELN_SCALE` stays in `src/post/ctx_armc.py`, default off. It is correct by construction, so it is worth switching on in any future corrector rebuild.
