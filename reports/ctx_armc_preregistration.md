# Context corrector Arm C (NM-unmatched rows in training) — pre-registration

Written 2026-10-06, before any Arm C result. Changes made after results exist go
under Amendments, dated.

## Why

- `lower-rmse` §87 (PROGRESS §87 on main): the context corrector is trained on NM-matched rows only (`~nm`, `nm = FLIGHT_ID_mvt is null`) but applied to NM-unmatched rows.
  - On those rows 16 inputs are always null and never null in training, so the trees route them down NaN branches they never trained.
  - The resulting shift is a flat level with no row-level signal. Out of sample, LIRF gets worse in 9/10 months and LTFM better.
- Removing the shift on LIRF rows (v40 on `lower-rmse`) lost on the board. This arm does the opposite: it lets the corrector **learn** NM-unmatched rows, so their correction comes from data rather than extrapolation.
- **Team rule (PROGRESS §87):** no change may alter LIRF NM-unmatched departures. Both arms copy the BASE values back onto those rows at the end of the chain.

## Coverage check (Part 0, read-only, 2026-10-06)

Main has none of the arm's pieces. So both C1 and C2 are run.

| item | status | where |
|---|---|---|
| (a) context corrector trained with NM-unmatched rows | no | `src/post/ctx_corrector.py:79`, `tests/oof_context_corrector_stage2.py:42` (`fit = ~nm & ...`) |
| (b) `nm_unmatched` indicator in the corrector | no | `tests/context_corrector_test.py:build` has no such column |
| (c) per-(airport, operator) delayed-echo rate in the corrector | partial | `airline_block` (`al_copy`, `CTX_AIR=1`) is a copy rate over all delays, NM-matched rows only. It leaked holdout labels in its gate and was REJECTED (§81), so it is not in the BASE. The NM-unmatched carrier echo rate lives upstream in the CatBoost corrector (`src/post/corrector_v2.py`), not in the context corrector. |
| (d) anything bypassing or altering the corrector on NM-unmatched rows | no | only on `lower-rmse` (v39/v40), never merged |

## BASE

- **The best uploaded file whose chain is on main: v41 = 253.68** (§86).
- Its corrector recipe is `ctxsel`: the `ctxcorr2rb` corrector (ensemble2, blocks 1–7 + block 9 ADS-B, `CTX_ENS=2 CTX_RWY=1 CTX_ADSB=1`) at EHAM/LSZH/EDDF/LEBL, and the `ctxcorr2r` corrector (v37: ensemble2, blocks 1–7, `CTX_ENS=2 CTX_RWY=1`) elsewhere.
- v41's chain: `stack_submit --cat-corrected --adsb-v3` -> `adsb_combiner --full` -> `neighbour_state --v2` -> `lobt_window` -> `nm_lines`, with `PRC_LGB=ctxsel`.
- **Corrector recipe kept fixed in both arms:**
  - LightGBM ensemble2 (global + per-airport mean), `CC.PARAMS` / `CC.PARAMS_APT`, early stopping on the inner `day % 5 == 0` split;
  - the same five two-month OOF blocks plus the 10-month fit for the holdout;
  - `objective="regression"`, i.e. **L2**. The task brief says "Huber". The context corrector has always been L2; Huber belongs to the upstream CatBoost corrector (corrector_v2). Changing the loss would be a second change, so L2 stays.
- Both member correctors (`ctxcorr2r` and `ctxcorr2rb`) are retrained with each arm's change, then combined per airport exactly as `ctxsel`.

## Arms

Definitions: `nm` = `FLIGHT_ID_mvt` is null (the corrector's own flag). `line_rows` = `nm & (T − SCHED > 6 h)`, rows that nm_lines serves; these stay excluded from training and from application, as in the BASE. `s` = the stack prediction, the corrector's input.

**C1.**
- Training rows = the BASE rows (`~nm & y ≤ 18,000 & ~line_rows`, unchanged) **plus** `nm & ~line_rows & s ≤ 7,200`.
- The NM-unmatched rows are selected on the BASE prediction `s`, never on the label.
- One new input: `nm_unmatched` (1.0 / 0.0).
- Nothing else changes.

**C2.** C1 plus an operator echo block:
- `ec_rate`: per (airport, `AIRCRAFT_OPERATOR_flt`), among labelled departures with T − SCHED ≥ 1,800 s, the share with |BLOCK − SCHED| < 30 s. BLOCK = T − y, so this is |(T − SCHED) − y| < 30.
- `ec_n`: the number of such delayed departures.
- Null where the operator is null, which includes every NM-unmatched row. `ec_rate` is also null when `ec_n = 0`.
- **Out of fold by month:**
  - In the gates, the table never contains a holdout month (Jan or Jul 2025; the §81 lesson). OOF-month rows use the other 9 training months, holdout rows use all 10.
  - In production, training rows use the other 11 months, ranking rows use all 12.

**Both arms:** after the chain, LIRF NM-unmatched rows take the BASE chain's value (a post-chain copy). This holds on the holdout as well as in the ranking build.

## Gate

`tests/ctx_armc_test.py`. Base = the BASE chain (`PRC_LGB=ctxsel`) cross-fit on the holdout, with the `ctxsel` holdout and OOF inputs. Treatment = the same chain with the arm's corrector inputs. LOBT window + nm_lines (10-month fit) are applied to both, as in `oof_context_corrector_stage2.py`. Jan<->Jul cross-fit in the chain.

**Reported:**
- pooled and per-month body RMSE (NM-matched, y ≤ 1 h), trimmed RMSE (y ≤ 5 h) and full RMSE;
- NM-unmatched rows, LIRF excluded, per airport;
- P(worse) from the (airport, day) cluster bootstrap (`adsb_v3_test.score`).

**Adopt rule, per arm** (deltas = treatment − base):
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. pooled full delta < 0;
3. neither month worse in full RMSE by > 0.10;
4. full delta still < 0 after dropping the 5 days with the largest contribution to the gain. Days are calendar days pooled over airports, ranked by Σ(se_new − se_base), most negative first;
5. NM-unmatched non-LIRF rows not worse in full RMSE: P(worse) < 0.9.

**If both pass:** prefer C1, unless C2 beats C1 by ≥ 0.2 pooled trimmed with P(worse) < 0.05 (C2 vs C1).

**Out-of-sample check (reported, not decisive).** On the 10 training months, from the five OOF-block fits, per airport: does the retrained corrector (the `ctxsel` combination) reduce SSE on NM-unmatched non-line rows, against the uncorrected stack `s`, in ≥ 7/10 months? The BASE corrector is reported alongside.

**Ranking diagnostics (label-free, reported):**
- share of NM-unmatched ranking rows whose input null pattern never occurs among the new corrector's training rows (BASE vs arm);
- corrector shift on NM-unmatched ranking rows per airport (mean, RMS), BASE vs arm.

## Build (only if an arm passes)

1. Production refit on 12 months (as `ctx_corrector.py`) of both members with the arm's change, then the `ctxsel` combination.
2. v41's chain on it.
3. Copy the BASE file's values onto LIRF NM-unmatched rows -> `data/submissions/smart-jigsaw_v45.parquet`.

Checks: 344,841 rows; MVT_ID set equal to `submitting.parquet`; no nulls or negatives; LIRF NM-unmatched rows byte-identical to the BASE.

Nothing is uploaded. Nothing is fitted to board scores. All fits use 2025 only; judgement is on the Jan/Jul 2025 holdout only.

## Amendments

(none)

## Results

(pending)
