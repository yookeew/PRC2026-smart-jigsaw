# Base regressor: clip long labels instead of dropping, and L2 vs Huber — pre-registration (screen)

Written 2026-10-06, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **outgoing-boat's base regressor** (their `models.py`, `TwoStage`, GPLv3) differs from ours in two ways that bear on large labels:
  - it uses plain L2 where we use Huber α = 800;
  - it clips the training target at 7,200 s, where we drop rows with taxi > 7,200 s.
- **The mechanism worth testing:**
  - dropping long labels tells the model that disrupted conditions never produce long taxis;
  - our §78 oracle found most of the remaining holdout error is whole disrupted days, which we under-predict by 15–30%;
  - clipping keeps "this context is very slow" in the training data.
- **Prior evidence against L2:** §22 found L2 worse than Huber (+0.79, CI [+0.18, +1.41]). That test used a single regressor, before the copy classifier, the stack and the corrector existed.
- **Stated up front:** only 584 of 2,085,047 labelled 2025 departures have taxi > 7,200 s. Clipping alone touches 0.03% of rows, so arm (b) is expected to be small. Most of any change in arm (c) will come from the loss.

## Change

`PRC_LABEL_CLIP=1` in `src/models/fit.py` (`_train_target`):
- the regressor keeps rows with taxi > 7,200 s;
- the direct target becomes min(taxi, 7,200);
- the flip target becomes d + max(taxi − 7,200, 0), the d implied by the clipped taxi, since d + taxi = offset exactly;
- the copy classifier, the reconstruction and the prediction clips are unchanged.

## Arms

All arms are the LightGBM base of `lgb_mixed_nb` (target `mixed`, seed 42, `cache/features_nb/`, holdout Jan+Jul 2025, 10-month fit). Each one runs `src/models/run_label_clip.py`.

| arm | label rule | loss | output |
|---|---|---|---|
| a (reference) | drop > 7,200 | Huber 800 | the existing `cache/eval/lgb_mixed_nb_holdout_ev.parquet`, not refitted. `fit.py` hasn't changed since it was written, and runs are deterministic. |
| c | clip at 7,200 | L2 | `cache/eval/lgb_mixed_nb_clipl2_holdout_ev.parquet` |
| b | clip at 7,200 | Huber 800 | `cache/eval/lgb_mixed_nb_cliphub_holdout_ev.parquet` (run later, when the laptop is on mains power) |

## Measurement

`src/models/run_label_clip.py compare <arm>` reports the arm vs (a) at base level (the LightGBM prediction `pred`):
- body RMSE (NM-matched, y ≤ 1 h), pooled / Jan / Jul;
- trimmed (y ≤ 5 h) and full RMSE;
- per-airport body;
- **disrupted-day slice:** the airport-days whose mean true taxi is in the top 5% of airport-days within that airport, over the holdout.

## Decision (screen only — nothing is submitted from this)

An arm qualifies for a full rebuild (both engines, out-of-fold months, corrector and chain, on Colab) iff, vs (a):
1. pooled body gain ≥ 1.0 s;
2. body Jan > 0 and Jul > 0;
3. full RMSE not worse.

**The bar is higher than the corrector blocks' 0.5 s** because base-level gains shrink through the stack and corrector. The rebuild would then get its own stage-2 check against v37's chain.

**If neither arm qualifies**, the L2/clip question is closed for this competition.

## Amendments

(none)

## Results

**Arm c** (clip + L2; 2026-10-06; logs `logs/base_clipl2.log`, `logs/base_clipl2_compare.log`):

| slice | a (Huber, drop) | c (L2, clip) | delta |
|---|---|---|---|
| body pooled | 215.86 | 218.82 | +2.96 |
| body Jan | 197.82 | 200.93 | +3.11 |
| body Jul | 229.53 | 232.39 | +2.86 |
| trimmed | 271.99 | 276.55 | +4.55 |
| full | 334.33 | 342.47 | +8.14 |
| disrupted days | 500.91 | 532.35 | +31.44 |

- Worse at every airport (LIRF +12.54, the rest +0.3 to +3.4), and worst on the disrupted days it was meant to help.
- **L2 early-stops much sooner:** best_iter 1,049 (flip) / 1,516 (direct), against 7,996 / 2,896 for Huber.
- **Does not qualify.** This repeats §22's L2-vs-Huber result on the current base, at about 4× the size.

**Arm b: not run (decided 2026-10-06).** Clipping touches 584 of 2,085,047 training rows, so arm b cannot plausibly reach the 1.0 s bar. Question closed.
