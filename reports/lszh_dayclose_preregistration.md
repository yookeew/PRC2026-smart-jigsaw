# LSZH level from the daily sums (v69) -- pre-registration

Written 2026-10-12, **before** the 2025 shrink check and the v69 build below were run. Base: v68 (team build, current best).

## Why

LSZH is the one airport the day stage skips (§125: "the daily sum would almost exactly undo v49's board-validated LSZH
fix"). v49's fix (−38.9 s Jan / −28.1 s Jul on NM-matched LSZH rows, §121) rests on the T − AOBT_3 anchor alone, and its
board read (−0.49) was a bundle with the EHAM storm rows and the capped LIRF rows, so the LSZH part was never read on its
own. Numbers already computed (scratch, disclosed here):

- **2025 has no LSZH gap.** Point solve Σ max(0, T − S − p + λ) = 60 D per LSZH day on the 2025 chain (`v57_dsm.frame25`,
  LSZH not in G2): month medians of λ between −9 and +10 s in all 12 months, 27–73% of days positive.
- **2026: the gap appears exactly at v49.** Month median λ (negative = taxi predicted too low):

  | LSZH file | Jan | Jul |
  |---|---|---|
  | v48 (before v49) | −6 | +3 |
  | current chain `chain_ctxc1sel_rh2_final` | −5 | +3 |
  | v49 | −44 | −24 |
  | v68 | −47 (p90 −39) | −20 (p90 −12) |

  `blind_v5`: Jan −46, Feb −15, Jun −10, Jul −20. Every 2026 day in Jan and Jul is negative on v68.
- The same 2025 check at the eight G2 airports finds structural gaps (LTFM +8 to +42 s, EGLL +15 to +40, every month)
  whose closure hurts 2025 (+0.39 s trimmed); 2026's LTFM gap is inside that range. **The candidate is LSZH only.**
- Other witnesses for LSZH 2026 being higher than production: OPDI (+46 Jul), ADS-B appear/dwell medians against the 2025
  offset (Jan +15 to +37), PRU monthly taxi-out (+19 Jul). Against: the anchor (v49) and §125's 1–9 January ADS-B mean.
- On 2025 the published LSZH sums match our labels within minutes per day (§125), and our rows / published flights is
  1.01 in both years.

## Candidate

**v69** = v68 with LSZH rows re-levelled per airport-day by the published sums (`src/post/lszh_dayclose.py`):
- days and sums as the production engine: `pru_recon2.solve_days` point solve (R0), day = MVT_TIME − p, population check
  as in the engine; only LSZH solved;
- shift on day d of month m: s_d = λ̄_m + a·(λ_d − λ̄_m), λ̄_m = median λ_d over the month's solved days; new taxi =
  max(p − s_d, 0) on every LSZH row of the day (no row class frozen, as R0); LOBT window, clip at 0, round
  (`final4m_v62.finish`);
- every non-LSZH row byte-identical to v68.

**Choice of a (fixed rule, 2025 LSZH rows, the 12-month frame above, no labels of 2026):** compute LSZH full RMSE for
a ∈ {0.5, 0.75, 1.0} with the same formula on 2025 (λ̄_m and λ_d from 2025's sums). Take a = 1.0 if it is no worse than
0.75 and 0.5 on full RMSE and better than no stage in ≥ 9 of 12 months; otherwise the better of 0.5 / 0.75. The level
term λ̄_m is applied in full in every arm: 2025 cannot test it (its LSZH levels are ≈ 0), and that is the point of the
candidate.

## Stop rules (label-free, checked before upload)

1. Non-LSZH rows identical to v68; IDs exactly the template's; no nulls; no LOBT-window violations.
2. Re-solving LSZH on v69 gives month medians |λ̄_m| ≤ 5 s in Jan and Jul.
3. Mean LSZH change per month within [+15, +60] s (Jan) and [+5, +40] s (Jul); otherwise stop and look.

## Upload rule

Separate upload so the LSZH part is read alone: **keep iff board < v68's score.** Expected if the sums are right ≈ −0.16 s;
if the anchor was right, about +0.16 s.

## Blind board

The same script with `PRC_SCOPE=final4m` on the current blind file (`blind_v5`) → `final4m/blind_v5_lszh.parquet`.
LSZH rows are identical across v64–v68 builds, so the LSZH delta carries onto any later blind file (rerun the script on
it). Used for the blind board only if v69 is kept.

## Outcome (2026-10-12; `logs/v69/`)

- **a:** 2025 LSZH (134,850 rows, 365 solved days) vs no stage, full / trimmed / body: a 0.5 −0.391 / −0.744 / −0.770,
  **a 0.75 −0.443 / −0.855 / −0.867**, a 1.0 −0.418 / −0.821 / −0.805; 12/12 months better in every arm. Rule → **a = 0.75**.
- **v69** (`smart-jigsaw_v69.parquet`, md5 fc4de5dd3b46b3cf264ecf500cb66a66): 23,150 LSZH rows changed, Jan mean +49.3 s
  (RMS 50.3), Jul +20.6 (RMS 21.3); all other rows identical to v68; RMS over all rows 9.5 s.
  Stop rules: (1) pass, 0 window violations; (2) month medians after −0.0 / +0.1; (3) pass. **Keep iff board < v68's score.**
- **Blind:** `final4m/blind_v5_lszh.parquet` (md5 a2cc71a0ab68ee4b885ab09e1c7f30dd): Jan +49.2, Feb +14.1, Jun +8.9,
  Jul +20.6; medians after within ±0.2 s; Jan/Jul LSZH rows equal v69's within 1 s. `blind_v5` already had 16 LOBT-window
  violations (2 LSZH, now fixed; 14 at other airports, untouched here).
- **Board (2026-10-12): v69 = 236.5174 vs v68 = 236.6730 (−0.156). Kept; new best.** On the −0.16 projection, in the
  direction of the sums. `final4m/blind_v5_lszh.parquet` (or the script rerun on the next blind file) carries LSZH for the
  blind board.
