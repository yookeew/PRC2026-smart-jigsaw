# Pre-registration: average of two independent builds on Jan / Jul (v76)

Written 2026-10-13, before v76 is uploaded. PROGRESS.md §163.

## Lever

For Jan / Jul 2026 two full builds of the same recipe exist:
- the live board's `smart-jigsaw_v75` (teammate's base fits and corrector, first release);
- the four-month build's Jan / Jul rows, `final4m/f4m_v75` (this machine's LightGBM refit, CatBoost Colab rerun,
  our three corrector seeds, final release).

They differ by refit noise (LightGBM 20 s RMS, CatBoost GPU 33 s, corrector seeds), not by recipe:
no level shift at any airport (|mean| ≤ 0.3 s on NM-matched rows), airport-day means within 2.1 s RMS (the day stage pins both).

For any two predictions p, q: MSE((p + q) / 2) = (MSE(p) + MSE(q)) / 2 − E[(p − q)²] / 4. So the average beats p
unless q is worse than p by more than E[(p − q)²] / 2.

## Scope

NM-matched departures at the nine non-LIRF airports (313,042 of 344,841 rows). Every other row keeps v75's value:
- LIRF: the two builds follow different recipes there. Live LIRF NM-unmatched rows carry the legacy v41 corrector,
  frozen since v42; live LIRF NM-matched rows carry the `ctxc1sel` chain. The four-month build uses the `ro` chain for both.
- NM-unmatched rows elsewhere: no label-free proxy to check them.

## Numbers before the build

- (p − q) on the scope rows: RMS 25.47 s. At equal quality the gain is E[d²]/4 over all rows, **−0.314 s**.
- Breakeven: q worse than p by ≈ 324 s² per scope row (≈ 0.6 s of RMSE).
- Equal quality, evidence:
  - CatBoost GPU reruns on the 2025 holdout: 273.03 / 273.41 / 273.01.
  - Label-free proxy T − AOBT_3 (2026, NM-matched, 60–3,600 s, 311,547 rows): MSE(q) − MSE(p) = −30 ± 74 s²; the average's
    proxy RMSE is 324.08 vs 324.34 (p) and 324.30 (q).
  - Airport × month proxy gaps go both ways (EGLL Jul q worse 1,267 ± 400; LTFM Jan q better 1,734 ± 557; LFPG q better
    both months). They start at the corrector seeds. Row-level standard errors ignore within-day clustering. No
    per-airport weights or selection, because that would fit the proxy.

## Build

`src/post/twin_average.py smart-jigsaw_v75.parquet final4m/f4m_v75.parquet smart-jigsaw_v76.parquet`:
v76 = rint((p + q) / 2) on scope rows, then the live LOBT window (first release) on those rows.

## Stop rules (any failure: no upload)

1. IDs exactly `submitting.parquet`'s; no nulls, no negatives.
2. Rows outside the scope byte-identical to v75.
3. Change on scope rows: RMS 12.7 ± 1 s, |mean| < 0.5 s.
4. Every scope row inside its live LOBT window.

## Decision

Keep iff the board scores v76 < 234.5098 (v75). The board gives a keep/reject verdict only.

If v76 is kept, the blind file `blind_v8` = Jan / Jul from v76, Feb / Jun unchanged from `f4m_v75`. Feb / Jun has
one build only. A second four-month build needs new base fits (LightGBM ≈ 80 min, CatBoost on Colab) and the chain. So
Jan / Jul would carry one stage Feb / Jun lacks, and the team decides on that before any blind overwrite.

## Result

- Pre-upload checks: `tests/v76_checks.py`, `logs/v76/v76_checks.log`. Every stop rule passed.
- **Board 2026-10-13: v76 = 234.2695 vs v75 = 234.5098 (−0.240). Kept.**
- The final blind file is `blind_v8`, uploaded as `smart-jigsaw_final.parquet`: Jan / Jul from v76, Feb / Jun unchanged
  from `f4m_v75`.
