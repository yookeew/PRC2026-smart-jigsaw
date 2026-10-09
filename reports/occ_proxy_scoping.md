# Label-free occupancy at LFPG / LTFM -- scoping and stop bar

Written 2026-10-08 **before any proxy number was computed**. Follows `reports/adsb_queue_scoping.md` (PROGRESS §126).

## Target (already measured, `logs/occ_oracle_split.log`)

Oracle O1 (occupancy and queue-ahead from other flights' TRUE off-blocks) vs control C, body RMSE, 10 OOF months,
115 ADS-B days, non-LIRF: pooled −4.09 s, of which **LTFM −1.65 s and LFPG −1.18 s** (≈ 70%). LTFM is steady by
month (−3 to −16 s on its own body rows), LFPG volatile (−0.3 to −17.8 s).

## AOBT_3 check (`tests/occ_aobt3_check.py`, `logs/occ_aobt3_check.log`, label-free except 2025 accuracy)

- LTFM: T − AOBT_3 sits on one default minute pair per runway (19/20 or 24/25 min) for 80–90% of rows, in 2025 and
  2026 alike; |AOBT_3 − true off-block| ≤ 120 s on only 10–20%. **AOBT_3 carries almost no off-block information
  at LTFM.**
- LFPG: AOBT_3 is a real measurement (≤ 120 s on 36–46%; smooth minute histogram, top minute 10–13%). The median
  T − AOBT_3 steps by 60 s between months (≈ 897 s Jan–Jun, Dec; ≈ 957 s Aug–Nov and **Jul 2026**). Its bias against
  the label is −55 s in Jul–Sep 2025. These are whole-minute recording offsets (§98's suspicion); small for occupancy.

## Proxies (built identically in training and production; same rows, folds and models as the oracle)

Each proxy replaces the true off-block B_j of OTHER flights by an estimate, then rebuilds the O1 feature set
(occupancy at T − x, x ∈ {0, 5, 10, 15, 20, 30} min, airport and own runway, and queue-ahead on the own runway),
plus a traffic-normalised copy (occupancy ÷ max(1, takeoffs in (t, t + 30 min])):
- **Pa3:** B̂_j = AOBT_3_j where present, else T_j − s_j;
- **Ppred:** B̂_j = T_j − s_j, s_j = the out-of-sample stack prediction (OOF in training, out-of-sample on the
  holdout and in 2026);
- **Pboth:** both sets.

Windows are in absolute time (±30 min), so the Jan → Jul 2026 seam cannot link flights more than 24 h apart.
Movement-based counts do not depend on ADS-B coverage. Traffic-level dependence is handled by the normalised copy.

## Stop bar (fixed now)

Proceed to a pre-registered gate on v50's chain only if the best of {Pa3, Ppred, Pboth} (choosing among three arms;
disclosed) gives, against C on the OOF months:
1. pooled non-LIRF body ≤ −0.5 s, **and**
2. ≥ 7 of 10 months improving, **and**
3. a LFPG + LTFM contribution to the pooled body gain ≤ −0.3 s.

Otherwise STOP. Also reported: the fraction of the O1 oracle each proxy recovers, per airport and per month at
LFPG / LTFM.

## Results

(appended below)

### Result (`tests/occ_proxy_test.py`, `logs/occ_proxy_test.log`) -- STOP

Deltas vs C, 609,238 rows (non-LIRF, 115 OOF ADS-B days), leave-one-month-out:

| slice | C | O1 | Pa3 | Ppred | Pboth |
|---|---|---|---|---|---|
| full | 184.56 | −5.77 | −1.22 | −1.67 | −1.67 |
| body | 167.43 | −4.09 | −0.30 | −0.29 | **−0.38** |
| NM-matched 30-60 min | 515.19 | −11.83 | +0.52 | −0.27 | −0.39 |

- Bar: (1) body ≤ −0.5 → **−0.38, FAIL**; (2) ≥ 7/10 months → 9/10 (Dec +3.05); (3) LFPG + LTFM contribution
  ≤ −0.3 → −0.35, pass. **STOP; no gate.**
- Fraction of the O1 oracle recovered (own-row body): **LFPG ≈ 0.20** (−1.5 s, 8/10 months better); **LTFM 0.02-0.07**
  (Pa3 0.02, as the default-spike finding predicts; Pboth 0.07), with December 2025 unstable (+9.8 to +14.7 s).
  Elsewhere ≈ 0 or negative (EDDM, EHAM, LEBL worse).
- The full-RMSE gain (−1.7) is larger than the body gain, so it sits in the tail. That is the population where
  holdout gains have transferred poorly (§123 notes; outgoing-boat's transfer ratio).
- An LFPG-only arm would be scope chosen after the result. Not pursued.
- **Conclusion.** Neighbours' estimated off-blocks (AOBT_3 or predicted) carry about a fifth of the oracle's
  occupancy information at LFPG and almost none at LTFM. The oracle's value is in the true off-block times
  themselves, which no label-free source at these two airports reproduces.
