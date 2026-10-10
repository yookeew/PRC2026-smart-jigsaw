# v57–v59: a 2026 day-sum drift model (main) + LIRF ordinary-row shift + LSZH day variation -- pre-registration

Written 2026-10-10 **before any number below was computed**. Base: v55 = 239.40 (PROGRESS §130). Engine
`src/post/pru_recon2.py` (expectation solve, §129). Three candidates on **disjoint rows**, so each upload is a clean
read against v55. LIRF and LSZH keep v55's values outside their own candidate.

## Common definitions

- Airport-day g: rows assigned by predicted off-block date; kept iff row count within ±5% of `FLT_DEP_IFR_2` (§125).
- λ̂_g: expectation solve, uniform: Σ_i G(d̂ᵢ + λ) = 60·D_g, G(x) = E_η max(0, x + η). η per airport, tracked
  (adsb.lol dwell / pass / fb_dwell) vs untracked. For 2026 η comes from the Jan + Jul 2025 holdout chain
  (`cache/v50/chain_all`); for 2025 periods, from other data (below). Shifts are in d units: new taxi = max(0, p − δ).
- δ*_g (2025 only, labels): mean over the day's rows of (p − y), i.e. the uniform shift that minimises the day's SSE.
- Measurement noise s²_a (2025, per airport): variance over airport-days of (λ̂_g − δ*_g) after removing its mean per
  (airport, period).
- **R0** = v55's recipe (point solve, uniform, 0.75·λ).

## C1 (main): day-sum drift model on the 8 airports v55 shifts (not LIRF, not LSZH)

Per airport, over the target period's days (2026: Jan + Jul, ≈ 62 days):
1. λ̂_g as above.
2. Level structure, OLS across the airport's days with month fixed effects:
   λ̂_g = μ_{a,m} + Σ_k β_{a,k} w_{g,k} + u_g, where w_{g,k} = share of the day's sensitivity Σ P_η(d̂ᵢ + η > 0) that
   falls in group k (computed at λ = 0, so it does not depend on D_g), reference group excluded.
3. Day residual shrunk by empirical Bayes: a_a = max(0, 1 − s²_a / Var_g(û_g)) (the 2026 dispersion, net of 2025
   measurement noise).
4. Row shift δᵢ = μ̂_{a,m} + β̂_{a,k(i)} + a_a·û_g. Early-predicted flights (d̂ < 0, invisible to the sums) are in the
   reference group: they get μ̂ + a·û, no extrapolated group term.

Families of shared terms (fixed now):
- **F0** none (level + EB day only);
- **F1** predicted-delay band: d̂ < 15 min (ref), 15–60 min, ≥ 60 min;
- **F2** NM-unmatched vs matched;
- **F3** observed (adsb.lol pushback tier or OPDI ground event ≥ 60 s before T) vs not;
- **F4** time of day (UTC): [0, 9) (ref), [9, 15), [15, 24).

**Validation 1 (2026, label-free): leave-one-day-out prediction of the published sums.** For each airport-day, fit
step 2 on the airport's other days and predict λ̂_g (equivalently D_g) from μ̂_{a,m} + β̂·w_g. A family is **kept** iff
it lowers the pooled leave-one-day-out RMSE of λ̂ by ≥ 5% vs F0 **and** lowers it at ≥ 5 of the 8 airports. F* = the
union of kept families (re-checked jointly with the same test; if the union fails it, the single best kept family).
Input check (§86 / v37 lesson): an airport's shared terms are dropped (F0 there) if any of its 2026 day shares w_{g,k}
falls outside the 2025 range of that airport's shares by more than 0.05.

**Validation 2 (2025, labels, decides): all 12 months, per-row RMSE vs R0.** Periods mimic 2026's two months:
(Jan, Jul) on the holdout chain `cache/v50/chain_all`; (Feb, Mar), (Apr, May), (Jun, Aug), (Sep, Oct), (Nov, Dec) on the
corrector OOF predictions `cache/oof/ctxc1sel`. Each period runs the whole procedure on its own published sums.
η: holdout period cross-month (Jan from Jul and vice versa), OOF periods from the other OOF periods; s²_a from the
other periods. Rows: the 8 airports.

**C1 PASS iff** (variant = F* if F* ≠ F0 and F* also satisfies the conditions, else F0):
1. pooled 12-month full RMSE (C1 − R0) ≤ −0.10 s;
2. pooled body RMSE (C1 − R0) ≤ 0 (body = NM-matched, y ≤ 1 h);
3. full RMSE better than R0 in ≥ 8 of the 12 months.
Why this evidence: the 2025 run tests the method as it will be used (a period's own sums, no labels), across 12 months
rather than the January-distorted holdout. The Jan/Jul months are 2 of 12 and are also reported separately, but
they do not gate on their own.

**Build:** v57 = v55 with the 8 airports' rows re-solved from v54 by C1 (LIRF, LSZH = v55), LOBT window last.
**Keep iff board < 239.40.**

## C3 (secondary): LSZH day-to-day variation only

Shift LSZH rows by δᵢ = a_LSZH·(λ̂_g − λ̄_{LSZH,m}) (expectation solve, month mean removed, so v49's level stays),
a_LSZH from the same EB rule (F0). Evidence: 2025, 12 months, LSZH rows, same periods / η / s² rules as C1, vs no change.
**PASS iff** pooled full ≤ −0.05 s and pooled body ≤ 0 on LSZH rows and full better in ≥ 8/12 months.
**Build:** v58 = v55 with only LSZH rows changed. **Keep iff board < 239.40.** Reported, not used: LSZH's month
levels under the point and expectation solves, set against §129's OPDI witness (Jul −46 s, Jan +0.7 s).

## C2 (secondary): LIRF ordinary-row day shift, no per-row bets

Ordinary LIRF rows = NM-matched, not hedged (§125 definition: 45 min ≤ T − S < 12 h and TAU + 60 < p < T − S − 60),
T − S < 12 h. Per LIRF day, λ solves
Σ_ordinary G(d̂ + λ) + Σ_hedged (1 − π)·max(0, T − S − N) + Σ_other non-hedged (T − S < 12 h) max(0, d̂) + Σ_{T − S ≥ 12 h} 0 = 60·D,
with π, N from §125's prior (isotonic map + 650 s floor; calibrated on the LIRF holdout chain `cache/v54/v48m_chain_holdout`,
cross-month for 2025). Only ordinary rows move: δ = μ̂_m + a_LIRF·û_g (F0). Hedged and other rows are unchanged.
Evidence: 2025, ordinary LIRF rows, same periods; LIRF predictions: the v48-method holdout chain for (Jan, Jul), the
corrector OOF elsewhere. **PASS iff** pooled full ≤ −0.10 s and pooled body ≤ 0 on ordinary LIRF rows and full better
in ≥ 8/12 months.
**Build:** v59 = v55 with only ordinary LIRF rows changed. **Keep iff board < 239.40.**

## Reported for the "2026 is different" reading (label-free)

Per airport: 2026 Var_g(λ̂) and mean λ̂ against the 2025 holdout chain's (same solve, same η source); a_a; how many
2026 days fall outside the 2025 range.

## Upload order (fixed now, ranked by the expected size of the read)

C1 first if it passes (the 8 airports that moved −2.36 on v55), then C2 / C3 in the order of their 2025 full-RMSE gain
scaled to 2026 row counts.

## Outcome C1 / C3 (2026-10-10, added after `tests/v57_dsm.py`, `logs/v57_dsm.log`)

- Validation 1 (2026): pooled leave-one-day-out RMSE of λ̂: F0 37.5 s, F1 34.2, F2 34.3, F4 35.8, F3 undefined (LTFM has
  no observed rows; not kept). F* = F1 + F2 (union 34.2, 7/8 airports). Input check: EHAM (band ≥ 60 min, NM-unmatched)
  and LFPG (band ≥ 60 min) 2026 shares exceed their 2025 ranges → F0 there.
- **Validation 2 (2025, 12 months, 8 airports) vs R0: F* full +43.7 s, body +14.3 (0/12 months); F0 full +0.42,
  body +1.10 (3/12 months). C1 FAILS** (both forms). The shared terms' β, identified from small day-to-day share
  variation, are huge, and applying them per row amplifies them; predicting sums (validation 1) cannot see that.
- **C3 (LSZH day variation) vs no change: full −0.311, body −0.651, 12/12 months better. PASS.**

## Amendment A1 (2026-10-10, after C1 failed, before any C1b number): new candidate C1b

Diagnostic (`logs/v57_dsm_bias_diag.log`, 2025, 8 airports, ≈ 2,900 airport-days): regressing the MSE-optimal uniform
day shift δ*_g on λ̂_g gives a pooled slope of 0.71 (expectation solve) / 0.70 (point solve), 0.4–0.9 per airport, at
correlations 0.7–1.0. The day sums react more than the day's mean residual. v55's 0.75 is close to the slope, and C1's
full application of the level overshoots. Likely mechanism: the sums see only delayed flights, whose taxi errors move
more than the average row's. So the allocation should be **learned from 2025 labels**, not identified from 2026 mix.

**C1b.** Shift δᵢ = s_{a,b(i)} · λ̂_g (no intercept), λ̂_g = expectation solve (uniform), b = predicted-delay band of
d̂ ∈ {< 0, 0–15 min, 15–60 min, ≥ 60 min}. Slopes s_{a,b} by OLS through the origin of (pᵢ − yᵢ) on λ̂_g over the
rows of airport a, band b with y ≤ 18,000 s, on **the other five 2025 periods** (leave-period-out); production slopes
from all six. Same rows, periods, η rules and **the same PASS conditions as C1** (pooled 12-month full vs R0 ≤ −0.10 s,
pooled body ≤ 0, full better in ≥ 8/12 months). Reported: the slopes, per airport, Jan/Jul. Known transfer risk, stated
now: 2026 has many more ADS-B-tracked rows than the 2025 data the slopes come from. **Build** v57 as C1 (8 airports
re-solved from v54; LIRF, LSZH = v55). Keep iff board < 239.40.

### Outcome C1b (`tests/v57_dsm_c1b.py`, `logs/v57_dsm_c1b.log`)

vs R0, 2025, 12 months, 8 airports: **full −0.384, trimmed −0.404, body +0.285**, full better in 11/12 months.
**FAIL** (condition 2). Body by month: Jan +0.85, **Feb +4.74**, Dec +0.23, every other month −0.02 to −0.67. February is
LTFM's 19 and 23 Feb 2025 (≈ 200 taxis > 1 h each, day mean residual −800 s, ordinary flights −67 / +32 s): the sums
demand a huge shift that the slopes hand to ordinary rows too (`logs/v57_feb_diag.log`). Production slopes
(< 0 / 0–15 / 15–60 / ≥ 60 min): EDDF 0.33 / 0.68 / 1.27 / 1.31, EDDM 0.27 / 0.53 / 0.77 / 2.18, EGLL 0.42 / 0.81 /
1.19 / 1.28, EHAM 0.12 / 0.31 / 0.51 / 0.63, LEBL 0.10 / 0.57 / 1.13 / 1.16, LEMD 0.47 / 0.81 / 1.18 / 1.13,
LFPG 0.24 / 0.56 / 0.90 / 1.15, LTFM 0.09 / 0.30 / 0.79 / 1.63. Not built.

### Outcome C2 (`tests/v57_lirf_c2.py`, `logs/v57_lirf_c2.log`)

LIRF 2025: ordinary 87.8% of rows, hedged 12.0%. 365 days solved; 2025 noise sd of λ̂ vs the ordinary rows' optimal shift
211 s, and the sums sit 5–69 s below the ordinary rows' level in every period. vs no change on ordinary rows:
**full +30.6 s, body +32.2, 2/12 months better (July 2025 +245 s). FAIL**; v59 not built. The LIRF sums are dominated
by the hedged rows' prior uncertainty; on some days the solve hands ordinary rows an enormous shift.
