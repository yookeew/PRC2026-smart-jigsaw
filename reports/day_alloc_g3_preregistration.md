# Day-shift allocation with within-day inputs (G3) -- pre-registration

Written 2026-10-10 **before any G3 number was computed**. Base: v63 = 238.32 (§133, G2 at the eight airports).
**Disclosed:** the 2025 data below were already used for §130's C1b, §131's S1–S4 and §133's G1/G2. Two looks at the
Jan + Jul 2025 holdout-chain rows were taken first (scratch, analysis only, no candidate fitted on the 12-month frame):
- After G2, an oracle that removes the true airport-day mean residual gains only −0.02 s (trimmed, 12 months):
  G2 already takes the day-level error. The true airport-day × hour mean (leave-one-out) is worth −4.27 s
  (12 months) / −2.41 s (holdout-chain months only), LTFM −7.7 on its own rows.
- A Jan ↔ Jul cross-fit residual model with the hour-window inputs below, paired against the same model without
  them: −0.57 s trimmed (Jan −0.42, Jul −0.68), but the residual model itself is 1.7 s worse than the chain, and an
  hour-only stand-alone stage is +0.5 s worse. So the inputs carry signal but do not work as a separate stage.
This test asks whether they help **inside G2's allocation**, i.e. whether they tell which rows absorb the day's λ̂.

## Fixed parts (as §133 / tests/day_alloc_gbm.py)

λ̂_g, periods, leave-period-out rule, 8 airports, masks (fit on trimmed y ≤ 18,000 s, |λ̂| ≥ 1 s; body = NM-matched and
y ≤ 3,600 s), label (p − y)/λ̂ with weight λ̂², init 0.5, hyper-parameters (15 leaves, min 2,000 rows per leaf,
lr 0.05, 300 rounds, L2 10, deterministic, seed 0), clip of s to [0, 1].

## New inputs (label-free; movement data only, so identical code for 2025, Jan/Jul 2026 and Feb/Jun 2026)

For departure i at airport a with takeoff T_i, over the other departures j at a with T_j in [T_i − 30 min, T_i + 30 min]:
- `h_rl` mean(T_j − LOBT_j − p_j), `h_ra` mean(T_j − AOBT_3_j − p_j), `h_rs` mean(T_j − SOBT_j − p_j), `h_p` mean(p_j)
  (each needs ≥ 3 neighbours with the input, else missing), `h_n` count;
- `rwy30` / `rwy15` departures on i's runway in [T_i − 30 min, T_i) / (T_i, T_i + 15 min] (computed correctly, not via
  `_roll_count`);
- `bk_lobt` / `bk_sobt`: departures with LOBT (SOBT) in [T_i − 3 h, T_i] minus takeoffs in [T_i − 3 h, T_i];
- `h_tinx` mean taxi-in excess of arrivals landing in [T_i − 30 min, T_i + 30 min] (excess vs the 2025 airport ×
  runway median), ≥ 3 arrivals;
- own row: `rl` = T_i − LOBT_i − p_i, `ra` = T_i − AOBT_3_i − p_i.
p_j is the same input p the allocation is applied to (2025: the frame's p; 2026: the production p).

## Candidates

- **G3** = G2's inputs + the inputs above, at the eight airports.
- **G3h** = G2's inputs + the hour-window inputs only (no own-row `rl`, `ra`), at the eight airports.

## Decision (2025, all 12 months, 8 airports' rows, vs v63's recipe = G2x8, same frame)

A candidate passes iff: (1) pooled full ≤ −0.10 s; (2) pooled body ≤ 0; (3) pooled trimmed ≤ 0; (4) full better in
≥ 8/12 months; (5) February and June each not worse on full; (6) January and July (the holdout-chain months, the
closest to production inputs) each not worse on full. **Pick the passing candidate with the lowest pooled full.**
None passes → nothing is built from this. Reported for information: per airport, per month, 2026 projection.

## Build if a candidate passes

v64 = v63's build (`src/post/day_alloc_build.py`) with the G3 slope model fitted on all six 2025 periods; LSZH, LIRF
rows as v63; LOBT window last. A reproduction check first: the same script with G2 must reproduce v63 exactly.
**Keep iff board < 238.32.** Blind board: the same on the four-month chain (`PRC_SCOPE=final4m`, input `f4m_e`).

## Outcome (2026-10-10; `logs/day_alloc_g3.log`)

| vs G2x8 | full | body | trimmed | months better | Feb | Jun | Jan / Jul | pass |
|---|---|---|---|---|---|---|---|---|
| G3 | −0.013 | −0.072 | −0.016 | 9 | +0.56 | +0.02 | +0.02 / −0.16 | no (1, 5) |
| G3h | −0.247 | −0.147 | −0.266 | 11 | −2.43 | **+0.05** | −0.01 / −0.14 | **no (5: June)** |

**Neither passes; nothing is built.** G3h's gain is mostly February (LTFM's snow days, as for G2); without February
it is ≈ −0.07. Per airport G3h full: LTFM −1.18, EDDM −0.13, LEBL −0.08, EDDF −0.07, EGLL −0.05, others ≈ 0.
2026 projection −0.19 (G3h), −0.01 (G3). Reading: the within-day error is not proportional to the day's λ̂, so an
allocation of λ̂ cannot take it; it would have to be learned per row (corrector).
