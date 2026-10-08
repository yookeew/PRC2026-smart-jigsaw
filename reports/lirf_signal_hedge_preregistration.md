# LIRF NM-unmatched echo hedge with label-free movement signals — pre-registration

Written 2026-10-07, before any gate result. Changes made after results exist go
under Amendments, dated.

## Why, and the team rule

- **Team rule (PROGRESS §87):** no change may alter LIRF NM-unmatched departures.
  **The team lifted it for this one test on 2026-10-07** (user decision), after
  the exploration below.
- §101: the LIRF NM-unmatched echo hedge is the largest identified error pool
  (validated 2026 bands: 1–3 h 8.7%, 3–6 h 6.9% of the board's SSE).
- §102, §103 and §111 found no usable label-free signal for these rows. Every
  signal tried improved log-loss but worsened the offset-weighted squared error
  out of month (the v22 pattern).

## Exploration already done (disclosed: all 12 months of 2025 have been seen)

Scratch scripts, 2026-10-07 (PROGRESS §113). LIRF NM-unmatched departures with
T − SCHED ≥ 1 h, 2025 (1,375 rows, echo rate 0.49). p is the OOF stack echo
probability.

- **The signals separate echo in 2025:**
  - no other departure with the same flight number within ±36 h: echo 0.74, against 0.26 with two or more;
  - another departure from the same stand between SCHED + 10 min and T − 30 min: echo 0.74–0.89, against 0.45;
  - an arrival at the same stand in that window: echo 0.75–0.84, against 0.34.
- **Leave-one-month-out logistic on [logit p, signals]**, offset-weighted
  (q − echo)²:
  - pooled over all offsets: 1.11× raw p (worse);
  - by band: 1–3 h 0.84× (2/12 months worse), 3–6 h 0.96× (4/12), 6–12 h 0.56× (0/11), > 12 h 1.26× (6/10).
- **The < 6 h scope below was chosen after seeing that band table.** No result
  on 2025 in this gate is independent of that choice. The leaderboard is the
  only clean test.
- On LIRF NM-matched rows the same signals add nothing, because p already
  prices them. So the scope is NM-unmatched only.

## Treatment

**Population (scope):** departures with ADEP = LIRF, FLIGHT_ID_mvt null, and
3,600 ≤ off < 21,600 s, where off = MVT_TIME − SCHED_TIME.
- Rows with off ≥ 6 h are untouched; nm_lines serves them.
- Rows with stack p ≥ 0.99 are also untouched.

**Signals.** Label-free, computed from the movement table only. Neighbours
count only if their MVT_TIME falls in the same calendar month (UTC) as the row's
MVT_TIME. The 2026 ranking holds only January and July, selected by MVT_TIME
month, so this gives the same definition in 2025 and 2026.
- `nF`: other LIRF departures with the same `FLIGHT_mvt` and |SCHED' − SCHED| < 36 h.
- `nF7`: the same with |SCHED' − SCHED| < 7 days.
- `nD`: other LIRF departures with the same `STAND_mvt` and MVT_TIME' in (SCHED + 600, MVT_TIME − 1,800).
- `nA`: LIRF arrivals with the same `STAND_mvt` and BLOCK_TIME' (in-block) in the same window.
- `fnull`: `FLIGHT_mvt` is null.

Null keys never match.

**Classifier.** sklearn `LogisticRegression(C=1.0, max_iter=2000)`, unweighted.
- **Inputs:** [logit p, log1p nF, nF == 0, log1p nF7, log1p nD, nD > 0, log1p nA, fnull].
- **Labels:** echo = |BLOCK − SCHED| < 30 s.
- **Training rows:** LIRF NM-unmatched departures with off ≥ 3,600 and a block time. This is all offset bands, as in the exploration.
- **p** = 0.577 · p_lgb + 0.428 · p_cat (§101/§103 convention), clipped to [1e-4, 1 − 1e-4].
  - 2025: the OOF fold files plus the holdout_ev files.
  - 2026: `lgb_direct_nolirf_nb_rank` / `cat_mixed_rank`. The LightGBM echo classifier is shared across variants; `echo_prob` is identical on the holdout.

**Prediction change.** It moves only the echo weight and keeps every chain
correction inside the non-echo branch:
- offc = clip(off, 0, 140,000);
- P_ne = (P − p · offc) / (1 − p);
- **P' = P + (q − p) · (offc − P_ne)**;
- then P' = max(P', 0).

## Gate (`tests/lirf_signal_hedge_test.py`)

**A. Holdout, chain level.**
- q is fitted on the 10 non-holdout months and applied to in-scope Jan+Jul 2025 rows.
- P = v42's holdout chain (`cache/eval/v42_chain_holdout.parquet`).
- Reported: full RMSE on all holdout rows, pooled / Jan / Jul; in-scope SSE.

**B. Leave-one-month-out, base level (12 months).**
- For each month, q is fitted on the other 11 months.
- P = base stack 0.577 · pred_lgb + 0.428 · pred_cat, from the OOF/holdout_ev files.
- Reported: in-scope SSE per month.

**Adopt rule** (all must hold):
1. A: pooled full-RMSE delta < 0.
2. A: Jan delta ≤ 0 and Jul delta ≤ 0.
3. B: pooled in-scope SSE lower, and lower in ≥ 9 of 12 months.
4. B: the pooled in-scope SSE gain is still < 0 after removing the 3 rows with the largest individual gain.

## Build (only if the gate passes)

- The production classifier is fitted on all 12 months.
- It is applied to in-scope 2026 rows of `smart-jigsaw_v42.parquet`, with P = v42's values.
- Output: `data/submissions/smart-jigsaw_v45.parquet`.

**Checks:**
- 344,841 rows; MVT_ID set equal to `submitting.parquet`; no nulls or negatives;
- only in-scope LIRF NM-unmatched rows differ from v42;
- report the rows changed, RMS and mean change, and the q − p distribution.

**Keep iff the board < 249.3128 (v42).** One upload. Nothing is fitted to board
scores.

## Amendments

(none. The clipped-branch variant below was measured after the result and is reported only; it is not used.)

## Results (2026-10-07)

Logs: `logs/lirf_signal_hedge_test.log`, `logs/v45_build.log`,
`logs/lirf_signal_hedge_clipvar.log`.

**Gate: PASS (all four rules).**

**A. Holdout chain** (10-month fit):
- 354 in-scope rows, echo 0.316; mean p 0.441 → q 0.376.
- In-scope SSE 3.94e9 → 3.50e9.
- Full RMSE: pooled 304.075 → 301.966 (**−2.11**), Jan −0.89, Jul −3.28.

**B. Leave-one-month-out** (base level):
- In-scope SSE 1.826e10 → 1.729e10 (0.947).
- 9/12 months better; worse in Feb (1.051), Jun (1.019) and Nov (1.056).
- The gain without the 3 largest single-row gains is still −8.3e8.

**v45 build** (`data/submissions/smart-jigsaw_v45.parquet`):
- **12-month fit:** 1,375 rows. Coefficients [logit p 1.39, log1p nF −0.95, nF=0 −0.49, log1p nF7 −0.48, log1p nD −1.40, nD>0 +1.25, log1p nA +0.67, fnull +0.81]; intercept 0.96.
- **2026 scope:** 322 rows (1–3 h 274, 3–6 h 48). Mean p 0.428 → q 0.370; q − p p10 −0.23 / p90 +0.13.
- **Change on scope:** mean −279 s, RMS 1,097 s, max 4,175 s. RMS over all rows 33.5 s.
- **Checks:** 321 rows differ from v42, all in scope; 344,841 rows; ID set = template; no nulls or negatives.
- **Keep iff the board < 249.3128.**

**Label-free diagnostics, 2026 vs 2025:**
- **Signal mix stable** (share > 0, 2025 → 2026):
  - nF 0.633 → 0.634;
  - nF7 0.790 → 0.817;
  - nD 0.115 → 0.152;
  - nA 0.347 → 0.317;
  - fnull 0.019 → 0.022.
- **Implied non-echo branch P_ne < 0 on 20.5% of 2026 scope rows**, against 7.1% on the holdout. The v41 chain values copied into v42 sit lower relative to p · offset in 2026, so the step (q − p)(offc − P_ne) is inflated on those rows.
  - A variant with P_ne clipped to [0, offc] in the step scores the same on the gate (−2.095 vs −2.109).
  - On 2026 it differs from v45 by 88 s RMS on 77 rows, ≈ 0.003 s of board RMSE.
  - So the deviation is immaterial and v45 is built as registered.

**Board (2026-10-07): v45 = 249.0543 (−0.26 vs v42's 249.3128). Kept; new best.**
- About 12% of the holdout gain (−2.11) transferred, in line with earlier tail
  results (nm_lines 7%).
- First LIRF NM-unmatched change to win on the board, after v20, §42, v22, v28 and v40.
