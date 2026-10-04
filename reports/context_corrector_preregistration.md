# Context corrector on the final chain — pre-registration

Written 2026-10-04, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **About two-thirds of our ordinary-flight gap to outgoing-boat sits where
  ADS-B can't help:** EGLL, LIRF, LFPG, LTFM. Measured as a share of our
  holdout SSE vs their v28: ~1.75% there, against ~0.95% at the ADS-B
  airports.
- **They reach it with the same raw data.** Their corrector adds feature
  blocks our models have never seen.
- **Our past correctors (§48–§51) only saw our existing base features.**
  This tests the blocks themselves, on top of the final chain, in the
  pattern that has transferred to the board (v30: a label-free corrector
  cross-fit Jan ↔ Jul).
- **Source of the block ideas:** <https://github.com/EnioAguiar/prc-taxiout-2026>
  (GPLv3; `refcel.py`, `plano13.py`, `stack.distancias_plano`, `pista.py`,
  `superficie.py`). Everything here is built by us from our own data.

## Blocks (`tests/context_corrector_test.py`; label-free unless stated)

1. **Stand prefix:** airport | leading letters of the stand (stand area).
2. **Cell statistics:** median, P90, SD and count of taxi per (airport,
   stand, runway).
   - Falls back to (airport, runway) and then the airport when a cell has
     fewer than 30 rows.
   - Fitted on the 10 training months' labels only, so the holdout is never
     read. Also the prediction minus the cell median.
3. **NM consistency:**
   - planned duration (ARVT_1 − EOBT_1), realised duration (ARVT_3 −
     AOBT_3), their difference and ratio;
   - ADES vs NM ADES, ADES_FILED vs ADES (diversion), aircraft type vs NM
     type.
4. **Distance to planned times:** prediction − (T − X) for X in SCHED,
   LOBT, IOBT, EOBT_1 and AOBT_3; the smallest absolute one; how many
   planned times exist.
5. **Retention at NM's off-block clock** (t = AOBT_3, same airport):
   - departures whose EOBT_1 has passed but whose AOBT_3 hasn't, airport
     and runway;
   - departures with AOBT_3 ≤ t < T on the same runway;
   - mean (AOBT_3 − SCHED) of departures with AOBT_3 in [t − 15 min, t).
6. **Surface at the estimated pushback** (o = T − chain prediction):
   - departures with o_j ≤ o < T_j;
   - arrivals taxiing in (landing ≤ o < in-block);
   - takeoffs and landings in (o, T).

## Test

- **Input:** the v32/v33 method's holdout predictions
  (`cache/eval/v31_chain_holdout.parquet` + the LOBT window + the
  NM-unmatched lines), called p.
- **Corrector:** LightGBM, L2, on (y − p); 63 leaves, lr 0.05, min_data
  200. Rounds come from an inner day-of-month split (day % 5 == 0).
  - Training rows: NM-matched, y ≤ 5 h.
  - Fit on Jan, applied to Jul, and the reverse.
  - Corrected prediction = p + correction, then the LOBT window.
  - Rows served by a fixed line (NM-unmatched, d > 6 h) keep p.
- **Control:** the same with only p, airport, hour, `aobt3_taxi` and
  `sched_takeoff_offset`. **Treatment:** control + blocks 1–6.

## Decision

**Primary: body RMSE** (NM-matched, y ≤ 1 h), all airports, treatment vs
control. **Adopt iff:**
1. pooled gain ≥ 1.5 s;
2. the gain holds in both Jan and Jul;
3. trimmed RMSE is not worse than control.

**If adopted:**
- A per-block ablation (drop one block at a time) is reported, not decided
  on.
- Then the ranking build: the corrector fit on both holdout months, applied
  to v33's inputs, as the next version.
- **Keep iff the board < the best at the time.**

## Amendments

**2026-10-04, after results.** The rule compared treatment with control only.
It should also have required beating the input p. Both arms make p worse, so
this is **rejected** whatever the scripted verdict says.

## Results (2026-10-04, `tests/context_corrector_test.py`, log `logs/context_corrector_test.log`)

| | p (input) | control | treatment |
|---|---|---|---|
| body | 199.59 | 204.18 | 201.57 |
| trimmed | 248.84 | 253.01 | 251.38 |
| LTFM body | 236.73 | 249.45 | 239.63 |

- **The script says ADOPT** (treatment beats control by 2.61; Jan 3.54,
  Jul 1.98).
- **But both correctors hurt p.** A corrector fit on one holdout month and
  applied to the other learns that month's residual bias, which doesn't carry
  over.
- **The blocks carry signal:** treatment recovers most of control's damage,
  most of all at LTFM.
- **The single-month setup can't use them.** Next: fit on the 10 OOF months
  (more data, every season), on the stack, the way outgoing-boat does.
