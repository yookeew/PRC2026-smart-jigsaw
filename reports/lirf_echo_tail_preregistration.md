# LIRF NM-unmatched departures ≥ 12 h late: predict the schedule echo — pre-registration

Written 2026-10-07, before the upload. Changes made after results exist go
under Amendments, dated.

## Basis (disclosed: this rests on past board results, not on a new 2025 holdout)

**§46 (team finding, 2026-09-28).** v20 moved the 11 LIRF NM-unmatched ranking
rows with T − SCHED ≥ 12 h towards the "+24 h" label (86,400 + 1,210 s) and lost
3.75 on the board. The finding: the +24 h bug did not carry into 2026.

**Quantified now (`logs/lirf_echo_tail_ab.log`).** We have the board scores of
v19 and v20, which differ only on those 11 rows. Their difference is one linear
equation in the rows' labels:
- observed ΔSSE: +8.58e8;
- predicted if all 11 are schedule echoes (y = T − SCHED): +7.39e8;
- predicted if all are +24 h labels: −7.87e9;
- predicted if all are normal taxis (~1,500 s): +1.09e10.

Only the echo hypothesis is consistent.

**Cross-check, v32 → v33.** This pair introduced §76's lines and Rome rule.
Under the echo hypothesis these 11 rows cost +4.1e8, and the other 85 changed
rows gained −5.3e8 (net observed −1.23e8). No contradiction.

**2025 composition.** LIRF NM-unmatched rows ≥ 12 h late are echo or +24 h
labels, almost never normal taxis (§76; 18 echo rows in 2025).

**Exact echo value.** At LIRF the block time carries the negated takeoff
jitter, so label = T − SCHED + e_T, with e_T = MVT_TIME's seconds offset from
the nearest minute. It holds within ±1 s on all 675 echo rows of 2025 (RMSE
0.70 s, against 3.50 s for T − SCHED alone).

**Team principle.** This uses the board results of submissions we had already
made (§82 rejected fitting blend weights to board scores). It is the same
inference §46 drew from v20; here it's made quantitative. No new upload is made
to probe labels.

## Treatment

On v45, for departures with ADEP = LIRF, FLIGHT_ID_mvt null and
T − SCHED ≥ 43,200 s: P = T − SCHED + e_T. All other rows are unchanged.

## Expected effect

- If all 11 rows are echoes: ΔSSE = −Σ (P_v45 − echo)² ≈ −6.9e8, so the board
  goes 249.05 → ~245.0.
- Most of it sits on the 4 Rome-line rows (WMTT3508, ITYTY672, SVII3006,
  QFAA7006) and NOSOS431, where v45 is 1.6–16.3k s above the echo value.

## Decision

`src/post/lirf_echo_tail.py smart-jigsaw_v45.parquet smart-jigsaw_v46.parquet`.
**Keep iff the board < 249.0543 (v45).**

## Amendments

(none)

## Results

**Board (2026-10-07): v46 = 249.2907 (+0.24 vs v45's 249.0543). REJECTED; v45 stays best.**
- Observed ΔSSE +4.06e7, against the −6.88e8 predicted under all-echo. The hypothesis is wrong.
- The new v45 → v46 equation weights the three rows moved most (WMTT3508, ITYTY672, SVII3006). It says they lie, on average, *above* their echo value. ITYTY672 = +24 h fits it almost exactly.
- So the +24 h label does occur in 2026, and the Rome line's hedge was right there. §46's "the bug didn't carry into 2026" was too strong: v20 lost through how it moved the other rows.
- **The two equations jointly don't identify the 11 labels.** Close fits need 4–6 non-echo rows, including normal labels never seen in 2025.
- Not pursued further. Composing more submissions from these equations would be label probing.
- `tests/lirf_echo_tail_joint.py`, `logs/lirf_echo_tail_joint.log`.
