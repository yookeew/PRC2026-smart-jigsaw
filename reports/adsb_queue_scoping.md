# ADS-B surface state of OTHER aircraft (runway queues, de-icing stops) -- scoping and bound

Written 2026-10-08 **before any number below was computed**. Steps 1-2 decide whether a gate is worth writing.

## Why this is not settled

- §114 dismissed an ADS-B runway queue because §23's "realised queue" found nothing. §23 built that queue from
  AOBT_3, and §114 showed AOBT_3 is often takeoff minus a default taxi time, so §23 largely re-measured takeoff
  counts.
- Takeoff counts saturate at runway capacity under congestion; queue length and holding time do not.
- adsb.lol sees aircraft near runways much better than at stands (§52, §60), so this could reach far more rows than
  the ~28% with a detected pushback.

## Step 1 -- coverage (label-free)

Per airport and month, 2025 (Jan, Jul and the two non-holdout raw days 09-15, 11-15) and 2026 (Jan, Jul):

- every adsb.lol surface run that ends in a takeoff (the detector's `surface_runs`) gets a runway (centreline within
  150 m and heading within 25 deg at the end of the run), a roll-start point, how far back it is visible, the time
  since it came within 1 km / 500 m of its roll start (only when it was seen outside that radius first), the time it
  spent slow (< 3 kt) within 1 km of the roll start, and long stops (>= 180 s) > 1 km from the roll start that are not
  the run's first stop (de-icing candidates);
- runs are matched to departures (same airport, |run end - MVT_TIME| <= 180 s, greedy nearest);
- reported: share of departures with a matched run; with a measurable 1 km entry; and the **neighbour reach** = share
  of departures with >= 1 same-runway run with a measurable entry taking off in [T - 15 min, T + 10 min], own run
  excluded.

Raw positions on this machine: Jan + Jul 2025, Jan + Jul 2026, 2025-09-15, 2025-11-15 only (other 2025 months exist
only as v4 detector output). Coverage numbers use no labels.

## Step 2 -- oracle bound (labels of OTHER flights only; 10 OOF months, no holdout row read)

Rows: non-LIRF departures on the 115 OOF ADS-B days (`good_days()`, v4 detections exist), with the OOF stack
`s` = NNLS(ctxc1sel OOF, cat_mixed OOF) and residual r = y - s. Residual models are LightGBM (Huber), fit
leave-one-month-out over the 10 months on rows with y <= 5 h, evaluated on every row of the held-out month.

- **Control C:** s, airport, hour, T - SCHED, (T - AOBT_3) - s, same-runway departures in +-15 min, the mean
  NM pseudo-residual (T_j - AOBT_3_j - s_j) of same-runway departures in +-15 min (stage-b `nbp` analogue), and
  the **true** residual mean / count of same-airport departures in +-30 min whose pushback v4 observed
  (dwell/pass/appear) -- an oracle version of the existing ADS-B neighbour features (`nbq`). C therefore credits
  the chain with more than it has.
- **O1 = C + realised surface occupancy from other flights' TRUE off-blocks** B_j = T_j - y_j (never the own label):
  for x in {0, 5, 10, 15, 20, 30} min, aircraft taxiing at T - x (B_j <= T - x < T_j) at the airport and on the own
  runway, and the subset of them that take off before T (queue ahead).
- **O2 = O1 + true residuals of the 3 previous and 3 next departures on the own runway** (all flights, any ADS-B).

O2 - C bounds what any feature built from other aircraft's surface state can add beyond the runway block, the NM
neighbours and the existing ADS-B neighbours, at full coverage and perfect measurement.

### Stop rule (fixed now)

Metrics pooled over the 10 held-out months: body RMSE (NM-matched, y <= 1 h) and trimmed RMSE (y <= 5 h).

- **STOP** if the O2 - C body gain is smaller than 1.0 s, or if fewer than 7 of 10 months improve.
- Otherwise project the board gain as (O2 - C body gain) x R x 0.5, with R the 2026 neighbour reach from step 1
  (non-LIRF departures weighted by 2026 counts) and 0.5 an assumed proxy fidelity. **STOP** if the projection is
  below 0.4 s.
- If it survives: label-free features that do not scale with coverage, a 2025-vs-2026 distribution check, a
  separate pre-registration, then the v50-chain gate.

## Fallbacks if killed

1. 2026 arrivals keep their block times: check ADS-B surface-timing drift (taxi-in) against real 2026 labels.
2. Per-airport comparison with outgoing-boat on the v50 chain; read their code where we trail (EDDF).

## Results

(appended below after running)

### Step 1 result (2026-10-08; `tests/adsb_queue_runs.py`, `tests/adsb_queue_coverage.py`, `logs/adsb_queue_coverage.log`)

- Run extraction: runway assignment agrees with `RUNWAY_mvt` on 69-100% of time-only matches. The disagreements were
  wrong matches when several runways are in use at once, so the match requires the same runway (±120 s).
  Runs with no runway are 2-6 point fragments (mostly EDDM) and are dropped.
- **Neighbour reach** (>= 1 other same-runway run with a measured 1 km entry in [T - 15, T + 10] min), non-LIRF:
  **2025 Jan 0.387 / Jul 0.372 → 2026 Jan 0.538 / Jul 0.541.** Own matched run: 0.375 / 0.335 → 0.517 / 0.535.
  - Coverage-stable: EHAM 0.95-0.97 in all four months; LSZH 0.69-0.83; EDDM January 0.95 / 0.96.
  - Coverage-shifted: EGLL 0.18 / 0.005 → 0.71 / 0.998; LEMD 0.00 / 0.01 → 0.89 / 0.32; EDDM July 0.37 → 0.98;
    LEBL July 0.56 → 0.75; EDDF *down* 0.74 / 0.86 → 0.50 / 0.44.
  - None: LTFM (no adsb.lol data), LFPG (< 0.5%). LIRF ≤ 0.32 and 0 in July 2026.
- Reach is larger than the own-pushback share, as expected, but it moves 2025 → 2026 by airport as much as the
  coverage that sank v40. The projection uses R = 0.54 (2026, non-LIRF).

### Step 2 result -- oracle (`tests/adsb_queue_oracle.py`, `logs/adsb_queue_oracle.log`)

609,238 rows (non-LIRF, 115 OOF ADS-B days). Own-label invariance asserted on all 27 features.

| slice | s | C | O1 - C | O2 - C |
|---|---|---|---|---|
| full | 186.47 | 184.56 | −5.77 | −14.27 |
| body | 170.50 | 167.43 | −4.09 | −10.31 |
| body, no own pushback | 184.37 | 184.64 | −4.95 | −12.42 |
| body, own pushback | 117.08 | 95.55 | +0.22 | +0.27 |
| NM-matched 30-60 min | 497.62 | 515.19 | −11.83 | −35.77 |

- 10/10 months improve (O2 − C −6.1 to −14.0; O1 − C −1.2 to −5.5).
- Per airport (O1 − C / O2 − C): LTFM −9.3 / −22.5, LFPG −7.9 / −18.7, EGLL −3.7 / −8.0, LEMD −2.6 / −8.6, LEBL −1.6 / −3.8,
  EDDF −1.5 / −5.8, LSZH −1.3 / −4.7, EDDM −0.6 / −2.5, EHAM −0.1 / −1.0. **The largest oracle gains are at the two
  airports adsb.lol does not see.** Where an own pushback is observed, nothing is left.
- **Stop rule: PASS** (O2 − C −10.31 ≥ 1.0, 10/10 months; projection 10.31 × 0.54 × 0.5 = 2.8 s ≥ 0.4).

### Step 2b -- real label-free proxy on the same rows (added after the oracle result; rule fixed before running)

The v4 detector output exists for all 115 OOF days: each matched departure's first surface sample
(`adsb_first_ts`). So the O1 quantities have a label-free ADS-B version there, with no holdout row read:
- visible occupancy at T - x (matched other departures with first_ts <= T - x < T_j), airport / own runway / ahead
  of the own takeoff; the same divided by the matched share of departures within ±1 h (coverage-normalised);
- visible surface time L_j - s_j of the 3 previous / 3 next matched same-runway departures.
C' = C + own (T - first_ts) - s (the combiner already has own first-sighting information). Arms: C', P (C' + proxy),
O1' (C' + O1).

**Rule:** a gate is worth writing only if **P - C' body ≤ −0.5 s pooled with ≥ 7/10 months improving**, and the gain
is not confined to airports with no 2026 reach. Otherwise STOP and go to the fallbacks.

### Step 2b result (`tests/adsb_queue_proxy.py`, `logs/adsb_queue_proxy.log`) -- STOP

| slice | C' | P - C' | O1' - C' |
|---|---|---|---|
| full | 183.47 | −0.10 | −5.66 |
| body | 166.26 | **−0.23** | −3.91 |
| body, no own pushback | 183.70 | −0.34 | −4.75 |
| NM-matched 30-60 min | 513.18 | −1.93 | −11.26 |

- 8/10 months improve, but pooled −0.23 misses the −0.5 bar. EGLL +0.80 (the coverage-shifted airport).
  Coverage-normalised and raw counts are the same (−0.23 / −0.25).
- **Why the bigger oracle doesn't matter.** At the airports adsb.lol covers, the O1' − C' oracle is small: EDDF −1.4,
  EGLL −3.6, LEMD −2.0, LSZH −1.1, LEBL −1.4, EHAM −0.2, EDDM +0.1. Weighted by 2026 reach, a perfect occupancy measurement
  is worth ≈ 0.7 s of body RMSE. The pooled oracle is driven by LTFM (−9.3) and LFPG (−7.6), which have no
  coverage, and by O2's neighbour residuals, which are labels. The proxy recovers ~6% of O1'.
- Runway-entry (1 km) features would need the raw ADS-B of the other 2025 months, which is not on this machine. Even
  at 3× the proxy's fidelity the projection stays under 0.4 s, so the files are not worth fetching for this.
- **Conclusion: §114's dismissal stands, now for the right reason.** §23's queue did re-measure takeoff counts, but
  realised occupancy is worth little beyond the chain at the airports ADS-B sees. Most of what is left sits at LTFM
  and LFPG, plus per-flight residuals of neighbours that no surface observation reproduces. De-icing was not bounded
  separately: long remote stops are as common in July as in January (EHAM 12.5% / 8.8% of matched runs), so they are
  mostly taxiway holds, and the regime part is inside O2.

## Fallback 1 -- ADS-B surface-timing drift on 2026 arrivals (`tests/adsb_arrival_drift.py`, `logs/adsb_arrival_drift.log`)

New label-based check: 2026 arrivals keep their block times. Landing run (gs falls through 40 kt) matched on
airport + runway + |run start − MVT| ≤ 120 s; ADS-B in-block = first stationary sample within 100 m of the own stand.
lag = true taxi-in − ADS-B taxi-in, arrivals with y ≤ 1 h, median (IQR 23-52 s in every cell):

| airport | 2025-01 | 2025-07 | 2026-01 | 2026-07 |
|---|---|---|---|---|
| EDDF | 34.1 | 33.3 | 35.3 | 31.8 |
| EDDM | −31.4 | −29.3 | −29.6 | −30.2 |
| EGLL | −26.5 | (−38.5, thin) | −24.5 | −23.0 |
| EHAM | −33.5 | −30.8 | −31.0 | −31.3 |
| LEBL | −67.5 | −59.7 | −67.6 | −64.5 |
| LEMD | -- | −63.1 | −67.8 | −67.0 |
| LSZH | 1.2 | 2.0 | 7.7 | 7.4 |

- **No receiver-timing drift at stands at any covered airport** (≤ ±6 s; LSZH +6). The EGLL shift that §40 suspected
  from the AOBT_3 yardstick does not appear on arrivals (−26.5 → −24.5 / −23.0).
- So the combiner's 2025-fitted ADS-B lags should still hold in 2026. Nothing to correct. **Closed.**
