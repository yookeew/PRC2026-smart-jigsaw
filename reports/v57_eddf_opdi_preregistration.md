# v57 part A: OPDI ground events at EDDF -- witness, leave-one-month-out gate, build rule

Written 2026-10-09 **before any EDDF OPDI-vs-label number and before the witness was computed**. Follows PROGRESS
§128.8 (EDDF left open: 2026 coverage, Jan/Jul 2025 holdout share 0.025, so the holdout cannot measure it).

## Facts fixed before writing (counts only, no labels)

- EDDF OPDI ground events switch on **2025-09-16** (daily share of departures with a ground event ≥ 60 s before T:
  ≤ 0.007 up to 09-15, 0.16 on 09-16, 0.29–0.73 afterwards). Missing: 2025-10-02..10-11 (file 404 upstream).
  2026: none on 01-02..01-10 (sparse OPDI period), 0.5–0.8 on later January days, 0.55 in July.
- Monthly share: Sep 0.24, Oct 0.29, Nov 0.43, Dec 0.51 (2025); Jan 0.36, Jul 0.55 (2026).
- EDDF rows with a ground event on the OOF ADS-B days the post-stages train on: Sep 1,567, Oct 2,139, Nov 3,089,
  Dec 2,521 (≈ 9.3k on ≈ 28 days). 2026: Jan 5,589, Jul 11,411.
- v52's production run (`src/post/v52_comb_delta.py`, `v52_nb_delta.py`, PRC_LGB=ctxc1sel) already fit the
  treatment unscoped; its EDDF rows were zeroed by the LSZH scope. `cache/v52/comb_delta_opdi.parquet` and
  `cache/v52/nb_delta_opdi.parquet` hold `delta` for every airport.

## W -- label-free drift witness (stop rule fixed now)

`tests/v57_opdi_witness.py`. Periods: P25 = 2025-09-16..12-31, J26 = 2026-01-11..31, U26 = 2026-07.
- **W1 (arrivals, decisive).** Arrival rows keep their in-block time in 2026. Match each EDDF / LSZH arrival to an
  OPDI flight (callsign = CALLSIGN_flt, else FLIGHT_mvt; airport = ADES; nearest exit-runway event within 300 s of
  the landing time). Lag = first entry-parking_position event after landing − BLOCK_TIME. Median and IQR per period.
- **W2 (departures, reported).** On rows with `o_lead_exitpark`: (T − AOBT_3) − o_lead_exitpark, and on adsb.lol
  dwell/pass rows adsb_taxi − o_lead_exitpark; medians per period.
- **STOP (no EDDF build)** if the EDDF W1 median lag moves by more than 45 s between P25 and J26 or between P25 and
  U26, unless LSZH (whose OPDI delta transferred on the board, v52) moves in the same direction by at least half
  as much. Also STOP if the EDDF W1 match share of arrivals differs by more than a factor 2 between P25 and 2026.

### Amendment W-A1 (2026-10-09, after W1/W2 were computed, before W3 and before any gate number)

W1 passed (EDDF median lag −215 s P25, −185 J26, −219 U26; match share 0.905 → 0.843 / 0.806). W2 (reported
only) flags a move: EDDF median (T − AOBT_3) − o_lead_exitpark = 400 / 420 / 390 / 440 s in Sep–Dec 2025, then
525 (Jan) / 535 (Jul) 2026, in every stand group; LSZH flat (240–275 → 275 / 225). Either NM's AOBT_3 or OPDI's
exit-parking event moved at EDDF. Added witness, stop rule fixed before computing it:
- **W3.** On EDDF departures with `o_lead_exitpark`, anchor x̂ = o_lead_exitpark + c, c = median(y − o_lead_exitpark)
  over P25 (labels, 2025 only). 2025: mean(y − x̂) on P25 is ≈ 0 by construction; 2026: mean(v54 − x̂) per month,
  where v54's EDDF level is checked against the published daily sums (v55's EDDF λ; v54 is level-correct to within
  its λ). The same with the AOBT_3 anchor â = (T − AOBT_3) + c₃.
- **STOP** the EDDF build if |median over 2026 days of the daily mean (v54 − x̂) − (EDDF λ shift that v55 applied)|
  > 45 s in January or in July (the OPDI clock is off by more than the W1 tolerance), and report which clock moved.

### Amendment W-A2 (2026-10-09): W3 outcome -- EDDF build STOPPED

`tests/v57_opdi_witness_w3.py`, `logs/v57_opdi_witness_w3.log`. c = 377 s, c₃ = −58 s. Median over days of the daily
mean on EDDF exit-parking rows:

| month | truth (2025) / v55 (2026) − x̂ | − â | OOF (2025) / v54 (2026) − x̂ | − â |
|---|---|---|---|---|
| 2025-09 / 10 / 11 / 12 | −13.7 / −25.4 / −25.0 / −1.2 | 38.2 / 22.8 / 29.4 / 20.1 | −23.5 / −30.5 / −30.5 / 3.5 | 39.0 / 14.4 / 15.1 / 27.8 |
| 2026-01 | 41.3 | −19.7 | 40.4 | −20.2 |
| 2026-07 | 43.4 | −25.6 | 46.5 | −24.2 |

- The rule as written compares 2026 with 0 (41.3 and 43.4 < 45: no stop). Its premise, "≈ 0 by construction" in
  2025, is wrong for a median of daily means: the 2025 reference is −16 (range −25 to −1). Against the 2025
  reference the OPDI clock moved **+57 s (Jan) / +59 s (Jul)**, above the 45 s tolerance the rule was meant to
  enforce. **Applied as intended: STOP.** No EDDF delta enters v57.
- Reading: if v55 is unbiased on these rows (its EDDF level is pinned by the daily sums: λ 7.9 / 2.0 s), the
  exit-parking event fires ~58 s later after the true off-block in 2026 than in late 2025, and NM's AOBT_3 moved
  ~48 s earlier (2025 +28 → 2026 −20/−26). Both shifts together match W2's +110–120 s. adsb.lol dwell/pass rows
  agree for July (+134 s, n 146), not for January (+8 s, n 60).
- The leave-one-month-out run below is still run: its control arm supplies part B's chain-like OOF predictions,
  and the EDDF treatment numbers are reported as information only. They cannot reverse this stop.

## Coverage rule (§86, v51 lesson), fixed now

Reference population = EDDF departures on OOF ADS-B days ≥ 2025-09-16 whose airport-day OPDI share ≥ 0.2.
EDDF passes iff (a) its ground-event share there ≥ 0.2; (b) the 2026 share (days with share ≥ 0.2) is within
[0.5, 2] × (a); (c) the standardised mean difference of `o_lead_first` on covered rows (reference vs 2026) ≤ 0.25.

## G -- leave-one-month-out gate (`tests/v57_eddf_lomo.py`)

The v52 production recipe, refit with one OOF month held out:
- Training rows as in `src/post/v52_comb_delta.py` / `v52_nb_delta.py`: the Jan+Jul 2025 holdout rows (holdout stack,
  `ctxc1sel` holdout predictions + catcorr, NNLS) plus the 115 OOF ADS-B days (`ctxc1sel` OOF stack), minus every
  row of the held-out month m. Combiner: non-LIRF, taxi ≤ 18,000 s. Stage b: taxi ≤ 18,000 s.
- Arms (same rows, same seeds): **ctl** = v50's inputs (combiner POS + RO, stage b RO); **trt** = ctl + the OPDI
  block O + REL (exactly v52's treatment).
- Prediction for the held-out month's rows: rows with ADS-B information (`has_adsb`, non-LIRF) get s + combiner;
  the others get s + stage b, day-demeaned over non-ADS-B rows per airport-day (as production). Clip [0, 140,000].
- m runs over all 10 OOF months (the ctl arm also serves as chain-like OOF predictions for part B's development).

**Primary population P:** EDDF departures in months 2025-09..12 on OOF ADS-B days ≥ 2025-09-16 whose airport-day
OPDI ground-event share ≥ 0.2. Body = NM-matched and y ≤ 3,600 s; trimmed = y ≤ 18,000 s.

**PASS iff all of:**
1. body RMSE (trt − ctl) on P ≤ −2.0 s;
2. body delta < 0 in at least 3 of the 4 months Sep–Dec;
3. day-cluster bootstrap (P's days, 2,000 draws) 95% CI upper bound of the body delta < 0;
4. full RMSE delta on P ≤ 0;
5. **validity:** the positive control, LSZH body delta pooled over the 10 held-out months, is < 0. If 5 fails the
   design cannot see a known effect: no build, whatever 1–4 say.

Reported: trimmed; per month; combiner-served vs stage-b rows; covered vs uncovered rows; top-20 share of the
gain; the other airports (unscoped); and the 2026 projection: P's per-row ΔMSE in (covered, uncovered) cells ×
the 2026 EDDF row counts in those cells (days with share ≥ 0.2), converted to the board RMSE at 240.7425.

## Build if W and G pass

EDDF delta = EDDF rows of `cache/v52/comb_delta_opdi.parquet` + `cache/v52/nb_delta_opdi.parquet` (v52's
production treatment − control; the same way v52's LSZH delta was carried into v53–v56), applied only on 2026
EDDF airport-days with OPDI share ≥ 0.2 (drops 2026-01-02..10). Zero on NM-line rows. It goes in **before** the
daily reconciliation, which is re-solved last (part B); every LIRF row keeps v56's value; LOBT window last.

## Outcome (2026-10-09; PROGRESS §129)

- W1 PASS, W2 flag, **W3 STOP** (amendment W-A2): no EDDF build.
- G, information only (`tests/v57_eddf_lomo_eval.py`, `logs/v57_eddf_lomo_eval.log`), P = 19,332 EDDF rows on 31 days,
  covered share 0.472: body 119.41 → 114.30 (**−5.11**; covered −11.59, uncovered −0.01), months −4.92 / −4.08 /
  −6.95 / −4.21, day-cluster CI [−6.51, −3.83], full −4.96, top-20 share 0.40. Conditions 1–4 would pass.
  LSZH positive control (Sep–Dec only; the treatment arm was not run on the other six months): body −23.99,
  CI [−28.94, −19.17].
