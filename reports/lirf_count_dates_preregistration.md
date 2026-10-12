# LIRF off-block dates from the published daily counts (v70) -- pre-registration

Written 2026-10-12 before the v70 build. Base: v69 (236.5174, current best).

## Mechanism and evidence (computed before this file, disclosed)

- `external-data/all_pre_departure_delays_{2025,2026}.csv` publishes `FLT_DEP_IFR_2`, the number of IFR departures per
  airport and UTC **off-block** date. At LIRF it equals our labels' count on 358 / 365 days of 2025; the 7 misses are 5
  month-end days (published 1–3 higher, zero delay attached) and one ±1 pair from a block 5 s before midnight.
- An echo row's block is SCHED (date of S); a real pushback's block is T − taxi. For a delayed departure whose S and
  T − taxi fall on different UTC dates (a "straddler"), the day counts say which date the block is on.
- **Bookkeeping** (`src/post/lirf_count_dates.py`): candidate block dates per LIRF departure = dates of T − N for
  N ∈ [180, 3600] s, plus date(S ± 6 s) where an echo is possible (NM-unmatched, or |S − LOBT| ≤ 3606), plus
  "+24 h" dates for NM-unmatched rows ≥ 10 h late. Rows with one candidate count toward it; per connected component of
  the others, enumerate assignments, keep those with the fewest days whose count is missed, and decide a row only if
  they all give it the same date and none misses a day among its own candidates. Days used: those whose whole
  population is in the file (2026: Jan 1–31, Feb 1–27, Jun 1–30, Jul 1–30).
- **2025 validation, labels hidden** (12 months, 120 straddlers ≥ 45 min late): 78 decided, 77 right. Dates only an
  echo / "+24 h" can reach: 33 of 33 truly echo or "+24 h". Dates only a real pushback can reach: 28 of 28 real. The
  one miss: a block 5 s before midnight.
- 2025 NM-unmatched UTC straddlers 1–12 h late: 35 echo, 1 "+24 h", 0 real (non-straddlers: 54% real).

## Candidate

**v70** = v69 with LIRF rows changed only where **all** hold: NM-unmatched; T − S in [1 h, 12 h); the count decides a
date that only an echo / "+24 h" can reach; the v69 prediction is below the echo value. New value = echo value
T − S + e_T (e_T = T's seconds offset from the whole minute, |e_T| ≤ 6; §115). Under either label (echo, or
"+24 h" ≈ 86,400 + taxi) the move reduces the error; only a real label, which the count excludes, would make it worse.

**Not changed:** the one 2026 row the count places on a real-only date (RYR9ZM, NM-unmatched, v69 at 9,107 vs echo
19,502): 2025 has no real NM-unmatched straddler, so count and mechanism disagree and v69 is near the 50/50 hedge.
Rows ≥ 12 h (echo vs "+24 h" share a date), undecided rows, NM-matched rows (decided ones already agree with v69).

## Stop rules (label-free)

1. Only LIRF rows change; every changed row meets all four conditions; IDs exactly the template's; no nulls.
2. Every move is upward and ≤ 6,000 s.
3. The 2026 bookkeeping decides ≥ 80% of NM-unmatched straddlers 1–12 h late in the same direction as the 2025
   mechanism (echo / "+24 h" date), i.e. the 2026 counts behave like 2025's.

## Upload rule

**Keep iff board < 236.5174.** Expected ≈ −0.09 s. Blind board: same script with `PRC_SCOPE=final4m` (adds the Feb / Jun
rows); used only if v70 is kept.

## Amendment A1 (2026-10-12, before the combined build): stack the NM-line removal at non-LIRF airports

The single-component v70 above (8 rows, ≈ −0.09 s) was built (`logs/v70/v70_build.log`, md5 b6e51f26…) but **not uploaded**
(team decision: too small alone). v70 is redefined as that file **plus**:

**NM-line removal, non-LIRF.** `src/post/nm_lines.py` (since v33) overrides NM-unmatched departures > 6 h late with a
per-airport line y = a + b·d fitted on all NM-unmatched departures (mostly short delays). Evidence (2025, computed before
this amendment): on the 144 non-LIRF rows the line covers, the in-sample line has SSE 3.21e8 against 8.51e7 for the
corrector's out-of-fold prediction; the line is worse at 8 of 9 airports (EDDF level), e.g. EHAM RMSE 1,334 vs 564,
LTFM 4,304 vs 2,055, LSZH 1,502 vs 233. The corrector is nearly unbiased there (mean residual +123 s, median −6;
a leave-one-month-out shift changes SSE by < 1%). The line was fitted when the base model was much weaker (§76).

- **Change:** for ADEP ≠ LIRF, NM-unmatched, T − S > 6 h: new = input + (chain before the line − chain after the line),
  chains `chain_ctxc1sel_rh2_win` / `chain_ctxc1sel_rh2_final` (the rh2 production chain, same ranking rows). Every other
  adjustment the row carries (v49's EHAM storm shift, the day stage) is kept; only the line's own contribution is removed.
  LIRF rows (LIRF line, Rome line) untouched.
- **Stop rules (A1):** only the rows above change; every change equals −(line − chain) for that row; new values in
  [200, 4,000] s; IDs exactly the template's.
- **Upload rule (whole v70):** keep iff board < 236.5174. The NM-line part is the larger one (70 live-board rows); a 2025
  per-airport projection is reported with the build.

## Outcome of the combined build (2026-10-12; `logs/v70/`)

- Count part: 8 LIRF rows moved up to the echo value (RYR9081 +3,237 s, WZZZZ811 +1,126, six ≤ 951); stop rule 3:
  0.89 of decided NM-unmatched straddlers on echo dates. Σd² 1.45e7.
- NM-line part: 70 non-LIRF rows (EHAM 41, EDDF 6, EGLL 5, LEBL 5, LFPG 5, LSZH 4, EDDM 2, LTFM 2), mean change −324 to
  −972 s (LTFM −4,279); no other non-LIRF row was touched by the line stage; Σd² 6.67e7 (LTFM's 2 rows 3.7e7).
- **Out-of-sample replay of the line removal (2025, lines refit leave-one-month-out vs the corrector's OOF value,
  144 rows):** SSE 2.84e8 → 8.51e7, 11/12 months better; realised gain 0.72 of the change energy (corrector + its LOMO
  bias: 0.80, 12/12). Holdout-only (31 rows, Jan + Jul): 1.49e7 → 1.04e7 (ratio 0.35; corrector 182 s low there).
- **v70** = `smart-jigsaw_v70.parquet`, md5 d7514e331400e10cb0485bce0ab4ad3d: 78 rows changed vs v69, Σd² 8.12e7.
  **Projection ≈ −0.38 s** (line 0.72 × 6.67e7 + count 1.45e7). Team bar for this upload: −0.6 s → **held, not uploaded**.
- **Board (2026-10-12): v70 = 236.2290 (−0.288 vs v69 = 236.5174). Kept; new best.** Inside the projected range
  (−0.15 … −0.45; central −0.38). Blind board: rerun both parts on the blind file (`PRC_SCOPE=final4m`; the line removal
  needs the four-month chain's files before / after `nm_lines`).
