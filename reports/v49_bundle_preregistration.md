# v49: three label-free 2026 fixes on v48 — pre-registration

Written 2026-10-08, before the build script was run. Changes made after results
exist go under Amendments, dated.

## Basis

These are 2026-only misreadings that the 2025 holdout can't see by construction
(the v37 / v41 / v42 class). Each rests on label-free 2026 witnesses (PROGRESS
§119–§120); no board information is used.

The arrival-echo feature from §119 is **not** included: it made January 2025
worse in the leave-one-month-out test.

## Fix 1 — LSZH level (two independent witnesses)

**Witnesses:**
- **Anchor** a = T − AOBT_3: v42's 2026 offset vs a was +72 / +63 s against 2025 truth +33 / +35 (§98).
- **ADS-B:** v48 − ADS-B taxi in 2026 vs truth − ADS-B taxi in 2025: dwell +69 / +90, appear +11 / +7 (§120).

**Rule (anchor witness, the one covering all NM-matched rows):** per month m ∈ {Jan, Jul}:
- o26 = clipped mean (±300 s) of (P_v48 − a) over LSZH NM-matched 2026 departures with AOBT_3;
- o25 = the same for (y − a) over LSZH NM-matched departures in month m of 2025;
- **shift_m = o25 − o26**, added to every LSZH NM-matched row in month m.

The ADS-B dwell witness is larger, so this is the conservative choice.

## Fix 2 — EHAM storm-day NM-unmatched rows without ADS-B

- **Storm days:** EHAM days in 2026 with ≥ 40 NM-unmatched departures (label-free).
- **Tracked** = ADS-B tier dwell or pass with 60 ≤ T − pushback ≤ 7,200 s.
- **Lag:** median (y − ADS-B taxi) over tracked EHAM January 2025 departures.
- **Corrector bias:** δ_C = median (ADS-B taxi + lag − C) over tracked storm-day NM-unmatched 2026 rows, with C = the context corrector output before the ADS-B stages (`data/submissions/ctxc1sel.parquet`).
- **Treated rows:** storm-day NM-unmatched rows with no ADS-B tier at all.
- **Rule:** P' = P_v48 + δ, with δ = δ_C − median(P_v48 − C) over the treated rows. This leaves whatever the later stages already added.

## Fix 3 — dominated rows

For LIRF NM-unmatched departures with off = T − SCHED ≥ 87,000 s, both
possible labels (echo off + e_T, or +24 h ≈ 86,400 + taxi) are ≤ off + e_T.
Rule: P' = min(P, off + e_T).

## Decision

- `src/post/v49_bundle.py smart-jigsaw_v48.parquet smart-jigsaw_v49.parquet`.
- Report each fix's rows, shift and Σ(ΔP)².
- **Keep iff the board < 245.8701 (v48).** One upload.

## Amendments

(none)

## Results

**Build (2026-10-08, `logs/v49_build.log`):**
- **Fix 1, LSZH:** Jan o26 +72.4 vs o25 +33.5 → **−38.9 s** on 9,695 rows; Jul +63.1 vs +35.0 → **−28.1 s** on 12,796 rows.
- **Fix 2, EHAM storm days** (2, 3, 4, 5, 6, 9 and 29 Jan 2026):
  - lag 177 s;
  - corrector bias on 275 tracked storm NM-unmatched rows: **+490 s**;
  - 116 untracked rows, median(P − C) +20 → **δ +470 s**.
- **Fix 3, dominated rows:** NOSOS431 −4,116 s, QFAA7006 −1,619 s.
- **Total:** 22,609 rows changed, Σ(ΔP)² 6.99e7.
- **Keep iff the board < 245.8701.**

**Board (2026-10-08): v49 = 245.3839 (−0.49 vs v48's 245.8701). Kept; new best.** The label-free 2026 witnesses held.
