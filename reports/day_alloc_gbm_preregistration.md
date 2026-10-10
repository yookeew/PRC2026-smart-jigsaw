# Learned day-shift allocation with a boosted slope model -- pre-registration

Written 2026-10-10 **before any number below was computed**. Base: v62 = 238.79 (§131): S3 at seven airports
(slopes per airport × delay band, clipped to [0, 1]), R0 (v55's uniform 0.75) at LTFM.
**Disclosed:** the 2025 data below were already used for §130's C1b and §131's S1–S4. This continues that line:
S3's slopes are one number per (airport, band); here the slope is a function of the row's inputs.

## Fixed parts (as §131 / tests/v61_learned_dayshift.py)

λ̂_g per airport-day from the cached 2025 solve (`cache/v57/lam_row_2025.parquet`), same periods, same leave-period-out
rule (fit on the other five periods, apply to the held-out one), same 8 airports, same masks (trimmed y ≤ 18,000 s for
fitting; body = NM-matched and y ≤ 3,600 s).

## Model

Shift δ = s(x) · λ̂_g, q = max(p − δ, 0). s(x) is a LightGBM regression with label (p − y) / λ̂ and weight λ̂², which is
exactly the loss Σ (p − y − s·λ̂)² (rows with |λ̂| < 1 s excluded from fitting). At apply time s is clipped to [0, 1] as
in S3. Hyper-parameters fixed now: 15 leaves, min 2,000 rows per leaf, learning rate 0.05, 300 rounds, L2 10,
init 0.5, deterministic, seed 0.

- **G1** inputs: airport (categorical), d̂ = (T − S) − p (continuous, replaces the bands), hour of takeoff, NM-unmatched
  flag, ADS-B tracked flag, OPDI flag, p.
- **G2** = G1 + |λ̂_g| and sign(λ̂_g) (regime-dependent allocation; S2's idea).

Each is evaluated at all 8 airports (**G·8**) and with LTFM kept on R0 (**G·7**, v62's scope), four candidates in all.

## Decision (2025, all 12 months, 8 airports' rows, vs v62's recipe = S3 at 7 + R0 at LTFM)

A candidate passes iff: (1) pooled full ≤ −0.10 s; (2) pooled body ≤ 0; (3) pooled trimmed ≤ 0; (4) full better in
≥ 8/12 months; (5) February and June each not worse on full (the blind board's extra months). **Pick the passing
candidate with the lowest pooled full.** None passes → nothing is built from this. Reported for information: vs R0,
per airport, per month, and the 2026 projection (per-airport ΔMSE per row × 2026 row counts, onto 238.79).

## Build if a candidate passes

v62 with the 8 (or 7) airports' rows re-solved by the same production path as v61/v62 (`src/post/v61_build.py`),
δ = s(x)·λ̂ with s fitted on all six 2025 periods; LSZH, LIRF and every other row as v62; LOBT window last.
**Keep iff board < 238.79.** Then ported to Feb/Jun through `src/post/final4m_v62.py`.

## Outcome (2026-10-10; `logs/day_alloc_gbm.log`)

G2x8: full −0.433, body −0.288, trimmed −0.461, 12/12 months, Feb −2.76, Jun −0.30 → **PASS, picked**. G2x7 passes too
(full −0.108); G1x8 and G1x7 fail. 2026 projection −0.33 s. Built as v63 (PROGRESS §133).
