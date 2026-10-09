# v57 part B: daily reconciliation v2 -- development protocol, holdout gate, build rule

Written 2026-10-09 **before any number below was computed** (development or gate). Follows PROGRESS §125
(`src/post/pru_daily_recon.py`, v55 = v54 + day shift at 8 airports, a = 0.75).

## The v55 recipe (R0) and why it may be biased

Per airport-day g (rows assigned by predicted off-block date; days whose row count is within ±5% of the published
`FLT_DEP_IFR_2`), R0 solves Σ max(0, d̂ᵢ + λ) = 60·D_g (d̂ = (T − S) − p) and applies d̂ + 0.75λ. With residuals
dᵢ = d̂ᵢ + δ_g + ηᵢ, E[max(0, d̂ + δ + η)] > max(0, d̂ + δ) (Jensen), most for flights near on-time, so R0's λ is
biased towards more delay (lower taxi) by an amount that varies with each day's punctuality mix. The noise in λ
comes from Σ ηᵢ over delayed flights, dominated by the long-taxi tail; the fixed 0.75 may stand in for both effects.

## Variants (all non-LIRF; LIRF untouched; LSZH reported but the production skip stays)

- **R0** -- v55 recipe.
- **R1(a)** -- expectation solve: Σ G_k(d̂ᵢ + λ) = 60·D_g, G_k(x) = mean over the calibration sample of
  max(0, x + η), η = d-residual (p − y) minus its airport-day mean, airport k's sample (tracked / untracked rows
  separately where ADS-B tiers exist). Apply a·λ, a on a grid {0.5, 0.6, …, 1.0}.
- **R2** -- R1 with an empirical-Bayes shrink per day, a_g = τ²_k / (τ²_k + σ²_g): σ²_g = Σ Var_η[max(0, d̂ᵢ + λ + ηᵢ)]
  / (Σ P_η(d̂ᵢ + λ + ηᵢ > 0))² (the noise of λ̂ on that day), τ²_k = max(mean_g λ̂_g² − mean_g σ²_g, 25 s²) over the
  target period's days of airport k (label-free on the target period).
- **R3(w)** -- the best of R1/R2 with rows in an adsb.lol pushback tier (dwell / pass / fb_dwell) shifted by w·λ
  instead of λ, solved jointly; w ∈ {0, 0.25, 0.5, 0.75}. Only on predictions that carry ADS-B stages.
- **R4** -- the best of R1/R2 with a multiplicative allocation: shift ∝ pᵢ / mean_g(p).

## Development (no holdout labels)

- **D1:** the 10 OOF months (Feb–Jun, Aug–Dec 2025), every day, p = `cache/oof/ctxc1sel` (corrector OOF); η
  calibration leave-one-month-out.
- **D2:** the 115 OOF ADS-B days, p = control arm of `tests/v57_eddf_lomo.py` (v52's production post-stage
  recipe refit with the month held out); η calibration leave-one-month-out, by tracked / untracked.
- Metrics per variant: RMSE change vs the unshifted p (full, trimmed y ≤ 5 h, body = NM-matched y ≤ 1 h), per
  month; the oracle uniform day shift (mean residual per airport-day) as the ceiling.
- **Selection rule:** the variant V* with the lowest pooled body RMSE on D2, provided it also beats R0 on D1 body
  and on D2 body in ≥ 6 of 10 months. If no variant beats R0 that way, **STOP**: v57's reconciliation stays R0
  (re-solved on its base), and part B ships nothing new.

### Amendment B-A1 (2026-10-09, after D1, before D2 and before the holdout)

D1 (`logs/v57_recon2_dev_D1.log`): R2's empirical-Bayes shrink is far too weak (a ≈ 0.8–1.0; body +1.31 vs R0
+0.26), so the noise in λ̂ is not just η sampling; R2 is dropped and "eb" no longer qualifies R3/R4. The grid for
D2 is extended, fixed now: R1(a), R4(a) for a ∈ {0.5, …, 1.0}; R3(w) and R3+R4 (w_i = pᵢ/mean p × w for tracked
rows) at a ∈ {0.6, 0.75}, w ∈ {0, 0.25, 0.5, 0.75}. The selection rule is unchanged (lowest pooled D2 body among
these, beating R0 on D1 body and on D2 body in ≥ 6/10 months; for R3 variants the D1 condition uses the same
variant with w = 1, since D1 has no tracked rows).

### Amendment B-A2 (2026-10-09, after D2, before the holdout): V* selected

D2 (`logs/v57_recon2_dev_D2.log`, 115 OOF ADS-B days, chain-like predictions), pooled body vs unshifted: R0 −0.125;
best five: **R3+R4 w = 0.75, a = 0.6: −1.334 (beats R0 in 9/10 months)**; R4 a = 0.6: −1.330 (9/10); R3+R4
w = 0.5, a = 0.6: −1.328; R3+R4 w = 0.25, a = 0.6: −1.303; R4 a = 0.5: −1.297. D1 for the same variant with w = 1
(R4, a = 0.6): body −0.653 vs R0 +0.257, 10/10 months. **V* = `trkmult`, a = 0.6, w = 0.75**: the day's shift is
allocated ∝ pᵢ / mean p, tracked rows (adsb.lol dwell / pass / fb_dwell) take 0.75 of it, λ from the expectation
solve. (The margin over plain R4 is 0.004 s; the rule picks V* as written.)

## Gate (holdout Jan + Jul 2025, touched once)

Base: `cache/v50/chain_all.parquet` (v50's holdout chain, non-LIRF; the §125 base `h2/chain_all` does not exist).
η calibration cross-month (Jan from Jul's chain residuals and vice versa); τ² label-free on the target month.
V* vs R0, both on that base. **PASS iff:**
1. body RMSE (V* − R0) pooled ≤ −0.10 s;
2. body (V* − R0) ≤ 0 in January and in July;
3. full RMSE (V* − R0) ≤ 0;
4. airport-day cluster bootstrap (2,000 draws) P(V* worse than R0 on full) < 0.2.

Reported: trimmed; per airport; V* and R0 vs the unshifted chain; LSZH's implied shift under V* (information).

## Build if PASS

v57 = v54 with V* solved on v54's non-LIRF rows (LSZH skipped as in v55), every LIRF row = v56, LOBT window last;
η for 2026 from the holdout chain (both months). 2026 projection = V* − R0 on the holdout, scaled by 2026 row counts.
**Keep iff board < 240.7425 (v56).** v57 differs from v56 only on non-LIRF rows, so the read is clean.
If part B stops, there is no v57 from this plan.

## Outcome (2026-10-09, added after the gate run; PROGRESS §129)

`tests/v57_recon2_gate.py`, `logs/v57_recon2_gate.log`. Holdout non-LIRF rows 317,811 (base full 266.08, body 178.86).

| metric | R0 − base | V* − base | V* − R0 |
|---|---|---|---|
| full | −1.272 | −1.416 | −0.143 |
| trimmed | −1.796 | −1.939 | −0.143 |
| body | −1.902 | −1.826 | **+0.076** |
| body Jan / Jul | −2.578 / −1.420 | −2.262 / −1.516 | **+0.316** / −0.096 |
| full Jan / Jul | −1.168 / −1.475 | −1.111 / −1.880 | +0.057 / −0.406 |

Bootstrap V* − R0: full CI [−0.334, +0.006], P(worse) 0.035; body CI [−0.037, +0.188]. **GATE: 1 fails, 2 fails,
3 passes, 4 passes → FAIL.** No v57 from part B; v55's recipe (R0) stays. 2026 projection (information) −0.14 s.
LSZH under V*, if it were not skipped: +4.9 / +4.4 s on Jan / Jul 2025.
