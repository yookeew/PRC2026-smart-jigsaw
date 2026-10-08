# Context corrector averaged over LightGBM seeds — pre-registration

Written 2026-10-07, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **The corrector varies a lot between refits.** Separate processes on the same rows give 2026 corrector outputs 16–19 s RMS apart (`reports/celn_scale_preregistration.md` A1; v42's chain vs this machine's rebuild: 16.45 s).
- That spread is variance from the fitting, not signal.
- Averaging K fits that differ only in their random draws keeps the signal and divides that variance by K.
- **Rough expected size** (if the between-fit spread is all error variance): about 13² · (1 − 1/K) s² of MSE, i.e. 0.1–0.25 s of board RMSE for K = 3. These are the only parts of the chain this targets.
- **Public precedents:** kind-mango's 5-member LIRF model ("price was 74 MSE of variance reduction; the live score confirmed it"); outgoing-boat's seed averaging of their base (0.2–0.3 s, not adopted).

## Change

- `PRC_CTX_SEEDS=42,43,44` in `src/post/ctx_armc.py`: every corrector fit (global + per-airport, ensemble2) runs once per LightGBM seed, and the corrections are averaged. Default unchanged.
- **Source of diversity:** with feature_fraction 0.9 the seed changes the feature subsampling per tree. Inside one process, fits with the same seed are exactly reproducible (§107).
- **Recipe:** Arm C c1, both members, ctxsel, v42's stage 2. LIRF NM-unmatched rows from v41's chain (team rule).

## Gate (holdout, once)

- **Adopt iff,** vs v42's chain (`cache/eval/v42_chain_holdout.parquet`), holdout body RMSE gain > 0 pooled, in January and in July. Same rule as amendment A1 of `reports/base_comb_preregistration.md`.
- **Also reported:** trimmed, full, per airport; the RMS between seed fits.
- **If adopted:** production correctors with the same seeds, v42's chain, LIRF NM-unmatched rows from v41. Upload is the team's call; keep iff the board score is below 249.31.

## Amendments

(none)

## Results

**Gate** (2026-10-07; `logs/ctx_seedavg.log`, `tests/ctx_gate_v42.py` with `PRC_GATE_SEL=ctxc1sel_s3`):
- RMS between seed fits per corrector fit: 11.7–17.0 s.
- vs v42's chain (holdout), LIRF NM-unmatched rows from v41:

| slice | v42 chain | seed-averaged | delta |
|---|---|---|---|
| body pooled | 193.95 | 193.76 | −0.19 |
| body Jan | 175.34 | 174.83 | −0.51 |
| body Jul | 207.90 | 207.93 | **+0.03** |
| trimmed | 245.20 | 244.87 | −0.33 |
| full | 304.07 | 303.75 | −0.33 |

- **Per airport (body):** better at 8 of 10 (LTFM −1.04, LFPG −0.62); worse at LIRF +0.94 and EGLL +0.14.
- **GATE: REJECT** (July +0.03). Not built.
- The pooled, trimmed and full gains are of the expected variance-reduction size. July misses by 0.03 s; the rule is applied as written.

## Post-hoc exception (2026-10-07, decided by the team after the REJECT)

The gate rejected on July by +0.03 s, so this build is **outside the pre-registered rule**. It is recorded here and in PROGRESS §109 as a deliberate team decision, not a gate pass.
- **Rationale:** averaging seeds lowers variance with no tuned parameter. The July miss is far smaller than the refit noise.
- **Precondition:** the EGLL ADS-B learning curve, the other open option, came out flat (PROGRESS §109).
- **Build:**
  - `tests/seedavg_one_fit.py` fits seeds 42/43/44 in ONE process per member (r, rb). It writes the seed-42 correction as the control and the 3-seed mean as the treatment.
  - The members are combined with ctxsel into `ctxc1sel_1fits42` and `ctxc1sel_1fits3`.
  - v42's chain is run on both (`logs/seedavg_chain.log`).
  - v44 = v42 + (chain_s3 − chain_s42), with LIRF NM-unmatched rows byte-identical to v42 (`src/post/celn_build.py ctxc1sel_1fits42 ctxc1sel_1fits3 smart-jigsaw_v44`).
- **Board rule:** keep iff the board score is below 249.31.

**Board result (2026-10-07):** v44 scored 249.5 vs v42's 249.31 (+0.19). **Not kept**; v42 remains the best submission.
