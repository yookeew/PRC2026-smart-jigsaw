# Context-corrector hyperparameters (Arm C c1) — pre-registration

Written 2026-10-07, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **The corrector is where gains reach the board.** v35–v42 were all corrector changes. A base change keeps ~7% of its gain through the chain (§104).
- **Its LightGBM parameters were never tuned:**
  - global model: `num_leaves` 63, `learning_rate` 0.05, `min_data_in_leaf` 200, `feature_fraction` 0.9, `lambda_l2` 10;
  - per-airport models: the same with `min_data_in_leaf` 100.
- **Public precedent:** kind-mango's base tuning (v76) gave −0.88 s live.

## Selection (no holdout rows)

- **Inputs:** `ctx_armc` holdout-mode inputs for the `r` member (`ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1`), i.e. `oof_context_corrector_test.prepare()` + the Arm C c1 training rows and indicator.
- **Fit / validate:**
  - fit on the c1 trainable rows of the OOF months except the validation months;
  - validate on February + August 2025, the months closest to the board's January / July;
  - early stopping on the existing day-of-month inner split inside the fit rows.
- **Metric:** validation body RMSE (NM-matched, y ≤ 1 h) after the LOBT window, plus trimmed (y ≤ 5 h). Per month.
- **Configurations** (global; per-airport gets the same changes, its `min_data_in_leaf` kept at half the global one):

| id | change from current |
|---|---|
| c0 | current |
| c1 | num_leaves 127 |
| c2 | num_leaves 255, min_data_in_leaf 400 |
| c3 | num_leaves 31 |
| c4 | learning_rate 0.025 |
| c5 | feature_fraction 0.7 |
| c6 | min_data_in_leaf 50 |
| c7 | lambda_l2 1 |
| c8 | num_leaves 127, learning_rate 0.025, feature_fraction 0.7 |

- **Chosen configuration:** the lowest pooled validation body RMSE. Only taken forward if it beats c0 by ≥ 0.3 s pooled, with Feb > 0 and Aug > 0; otherwise stop.

## Gate (holdout, run once, for the chosen configuration only)

- Both Arm C members (`r`, `rb`) rebuilt with the chosen parameters, then ctxsel and v42's stage 2 (`tests/base_comb_v42_test.py` method).
- LIRF NM-unmatched rows from v41's chain (team rule).
- **Adopt iff,** vs v42's chain (`cache/eval/v42_chain_holdout.parquet`), holdout body RMSE gain > 0 pooled, in January and in July.
- **If adopted:** build v43 through v42's recipe; the upload is the team's call; keep iff the board score is below 249.31.

## Amendments

**A1 (2026-10-07, before any selection result).** Refitting the same corrector on the same rows on this machine is not
reproducible (`reports/celn_scale_preregistration.md` A1: early stopping at 747 vs 834 rounds). A second run of the
current configuration, `c0b`, is added as a refit-noise yardstick.
- **Selection rule, tightened:** the chosen configuration must beat c0 by ≥ 0.3 s pooled, with Feb > 0 and Aug > 0,
  **and** by more than 2 × |c0 − c0b| pooled.

## Results

**Selection** (2026-10-07; `tests/ctx_tune_select.py`, `logs/ctx_tune_select.log`): fit rows 1,405,419; validation Feb + Aug 2025, body n 330,720.
- The stack alone scores body 203.11.

| id | best_iter | body | Feb | Aug | trimmed |
|---|---|---|---|---|---|
| c0 | 1456 | 197.43 | 187.54 | 204.56 | 252.06 |
| c0b | 1456 | 197.43 | 187.54 | 204.56 | 252.06 |
| c1 127 leaves | 1009 | 197.40 | 187.28 | 204.70 | 251.74 |
| c2 255 leaves, min 400 | 667 | 197.21 | 186.95 | 204.59 | 251.81 |
| c3 31 leaves | 853 | 197.74 | 187.60 | 205.04 | 252.29 |
| c4 lr 0.025 | 2356 | 197.18 | 187.18 | 204.39 | 251.51 |
| c5 ff 0.7 | 1062 | 197.41 | 187.32 | 204.67 | 252.53 |
| c6 min 50 | 1150 | 197.57 | 187.76 | 204.65 | 251.90 |
| c7 l2 1 | 629 | 197.42 | 187.48 | 204.60 | 251.62 |
| c8 127 leaves, lr 0.025, ff 0.7 | 1297 | **197.14** | 187.01 | 204.44 | 251.63 |

- **Refit noise within one process: 0.000** (c0 = c0b exactly). The nondeterminism of A1 is between processes only.
- **Chosen c8: gain +0.29 (Feb +0.53, Aug +0.12), below the 0.3 bar. STOP.** No holdout gate run.
- **Reading:** the corrector is close to its optimum. Lower learning rate and more leaves help slightly, mostly in February; fewer leaves hurts.
