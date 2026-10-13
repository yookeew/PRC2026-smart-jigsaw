# Reproducing this project

Sections 0–8 rebuild **v30** (leaderboard RMSE 261.40) from the raw
challenge data, step by step. Section 9 maps the stages from v30 to the
final submissions: **v76** on the live board (234.2695) and
`smart-jigsaw_final.parquet` on the blind board. It also gives the
four-month driver scripts that run the whole recipe. The design is in [DESIGN.md](DESIGN.md). The
experiment log, including every rejected idea, is in
[PROGRESS.md](PROGRESS.md). Every adopted change has a pre-registration in
`reports/*_preregistration.md`.

**The pipeline:**

1. External data: stands, runways, METAR, ADS-B.
2. ADS-B pushback detection (four detector passes).
3. Features, built locally and split-blind.
4. Two base models: LightGBM (local) and CatBoost (Colab GPU).
5. A second-stage corrector on CatBoost, trained on out-of-fold predictions.
6. An NNLS stack of the two models, then the ADS-B stages → v23.
7. A learned ADS-B combiner on rows that have ADS-B information, trained on
   the holdout plus a sample of ADS-B days from every other 2025 month → v27.
8. A neighbour-state corrector on rows without ADS-B information. It uses
   label-free signals from nearby flights, and its correction is
   zero-mean per airport and day → v30.

Everything under `cache/` is scratch space, gitignored, and rebuilt by the
steps below.

**Rough cost:**

| Step | Where | Time |
|---|---|---|
| ADS-B download (126 + 122 days) | Colab or any machine | several hours, network-bound |
| ADS-B detection (4 passes) | local | ~1 h |
| LightGBM | local (tested on 22 cores / 32 GB RAM) | ~2 h |
| CatBoost (3 runs) | Colab T4 | ~4 h |
| Everything else | local | < 1 h |

Unless a step says otherwise, run commands from the repo root with the uv
environment's Python: `.venv/Scripts/python.exe` on Windows,
`.venv/bin/python` elsewhere.

## 0. Environment and challenge data

1. `uv sync` installs the pinned Python 3.11 environment from
   `pyproject.toml` / `uv.lock`.
2. Configure the S3 credentials for the challenge bucket (see
   [README.md](README.md), "Get your own OpenSky/challenge credentials").
   They're only used to download the challenge files. No OpenSky flight or
   trajectory data is used anywhere in the pipeline.
3. `python -m src.ingest.fetch_challenge_data` downloads the monthly
   training files to `data/raw/`, and `ranking.parquet` and
   `submitting.parquet` to `data/ranking/`.

## 1. External data

Each source, with its licence, is listed in [DATA_SOURCES.md](DATA_SOURCES.md).

1. **Stands and runway thresholds** (X-Plane Scenery Gateway, GPLv2):
   `src/ingest/fetch_stands.py` writes `data/external/stands.csv` and
   `data/external/runways.csv`. Both are committed, so you can skip this
   step.
2. **METAR weather** (Iowa Environmental Mesonet, republishing NOAA/NWS
   public-domain data): `src/ingest/fetch_metar_data.py` writes to
   `external-data/metar/`. Used by feature family 7.
3. **ADS-B surface tracks** (adsb.lol, ODbL), on Colab or any machine with
   good bandwidth:

   ```
   python src/ingest/fetch_adsb.py --out external-data/adsb-fetch/ \
       --days 2025-01-01:2025-01-31 2025-07-01:2025-07-31 \
              2026-01-01:2026-01-31 2026-07-01:2026-07-31 \
              2025-09-15 2025-11-15
   ```

   - Setting `GITHUB_TOKEN` avoids the GitHub API's rate limit.
   - Finished days are skipped, so an interrupted run can be restarted.
   - The Colab cells we used are in
     [notebooks/colab_train.py](notebooks/colab_train.py).
   - The two single days (2025-09-15, 2025-11-15) were used to explore
     designs outside the holdout. The stand inference in step 2 also reads
     them.
4. **The rest-of-2025 sample** for the combiner (§64-65): for every
   non-holdout month (Feb-Jun, Aug-Dec), the 1st-4th, then every third day
   to the 28th; plus the 15th of Sep and Nov.
   - It goes into `external-data/adsb-restofyear/` (Colab cell "ADS-B
     fetch, the other 2025 months").
   - Keep `manifest.csv`: the pre-registered thin-day rule reads it and
     keeps 115 of the 122 days.
5. `src/ingest/normalise_adsb.py` writes the tracks to
   `data/external/adsb/day=*/`. It reads `external-data/adsb/`,
   `external-data/adsb-fetch/` and `external-data/adsb-restofyear/`.

## 2. ADS-B pushback detection (local, ~1 h)

`src/link/adsb_pushback.py` is split-blind. It reads only takeoff time,
stand, ADS-B points and runway geometry, never an off-block time. Run these
in order, because later passes depend on earlier outputs:

1. `src/link/adsb_pushback.py` → `cache/adsb_pushback/`, the original
   stand-appearance detector (§37). The holdout loaders read it.
2. `src/ingest/stand_infer.py` → `data/external/stands_inferred.csv`
   (committed). Positions for stands missing from the Gateway table, from
   stationary first ADS-B samples (§56; ODbL, see `data/external/LICENSE`).
3. `src/link/adsb_pushback.py --v3` → `cache/adsb_pushback_v3/`: inferred
   stands plus the fallback match for unmatched departures (§56).
4. `src/link/adsb_pushback.py --v4` → `cache/adsb_pushback_v4/`: v3 plus
   the fallback for matched rows with no at-stand pushback (§57).

Run pass 4 after the rest-of-2025 days are normalised so they get v4
detections too. The `--v5` pass (runway-ending matches, §60) was tested and
rejected, so nothing downstream needs it.

## 3. Features (local, a few minutes)

`src/features/export_model_inputs.py` writes `train2025.parquet`,
`holdout_gap2025.parquet`, `ranking.parquet` and `labels2025.parquet` to
`cache/features/`.

- The builder is split-blind. Departure off-block and taxi times are
  removed before any feature is computed, in training and ranking data
  alike.
- Labels are stored separately, keyed on `MVT_ID_mvt`.

## 4. Base models

Both base models use `target="mixed"`: the model predicts pushback delay at
LIRF and taxi time directly everywhere else. The holdout is Jan + Jul 2025,
trained on the other 10 months. After scoring, each model is refit on all of
2025 to predict the ranking set.

**LightGBM, local (~2 h):**

```python
import sys; sys.path.insert(0, "src")
from models.fit import run
run(engine="lgb", target="mixed", seed=42, name="lgb_mixed", submit=True,
    ev_out="cache/eval/lgb_mixed_holdout_ev.parquet")
```

This writes `cache/eval/lgb_mixed_holdout_ev.parquet` and
`data/submissions/lgb_mixed.parquet`.

**CatBoost, Colab T4.** Upload `src/` and `cache/features/`, then follow the
setup in [notebooks/colab_train.py](notebooks/colab_train.py). Run these
three cells and copy the outputs back to the same paths locally:

1. Out-of-fold predictions for the corrector: 10 folds, one per training
   month, ~3 h. Each fold is saved as it finishes, so a rerun resumes.

   ```python
   run_oof(engine="cat", target="mixed", eta=0.05)   # -> cache/oof/cat_mixed/fold=*.parquet
   ```

2. The CatBoost baseline that the corrector's gate is scored against:

   ```python
   run(engine="cat", target="mixed", seed=42, eta=0.05, name="cat_mixed", submit=True,
       ev_out="cache/eval/cat_mixed_holdout_ev.parquet")
   ```

3. The CatBoost run used from here on. It also saves the per-row ranking
   predictions the corrector needs:

   ```python
   run(engine="cat", target="mixed", eta=0.05, name="cat_mixed_rerun", submit=True,
       ev_out="cache/eval/cat_mixed_rerun_holdout_ev.parquet",
       rank_out="cache/eval/cat_mixed_rank.parquet")
   ```

CatBoost on the GPU isn't bit-for-bit deterministic. Runs 2 and 3 differed
by 27 s RMS for us, and our submissions use run 3 throughout
(`reports/oof_corrector_v2_preregistration.md`). A fresh reproduction will
land close to our scores but won't be byte-identical. Everything
downstream of the CatBoost runs is deterministic.

**The LightGBM out-of-fold folds are required from step 7 on.** Steps 7 and
8 train on the rest-of-2025 ADS-B days, using the out-of-fold LightGBM and
CatBoost predictions as their stack input:
- the command is `run_oof(engine="lgb", target="mixed", seed=42)`;
- it writes `cache/oof/lgb_mixed/`;
- ~30-50 min per fold locally.

## 5. OOF corrector on CatBoost (local, ~30 min)

1. `tests/oof_corrector_v2_test.py` runs the pre-registered holdout gate and
   writes the corrector's round count to `cache/oof/corrector_v2_meta.json`.
2. `src/post/corrector_v2.py` refits the corrector on all 12 months and
   applies it. It writes `cache/eval/catcorr_mixed_holdout_ev.parquet` and
   `data/submissions/catcorr_mixed.parquet`.

## 6. Stack and ADS-B stages → v23 (local, a few minutes)

```
python src/post/stack_submit.py --cat-corrected --adsb-v3
```

This runs, in order:

1. An NNLS stack of LightGBM and corrected CatBoost, with weights fit on the
   holdout (~0.46 / 0.56).
2. The ADS-B pushback blend (`src/post/adsb_blend.py`) with
   quality-modulated weights.
3. The partial-track estimate (`src/post/adsb_partial.py`).
4. The gated fallback blend (`src/post/adsb_fallback.py`).

It reads `cache/adsb_pushback_v3/` and writes
`data/submissions/stack_lgb_catcorr_v3_adsbq_partial_fb.parquet`. Copy it
to `data/submissions/smart-jigsaw_v23.parquet`, because step 7 reads that
name.

Without `--adsb-v3`, the same script rebuilds v21 byte for byte
(`stack_lgb_catcorr_adsbq_partial.parquet`).

## 7. Learned ADS-B combiner → v27 (local, ~40 min)

```
python src/post/adsb_combiner.py --full
```

- This is v25's restricted combiner, trained on two sets of rows:
  - the Jan+Jul 2025 holdout;
  - the 115 rest-of-2025 days that pass the pre-registered thin-day rule.
- The stack prediction for those days comes from the month-wise
  out-of-fold LightGBM and CatBoost (`cache/oof/{lgb,cat}_mixed/`).
- It writes `data/submissions/smart-jigsaw_v27.parquet` (262.13).
  Pre-registrations: `reports/adsb_combiner_12m_preregistration.md` and
  `reports/adsb_combiner_fullyear_preregistration.md`.
- `--m12` builds v26 (Sep-Dec days only, 262.70).

## 8. Neighbour-state corrector → v30 (local, ~40 min)

```
python src/post/neighbour_state.py --v2
```

Departures close together in time share errors that the base models
miss (§70). This step adds **28 label-free neighbour features** to every
departure, at ±15/30/60 min, by airport and by runway:
- neighbours' network-manager off-block error relative to the stack;
- neighbours' ADS-B pushback error, from flights with an observed pushback;
- neighbours' delay;
- arrivals' taxi-in excess.

How v30 uses them:
- A LightGBM corrector on `taxi − stack` is trained on the holdout plus the
  115 days. It's applied only to rows **without** ADS-B information.
- Its correction is **demeaned per (airport, UTC day)**. It can only
  redistribute error within a day, never move an airport's level.
- Rows with ADS-B information keep v27's value. The step reads
  `data/submissions/smart-jigsaw_v27.parquet`.
- Output: `data/submissions/smart-jigsaw_v30.parquet` (261.40).
  Pre-registrations: `reports/neighbour_state_preregistration.md` and
  `reports/neighbour_state_v2_preregistration.md`.
- Without `--v2` it builds the rejected v29.

### v25 (an earlier best)

```
python src/post/adsb_combiner.py --restricted
```

- One LightGBM is fit on `taxi − stack` over the Jan+Jul 2025 holdout. Its
  inputs are the stack prediction and every v4 detector field
  (`reports/adsb_combiner_preregistration.md`).
- It's applied to 2026 ranking rows that have ADS-B information (a matched
  track or a fallback match). Every other row, and all of LIRF, keeps v23's
  value (`reports/adsb_combiner_restricted_preregistration.md`).
- Output: `data/submissions/smart-jigsaw_v25.parquet` (264.39).
- Without `--restricted` it builds v24 (265.66), which applies the
  combiner to every non-LIRF row.

The build imports helper code from `tests/` (`ltfm_egll_anatomy.py`,
`adsb_v3_test.py`, `adsb_combiner_test.py`). That's deliberate: the
production build uses exactly the same functions as the validated gate.

Upload with `python -m src.post.upload <file>` (credentials in `.env`, see
`.env.example`).

## 9. From v30 to the final submissions (v76 live, `smart-jigsaw_final` blind)

Final files (`data/submissions/` is gitignored; checked by `tests/v76_checks.py`, log `logs/v76/v76_checks.log`):

| board | file | md5 | score |
|---|---|---|---|
| live (Jan + Jul 2026) | `smart-jigsaw_v76.parquet` | `9b818fb96911e546ebc591a81830bec8` | 234.2695 |
| blind (Jan + Feb + Jun + Jul 2026) | `final4m/smart-jigsaw_final.parquet` (= `blind_v8`) | `08adc51280d01f1521e27e72f06ce21c` | not shown |

The stages after v30, in order, with their PROGRESS.md sections. Each has a pre-registration in `reports/`.
- **Chain:** the LOBT window (`lobt_window.py`); the context corrector with NM-unmatched rows in training
  (`ctx_armc.py`, §88); NM long-delay lines (`nm_lines*.py`, §76); stand re-occupation and identity deltas
  (`v50_*`, §123–§124).
- **LIRF NM-unmatched hedges:** `lirf_signals.py`, `lirf_twin_bounds.py`, `lirf_hedge2.py` (§113–§117); v49's fixes
  (`v49_bundle.py`).
- **Day level:** EUROCONTROL daily-sum reconciliation (`pru_daily_recon.py`, §125); v62's recipe with OPDI deltas
  (`final4m_v62.py`, §128–§131); the learned allocation G2 (`day_alloc_build.py`, §133).
- **Corrector blocks 11–13, averaged over three seeds** (`ctxc1sel_rf3`, §135, §143, §152, §154).
- **Post-stages:** `lszh_dayclose.py` (§149); `lirf_count_dates.py` and `nm_lines_remove.py` (§150);
  `lirf_hedge_repair.py` (§156); `lirf_airline_record.py`, then with `ALR_L2LOOSE=1` (§158, §161).
- **Average of two independent builds** on Jan / Jul (`twin_average.py`, §163).

**The four-month build runs the whole recipe from scratch** on `final_ranking.parquet` (`PRC_SCOPE=final4m`, §132,
§162). It is the reproduction path:

1. Features, ADS-B detections, LightGBM and CatBoost for the scope: see the header of `run_final4m.sh` (§0–§4 with
   `PRC_SCOPE=final4m`; CatBoost from the Colab cell "CatBoost for the four-month pipeline").
2. `bash run_final4m.sh` → `final4m/f4m_e.parquet` (chain, identity deltas, LIRF hedges, v49's fixes) and `f4m_full.parquet`.
3. The OPDI deltas on that chain (`PRC_SCOPE=final4m PRC_LGB=ctxc1sel_ro`, `v52_comb_delta.py`, `v52_nb_delta.py`), then
   `PRC_SCOPE=final4m python src/post/final4m_v62.py data/submissions/final4m/f4m_e.parquet data/submissions/final4m/f4m_full.parquet data/submissions/final4m/f4m_v62.parquet`.
4. `bash run_v75_final4m.sh B` → `final4m/f4m_v75.parquet`. It needs the corrector's holdout and OOF inputs, made once by
   the script's `inputs` step.
5. `python src/post/twin_average.py data/submissions/smart-jigsaw_v75.parquet data/submissions/final4m/f4m_v75.parquet data/submissions/smart-jigsaw_v76.parquet`.
6. `PRC_SCOPE=final4m python src/post/final4m_compose.py data/submissions/smart-jigsaw_v76.parquet data/submissions/final4m/f4m_v75.parquet data/submissions/final4m/blind_v8.parquet`,
   then copy it to `final4m/smart-jigsaw_final.parquet`. Check it with `tests/blind_qa.py` and `tests/v76_checks.py`.

**The live v75** came from the teammate's machine. `run_v75_final4m.sh A` rebuilds its v68 from their corrector files
(63 of 344,841 rows differ, max 4 s), and the post-stages v68 → v75 replay byte for byte (§162).

A rebuild matches these files only up to refit noise. LightGBM is deterministic only on the same machine and thread
count, and GPU CatBoost differs by 25–33 s RMS from run to run. Two independent builds differ by ≈ 25 s RMS per row (§163).

## Leaderboard history

| version | change | RMSE |
|---|---|---|
| v12 | LightGBM, mixed target | 302 |
| v15 | + CatBoost stack | 296 |
| v16 | + ADS-B pushback blend | 277 |
| v17 | + partial-track estimate | 275 |
| v19 | + quality-modulated ADS-B weights | 274.25 |
| v21 | CatBoost replaced by the OOF-corrected CatBoost | 270.2 |
| v23 | + inferred stands and gated fallback match (v3 detections) | 269.50 |
| v24 | + learned ADS-B combiner (v4 detections) | 265.66 |
| v25 | combiner restricted to rows with ADS-B information | 264.39 |
| v26 | combiner retrained with Sep–Dec 2025 ADS-B | 262.70 |
| v27 | combiner retrained with all of 2025's ADS-B sample | 262.13 |
| v30 | + within-day neighbour-state corrector on rows without ADS-B | 261.40 |
| v31–v74 | see PROGRESS.md (stages listed in section 9) | |
| v75 | + LIRF airline echo record with loose shifted blocks | 234.5098 |
| **v76** | + average of two independent builds on Jan / Jul (final live submission) | **234.2695** |

Rejected submissions:
- v18, v20 and v22: PROGRESS.md §46 and §55;
- v28 (a LIRF-specific combiner, 262.36): §66;
- v29 (neighbour state without the within-day constraint, 262.33): §71.

## Earlier stages (not needed for v30)

These stages produced the reports that shaped the design:

- `src/ingest/audit_ranking.py` writes `reports/step0_audit.md` (leak and
  assumption checks on the ranking data).
- `src/link/validate_links.py` writes `reports/stage1_link_validation.md`
  (stand-linking bounds).
- `src/models/baseline.py` is the group-mean baseline (holdout RMSE
  ~400 s).
- The holdout harness (paired (airport, day) cluster bootstrap) is
  `tests/_harness.py`.
