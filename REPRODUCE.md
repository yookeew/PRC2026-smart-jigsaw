# Reproducing our submissions

This rebuilds our final files from the raw challenge data:

- **Jan + Jul 2026:** `data/submissions/smart-jigsaw_v75.parquet`
- **Jan + Feb + Jun + Jul 2026 (final ranking):** Jan/Jul rows from v75, Feb/Jun rows from the same
  pipeline run on `final_ranking.parquet` (step 9).

The design is in [DESIGN.md](DESIGN.md).

## Before you start

- Run everything from the repo root. In the commands below:

  ```bash
  export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
  PY=.venv/Scripts/python.exe          # .venv/bin/python on Linux / macOS
  S=data/submissions
  ```

- Some scripts take bare file names and resolve them in `data/submissions/`. Others take paths. The
  commands below use each script's own convention.
- `cache/`, `models/` and `data/submissions/` are local working folders, rebuilt by these steps.
- **The build is incremental.** Most post-stages take the previous stage's file as input. Run the
  steps in the order given.
- **Determinism.** LightGBM runs are deterministic. CatBoost on the GPU is not (two identical runs
  differed by ~27 s RMS), so a fresh rebuild lands close to our files, not byte-identical.

Approximate cost: ADS-B download, many hours (network-bound); LightGBM base fits ~6–10 h locally
(16 cores, 32 GB RAM); CatBoost ~4 h on a Colab T4; each context corrector ~1–2 h; each production
chain ~30–40 min.

---

## 1. Ingest

### 1.1 Environment and challenge data

```bash
uv sync
$PY -m src.ingest.fetch_challenge_data
```

`fetch_challenge_data` needs the challenge S3 credentials (see [README.md](README.md)). It writes the
monthly training files to `data/raw/`, and `ranking.parquet`, `submitting.parquet`,
`final_ranking.parquet` and `final_submitting.parquet` to `data/ranking/`. Keep all four ranking
files: some live-board stages read `final_ranking.parquet` too.

### 1.2 External data

Licences are in [DATA_SOURCES.md](DATA_SOURCES.md).

```bash
$PY src/ingest/fetch_stands.py          # data/external/stands.csv, runways.csv (already committed)
$PY src/ingest/fetch_metar_data.py      # external-data/metar/
$PY src/ingest/fetch_atfm_data.py       # external-data/all_pre_departure_delays_{2025,2026}.csv (+ unused ATFM tables)
$PY src/ingest/fetch_opdi_events.py --months 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 \
    2025-07 2025-08 2025-09 2025-10 2025-11 2025-12 2026-01 2026-02 2026-06 2026-07   # cache/opdi_events/
```

### 1.3 ADS-B tracks (adsb.lol)

These are best fetched on Colab or another machine with fast bandwidth. Set `GITHUB_TOKEN` to avoid
GitHub's rate limit. Finished days are skipped, so an interrupted run can simply be restarted.

```bash
# Jan + Jul 2025 / 2026, and two extra days
$PY src/ingest/fetch_adsb.py --out external-data/adsb-fetch/ --days \
    2025-01-01:2025-01-31 2025-07-01:2025-07-31 2026-01-01:2026-01-31 2026-07-01:2026-07-31
$PY src/ingest/fetch_adsb.py --out external-data/adsb-restofyear/ --days 2025-09-15 2025-11-15
# rest-of-2025 sample: for 2025-02..06 and 2025-08..12, days 01 02 03 04 07 10 13 16 19 22 25 28
$PY src/ingest/fetch_adsb.py --out external-data/adsb-restofyear/ --days 2025-02-01 2025-02-02 ...
# Feb + Jun 2026 (four-month file only)
$PY src/ingest/fetch_adsb.py --out external-data/adsb-final2026/ --days 2026-02-01:2026-02-28 2026-06-01:2026-06-30
```

- The Colab cell "ADS-B fetch, the other 2025 months" in
  [notebooks/colab_train.py](notebooks/colab_train.py) runs the rest-of-2025 sample in parallel.
- Keep each folder's `manifest.csv`, because the combiner's thin-day rule reads the
  rest-of-2025 one.
- If a day comes back truncated, re-fetch it with `--force`.

Then normalise:

```bash
$PY src/ingest/normalise_adsb.py          # -> data/external/adsb/day=*/
```

---

## 2. ADS-B pushback detection

Run these in this order (each pass reads the previous outputs):

```bash
$PY src/link/adsb_pushback.py             # cache/adsb_pushback/
$PY src/ingest/stand_infer.py             # data/external/stands_inferred.csv (committed)
$PY src/link/adsb_pushback.py --v3        # cache/adsb_pushback_v3/
$PY src/link/adsb_pushback.py --v4        # cache/adsb_pushback_v4/
$PY src/link/adsb_pushback.py --v6        # cache/adsb_pushback_v6/   (read by v49_bundle.py)
$PY src/link/adsb_pos.py                  # cache/adsb_pos_v4/        (first-sighting positions)
```

---

## 3. Feature building

```bash
$PY src/features/export_model_inputs.py   # cache/features/{train2025,holdout_gap2025,ranking,labels2025}.parquet
$PY src/features/neighbour_feats.py       # cache/features_nb/ (the same files + 13 label-free neighbour features)
$PY src/features/opdi_feats.py            # cache/opdi_feats/opdi_feats.parquet
$PY src/features/reocc_feats.py           # cache/v50/reocc_feats3.parquet (stand re-occupation)
$PY -c "import polars as pl; c=['MVT_ID_mvt','ro_gap_last','ro_tm_last','ro_om_last','ro_two_dep','ro_tm_next','ro_gap_prev','ro_tm_prev','ro_n_since_s','ro_first_since_s']; pl.read_parquet('cache/v50/reocc_feats3.parquet').select(c).write_parquet('cache/v50/reocc_feats.parquet')"
$PY src/features/id_feats.py              # cache/v50/reocc_feats6.parquet (identity of the first arrival at the stand)
```

`export_model_inputs.py` is split-blind: departure off-block and taxi times are removed before any
feature is computed, in training and ranking data alike. Labels go to a separate file.

---

## 4. Base models

### 4.1 LightGBM (local)

The main base model `lgb_mixed_nb` predicts pushback delay at LIRF and taxi time elsewhere. It is
trained on `cache/features_nb/`.

```bash
$PY src/models/run_neighbour_base.py holdout   # fit on 10 months, predict Jan+Jul 2025 -> cache/eval/lgb_mixed_nb_holdout_ev.parquet
$PY src/models/run_neighbour_base.py oof       # one fold per training month   -> cache/oof/lgb_mixed_nb/fold=*.parquet
$PY src/models/run_neighbour_base.py prod      # refit on all 2025, predict 2026 -> $S/lgb_mixed_nb.parquet, models/lgb_mixed_nb/
```

The LIRF stages also need 2026 echo probabilities from a second LightGBM, `lgb_direct_nolirf_nb`:

```bash
$PY - <<'EOF'
import sys; sys.path.insert(0, "src")
from pathlib import Path
from models.fit import run
F, EV, RK = Path("cache/features_nb"), Path("cache/eval/lgb_direct_nolirf_nb_holdout_ev.parquet"), Path("cache/eval/lgb_direct_nolirf_nb_rank.parquet")
run(engine="lgb", target="direct_nolirf", seed=42, feat_dir=F, name="lgb_direct_nolirf_nb", submit=False, ev_out=EV)
run(engine="lgb", target="direct_nolirf", seed=42, feat_dir=F, name="lgb_direct_nolirf_nb", submit=True, ev_out=EV,
    rank_out=RK, resume=True)
EOF
```

### 4.2 CatBoost (Google Colab, T4 GPU)

1. Copy to Google Drive, at the same relative paths:
   - the `src/` tree;
   - `cache/features/{train2025,holdout_gap2025,ranking,labels2025}.parquet`;
   - `data/ranking/submitting.parquet`.
2. In a T4 runtime:

   ```python
   !pip -q install polars==1.44.1 lightgbm==4.7.0 catboost
   from google.colab import drive; drive.mount('/content/drive')
   import os, sys; os.chdir('/content/drive/MyDrive/smart-jigsaw'); sys.path.insert(0, 'src')
   from models.fit import run, run_oof

   # out-of-fold predictions, 10 folds (~3 h; finished folds are skipped on a rerun)
   run_oof(engine="cat", target="mixed", eta=0.05)                       # cache/oof/cat_mixed/fold=*.parquet
   # holdout + refit used by the context correctors
   run(engine="cat", target="mixed", seed=42, eta=0.05, name="cat_mixed", submit=True,
       ev_out="cache/eval/cat_mixed_holdout_ev.parquet")                 # + data/submissions/cat_mixed.parquet
   # holdout + refit with per-row ranking outputs, used by the OOF corrector and the LIRF stages
   run(engine="cat", target="mixed", eta=0.05, name="cat_mixed_rerun", submit=True,
       ev_out="cache/eval/cat_mixed_rerun_holdout_ev.parquet",
       rank_out="cache/eval/cat_mixed_rank.parquet")
   ```

3. Copy back to the same local paths: `cache/oof/cat_mixed/`, `cache/eval/cat_mixed_*`,
   `data/submissions/cat_mixed.parquet`, `data/submissions/cat_mixed_rerun.parquet`.

### 4.3 OOF corrector on CatBoost (local)

```bash
$PY tests/oof_corrector_v2_test.py   # holdout gate; writes the round count to cache/oof/corrector_v2_meta.json
$PY src/post/corrector_v2.py         # cache/eval/catcorr_mixed_holdout_ev.parquet, $S/catcorr_mixed.parquet
```

---

## 5. Context correctors and seed averaging

A context corrector is a LightGBM on the residual of the base stack (LightGBM + CatBoost). It is
trained on the out-of-fold months. Each corrector writes three things:

- `cache/eval/<name>_holdout_ev.parquet`
- `cache/oof/<name>/fold=*.parquet`
- `$S/<name>.parquet`

The production chain (step 6) reads all three.

The final submission carries six correctors. Each one is still needed, because later stages keep
some of the rows built from an earlier one.

**5.1 `ctxsel`** (blocks 1–9; its chain gives v41):

```bash
for B in "" "CTX_ADSB=1"; do
  env CTX_ENS=2 CTX_RWY=1 $B $PY -c "import sys; sys.path.insert(0,'tests'); import oof_context_corrector_stage2 as S; S.build_inputs()"
  env CTX_ENS=2 CTX_RWY=1 $B $PY src/post/ctx_corrector.py
done                                           # -> ctxcorr2r and ctxcorr2rb
PRC_LGB=ctxcorr2r $PY tests/ctx_adsb_stable_test.py   # -> ctxsel (rb at EHAM/LSZH/EDDF/LEBL, r elsewhere)
cp cache/eval/ctxcorr_chain_holdout.parquet cache/eval/v41_chain_holdout.parquet
```

`ctx_adsb_stable_test.py` also replays `ctxsel`'s chain on the 2025 holdout
(`ctxcorr_chain_holdout.parquet`). Later holdout replays read that file as v41's chain.

**5.2 `ctx_armc.py` correctors.** Each corrector has two members: `r`, and `rb`, which adds ADS-B
inputs. For each member, `inputs` fits the holdout and OOF predictions and `prod` fits the
production model. `sel` then combines the two members per airport.

```bash
armc () {   # armc "<flags>" <tag>
  for B in "" "CTX_ADSB=1"; do
    env ARMC_ARM=c1 $1 $B PRC_ARMC_TAG=$2 $PY src/post/ctx_armc.py inputs
    env ARMC_ARM=c1 $1 $B PRC_ARMC_TAG=$2 $PY src/post/ctx_armc.py prod
  done
  env ARMC_ARM=c1 PRC_ARMC_TAG=$2 $PY src/post/ctx_armc.py sel
}
armc "CTX_ENS=2 CTX_RWY=1"                                          ""     # -> ctxc1sel
armc "CTX_ENS=2 CTX_RWY=1 CTX_RO=1"                                 _ro    # -> ctxc1sel_ro
armc "CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1"          _rh2   # -> ctxc1sel_rh2
```

**5.3 Final corrector `ctxc1sel_rf3`, averaged over three seeds.** It has blocks 1–13. The seed-42
fits are averaged 1:2 with the in-process mean of seeds 43 and 44:

```bash
F="ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_FLOW=1 PRC_ARMC_TAG=_rf"
for SEEDS in "" "PRC_CTX_SEEDS=43,44"; do          # seed 42 -> ctxc1{r,rb}_rf;  seeds 43,44 -> ctxc1{r,rb}_rf_s2
  for B in "" "CTX_ADSB=1"; do
    env $F $SEEDS $B $PY src/post/ctx_armc.py inputs
    env $F $SEEDS $B $PY src/post/ctx_armc.py prod
  done
done
$PY src/post/ctx_seedavg_merge.py ctxc1r_rf  1 ctxc1r_rf_s2  2 ctxc1r_rf3
$PY src/post/ctx_seedavg_merge.py ctxc1rb_rf 1 ctxc1rb_rf_s2 2 ctxc1rb_rf3
ARMC_ARM=c1 PRC_ARMC_TAG=_rf3 $PY src/post/ctx_armc.py sel          # -> ctxc1sel_rf3
```

---

## 6. Production chain

For a corrector `N`, the production chain runs:

1. the stack of `N` and the corrected CatBoost, with the ADS-B pushback stages;
2. the learned ADS-B combiner;
3. the within-day neighbour corrector;
4. the LOBT window;
5. the NM-unmatched long-delay lines.

`ids` fits the identity-arm deltas on that chain. The delta scripts need them.

```bash
chain () {   # chain N
  PRC_LGB=$1 $PY src/post/stack_submit.py --cat-corrected --adsb-v3     # $S/stack_lgb_catcorr_v3_$1_adsbq_partial_fb.parquet
  PRC_LGB=$1 $PY src/post/adsb_combiner.py --full                       # $S/smart-jigsaw_v27$1.parquet
  PRC_LGB=$1 $PY src/post/neighbour_state.py --v2                       # $S/chain_$1.parquet
  $PY src/post/lobt_window.py $S/chain_$1.parquet $S/chain_$1_win.parquet
  $PY src/post/nm_lines_nocheck.py $S/chain_$1_win.parquet $S/chain_$1_final.parquet
}
ids () {     # ids N DIR  -> DIR/comb_delta_all_id.parquet, DIR/nb_delta_ro_id.parquet
  mkdir -p $2 && cp cache/v50/reocc_feats6.parquet $2/
  PRC_LGB=$1 SCR=$2 ARM=all_id    RO_FILE=reocc_feats6.parquet $PY src/post/v50_comb_delta.py
  PRC_LGB=$1 SCR=$2 NB_ARM=ro_id  RO_FILE=reocc_feats6.parquet $PY src/post/v50_nb_delta.py
}

chain ctxsel
chain ctxc1sel;      ids ctxc1sel cache/v53
cp $S/chain_ctxc1sel_final.parquet $S/chain_ctxc1selcp_final.parquet     # reference name read by v54_delta.py
chain ctxc1sel_ro;   ids ctxc1sel_ro cache/v64/id_ctxc1sel_ro
chain ctxc1sel_rh2                                                       # its chain files are read by step 8 (v70)
chain ctxc1sel_rf3;  ids ctxc1sel_rf3 cache/v64/id_ctxc1sel_rf3
```

---

## 7. Holdout replays used for calibration

The day-level stages calibrate on the production recipe replayed over the Jan + Jul 2025 holdout,
plus the out-of-fold months:

```bash
PRC_LGB=ctxcorr2r CORR=ctxc1sel SCR=cache/v50 ARMS=all $PY tests/v50_chain_gate.py   # cache/v50/chain_all.parquet
```

`pru_daily_recon.py` (v55, below) also needs two holdout files: a non-LIRF chain with column `pred`
and a LIRF chain with column `v48`. With `PRU_LIRF=0`, as used here, they only feed the LIRF
calibration, which is then not applied. They do not change the output.

```bash
mkdir -p cache/v54
cp cache/v50/chain_all.parquet cache/v54/ctxc1sel_ro_chain_all_holdout.parquet
$PY -c "import polars as pl; pl.read_parquet('cache/v50/chain_all.parquet').select('MVT_ID_mvt', v48='pred').write_parquet('cache/v54/v48m_chain_holdout.parquet')"
```

---

## 8. Post-stages, in order

```bash
# v41 = ctxsel's chain; v42 = ctxc1sel's chain with the LIRF NM-unmatched rows kept from v41
cp $S/chain_ctxsel_final.parquet $S/smart-jigsaw_v41.parquet
$PY src/post/armc_build.py smart-jigsaw_v41.parquet chain_ctxc1sel_final.parquet smart-jigsaw_v42.parquet

# LIRF NM-unmatched hedges
$PY src/post/lirf_signals.py     $S/smart-jigsaw_v42.parquet $S/smart-jigsaw_v45.parquet
$PY src/post/lirf_twin_bounds.py smart-jigsaw_v45.parquet smart-jigsaw_v47.parquet
$PY src/post/lirf_hedge2.py      smart-jigsaw_v47.parquet smart-jigsaw_v48.parquet

# label-free fixes: LSZH level, EHAM storm days, dominated LIRF rows
$PY src/post/v49_bundle.py smart-jigsaw_v48.parquet smart-jigsaw_v49.parquet

# v50: stand re-occupation + ADS-B first-sighting position (combiner and neighbour-stage deltas)
PRC_LGB=ctxc1sel ARM=all     $PY src/post/v50_comb_delta.py      # cache/v50/comb_delta_all.parquet
PRC_LGB=ctxc1sel NB_ARM=ro_b $PY src/post/v50_nb_delta.py        # cache/v50/nb_delta_ro_b.parquet
FREEZE_LIRF=1 $PY src/post/v50_assemble.py $S/smart-jigsaw_v49.parquet $S/smart-jigsaw_v50.parquet \
    cache/v50/comb_delta_all.parquet cache/v50/nb_delta_ro_b.parquet

# v52: OPDI ground events, applied at LSZH only
PRC_LGB=ctxc1sel SCR=cache/v52 $PY src/post/v52_comb_delta.py    # cache/v52/comb_delta_opdi.parquet
PRC_LGB=ctxc1sel SCR=cache/v52 $PY src/post/v52_nb_delta.py      # cache/v52/nb_delta_opdi.parquet
$PY -c "
import polars as pl
for f in ('comb', 'nb'):
    d = pl.read_parquet(f'cache/v52/{f}_delta_opdi.parquet')
    d.with_columns(delta=pl.when(pl.col('ADEP_mvt') == 'LSZH').then(pl.col('delta')).otherwise(0.0)).write_parquet(f'cache/v52/{f}_delta_opdi_lszh.parquet')"
FREEZE_LIRF=1 $PY src/post/v50_assemble.py $S/smart-jigsaw_v50.parquet $S/smart-jigsaw_v52.parquet \
    cache/v52/comb_delta_opdi_lszh.parquet cache/v52/nb_delta_opdi_lszh.parquet

# v53: identity of the first arrival at the stand = v52 + (v49 with the identity deltas - v50)
FREEZE_LIRF=1 $PY src/post/v50_assemble.py $S/smart-jigsaw_v49.parquet cache/v53/id_only.parquet \
    cache/v53/comb_delta_all_id.parquet cache/v53/nb_delta_ro_id.parquet
$PY -c "
import polars as pl
r = lambda f, c: pl.read_parquet(f).select(pl.col('MVT_ID_mvt').cast(pl.Float64), pl.col('TAXITIME_SEC_mvt').cast(pl.Float64).alias(c))
d = r('cache/v53/id_only.parquet', 'a').join(r('data/submissions/smart-jigsaw_v50.parquet', 'b'), on='MVT_ID_mvt')
d.select('MVT_ID_mvt', delta=pl.col('a') - pl.col('b')).write_parquet('cache/v53/delta_v53_minus_v50.parquet')"
FREEZE_LIRF=1 $PY src/post/v50_assemble.py $S/smart-jigsaw_v52.parquet $S/smart-jigsaw_v53.parquet \
    cache/v53/delta_v53_minus_v50.parquet

# v54: stand-aware corrector (ctxc1sel_ro) in place of ctxc1sel
$PY src/post/v54_delta.py cache/v64/id_ctxc1sel_ro cache/v54/delta.parquet
FREEZE_LIRF=1 $PY src/post/v50_assemble.py $S/smart-jigsaw_v53.parquet $S/smart-jigsaw_v54.parquet cache/v54/delta.parquet

# v55: day-level reconciliation with EUROCONTROL's daily pre-departure delay (LIRF untouched, LSZH skipped)
PRU_LIRF=0 $PY src/post/pru_daily_recon.py $S/smart-jigsaw_v54.parquet $S/smart-jigsaw_v55.parquet \
    cache/v54/v48m_chain_holdout.parquet cache/v54/ctxc1sel_ro_chain_all_holdout.parquet

# v58 / v60 / v61 read a frame that also joins smart-jigsaw_v56.parquet (a rejected file whose values they never use);
# a copy of v55 satisfies the reader.
cp $S/smart-jigsaw_v55.parquet $S/smart-jigsaw_v56.parquet
$PY tests/v57_dsm.py                                    # cache/v57/dsm_params.json (2025 noise and drift parameters)
$PY src/post/v58_build.py $S/smart-jigsaw_v58.parquet   # LSZH day-to-day level, on v55
$PY src/post/v60_build.py $S/smart-jigsaw_v60.parquet   # EDDF OPDI delta demeaned per day, on v55
$PY src/post/v61_build.py $S/smart-jigsaw_v61.parquet   # learned day shift at 8 airports, LSZH from v58

# v62: v61 with every LTFM row taken from v60
$PY -c "
import polars as pl
S = 'data/submissions/'
v61 = pl.read_parquet(S + 'smart-jigsaw_v61.parquet')
v60 = pl.read_parquet(S + 'smart-jigsaw_v60.parquet').rename({'TAXITIME_SEC_mvt': 'v60'})
ltfm = pl.read_parquet('data/ranking/ranking.parquet').filter(pl.col('ADEP_mvt') == 'LTFM')['MVT_ID_mvt'].cast(pl.Float64)
(v61.join(v60, on='MVT_ID_mvt', how='left', maintain_order='left')
    .with_columns(pl.when(pl.col('MVT_ID_mvt').is_in(ltfm.implode())).then(pl.col('v60'))
                  .otherwise(pl.col('TAXITIME_SEC_mvt')).alias('TAXITIME_SEC_mvt'))
    .drop('v60').write_parquet(S + 'smart-jigsaw_v62.parquet'))"

# v68: final corrector (ctxc1sel_rf3) as a delta on v54, then the learned day allocation (G2) on v62's recipe
mkdir -p cache/v68
$PY src/post/v64_delta.py cache/v68/delta_rf3.parquet ctxc1sel_rf3 cache/v64
FREEZE_LIRF=1 $PY src/post/v50_assemble.py $S/smart-jigsaw_v54.parquet cache/v68/p.parquet cache/v68/delta_rf3.parquet
$PY src/post/day_alloc_build.py cache/v68/p.parquet $S/smart-jigsaw_v62.parquet $S/smart-jigsaw_v68.parquet

# v69: LSZH re-levelled per day from the daily sums
$PY src/post/lszh_dayclose.py $S/smart-jigsaw_v68.parquet $S/smart-jigsaw_v69.parquet 0.75

# v70: LIRF day counts, then the NM-unmatched line removed outside LIRF
$PY src/post/lirf_count_dates.py $S/smart-jigsaw_v69.parquet cache/v70_counts.parquet
$PY src/post/nm_lines_remove.py cache/v70_counts.parquet $S/smart-jigsaw_v70.parquet \
    $S/chain_ctxc1sel_rh2_win.parquet $S/chain_ctxc1sel_rh2_final.parquet

# v71 -> v75: LIRF NM-unmatched recording-error stages
$PY src/post/lirf_hedge_repair.py   $S/smart-jigsaw_v70.parquet $S/smart-jigsaw_v71.parquet
$PY src/post/lirf_airline_record.py $S/smart-jigsaw_v71.parquet $S/smart-jigsaw_v72.parquet
ALR_L2LOOSE=1 $PY src/post/lirf_airline_record.py $S/smart-jigsaw_v72.parquet $S/smart-jigsaw_v75.parquet
```

`smart-jigsaw_v75.parquet` is the final Jan/Jul file. Our copy has md5
`027b245e7a54c99dc753040fb8317f4c`. Every stage prints its own checks. Each output must have the
template's `MVT_ID_mvt` exactly, no nulls and integer seconds.

---

## 9. Four-month file (Jan + Feb + Jun + Jul 2026)

The same pipeline, with `PRC_SCOPE=final4m`. Holdout and OOF fits do not depend on the scope and
are reused. Outputs go to `final4m/` subfolders.

1. With `PRC_SCOPE=final4m`, run:
   - `export_model_inputs.py` and `neighbour_feats.py`;
   - the step 2 detection passes for the 2026 days;
   - `opdi_feats.py`, `reocc_feats.py` and `id_feats.py`;
   - `run_neighbour_base.py prod`.
2. Colab: the cell "CatBoost for the four-month pipeline" in
   [notebooks/colab_train.py](notebooks/colab_train.py) writes
   `data/submissions/final4m/cat_mixed_f4m.parquet` and `cache/eval/final4m/cat_mixed_f4m_rank.parquet`.
3. `bash run_final4m.sh`. This runs the corrector, `ctxc1sel_ro`, the chain and the step 8 stages up to
   v55's recipe, and writes `data/submissions/final4m/f4m_e.parquet`.
4. `PRC_SCOPE=final4m $PY src/post/final4m_v62.py` writes `final4m/f4m_v62.parquet`, the four-month
   counterpart of v62.
5. Final corrector and G2: follow `run_v64_final4m.sh` with `ctxc1sel_rf3` in place of
   `ctxc1sel_rohb`. Before that, run `ctx_armc.py prod` for both members and both seed settings, and
   `ctx_seedavg_merge.py`, with `PRC_SCOPE=final4m`.
6. With `PRC_SCOPE=final4m`, run the v69–v75 commands of step 8 on that file, in the same order. For
   `nm_lines_remove.py`, use the four-month chain's `_win` / `_final` files.
7. Compose the upload. This takes Jan/Jul rows from v75 and Feb/Jun rows from the four-month build,
   and checks the IDs against `final_submitting.parquet`:

   ```bash
   $PY src/post/final4m_compose.py $S/smart-jigsaw_v75.parquet <four-month build>.parquet <output>.parquet
   ```

---

## Notes

- Several production scripts import helper code from `tests/` on purpose (e.g.
  `context_corrector_test.py`, `adsb_combiner_test.py`, `v57_dsm.py`), so the build uses the same
  functions as the validated tests.
- `tests/flow_screen/`, most other `tests/*` scripts, and the root-level `run_*` scripts are the
  experiment history. Apart from the scripts named above, the build does not use them.
