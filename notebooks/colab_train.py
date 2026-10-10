# Colab model-training driver.
#
# Feature building stays local (Polars/CPU, ~30 s). Only the LightGBM/XGBoost
# fits move here — the local CPU fit is ~4 min each and the full pipeline
# (holdout + submission) is ~25 min, too slow to iterate on.
#
# ---------------------------------------------------------------------------
# SETUP (run once per Colab session)
# ---------------------------------------------------------------------------
# 1. Locally:  .venv/Scripts/python.exe src/features/export_model_inputs.py
#    -> writes cache/features/{train2025,ranking,labels2025}.parquet  (~90 MB)
# 2. Upload those 3 files + data/ranking/submitting.parquet + the src/ tree
#    to Google Drive (or `git clone` the repo and copy the parquets in).
# 3. In Colab:
#
#     !pip -q install polars==1.44.1 lightgbm==4.7.0 xgboost
#     from google.colab import drive; drive.mount('/content/drive')
#     import sys; sys.path.insert(0, '/content/drive/MyDrive/smart-jigsaw/src')
#     import os; os.chdir('/content/drive/MyDrive/smart-jigsaw')
#
# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
# %%
from models.fit import run

# LightGBM (CPU on Colab is still faster than the laptop; ~5 min total).
# Default loss is huber (alpha=800) -- beat plain L2 on the gap-realistic
# holdout (303.1s vs 309.2s), see reports/eval/lgb_colab_huber_a800.md. It
# hadn't fully plateaued at the local ROUNDS ceiling (8000); worth pushing
# rounds higher here where it's cheap to check whether it improves further.
model, ev = run(engine="lgb", name="lgb_colab")

# %%
# XGBoost on the Colab T4 GPU (~90 s per fit). Set Runtime > Change runtime
# type > T4 GPU first.
model_x, ev_x = run(engine="xgb", name="xgb_colab")

# %%
# ev / ev_x are Polars frames (MVT_ID_mvt, taxi, pred, ADEP_mvt, ym) for the
# Jan+Jul 2025 holdout — slice them for per-airport / per-decile analysis.
import polars as pl
print(ev.group_by("ADEP_mvt").agg(
    ((pl.col("pred") - pl.col("taxi")).pow(2).mean().sqrt()).alias("rmse"),
    pl.len(),
).sort("rmse", descending=True))

# %%
# OOF CatBoost predictions for the corrector (reports/oof_corrector_preregistration.md).
# T4 runtime, ~3 h for 10 folds x 3 fits. Each fold is saved to
# cache/oof/cat_mixed/fold=<month>.parquet as it finishes; if the session drops,
# rerun this cell and finished folds are skipped. Keep cache/ on Drive.
import os; os.environ["PRC_MEMLOG"] = "1"
from models.fit import run_oof
run_oof(engine="cat", target="mixed", eta=0.05)

# %%
# CatBoost rerun for v21 (reports/oof_corrector_v2_preregistration.md): the
# production CatBoost run, also saving the ranking rows' pred/echo_prob, which
# the corrector needs. T4 runtime, ~40 min (3 holdout fits + 3 refits).
# Named cat_mixed_rerun so data/submissions/cat_mixed.parquet (v19's input) is
# left alone. Copy back: cache/eval/cat_mixed_rerun_holdout_ev.parquet,
# cache/eval/cat_mixed_rank.parquet (and data/submissions/cat_mixed_rerun.parquet).
import os; os.environ["PRC_MEMLOG"] = "1"
from models.fit import run
run(engine="cat", target="mixed", eta=0.05, name="cat_mixed_rerun", submit=True,
    ev_out="cache/eval/cat_mixed_rerun_holdout_ev.parquet",
    rank_out="cache/eval/cat_mixed_rank.parquet")

# %%
# Stage-1 screen: CatBoost with categorical pairs (reports/cat_ctr2_preregistration.md).
# T4 runtime. Holdout fits only (submit=False, no refit): 2 regressors + echo classifier.
# The log prints cumulative seconds per fit ("holdout fit ...s") -- note them: if the
# mean fit time exceeds 25 min, stage 2 (30 OOF fits) is not feasible and we stop.
# Copy back: cache/eval/cat_ctr2_mixed_holdout_ev.parquet (+ the cell output as a log).
import os; os.environ["PRC_MEMLOG"] = "1"
from models.fit import run
run(engine="cat_ctr2", target="mixed", eta=0.05, seed=42, name="cat_ctr2_mixed", submit=False,
    ev_out="cache/eval/cat_ctr2_mixed_holdout_ev.parquet")

# %%
# ADS-B fetch, Jan + Jul 2025/2026 (src/ingest/fetch_adsb.py). DONE -- kept for reference.
# import subprocess, time, pathlib
# R   = '/content/drive/MyDrive/smart-jigsaw'
# OUT = f'{R}/adsb_fetch'          # fresh folder, don't reuse the old contaminated one
# jobs = {
#     'jan': '2025-01-01:2025-01-31 2026-01-01:2026-01-31',
#     'jul': '2025-07-01:2025-07-31 2026-07-01:2026-07-31',
# }
# procs = {k: subprocess.Popen(
#             f'python -u {R}/src/ingest/fetch_adsb.py --days {v} --out {OUT} > {R}/fetch_{k}.log 2>&1',
#             shell=True) for k, v in jobs.items()}
# while any(p.poll() is None for p in procs.values()):
#     time.sleep(300)
#     for k in procs:
#         lines = pathlib.Path(f'{R}/fetch_{k}.log').read_text().strip().splitlines()
#         print(time.strftime('%H:%M'), k, '|', lines[-1] if lines else '(starting)')
# print('all done'); print(open(f'{OUT}/manifest.csv').read())

# %%
# ADS-B fetch, the other 2025 months (PROGRESS.md §62): training data for the ADS-B
# combiner beyond the Jan/Jul holdout, from months whose receiver coverage looks like 2026.
# Days spread over each month: 01-03 (fetched first), then every third day 04..28, so
# the sample covers the month's weather/traffic/coverage swings. Sep-Dec first (closest to
# 2026), then Feb-Jun + Aug. Own OUT
# folder (adsb_restofyear), separate from the Jan/Jul pull; fetch_adsb.py skips days
# already in it, so after a disconnect just re-run this cell. Runs MAX_PAR jobs at a time
# (one month per job). Copy back: the whole adsb_restofyear folder ->
# external-data/adsb-restofyear/ (normalise_adsb.py reads it), then locally:
# normalise_adsb.py, then adsb_pushback.py --v4.
import subprocess, time, pathlib
from google.colab import drive
drive.mount('/content/drive')
R   = '/content/drive/MyDrive/smart-jigsaw'
OUT = f'{R}/adsb_restofyear'
MAX_PAR = 4
assert pathlib.Path(f'{R}/src/ingest/fetch_adsb.py').exists(), f'fetch_adsb.py not found under {R}'
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
months = ['2025-09', '2025-10', '2025-11', '2025-12',            # priority: closest to 2026
          '2025-08', '2025-06', '2025-05', '2025-04', '2025-03', '2025-02']
DAYS = [1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28]
EXTRA = {'2025-09': ['2025-09-15'], '2025-11': ['2025-11-15']}   # re-fetch: old extracts corrupted
queue = [(m, ' '.join([f'{m}-{d:02d}' for d in DAYS] + EXTRA.get(m, []))) for m in months]
running = {}
while queue or running:
    while queue and len(running) < MAX_PAR:
        m, days = queue.pop(0)
        running[m] = subprocess.Popen(
            f'python -u {R}/src/ingest/fetch_adsb.py --days {days} --out {OUT} > {OUT}/fetch_{m}.log 2>&1',
            shell=True)
    time.sleep(300)
    for m in list(running):
        log = pathlib.Path(f'{OUT}/fetch_{m}.log')
        lines = log.read_text().strip().splitlines() if log.exists() else ['(no log file yet)']
        print(time.strftime('%H:%M'), m, '|', lines[-1] if lines else '(starting)')
        if running[m].poll() is not None:
            print(time.strftime('%H:%M'), m, 'finished, exit', running[m].returncode)
            del running[m]
print('all done'); print(open(f'{OUT}/manifest.csv').read())

# %%
# SETUP FOR THE FOUR-MONTH PIPELINE - RUN THIS FIRST (fresh runtime, T4 GPU)
# Checks out the "febjune" branch into the Drive project folder (tracked files only: the
# git-ignored data under cache/, data/, models/ is left alone and must be uploaded separately).
# Does not import anything from src/, so the CatBoost cell below still sees a fresh scope.
'''from google.colab import drive
drive.mount('/content/drive')

PROJ = '/content/drive/MyDrive/smart-jigsaw'
%cd {PROJ}

import os, sys, subprocess
from google.colab import userdata
# private repo: the token comes from Colab secrets (key icon in the left bar, notebook access on) and is passed to
# this one fetch only, never written to .git/config on Drive and never printed
TOKEN = None
for k in ('GITHUB_TOKEN', 'GH_TOKEN', 'github_token', 'GITHUB_PAT'):
    try:
        TOKEN = userdata.get(k); print('using Colab secret', k); break
    except Exception:
        pass
assert TOKEN, 'no GitHub token in Colab secrets (tried GITHUB_TOKEN, GH_TOKEN, github_token, GITHUB_PAT)'
def git(*a):
    r = subprocess.run(['git', *a], capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(TOKEN, '***').strip()
    if out: print(out)
    assert r.returncode == 0, f'git {a[0]} failed'
if not os.path.exists(f'{PROJ}/.git'):
    git('init', '-q')
git('fetch', '-q', '--depth', '1', f'https://x-access-token:{TOKEN}@github.com/yookeew/dumb-puzzle.git', 'febjune')
git('checkout', '-q', '-f', '-B', 'febjune', 'FETCH_HEAD')
git('log', '-1', '--format=%h %ad %s', '--date=format:%Y-%m-%d %H:%M')

!pip -q install polars==1.44.1 lightgbm==4.7.0 xgboost catboost
sys.path.insert(0, f'{PROJ}/src')
for d in ('cache/eval/final4m', 'data/submissions/final4m', 'reports/eval/final4m', 'models'):
    os.makedirs(f'{PROJ}/{d}', exist_ok=True)
need = ['src/scope.py', 'data/ranking/final_submitting.parquet'] + \
       [f'cache/features/final4m/{f}.parquet' for f in ('train2025', 'holdout_gap2025', 'labels2025', 'ranking')]
for f in need:
    p = f'{PROJ}/{f}'
    print(f'{"ok     " if os.path.exists(p) else "MISSING"} {f}' + (f'  {os.path.getsize(p) / 1e6:.1f} MB' if os.path.exists(p) else ''))
!md5sum cache/features/final4m/ranking.parquet   # expect 04bcbf67...
'''
# %%
# CatBoost for the four-month pipeline (PROGRESS.md §132). T4 runtime, ~40 min.
# The production CatBoost was never saved and its round counts were not recorded, so this
# repeats the holdout fits (round counts -> .rounds.json) and the all-2025 refit, saves the
# refit to models/cat_mixed_f4m/ and predicts Jan+Feb+Jun+Jul 2026. New names only: nothing
# the live-board files were built from is touched. Start from a FRESH runtime (scope is read
# when models.fit is first imported) and run the setup cell above first (it checks out src/).
# Upload first (git-ignored, to the same paths under R):
#   cache/features/final4m/{train2025,holdout_gap2025,labels2025,ranking}.parquet,
#   data/ranking/final_submitting.parquet.
# Copy back: models/cat_mixed_f4m/ (whole folder), cache/eval/cat_mixed_f4m_holdout_ev.parquet
#   (+ .rounds.json), cache/eval/final4m/cat_mixed_f4m_rank.parquet,
#   data/submissions/final4m/cat_mixed_f4m.parquet, reports/eval/final4m/cat_mixed_f4m.md.
import os, sys
os.environ["PRC_SCOPE"] = "final4m"; os.environ["PRC_MEMLOG"] = "1"
R = '/content/drive/MyDrive/smart-jigsaw'
sys.path.insert(0, f'{R}/src'); os.chdir(R)
import scope; assert scope.FINAL and scope.TEMPLATE.exists(), 'fresh runtime + final_submitting.parquet needed'
from pathlib import Path
from models.fit import run
run(engine="cat", target="mixed", eta=0.05, name="cat_mixed_f4m", submit=True,
    feat_dir=Path("cache/features/final4m"),
    ev_out="cache/eval/cat_mixed_f4m_holdout_ev.parquet",
    rank_out="cache/eval/final4m/cat_mixed_f4m_rank.parquet",
    model_out="models/cat_mixed_f4m", resume=True)   # resume: a dropped session restarts at the refit

# %%
# ADS-B re-fetch of the short Feb/Jun 2026 days (2026-10-09). The first pull lost them to
# "Connection reset by peer": tolerant mode kept the truncated read (02-17, 02-20, 02-22,
# 06-23), and on 02-28 / 06-11 the fallback staging replica is itself thin. fetch_adsb.py now
# resumes a dropped download with an HTTP Range request and never keeps a network-truncated
# read. --force overwrites a day only if the new read has more points, so re-running is safe.
# Copy the updated src/ingest/fetch_adsb.py to Drive first (the assert checks).
# Copy back: the six adsb_2026MMDD.parquet files + manifest.csv -> external-data/adsb-final2026/.
import subprocess, time, pathlib
from google.colab import drive
drive.mount('/content/drive')
R   = '/content/drive/MyDrive/smart-jigsaw'
OUT = f'{R}/adsb_final2026'
MAX_PAR = 2   # the resets came with 4 parallel downloads
src = pathlib.Path(f'{R}/src/ingest/fetch_adsb.py')
assert '--force' in src.read_text() and 'RESUMES' in src.read_text(), 'old fetch_adsb.py on Drive; copy the new one'
jobs = [('feb_a', '2026-02-17 2026-02-20 2026-02-22'), ('feb_b', '2026-02-28'),
        ('jun',   '2026-06-11 2026-06-23')]
queue, running = list(jobs), {}
while queue or running:
    while queue and len(running) < MAX_PAR:
        k, days = queue.pop(0)
        running[k] = subprocess.Popen(
            f'python -u {src} --force --days {days} --out {OUT} > {OUT}/refetch_{k}.log 2>&1', shell=True)
    time.sleep(120)
    for k in list(running):
        lines = pathlib.Path(f'{OUT}/refetch_{k}.log').read_text().strip().splitlines()
        print(time.strftime('%H:%M'), k, '|', lines[-1] if lines else '(starting)')
        if running[k].poll() is not None:
            print(time.strftime('%H:%M'), k, 'finished, exit', running[k].returncode)
            del running[k]
import polars as pl
for d in ['20260217', '20260220', '20260222', '20260228', '20260611', '20260623']:
    a = pl.scan_parquet(f'{OUT}/adsb_{d}.parquet').group_by('airport').agg(
        pl.len().alias('n'), ((pl.col('ts') // 3600) % 24).n_unique().alias('hrs')).collect()
    print(d, f"{int(a['n'].sum()):>10,} pts  airports {a.height}  min hours {int(a['hrs'].min())}")
print('normal day: ~1.4-1.9M pts, 10 airports, 22-24 h')

# %%
# Submissions land in data/submissions/<name>.parquet — download and upload to
# the challenge portal as <team>_v<n>.parquet.
