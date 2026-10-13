# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Predicting `TAXITIME_SEC_mvt` for departures at 11 major European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTAI, LTFM, LSZH) for the
2026 PRC / OpenSky Network Data Challenge. Full design and rationale live in
[DESIGN.md](DESIGN.md).

## Setup (Windows)

### 1. Install uv

```powershell
winget install astral-sh.uv
```

### 2. Clone the repo and sync the environment

```powershell
git clone https://github.com/yookeew/smart-jigsaw.git
cd smart-jigsaw
uv sync
.\.venv\Scripts\Activate.ps1
```

`uv sync` reads `pyproject.toml` (which pins `requires-python = "==3.11.*"`)
and `uv.lock`. **You do not need to separately install Python 3.11** — if uv
doesn't find a matching interpreter on your system, it downloads one
automatically and creates `.venv` from it. This also installs every
dependency (pyopensky, traffic, polars, pyarrow, lightgbm, scikit-learn,
networkx, jupyter) at the exact locked versions. CatBoost is not in the
local environment: its fits run on Google Colab, which installs it
(see [notebooks/colab_train.py](notebooks/colab_train.py)).

### 3. Get your own OpenSky/challenge credentials

You need your own OSN account login and, separately, the MinIO/S3
`access_key` + `secret_key` for the `competition-data` bucket (from the
challenge organizers — this is the same thing as "S3 credentials", MinIO is
just the S3-compatible server OpenSky runs). These are per-person, not
shared — each person needs their own.

### 4. Create/edit the pyopensky config file

This file lives **outside the repo**, per-machine, and must never be
committed. Find (and auto-create) it by importing the package once:

```powershell
python -c "from pyopensky.config import opensky_config_dir; print(opensky_config_dir)"
```

On Windows this is `%LOCALAPPDATA%\pyopensky\pyopensky\settings.conf`. Open
it — the first import already generated a template — and fill it in. Note
`access_key`/`secret_key` go under a separate `[s3]` section, which starts
commented out; **uncomment it**:

```ini
[default]
username = your_osn_user
password = ...
client_id =
client_secret =

[s3]
access_key = ACCESS_KEY
secret_key = SECRET_KEY
```

(`client_id`/`client_secret` are for the REST API, unused here — fine to
leave blank.)

### 5. Download the challenge data

```powershell
python -m src.ingest.fetch_challenge_data
```

Downloads every object in the `competition-data` bucket, routed into
`data/raw/` (monthly training files) and `data/ranking/` (`ranking.parquet`,
`submitting.parquet`). Both are gitignored — this step is per-machine, not
committed. Safe to re-run: it skips files already present.

## Reproducing our submission

[REPRODUCE.md](REPRODUCE.md) rebuilds our final files from the raw data,
step by step. [DESIGN.md](DESIGN.md) describes the model, and
[PROGRESS.md](PROGRESS.md) is the full experiment log. Every adopted change
has a pre-registered test in `reports/*_preregistration.md`.

## Licence

The source code in this repository is licensed under the **GNU General
Public License v3.0** (see [LICENSE](LICENSE)), as the challenge rules
require. Data files are not covered by the code licence; each carries its
own (below).

## Data attribution

- **Challenge data:** PRC / OpenSky Network, under the challenge terms.
  Not redistributed here.
- **adsb.lol** ADS-B tracks: Open Database License (ODbL) v1.0, "Contains
  data from adsb.lol contributors". Our derived `data/external/stands_inferred.csv`
  is under the same licence. See [data/external/LICENSE](data/external/LICENSE).
- **X-Plane Scenery Gateway** stand and runway positions
  (`data/external/stands.csv`, `runways.csv`): GNU GPL v2, by Laminar
  Research and scenery contributors. See
  [data/external/LICENSE](data/external/LICENSE).
- **METAR** observations: NOAA/NWS (public domain), retrieved via the Iowa
  Environmental Mesonet, Iowa State University.
- **EUROCONTROL Aviation Intelligence Unit** daily all-causes pre-departure
  delay (used by the final submission's day-level stages) and three ATFM
  datasets (fetched and tested, not used): free to copy with attribution,
  non-commercial use only; cleared by the organisers for this challenge. See
  [external-data/LICENSE](external-data/LICENSE).
- **OPDI flight events** (EUROCONTROL PRC/PRU, Open Performance Data
  Initiative, based on OpenSky Network data): airport ground events, used at
  LSZH and EDDF. Free to use with attribution.

Per-dataset URLs, access dates and fetch scripts are in
[DATA_SOURCES.md](DATA_SOURCES.md). We do not query OpenSky Network
trajectory or state-vector data ourselves; the only OpenSky-derived inputs
are the challenge data and the organisers' published OPDI events.
