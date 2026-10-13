# Data Sources

Every external dataset used in this project: what it is, its licence, when we fetched it, what it
feeds, and how to fetch it again. Per the challenge rules, all external data must be openly
accessible/usable and documented here. The step numbers refer to [REPRODUCE.md](REPRODUCE.md).

## Used by the final submission

| Source | Licence | What it feeds | REPRODUCE step |
|---|---|---|---|
| PRC/OSN challenge data | Challenge terms | labels and every feature | 1.1 |
| METAR (NOAA/NWS via Iowa Environmental Mesonet) | Public domain | weather features of the base models | 1.2 |
| EUROCONTROL all-causes pre-departure delay (daily, per airport) | EUROCONTROL terms: attribution, non-commercial; cleared by the organisers | day-level reconciliation (v55), LSZH day level (v58, v69), learned day shift (v61, G2 in v68), LIRF day counts (v70) | 1.2 |
| OPDI flight events (EUROCONTROL PRC/PRU, based on OpenSky Network data) | OPDI open data: attribution | ground-event inputs at LSZH (v52) and EDDF (v60); coverage flags in the day-level stages | 1.2 |
| ADS-B surface tracks (adsb.lol) | ODbL 1.0 | pushback detection, partial-track estimate, ADS-B combiner, corrector inputs | 1.3, 2 |
| Stand and runway positions (X-Plane Scenery Gateway) | GPLv2 | own-stand coordinates for the ADS-B detector | 1.2 (committed) |
| Inferred stand positions (derived from adsb.lol) | ODbL 1.0 | coordinates for 50 stands missing from the Gateway table | 2 (committed) |

We do not query OpenSky Network trajectory or state-vector data ourselves. The only
OpenSky-derived inputs are the challenge dataset and the organisers' published OPDI events.

## Source details

| Name | Publisher | URL | Licence | Date accessed | Derived artefact | Fetch script |
|------|-----------|-----|---------|----------------|-------------------|---------------|
| PRC/OSN challenge data (movements + flight) | OpenSky Network / PRC | `prc-2026-datasets` S3 bucket via pyopensky | Challenge terms | — | `data/raw/training_*.parquet`, `data/ranking/{ranking,submitting,final_ranking,final_submitting}.parquet` (not redistributed) | `src/ingest/fetch_challenge_data.py` |
| METAR (hourly routine reports, 10 challenge departure airports) | Underlying data: NOAA/NWS (US federal government work, public domain, 17 U.S.C. §105). Archive and redistribution: Iowa Environmental Mesonet (IEM), Iowa State University. | https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py (`request/download.phtml` is the human-facing form for the same endpoint) | The observations are public domain. IEM states no further licence or attribution terms on its dataset page (`info/datasets/metar.html`); its site copyright notice covers the website, not the data. No account or API key. IEM recommends NCEI ISD as the source of record; we use IEM as a free, easy-to-query mirror. | 2026-09-20 (2025 + Jan/Jul 2026); 2026-10-09 (Feb/Jun 2026) | `external-data/metar/{STATION}_{window}.csv`, windows `2025_2026-01`, `2026-07`, `2026-02`, `2026-06` | `src/ingest/fetch_metar_data.py` |
| All-causes Pre-departure Delay (daily, per airport): `DLY_ALL_PRE_2`, `FLT_DEP_IFR_2` | EUROCONTROL, Aviation Intelligence Unit | https://www.eurocontrol.int/performance/data/download/csv/all_pre_departure_delays_2025.csv (+ `_2026`), listed on https://ansperformance.eu/csv/ | Free to copy with attribution, non-commercial use only. EUROCONTROL data cleared for this challenge by the organisers (Discord, 2026-09-17; reconfirmed 2026-10-09 for ranking-period aggregates: "no change. still ok"). | 2026-10-08 | `external-data/all_pre_departure_delays_{2025,2026}.csv` | `src/ingest/fetch_atfm_data.py` |
| OPDI flight events v0.0.2 (airport ground milestones: entry/exit parking_position, apron, taxiway, runway, deicing_pad) | EUROCONTROL Performance Review Commission / PRU (Open Performance Data Initiative), derived from OpenSky Network ADS-B | https://www.opdi.aero/flight-event-data.html; files `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{YYYYMMDD}_{YYYYMMDD+10}.parquet` | OPDI open data: "Open data can be freely used, reused, and distributed provided that the data source is attributed" (https://www.opdi.aero/about.html). Attribution: "OPDI (EUROCONTROL PRC/PRU), based on OpenSky Network data". This is the organisers' published derived product, not the OpenSky Trino / state-vector database. | 2026-10-08 (2025 + Jan/Jul 2026); 2026-10-10 (Feb/Jun 2026) | `cache/opdi_events/flight_events_*.parquet` (rows at the 10 airports only) → `cache/opdi_feats/opdi_feats.parquet` (per-departure ground-event leads) | `src/ingest/fetch_opdi_events.py` → `src/features/opdi_feats.py` |
| ADS-B/MLAT surface tracks (readsb `trace_full` per aircraft per day) | adsb.lol (community feeder network) | https://github.com/adsblol/globe_history_2025 and `globe_history_2026`: one GitHub release per day, split tar. Replica chosen via `PREFERRED_RELEASES.txt` in each repo root; fallbacks `prod-0` / `staging-0` / `prod-0tmp` / `staging-0tmp`. The tag used per day is recorded in each folder's `manifest.csv`. | Database: ODbL 1.0 (`LICENSE-ODbL.txt` ships inside each release); feeder contributions CC0. Derived tables carry ODbL and the attribution "adsb.lol contributors". | 2026-09-23 to 2026-09-27 (Jan/Jul 2025 and 2026); 2026-10-02 to 2026-10-03 (rest-of-2025 sample); 2026-10-09 to 2026-10-10 (Feb/Jun 2026, six days re-fetched) | Raw extracts (points within ±0.10° lat / ±0.15° lon of each aerodrome reference point) in `external-data/adsb-fetch/`, `adsb-restofyear/`, `adsb-final2026/` (and `adsb/`, the first Colab extracts) → normalised store `data/external/adsb/day=YYYY-MM-DD/` → detections `cache/adsb_pushback{,_v3,_v4,_v6}/`, positions `cache/adsb_pos_v4/` | `src/ingest/fetch_adsb.py` → `src/ingest/normalise_adsb.py` → `src/link/adsb_pushback.py`, `src/link/adsb_pos.py` |
| Stand/gate positions (apt.dat row codes 1300/1301) and runway ends (row code 100) | X-Plane Scenery Gateway (Laminar Research; community-contributed airport sceneries) | https://gateway.x-plane.com: public API via the `xplane_airports` package (MIT), the recommended scenery pack per airport, no account | GNU GPL v2. Each downloaded pack ships a `COPYING` file with the GPLv2 text; a copy is at `data/external/stands_LICENSE_GPLv2.txt`. Scenery IDs and approval dates are recorded per row in the CSV. | 2026-09-26 | `data/external/stands.csv`, `data/external/runways.csv` (committed); raw apt.dat per airport in `cache/stands/` | `src/ingest/fetch_stands.py` |
| Inferred stand positions (stands missing from the Gateway table) | Derived from adsb.lol (row above) and the challenge's `STAND_mvt` | — (computed locally) | ODbL 1.0, a derived database of adsb.lol; attribution "adsb.lol contributors" | 2026-10-02 | `data/external/stands_inferred.csv` (50 stands; median of matched runs' stationary first samples, n ≥ 3, spread ≤ 30 m; LFPG excluded) | `src/ingest/stand_infer.py` |

## Fetched or consulted, but not used by the final submission

| Name | Publisher / URL | Licence | Date accessed | Why it is here |
|---|---|---|---|---|
| ATFM Slot Adherence | EUROCONTROL AIU, https://ansperformance.eu/csv/atfm_slot_adherence_2025.csv (+ `_2026`) | Attribution, non-commercial; organiser-cleared (Discord, 2026-09-17) | 2026-09-19 | Tested as feature family 5 and rejected (PROGRESS.md §20: daily granularity caps any gain at ~2 s). The family is off by default (`build_features(atfm=False)`). Fetched by `src/ingest/fetch_atfm_data.py` into `external-data/`. |
| Arrival ATFM Delay by Cause (`apt_dly`) | EUROCONTROL AIU, https://ansperformance.eu/csv/apt_dly_2025.csv.bz2 (+ `_2026`) | as above | 2026-09-19 | as above |
| Airport Traffic | EUROCONTROL AIU, https://ansperformance.eu/csv/airport_traffic_2025.csv (+ `_2026`) | as above | 2026-09-19 | as above |
| OpenSky airport coverage table (2026) | EUROCONTROL PRU, https://euctrl-pru.github.io/opensky-airport-coverage/downloads/opensky-airport-coverage-all-aerodromes-2026.csv | Published by PRU with OPDI | 2026-10-08 | Read to check OPDI coverage per airport (PROGRESS.md §128); no derived artefact. |
| OurAirports airport coordinates | https://davidmegginson.github.io/ourairports-data/airports.csv | Public domain | 2026-10-12 | Departure-direction screen (PROGRESS.md §140, `tests/flow_screen/dir_screen.py`). The screen failed. Stored at `external-data/ourairports/airports.csv`, not committed. |
| OpenStreetMap | — | ODbL | — | Planned for routed taxi distances but never built: stand/runway geometry added nothing as a feature (PROGRESS.md §23, §38). No OSM-derived data exists in this repo. |
| `external-data/taxi_in_additional_time_2026.csv` | unknown | unconfirmed | — | Present on one machine, not read by any code. Its provenance and licence weren't confirmed, so it is not part of any submission. |

## Licence files and redistribution

- `LICENSE` (repo root): GNU GPL v3, covering this project's source code only.
- `data/external/LICENSE`: the committed derived tables.
  - `stands.csv` and `runways.csv` are under GPLv2 (Scenery Gateway), with the full text in
    `data/external/stands_LICENSE_GPLv2.txt`.
  - `stands_inferred.csv` is under ODbL 1.0 (a derived database of adsb.lol, attribution
    "adsb.lol contributors").
- `external-data/LICENSE`: the terms of everything fetched into `external-data/`:
  - the EUROCONTROL files (attribution, non-commercial; they cover those files only, not the code);
  - METAR (public domain);
  - the adsb.lol extracts (ODbL 1.0);
  - OurAirports (public domain).
- **Not committed**, rebuilt by the fetch scripts and kept under their own licences:
  - the challenge data;
  - the raw and normalised ADS-B data (`external-data/adsb*/`, `data/external/adsb/`; ODbL 1.0);
  - the EUROCONTROL CSVs;
  - the METAR CSVs;
  - the OPDI files;
  - every cache derived from them.

**Attribution strings:**

- "Contains data from adsb.lol contributors, made available under the ODbL."
- "Contains data from EUROCONTROL's Aviation Intelligence Unit" (https://ansperformance.eu).
- "OPDI (EUROCONTROL PRC/PRU), based on OpenSky Network data."
- METAR: NOAA/NWS, retrieved via the Iowa Environmental Mesonet, Iowa State University.
- Stand and runway positions: X-Plane Scenery Gateway, Laminar Research and scenery contributors
  (GPLv2).
