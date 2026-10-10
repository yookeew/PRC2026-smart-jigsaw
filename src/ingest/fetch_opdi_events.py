"""Fetch OPDI flight events (v0.0.2) for the challenge airports, reading only what is needed.

OPDI (Open Performance Data Initiative, EUROCONTROL PRC/PRU) publishes flight events derived from OpenSky
Network ADS-B as 10-day Parquet files (~550 MB each) on a grid anchored at 2022-01-01:
  https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{d0}_{d0+10}.parquet

Row groups are clustered by event type. Airport (ground) events -- entry/exit parking_position, apron, taxiway,
runway, deicing_pad -- are the only rows with a non-empty `info` JSON (osm_airport, osm_ref, osm_aeroway, osn_icao24,
osn_flight_id). So for each file we read the Parquet footer by HTTP range request, keep the row groups whose `info`
column has real content, read only the columns we need from them, and keep rows whose osm_airport is one of the 10
challenge airports. Default: one streaming GET per file to a temporary file that is deleted after filtering (the
server rate-limits bursts of range requests with a Cloudflare challenge); --ranged reads only the needed row groups
and columns by HTTP range request (~170 of ~550 MB per file).

Output: cache/opdi_events/flight_events_{d0}_{d1}.parquet with columns
  flight_id str (OPDI id, unique within a file) | type str | t f64 epoch s (UTC) | lon f64 | lat f64 | airport str | aeroway str | ref str |
  icao24 str | callsign str | time_in_use_s f64
Files already present are skipped.

Licence: OPDI open data, "freely used, reused, and distributed provided that the data source is attributed"
(https://www.opdi.aero/about.html). See DATA_SOURCES.md.

Usage:
  python src/ingest/fetch_opdi_events.py 2025-07-04 2026-07-09        # the 10-day files starting on these days
  python src/ingest/fetch_opdi_events.py --months 2025-01 2025-07       # every file overlapping these months
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import requests

BASE = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/"
GRID0 = dt.date(2022, 1, 1)
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
COLS = ["flight_id", "type", "event_time", "longitude", "latitude", "info"]
OUT = Path("cache/opdi_events")
UA = {"User-Agent": "prc-dc-2026-smart-jigsaw (OPDI open data; research)"}
PAUSE_S = 0.5


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos, self.nbytes = url, 0, 0
        self.s = requests.Session()
        r = self.s.head(url, headers=UA, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for k in range(10):
            time.sleep(PAUSE_S)                       # be polite: the server rate-limits bursts (HTTP 429)
            wait = 5 * (k + 1)
            try:
                r = self.s.get(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"}, timeout=600)
                if r.status_code == 206 and len(r.content) == end - self.pos + 1:
                    b = r.content
                    break
                if r.status_code == 429:
                    wait = max(60 * (k + 1), int(r.headers.get("Retry-After", 0) or 0))
                msg = f"status {r.status_code}, {len(r.content)} bytes"
            except requests.RequestException as e:
                msg = str(e)
            print(f"  retry {k + 1} {self.pos}-{end}: {msg}; waiting {wait} s", flush=True)
            time.sleep(wait)
        else:
            raise IOError(f"range {self.pos}-{end} failed")
        self.pos += len(b)
        self.nbytes += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[: len(b)] = b
        return len(b)


def file_name(d0: dt.date) -> str:
    return f"flight_events_{d0:%Y%m%d}_{d0 + dt.timedelta(days=10):%Y%m%d}.parquet"


def grid_start(d: dt.date) -> dt.date:
    return GRID0 + dt.timedelta(days=((d - GRID0).days // 10) * 10)


def files_for_month(ym: str) -> list[dt.date]:
    y, m = map(int, ym.split("-"))
    first = dt.date(y, m, 1)
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    d, out = grid_start(first), []
    while d < nxt:
        out.append(d)
        d += dt.timedelta(days=10)
    return out


def _download(url: str, dest: Path) -> None:
    """One streaming GET per file (few requests: the server rate-limits bursts of range requests)."""
    for k in range(8):
        try:
            with requests.get(url, headers=UA, stream=True, timeout=600) as r:
                if r.status_code == 200:
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(1 << 22):
                            f.write(chunk)
                    if dest.stat().st_size == int(r.headers.get("Content-Length", dest.stat().st_size)):
                        return
                    msg = "short read"
                elif r.status_code == 404:
                    raise FileNotFoundError(url)
                else:
                    msg = f"status {r.status_code}"
        except requests.RequestException as e:
            msg = str(e)
        wait = 120 * (k + 1)
        print(f"  retry {k + 1} {url.rsplit('/', 1)[1]}: {msg}; waiting {wait} s", flush=True)
        time.sleep(wait)
    raise IOError(f"download failed: {url}")


def fetch(d0: dt.date, ranged: bool = False) -> Path:
    name = file_name(d0)
    dest = OUT / name
    if dest.exists():
        print(f"skip  {name}", flush=True)
        return dest
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp_raw = OUT / f"_{name}.raw"
    if ranged:
        f = HttpFile(BASE + name)
        size, fetched = f.size, None
    else:
        _download(BASE + name, tmp_raw)
        f = open(tmp_raw, "rb")
        size = fetched = tmp_raw.stat().st_size
    pf = pq.ParquetFile(f)
    md = pf.metadata
    ci = {md.row_group(0).column(j).path_in_schema: j for j in range(md.num_columns)}
    rgs = [i for i in range(md.num_row_groups) if md.row_group(i).column(ci["info"]).total_compressed_size > 1000]
    parts = []
    for i in rgs:
        tb = pl.from_arrow(pf.read_row_group(i, columns=COLS))
        tb = tb.filter(pl.col("info").str.contains('"osm_airport":"(' + "|".join(AIRPORTS) + ')"'))
        parts.append(tb)
    if ranged:
        fetched = f.nbytes
    else:
        f.close()
        tmp_raw.unlink()
    ev = pl.concat(parts, how="vertical_relaxed")
    j = lambda k: pl.col("info").str.json_path_match(f"$.{k}")
    # 2025 files: flight_id uint64, event_time timestamp; 2026 files: int64 and "YYYY-mm-dd HH:MM:SS" strings
    et = pl.col("event_time")
    et = (et.str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC") if ev.schema["event_time"] == pl.Utf8
          else et.dt.replace_time_zone("UTC"))
    ev = ev.select(
        pl.col("flight_id").cast(pl.Utf8),
        "type",
        t=et.dt.epoch("ms") / 1000.0,
        lon=pl.col("longitude"),
        lat=pl.col("latitude"),
        airport=j("osm_airport"),
        aeroway=j("osm_aeroway"),
        ref=j("osm_ref"),
        icao24=j("osn_icao24"),
        callsign=j("osn_flight_id").str.strip_chars(),
        time_in_use_s=j("opdi_time_in_use_s").cast(pl.Float64, strict=False),
    ).filter(pl.col("airport").is_in(AIRPORTS))
    tmp = dest.with_suffix(".tmp")
    ev.write_parquet(tmp)
    tmp.replace(dest)
    print(f"fetch {name}: row groups {rgs}, {fetched / 1e6:.0f} of {size / 1e6:.0f} MB, "
          f"{ev.height:,} events, {time.time() - t0:.0f} s", flush=True)
    return dest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("starts", nargs="*", help="grid start days YYYY-MM-DD")
    ap.add_argument("--months", nargs="*", default=[], help="YYYY-MM: every file overlapping the month")
    ap.add_argument("--ranged", action="store_true", help="HTTP range reads instead of one GET per file")
    a = ap.parse_args(argv)
    days = {grid_start(dt.date.fromisoformat(s)) for s in a.starts}
    for ym in a.months:
        days.update(files_for_month(ym))
    missing = []
    for d0 in sorted(days):
        try:
            fetch(d0, a.ranged)
        except FileNotFoundError:                     # listed on opdi.aero but absent on the server (e.g. 2025-10-02)
            print(f"MISSING {file_name(d0)} (HTTP 404)", flush=True)
            missing.append(file_name(d0))
    print(f"done; missing {missing}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
