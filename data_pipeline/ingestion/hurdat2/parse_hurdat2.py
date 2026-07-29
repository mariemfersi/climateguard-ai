"""
Parse the raw HURDAT2 text format into a tidy pandas DataFrame.

HURDAT2 format reference: https://www.nhc.noaa.gov/data/hurdat/hurdat2-format.pdf

The file interleaves two record types with no explicit type marker other
than shape:

  Header record (one per storm), e.g.:
      AL092023,             LEE,     51,
      -> [basin+cyclone_no+year, name, number_of_best_track_entries]

  Data record (one per 6-hourly observation), e.g.:
      20120829, 1200,  , TS, 20.9N,  85.5W,  45, 1000, ...
      -> [date, time, record_identifier, status, lat, lon,
          max_wind_kt, min_pressure_mb, <13 wind-radii fields>]

This parser deliberately keeps only the core fields needed downstream
(storm_id, name, timestamp, lat, lon, max_wind_kt, min_pressure_mb, status)
per the roadmap's Task 1.1.1 acceptance criteria. Wind-radii fields are
parsed but not currently retained — extend `_parse_data_record` if a later
phase needs them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_HEADER_ID_PATTERN = re.compile(r"^[A-Z]{2}\d{6}$")


def _parse_latlon(value: str) -> float:
    """'20.9N' -> 20.9 ; '85.5W' -> -85.5"""
    value = value.strip()
    sign = -1.0 if value[-1] in ("S", "W") else 1.0
    return sign * float(value[:-1])


def _parse_header_record(fields: list[str]) -> dict:
    storm_id_raw, name, _num_entries = fields[0], fields[1], fields[2]
    return {"storm_id": storm_id_raw.strip(), "name": name.strip()}


def _parse_data_record(fields: list[str], storm_id: str, name: str) -> dict:
    date, time = fields[0].strip(), fields[1].strip()
    status = fields[3].strip()
    lat = _parse_latlon(fields[4])
    lon = _parse_latlon(fields[5])
    max_wind_kt = int(fields[6].strip())
    min_pressure_mb = int(fields[7].strip())

    timestamp = pd.Timestamp(
        year=int(date[0:4]),
        month=int(date[4:6]),
        day=int(date[6:8]),
        hour=int(time[0:2]),
        minute=int(time[2:4]),
        tz="UTC",
    )

    return {
        "storm_id": storm_id,
        "name": name,
        "timestamp": timestamp,
        "status": status,
        "lat": lat,
        "lon": lon,
        "max_wind_kt": max_wind_kt,
        "min_pressure_mb": min_pressure_mb,
    }


def parse_hurdat2(raw_text: str) -> pd.DataFrame:
    """
    Parse raw HURDAT2 text into a tidy DataFrame with one row per
    6-hourly storm observation.

    Raises:
        ValueError: if a data record appears before any header record
            (malformed input — fail loudly rather than silently dropping rows).
    """
    records: list[dict] = []
    current_storm_id: str | None = None
    current_name: str | None = None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        fields = [f.strip() for f in line.split(",")]
        # Trailing comma produces a trailing empty field — drop it.
        if fields and fields[-1] == "":
            fields = fields[:-1]

        if _HEADER_ID_PATTERN.match(fields[0]):
            header = _parse_header_record(fields)
            current_storm_id = header["storm_id"]
            current_name = header["name"]
            continue

        if current_storm_id is None:
            raise ValueError(
                f"Malformed HURDAT2 input: data record encountered before "
                f"any header record: {line!r}"
            )

        records.append(_parse_data_record(fields, current_storm_id, current_name))

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["storm_id", "timestamp"], keep="last")
        df = df.sort_values(["storm_id", "timestamp"]).reset_index(drop=True)
    return df


def parse_hurdat2_file(path: Path) -> pd.DataFrame:
    return parse_hurdat2(Path(path).read_text())
