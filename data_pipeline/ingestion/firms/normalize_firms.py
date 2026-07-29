"""
Normalize raw NASA FIRMS active-fire data into a tidy schema, consistent
with the (lat, lon, timestamp, ...) convention used by hurdat2/era5.

Kept separate from fetch_firms.py so this pure transformation logic can be
unit-tested without any network dependency.
"""

from __future__ import annotations

import pandas as pd


def normalize_firms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw FIRMS columns (latitude, longitude, acq_date, acq_time, ...)
    into the project's standard (lat, lon, timestamp, ...) schema.

    FIRMS reports acq_time as a zero-padded-on-read-but-not-on-write 4-digit
    HHMM integer/string (e.g. 907 means 09:07 UTC) — this is a common source
    of an off-by-a-digit bug if not explicitly zero-padded before parsing.
    """
    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "timestamp", "confidence", "frp"])

    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    acq_time_str = df["acq_time"].astype(int).astype(str).str.zfill(4)
    hours = acq_time_str.str[:2].astype(int)
    minutes = acq_time_str.str[2:].astype(int)

    df["timestamp"] = (
        pd.to_datetime(df["acq_date"])
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    ).dt.tz_localize("UTC")

    keep_cols = ["lat", "lon", "timestamp"]
    for optional_col in ["confidence", "frp", "satellite", "instrument"]:
        if optional_col in df.columns:
            keep_cols.append(optional_col)

    return df[keep_cols].reset_index(drop=True)