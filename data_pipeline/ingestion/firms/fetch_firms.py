"""
Fetch active-fire detection data from NASA FIRMS (Fire Information for
Resource Management System) — the "area" API, which returns CSV.

API docs: https://firms.modaps.eosdis.nasa.gov/api/area/

Per design doc §4, this is included for horizontal-expansion readiness
(wildfire peril) even though the vertical slice is hurricane-first — see
roadmap Milestone 1.3, flagged optional.

Usage:
    python -m data_pipeline.ingestion.firms.fetch_firms
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# FIRMS area-coordinates format is "West,South,East,North" — note this is a
# DIFFERENT field order than ERA5's GULF_COAST_BBOX ([North,West,South,East]).
# Kept as its own constant rather than reused/converted, to avoid a subtle
# bounding-box bug from mixing up the two conventions.
GULF_COAST_AREA = "-100,10,-60,35"

DEFAULT_SOURCE = "VIIRS_SNPP_NRT"
MAX_DAY_RANGE = 5  # hard limit enforced by the FIRMS API itself


def fetch_firms_raw(
    area_coordinates: str = GULF_COAST_AREA,
    day_range: int = 5,
    source: str = DEFAULT_SOURCE,
    date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch raw active-fire detections as a DataFrame.

    Args:
        area_coordinates: "West,South,East,North" bounding box string.
        day_range: number of days of data to return, 1-10 (FIRMS API limit).
        source: satellite/instrument source, e.g. "VIIRS_SNPP_NRT".
        date: optional end date "YYYY-MM-DD"; defaults to most recent
            available data if omitted.

    Raises:
        ValueError: if day_range is outside the FIRMS-supported 1-10 range.
        RuntimeError: if FIRMS_API_KEY is not configured, or the API
            returns an error payload instead of CSV data.
    """
    if not (1 <= day_range <= MAX_DAY_RANGE):
        raise ValueError(
            f"day_range must be between 1 and {MAX_DAY_RANGE} (FIRMS API limit), "
            f"got {day_range}"
        )

    settings = get_settings()
    map_key = settings.require("nasa_firms_map_key")

    url_parts = [FIRMS_BASE_URL, map_key, source, area_coordinates, str(day_range)]
    if date:
        url_parts.append(date)
    url = "/".join(url_parts)

    logger.info(
        "Fetching FIRMS active fires: source=%s area=%s day_range=%d",
        source,
        area_coordinates,
        day_range,
    )
    resp = requests.get(url, timeout=30)
    print("Status:", resp.status_code)
    print(resp.text)
    resp.raise_for_status()

    text = resp.text.strip()
    # FIRMS returns a plain-text error message (not CSV, not an HTTP error
    # code) for bad requests such as an invalid MAP_KEY — must be detected
    # explicitly or a bad key silently becomes an "empty result" bug.
    if text.lower().startswith(("invalid", "error", "no valid map_key")):
        raise RuntimeError(f"FIRMS API returned an error: {text[:200]}")

    if not text:
        logger.warning("FIRMS returned an empty response for this query.")
        return pd.DataFrame(
            columns=["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]
        )

    return pd.read_csv(io.StringIO(text))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_firms_raw()
    print(f"Fetched {len(df)} active-fire detections")
    print(df.head())