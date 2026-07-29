"""
Fetch economic indicator time series from FRED (Federal Reserve Economic
Data), the Federal Reserve Bank of St. Louis' public API.

Per Roadmap Task 1.4.1: parameterized by series ID so the same function
serves construction cost index, CPI, and interest rate series.

Usage:
    python -m data_pipeline.ingestion.fred.fetch_fred
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series used by ClimateGuard AI's macro-financial features (design doc §4).
# Key = our internal feature name, Value = FRED's series ID.
DEFAULT_SERIES = {
    "cpi_all_urban": "CPIAUCSL",  # CPI, all urban consumers — general inflation proxy
    "construction_cost_index": "WPUSI012011",  # PPI: construction materials
    "fed_funds_rate": "FEDFUNDS",  # effective federal funds rate
}


def fetch_fred_series(series_id: str, series_name: str) -> pd.DataFrame:
    """
    Fetch a single FRED series as a tidy DataFrame with columns
    [series_name, date, value].

    Raises:
        RuntimeError: if FRED_API_KEY is not configured (see .env.example).
        requests.HTTPError: if FRED returns a non-200 response (e.g. an
            invalid series_id) — surfaced directly rather than swallowed,
            since a silently-empty series would be a worse failure mode.
    """
    settings = get_settings()
    api_key = settings.require("fred_api_key")

    logger.info("Fetching FRED series '%s' (%s)", series_name, series_id)

    resp = requests.get(
        FRED_BASE_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()

    observations = resp.json()["observations"]
    df = pd.DataFrame(observations)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    # FRED encodes missing values as the literal string "." — convert to NaN
    # rather than silently coercing to 0 or dropping the row.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_name"] = series_name
    df["series_id"] = series_id

    return df[["series_id", "series_name", "date", "value"]]


def fetch_all_default_series() -> pd.DataFrame:
    """Fetch every series in DEFAULT_SERIES and concatenate into one table."""
    frames = [
        fetch_fred_series(series_id, series_name)
        for series_name, series_id in DEFAULT_SERIES.items()
    ]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_all_default_series()
    print(df.groupby("series_name")["date"].agg(["min", "max", "count"]))