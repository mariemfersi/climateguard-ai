"""
Fetch the raw HURDAT2 Atlantic hurricane database text file from NOAA.

HURDAT2 is a public static file — no API key required. See:
https://www.nhc.noaa.gov/data/#hurdat

Usage:
    python -m data_pipeline.ingestion.hurdat2.fetch_hurdat2
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# NOAA updates the filename to include the latest year covered — check
# https://www.nhc.noaa.gov/data/hurdat/ periodically and update this constant.
HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"

RAW_DOWNLOAD_PATH = Path("data_pipeline/bronze/hurdat2/_raw_hurdat2.txt")


def fetch_hurdat2_raw(url: str = HURDAT2_URL, dest: Path = RAW_DOWNLOAD_PATH) -> Path:
    """
    Download the raw HURDAT2 text file to disk.

    Kept deliberately separate from parsing (parse_hurdat2.py) so the two
    can be tested independently: this function is tested with a live-network
    integration test (skipped in restricted-network environments); the
    parser is tested against a local fixture with no network dependency.
    """
    logger.info("Fetching HURDAT2 from %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text)
    logger.info("Wrote raw HURDAT2 file to %s (%d bytes)", dest, len(resp.text))
    return dest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_hurdat2_raw()
