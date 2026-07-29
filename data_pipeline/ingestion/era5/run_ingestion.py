"""
Milestone 1.2 entrypoint: fetch ERA5, flatten it, write it to Bronze.

Usage:
    python -m data_pipeline.ingestion.era5.run_ingestion
"""

from __future__ import annotations

import logging

from data_pipeline.ingestion.common.bronze_writer import write_bronze
from data_pipeline.ingestion.era5.fetch_era5 import fetch_era5_raw
from data_pipeline.ingestion.era5.flatten_era5 import flatten_era5_file

logger = logging.getLogger(__name__)


def run(year: str = "2023", months: list[str] | None = None) -> None:
    months = months or ["06", "07", "08", "09", "10", "11"]
    raw_path = fetch_era5_raw(year=year, months=months)
    df = flatten_era5_file(raw_path)
    logger.info("Flattened %d ERA5 grid-cell/time records", len(df))

    out_path = write_bronze(
        df,
        source="era5",
        dataset="gulf_coast_reanalysis",
        dedupe_keys=["lat", "lon", "timestamp"],
    )
    logger.info("Wrote Bronze ERA5 data to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
