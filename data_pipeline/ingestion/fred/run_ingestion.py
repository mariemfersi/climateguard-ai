"""
Milestone 1.4 entrypoint: fetch all default FRED series, write to Bronze.

Usage:
    python -m data_pipeline.ingestion.fred.run_ingestion
"""

from __future__ import annotations

import logging

from data_pipeline.ingestion.common.bronze_writer import write_bronze
from data_pipeline.ingestion.fred.fetch_fred import fetch_all_default_series

logger = logging.getLogger(__name__)


def run() -> None:
    df = fetch_all_default_series()
    logger.info("Fetched %d total observations across all FRED series", len(df))

    out_path = write_bronze(
        df,
        source="fred",
        dataset="macro_series",
        dedupe_keys=["series_id", "date"],
    )
    logger.info("Wrote Bronze FRED data to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()