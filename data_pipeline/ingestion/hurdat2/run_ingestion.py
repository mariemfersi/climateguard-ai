"""
Milestone 1.1 entrypoint: fetch HURDAT2, parse it, write it to Bronze.

Usage:
    python -m data_pipeline.ingestion.hurdat2.run_ingestion
"""

from __future__ import annotations

import logging

from data_pipeline.ingestion.common.bronze_writer import write_bronze
from data_pipeline.ingestion.hurdat2.fetch_hurdat2 import fetch_hurdat2_raw
from data_pipeline.ingestion.hurdat2.parse_hurdat2 import parse_hurdat2_file

logger = logging.getLogger(__name__)


def run() -> None:
    raw_path = fetch_hurdat2_raw()
    df = parse_hurdat2_file(raw_path)
    logger.info("Parsed %d storm-observation records", len(df))

    out_path = write_bronze(
        df,
        source="hurdat2",
        dataset="tracks",
        dedupe_keys=["storm_id", "timestamp"],
    )
    logger.info("Wrote Bronze HURDAT2 tracks to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
