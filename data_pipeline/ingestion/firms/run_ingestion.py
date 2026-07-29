"""
Milestone 1.3 entrypoint: fetch NASA FIRMS active-fire data, normalize it,
write it to Bronze.

Usage:
    python -m data_pipeline.ingestion.firms.run_ingestion
"""

from __future__ import annotations

import logging

from data_pipeline.ingestion.common.bronze_writer import write_bronze
from data_pipeline.ingestion.firms.fetch_firms import fetch_firms_raw
from data_pipeline.ingestion.firms.normalize_firms import normalize_firms

logger = logging.getLogger(__name__)


def run() -> None:
    raw_df = fetch_firms_raw()
    df = normalize_firms(raw_df)
    logger.info("Normalized %d FIRMS active-fire detections", len(df))

    if df.empty:
        logger.warning(
            "No active fires found for the current query window — this can "
            "be a genuinely quiet fire period, not necessarily a bug. "
            "Skipping Bronze write (write_bronze rejects empty writes)."
        )
        return

    out_path = write_bronze(
        df,
        source="firms",
        dataset="active_fires",
        dedupe_keys=["lat", "lon", "timestamp"],
    )
    logger.info("Wrote Bronze FIRMS data to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()