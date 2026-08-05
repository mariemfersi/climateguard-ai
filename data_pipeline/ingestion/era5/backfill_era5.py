"""
ERA5 backfill script to complete full 74-year coverage (1950-2023).

Current coverage: 14 years (1955, 1961, 1969, 1975, 1983, 1989, 1995, 1998, 2001, 2004, 2005, 2008, 2012, 2023)
Target coverage: 74 years (1950-2023 hurricane seasons)

This script processes years incrementally to avoid MemoryError from accumulating
155M+ rows in memory. Each year is downloaded, flattened, and written to Bronze
as a separate parquet file, then combined in a post-processing step.

Usage:
    python -m data_pipeline.ingestion.era5.backfill_era5
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from data_pipeline.ingestion.common.bronze_writer import write_bronze
from data_pipeline.ingestion.era5.fetch_era5 import fetch_era5_raw, GULF_COAST_BBOX
from data_pipeline.ingestion.era5.flatten_era5 import flatten_era5_file

logger = logging.getLogger(__name__)

# Hurricane season months (June-November)
HURRICANE_SEASON_MONTHS = ["06", "07", "08", "09", "10", "11"]

# Year range for full backfill
BACKFILL_START_YEAR = 1950
BACKFILL_END_YEAR = 2023

# Years that already have coverage (from existing data)
EXISTING_COVERAGE_YEARS = {1955, 1961, 1969, 1975, 1983, 1989, 1995, 1998, 2001, 2004, 2005, 2008, 2012, 2023}

# Temporary directory for per-year parquet files
YEARLY_DIR = Path("data_pipeline/bronze/era5/_yearly")


def backfill_single_year(year: int) -> Path:
    """
    Download, flatten, and write ERA5 data for a single year.
    
    Args:
        year: Year to process (e.g., 1950)
    
    Returns:
        Path to the written parquet file
    """
    logger.info(f"Processing year {year}...")
    
    # Download raw NetCDF for this year
    raw_path = YEARLY_DIR / f"_raw_era5_{year}.nc"
    fetch_era5_raw(
        year=str(year),
        months=HURRICANE_SEASON_MONTHS,
        dest=raw_path,
        area=GULF_COAST_BBOX,
    )
    
    # Flatten to tabular format
    df = flatten_era5_file(raw_path)
    logger.info(f"Flattened {len(df)} rows for year {year}")
    
    # Write to Bronze as per-year file
    yearly_path = YEARLY_DIR / f"gulf_coast_reanalysis_{year}.parquet"
    yearly_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(
        yearly_path,
        index=False,
        engine="pyarrow",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    
    logger.info(f"Wrote {yearly_path}")
    
    # Clean up raw NetCDF to save disk space
    raw_path.unlink()
    
    return yearly_path


def combine_yearly_files() -> Path:
    """
    Combine all per-year parquet files into a single Bronze parquet file.
    
    This uses pyarrow's ParquetWriter to combine files out-of-core without
    loading everything into memory. Deduplication is skipped since each year
    has distinct timestamps (no overlap between years).
    
    Returns:
        Path to the combined parquet file
    """
    logger.info("Combining yearly parquet files out-of-core...")
    
    yearly_files = sorted(YEARLY_DIR.glob("gulf_coast_reanalysis_*.parquet"))
    if not yearly_files:
        raise ValueError("No yearly parquet files found to combine")
    
    # Use pyarrow ParquetWriter to combine without loading into memory
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Read first file to get schema
    first_df = pd.read_parquet(yearly_files[0])
    schema = pa.Table.from_pandas(first_df).schema
    
    # Create output file with schema
    out_path = Path("data_pipeline/bronze/era5/gulf_coast_reanalysis.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = pq.ParquetWriter(out_path, schema, compression='snappy')
    
    total_rows = 0
    for yearly_file in yearly_files:
        logger.info(f"Appending {yearly_file}...")
        df = pd.read_parquet(yearly_file)
        table = pa.Table.from_pandas(df)
        writer.write_table(table)
        total_rows += len(df)
    
    writer.close()
    logger.info(f"Combined {total_rows} total rows out-of-core")
    
    # Skip deduplication - each year has distinct timestamps, no overlap
    # This avoids loading 714M+ rows into memory for deduplication
    
    # Clean up yearly files
    for yearly_file in yearly_files:
        yearly_file.unlink()
    YEARLY_DIR.rmdir()
    
    return out_path


def run_backfill() -> Path:
    """
    Run the full ERA5 backfill for missing years.
    
    Returns:
        Path to the combined Bronze parquet file
    """
    # Determine which years need backfill
    missing_years = set(range(BACKFILL_START_YEAR, BACKFILL_END_YEAR + 1)) - EXISTING_COVERAGE_YEARS
    logger.info(f"Years to backfill: {sorted(missing_years)} ({len(missing_years)} years)")
    
    if not missing_years:
        logger.info("No years need backfill - coverage is complete")
        return Path("data_pipeline/bronze/era5/gulf_coast_reanalysis.parquet")
    
    # Process each year incrementally
    for year in sorted(missing_years):
        try:
            backfill_single_year(year)
        except Exception as e:
            logger.error(f"Failed to process year {year}: {e}")
            raise
    
    # Combine all yearly files into single Bronze file
    combined_path = combine_yearly_files()
    
    logger.info(f"Backfill complete: {combined_path}")
    return combined_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_backfill()
