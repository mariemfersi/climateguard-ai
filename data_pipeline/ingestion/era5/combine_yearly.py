"""
Resume ERA5 backfill by combining already-downloaded yearly parquet files.
Run this after backfill_era5.py fails during the combination step.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

YEARLY_DIR = Path("data_pipeline/bronze/era5/_yearly")

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
    
    logger.info(f"Found {len(yearly_files)} yearly files to combine")
    
    # Use pyarrow ParquetWriter to combine without loading into memory
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    
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
    
    logger.info(f"Backfill complete: {out_path}")
    return out_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    combine_yearly_files()
