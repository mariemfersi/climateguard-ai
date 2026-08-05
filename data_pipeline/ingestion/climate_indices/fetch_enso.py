"""
Fetch ENSO (El Niño-Southern Oscillation) index data from NOAA.

The Oceanic Niño Index (ONI) is NOAA's primary ENSO indicator: a 3-month running
mean of ERSST.v5 SST anomalies in the Niño 3.4 region (5°N-5°S, 120°-170°W).

ONI values are categorized as:
- El Niño: ≥ +0.5°C for at least 5 consecutive overlapping 3-month seasons
- La Niña: ≤ -0.5°C for at least 5 consecutive overlapping 3-month seasons
- Neutral: Between -0.5°C and +0.5°C

This script fetches historical ONI data from NOAA's CPC website and writes it
to the Bronze layer as a time-varying covariate for TFT modeling.

Usage:
    python -m data_pipeline.ingestion.climate_indices.fetch_enso
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_pipeline.ingestion.common.bronze_writer import write_bronze

logger = logging.getLogger(__name__)

# NOAA CPC ONI data source (historical monthly values)
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


def fetch_oni_data(url: str = NOAA_ONI_URL) -> pd.DataFrame:
    """
    Fetch historical ONI data from NOAA CPC.
    
    The ONI file format is fixed-width ASCII with columns:
    - SEAS: 3-letter season code (DJF, JFM, FMA, etc.)
    - YR: Year (4 digits)
    - TOTAL: Total SST value (not used)
    - ANOM: ONI anomaly value (this is what we need)
    
    Args:
        url: NOAA CPC ONI data URL
    
    Returns:
        DataFrame with columns [year, season, oni_anomaly_celsius, enso_phase]
    """
    logger.info(f"Fetching ONI data from {url}")
    
    # Read the fixed-width ASCII file
    # Column positions based on NOAA CPC format
    colspecs = [
        (0, 6),    # SEAS (season)
        (6, 10),   # YR (year)
        (10, 17),  # TOTAL (SST)
        (17, 23),  # ANOM (anomaly)
    ]
    
    col_names = ["season", "year", "total_sst", "oni_anomaly_celsius"]
    
    df = pd.read_fwf(url, colspecs=colspecs, names=col_names, header=None, skiprows=1)
    
    # Convert year to integer
    df["year"] = df["year"].astype(int)
    
    # Convert anomaly to float
    df["oni_anomaly_celsius"] = df["oni_anomaly_celsius"].astype(float)
    
    # Clean up: remove rows with missing values
    df = df.dropna(subset=["oni_anomaly_celsius"])
    
    # Add ENSO classification
    df["enso_phase"] = df["oni_anomaly_celsius"].apply(
        lambda x: "el_nino" if x >= 0.5 else ("la_nina" if x <= -0.5 else "neutral")
    )
    
    logger.info(f"Fetched {len(df)} ONI records from {df['year'].min()} to {df['year'].max()}")
    
    return df[["year", "season", "oni_anomaly_celsius", "enso_phase"]]


def run_enso_ingestion() -> Path:
    """
    Fetch ONI data and write to Bronze layer.
    
    Returns:
        Path to the written Bronze parquet file
    """
    df = fetch_oni_data()
    
    # Write to Bronze
    out_path = write_bronze(
        df=df,
        source="climate_indices",
        dataset="oni",
        dedupe_keys=["year", "season"],
    )
    
    logger.info(f"ONI data written to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_enso_ingestion()
