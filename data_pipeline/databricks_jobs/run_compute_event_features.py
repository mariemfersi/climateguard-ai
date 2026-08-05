"""
Generate event-focused climate features from HURDAT2 storm track data.

This script computes per-location-year hurricane risk features that capture
actual storm activity rather than simple climatological averages.

Usage:
    python -m data_pipeline.databricks_jobs.run_compute_event_features
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_pipeline.databricks_jobs.compute_event_features import (
    compute_yearly_event_features,
    compute_regional_encoding,
    load_hurdat2_tracks,
    validate_data_alignment,
)

logger = logging.getLogger(__name__)

BRONZE_HURDAT2_PATH = Path("data_pipeline/bronze/hurdat2/tracks.parquet")
SILVER_LOCATIONS_PATH = Path("data_pipeline/silver/locations.parquet")
SILVER_CLAIMS_PATH = Path("data_pipeline/silver/claims.parquet")
EVENT_FEATURES_OUTPUT_PATH = Path("data_pipeline/gold/event_features.parquet")

TRAINING_START_YEAR = 1950
TRAINING_END_YEAR = 2023


def run() -> pd.DataFrame:
    """Generate event-focused climate features and save to Gold layer."""
    
    # Check input files exist
    for path in [BRONZE_HURDAT2_PATH, SILVER_LOCATIONS_PATH, SILVER_CLAIMS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — earlier phase/milestone must be run first."
            )
    
    logger.info("Loading HURDAT2 storm tracks...")
    storm_tracks = load_hurdat2_tracks(BRONZE_HURDAT2_PATH)
    logger.info(f"Loaded {len(storm_tracks):,} storm track observations")
    
    logger.info("Loading locations...")
    locations = pd.read_parquet(SILVER_LOCATIONS_PATH)
    logger.info(f"Loaded {len(locations):,} locations")
    
    logger.info("Loading claims for validation...")
    claims = pd.read_parquet(SILVER_CLAIMS_PATH)
    logger.info(f"Loaded {len(claims):,} claim records")
    
    # Compute event features for each year
    years = list(range(TRAINING_START_YEAR, TRAINING_END_YEAR + 1))
    logger.info(f"Computing event features for {len(years)} years ({TRAINING_START_YEAR}-{TRAINING_END_YEAR})...")
    
    event_features = compute_yearly_event_features(
        storm_tracks=storm_tracks,
        locations=locations,
        years=years,
        influence_radius_km=500,  # 500km influence radius
    )
    
    logger.info(f"Generated {len(event_features):,} event feature rows")
    
    # Validate data alignment
    logger.info("Validating data alignment...")
    validation_results = validate_data_alignment(locations, claims, event_features)
    
    print("\n=== Data Alignment Validation ===")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    # Check for alignment issues
    if validation_results["orphaned_claims"] > 0:
        logger.warning(f"Found {validation_results['orphaned_claims']} orphaned claim records")
    if validation_results["orphaned_event_features"] > 0:
        logger.warning(f"Found {validation_results['orphaned_event_features']} orphaned event feature records")
    
    # Save event features
    EVENT_FEATURES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    event_features.to_parquet(EVENT_FEATURES_OUTPUT_PATH, index=False)
    logger.info(f"Saved event features to {EVENT_FEATURES_OUTPUT_PATH}")
    
    # Print summary statistics
    print("\n=== Event Features Summary ===")
    print(f"Total rows: {len(event_features):,}")
    print(f"Years covered: {event_features['year'].min()}-{event_features['year'].max()}")
    print(f"Locations covered: {event_features['location_id'].nunique():,}")
    
    print("\nStorm count distribution:")
    print(event_features["storm_count"].describe())
    
    print("\nLocations with 1+ storms nearby:")
    print(f"  { (event_features['storm_count'] > 0).sum():,} / {len(event_features):,} rows")
    print(f"  { (event_features.groupby('location_id')['storm_count'].max() > 0).sum():,} / {len(locations):,} locations")
    
    print("\nMax wind nearby distribution (for rows with storms):")
    nearby = event_features[event_features["storm_count"] > 0]
    if len(nearby) > 0:
        print(nearby["max_wind_nearby_kt"].describe())
    else:
        print("  No storms nearby in dataset")
    
    return event_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
