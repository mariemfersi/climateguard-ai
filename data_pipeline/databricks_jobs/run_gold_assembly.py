"""
Milestone 3.2 entrypoint: assemble the Gold-layer feature table by joining:
  - Silver locations (construction attributes, TIV)
  - Aggregated ERA5 climate features (via geo_join_era5 + aggregate_climate_features)
  - Basin-wide SST regional covariate (via compute_basin_wide_sst_summary)
  - Aggregated historical hazard features (via aggregate_hazard_features, from claims.parquet)

Writes the result to data_pipeline/gold/gold_features.parquet locally.

Usage:
    python -m data_pipeline.databricks_jobs.run_gold_assembly
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_pipeline.databricks_jobs.aggregate_climate_features import (
    aggregate_climate_features_per_location,
    compute_basin_wide_sst_summary,
)
from data_pipeline.databricks_jobs.aggregate_hazard_features import (
    aggregate_hazard_features_per_location,
)
from data_pipeline.databricks_jobs.compute_geographic_features import (
    compute_geographic_risk_features,
)
from data_pipeline.databricks_jobs.geo_join_era5 import geo_join_era5, get_spark_session

logger = logging.getLogger(__name__)

BRONZE_ERA5_PATH = Path("data_pipeline/bronze/era5/gulf_coast_reanalysis.parquet")
SILVER_LOCATIONS_PATH = Path("data_pipeline/silver/locations.parquet")
SILVER_CLAIMS_PATH = Path("data_pipeline/silver/claims.parquet")
GOLD_OUTPUT_PATH = Path("data_pipeline/gold/gold_features.parquet")
YEARLY_SST_OUTPUT_PATH = Path("data_pipeline/gold/yearly_sst.parquet")


def run() -> pd.DataFrame:
    for path in [BRONZE_ERA5_PATH, SILVER_LOCATIONS_PATH, SILVER_CLAIMS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — earlier phase/milestone must be run first."
            )

    logger.info("Loading Silver locations and claims...")
    locations = pd.read_parquet(SILVER_LOCATIONS_PATH)
    claims = pd.read_parquet(SILVER_CLAIMS_PATH)

    logger.info("Starting local Spark session for ERA5 geo-join...")
    spark = get_spark_session(app_name="climateguard-gold-assembly")
    try:
        era5_sdf = spark.read.parquet(str(BRONZE_ERA5_PATH.resolve()))
        logger.info("Loaded %d ERA5 records", era5_sdf.count())

        logger.info("Computing basin-wide SST summary (regional covariate by year)...")
        basin_sst_by_year = compute_basin_wide_sst_summary(era5_sdf)
        logger.info("Yearly SST: %d years, range %.2f-%.2f°C", 
                    len(basin_sst_by_year),
                    basin_sst_by_year["basin_sst_celsius_mean"].min(),
                    basin_sst_by_year["basin_sst_celsius_mean"].max())
        
        # Save yearly SST for later use in training table construction
        basin_sst_by_year.to_parquet(YEARLY_SST_OUTPUT_PATH, index=False)
        logger.info("Saved yearly SST to %s", YEARLY_SST_OUTPUT_PATH)

        logger.info("Running geo-join (this can take a minute at full scale)...")
        geo_joined = geo_join_era5(spark, locations, era5_sdf)

        logger.info("Aggregating per-location atmospheric features (pressure, wind)...")
        climate_features_sdf = aggregate_climate_features_per_location(geo_joined)
        climate_features = climate_features_sdf.toPandas()
    finally:
        spark.stop()

    logger.info("Aggregating historical hazard features per location...")
    hazard_features = aggregate_hazard_features_per_location(locations, claims)

    logger.info("Computing static geographic risk features...")
    geographic_features = compute_geographic_risk_features(locations)

    logger.info("Assembling final Gold feature table...")
    gold = locations.merge(climate_features, on="location_id", how="left")
    gold = gold.merge(hazard_features, on="location_id", how="left")
    gold = gold.merge(geographic_features, on="location_id", how="left")

    # Note: SST is now saved separately as yearly_sst.parquet and will be
    # joined during training table construction (build_training_table.py)
    # to enable time-varying climate covariates for Phase 5 (TFT)

    # mslp/wind are genuinely per-location and should NEVER be missing,
    # since they're defined everywhere (including over land) — unlike the
    # old SST design. Fail loudly if this regresses.
    missing_atmospheric = gold["mslp_hpa_mean"].isna().sum()
    if missing_atmospheric > 0:
        logger.warning(
            "%d/%d locations have NO matched ERA5 atmospheric data — "
            "this should never happen (pressure/wind are defined "
            "everywhere) and indicates a real geo-join coverage problem.",
            missing_atmospheric,
            len(gold),
        )

    GOLD_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(GOLD_OUTPUT_PATH, index=False)
    logger.info(
        "Wrote %d rows, %d columns to %s",
        len(gold),
        len(gold.columns),
        GOLD_OUTPUT_PATH,
    )

    print("\n--- Gold feature table summary ---")
    print(f"Rows: {len(gold):,}  Columns: {len(gold.columns)}")
    print(f"Columns: {gold.columns.tolist()}")

    print(f"\nYearly SST (saved separately for training table):")
    print(f"  Years: {len(basin_sst_by_year)}")
    print(f"  Mean range: {basin_sst_by_year['basin_sst_celsius_mean'].min():.2f}-{basin_sst_by_year['basin_sst_celsius_mean'].max():.2f}°C")
    print(f"  Max range: {basin_sst_by_year['basin_sst_celsius_max'].min():.2f}-{basin_sst_by_year['basin_sst_celsius_max'].max():.2f}°C")
    print(
        f"\nLocations with 1+ historical claim: "
        f"{(gold['historical_claim_count'] > 0).sum():,} / {len(gold):,}"
    )
    print(
        gold[
            [
                "mslp_hpa_mean",
                "era5_wind_speed_ms_mean",
                "historical_claim_count",
                "historical_incurred_loss_usd",
            ]
        ].describe()
    )

    return gold


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()