"""
Prepare multi-horizon time series dataset for Temporal Fusion Transformer (TFT).

This script constructs a pytorch-forecasting TimeSeriesDataSet with:
- Static covariates: region (from Phase 4 regional encoding)
- Time-varying covariates: SST anomaly (from ERA5), ENSO index (from ONI)
- Target variables: hurricane frequency and severity per region-year

Grain: (region, year) - one time series per region with yearly observations.

Usage:
    python -m ml.tft_climate_trend.prepare_tft_dataset
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

from data_pipeline.databricks_jobs.compute_event_features import compute_regional_encoding
from data_pipeline.ingestion.common.bronze_writer import write_bronze

logger = logging.getLogger(__name__)

# Data paths
GOLD_FEATURES_PATH = Path("data_pipeline/gold/gold_features.parquet")
CLAIMS_PATH = Path("data_pipeline/silver/claims.parquet")
ERA5_PATH = Path("data_pipeline/bronze/era5/gulf_coast_reanalysis.parquet")
ONI_PATH = Path("data_pipeline/bronze/climate_indices/oni.parquet")

# Time series parameters
TRAINING_START_YEAR = 1950
TRAINING_END_YEAR = 2023
N_REGIONAL_CLUSTERS = 10

# TFT dataset parameters
ENCODER_LENGTH = 10  # Look back 10 years
PREDICTOR_LENGTH = 5  # Forecast 5 years ahead
BATCH_SIZE = 64


def compute_regional_climate_summary(
    era5_path: Path,
    oni_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute yearly regional climate summaries from ERA5 and ONI data.
    
    Reads ERA5 data incrementally and aggregates per batch to avoid memory error.
    
    Args:
        era5_path: Path to ERA5 parquet file
        oni_df: ONI data with columns [year, season, oni_anomaly_celsius, enso_phase]
    
    Returns:
        DataFrame with columns [year, basin_sst_celsius_mean, basin_sst_celsius_max,
                               oni_anomaly_celsius, enso_phase]
    """
    logger.info("Computing regional climate summaries from ERA5 (incremental aggregation)...")
    
    # Read ERA5 in batches and aggregate incrementally
    import pyarrow.dataset as ds
    
    era5_dataset = ds.dataset(
        era5_path,
        format="parquet",
    )
    
    # Read in batches and aggregate incrementally
    batch_size = 5_000_000  # 5M rows per batch
    yearly_sst_sums = {}
    yearly_sst_max = {}
    yearly_counts = {}
    
    for batch in era5_dataset.to_table(columns=["timestamp", "sst_celsius"]).to_batches(max_chunksize=batch_size):
        batch_df = batch.to_pandas()
        batch_df["year"] = pd.to_datetime(batch_df["timestamp"]).dt.year
        
        # Filter to training year range
        batch_df = batch_df[
            (batch_df["year"] >= TRAINING_START_YEAR) &
            (batch_df["year"] <= TRAINING_END_YEAR)
        ]
        
        # Aggregate per year in this batch
        for year, group in batch_df.groupby("year"):
            if year not in yearly_sst_sums:
                yearly_sst_sums[year] = 0
                yearly_sst_max[year] = 0
                yearly_counts[year] = 0
            
            yearly_sst_sums[year] += group["sst_celsius"].sum()
            yearly_sst_max[year] = max(yearly_sst_max[year], group["sst_celsius"].max())
            yearly_counts[year] += len(group)
    
    # Compute final aggregates
    sst_yearly_data = []
    for year in sorted(yearly_sst_sums.keys()):
        sst_yearly_data.append({
            "year": year,
            "basin_sst_celsius_mean": yearly_sst_sums[year] / yearly_counts[year],
            "basin_sst_celsius_max": yearly_sst_max[year],
        })
    
    sst_yearly = pd.DataFrame(sst_yearly_data)
    
    # Ensure full year coverage across the training range: fill missing years
    all_years = pd.DataFrame({"year": list(range(TRAINING_START_YEAR, TRAINING_END_YEAR + 1))})
    sst_yearly_full = all_years.merge(sst_yearly, on="year", how="left")

    # Track which years were missing from ERA5 and will be filled/interpolated
    missing_years = sorted(list(set(all_years["year"]) - set(sst_yearly["year"])))
    if missing_years:
        logger.info(f"ERA5 missing years detected and will be filled/interpolated: {missing_years}")

    # Interpolate mean SST, and forward/backward fill extremes and any remaining gaps
    sst_yearly_full["basin_sst_celsius_mean"] = (
        sst_yearly_full["basin_sst_celsius_mean"].interpolate(method="linear").ffill().bfill()
    )
    sst_yearly_full["basin_sst_celsius_max"] = (
        sst_yearly_full["basin_sst_celsius_max"].ffill().bfill()
    )

    # Aggregate ONI to yearly values (use SON season as representative)
    oni_yearly = oni_df[oni_df["season"] == "SON"][['year', 'oni_anomaly_celsius', 'enso_phase']]

    # Merge SST and ONI
    climate_summary = sst_yearly_full.merge(oni_yearly, on="year", how="left")
    # Fill ONI gaps via interpolation/ffill/bfill
    climate_summary["oni_anomaly_celsius"] = (
        climate_summary["oni_anomaly_celsius"].interpolate(method="linear").ffill().bfill()
    )
    
    logger.info(f"Climate summary: {len(climate_summary)} years ({climate_summary['year'].min()}-{climate_summary['year'].max()})")
    
    return climate_summary


def build_regional_time_series(
    gold_features: pd.DataFrame,
    claims: pd.DataFrame,
    climate_summary: pd.DataFrame,
    n_clusters: int = N_REGIONAL_CLUSTERS,
) -> pd.DataFrame:
    """
    Build regional time series dataset for TFT.
    
    Args:
        gold_features: Gold feature table (one row per location)
        claims: Claims data (one row per claim event)
        climate_summary: Yearly climate summaries
        n_clusters: Number of regional clusters
    
    Returns:
        DataFrame with columns [region_id, year, frequency, severity,
                               basin_sst_celsius_mean, basin_sst_celsius_max,
                               oni_anomaly_celsius, enso_phase]
    """
    logger.info("Building regional time series...")
    
    # Compute regional encoding
    regional_encodings = compute_regional_encoding(
        gold_features[["location_id", "lat", "lon"]],
        n_clusters=n_clusters,
    )
    
    # Merge regional encoding to gold features
    gold_with_region = gold_features.merge(regional_encodings, on="location_id", how="left")
    
    # Aggregate claims to (location_id, year) grain
    claims["year"] = pd.to_datetime(claims["loss_date"]).dt.year
    claims_yearly = claims[
        (claims["year"] >= TRAINING_START_YEAR) &
        (claims["year"] <= TRAINING_END_YEAR)
    ].groupby(["location_id", "year"]).agg(
        frequency=("claim_id", "count"),
        severity=("damage_ratio", "mean"),
    ).reset_index()
    
    # Merge claims to gold features with regional encoding
    regional_claims = gold_with_region.merge(
        claims_yearly,
        on="location_id",
        how="left",
    )
    
    # Fill missing years with zero frequency/severity
    years = np.arange(TRAINING_START_YEAR, TRAINING_END_YEAR + 1)
    region_ids = regional_encodings["region_cluster"].unique()
    
    # Create complete (region, year) grid
    grid = pd.MultiIndex.from_product(
        [region_ids, years],
        names=["region_id", "year"]
    ).to_frame(index=False)
    
    # Convert region_id to string for consistency
    grid["region_id"] = grid["region_id"].astype(str)
    
    # Aggregate to regional level
    regional_ts = regional_claims.groupby(["region_cluster", "year"]).agg(
        frequency=("frequency", "sum"),
        severity=("severity", "mean"),
    ).reset_index()
    regional_ts = regional_ts.rename(columns={"region_cluster": "region_id"})
    
    # Convert region_id to string for consistency
    regional_ts["region_id"] = regional_ts["region_id"].astype(str)
    
    # Merge with grid to ensure all years present
    regional_ts = grid.merge(regional_ts, on=["region_id", "year"], how="left")
    regional_ts["frequency"] = regional_ts["frequency"].fillna(0)
    regional_ts["severity"] = regional_ts["severity"].fillna(0)
    
    # Merge with climate summaries
    regional_ts = regional_ts.merge(climate_summary, on="year", how="left")
    
    # Fill missing climate values with forward fill then backward fill
    # (handles years not in ERA5 backfill, e.g., 2023)
    # Use groupby to prevent cross-region contamination
    climate_cols = ["basin_sst_celsius_mean", "basin_sst_celsius_max", "oni_anomaly_celsius", "enso_phase"]
    for col in climate_cols:
        regional_ts[col] = regional_ts.groupby("region_id")[col].transform(lambda s: s.ffill().bfill())
    
    logger.info(f"Regional time series: {len(regional_ts)} rows ({len(region_ids)} regions × {len(years)} years)")
    
    return regional_ts


def stratified_year_split_tft(
    regional_ts: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split regional time series into train/test sets by year to prevent temporal leakage.
    
    Ensures no year appears in both train and test sets, which is critical for
    time series forecasting to avoid data leakage.
    
    Args:
        regional_ts: Regional time series DataFrame with region_id and year columns
        test_size: Fraction of years to reserve for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df) with disjoint year sets
    """
    import numpy as np
    
    unique_years = sorted(regional_ts["year"].unique())
    n_years = len(unique_years)
    
    if n_years < 2:
        raise ValueError("At least two different years required for splitting")
    
    # Calculate number of test years
    n_test_years = max(1, int(n_years * test_size))
    
    # Randomly select test years
    rng = np.random.default_rng(seed)
    test_years = rng.choice(unique_years, size=n_test_years, replace=False)
    
    # Split data
    test_df = regional_ts[regional_ts["year"].isin(test_years)].copy()
    train_df = regional_ts[~regional_ts["year"].isin(test_years)].copy()
    
    logger.info(f"Temporal split: {len(train_df)} train samples, {len(test_df)} test samples")
    logger.info(f"Train years: {sorted(train_df['year'].unique())}")
    logger.info(f"Test years: {sorted(test_df['year'].unique())}")
    
    return train_df, test_df


def assert_no_year_leakage_tft(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Assert that train and test sets have no overlapping years.
    
    Args:
        train_df: Training DataFrame with year column
        test_df: Test DataFrame with year column
    
    Raises:
        AssertionError: If any year appears in both train and test sets
    """
    train_years = set(train_df["year"].unique())
    test_years = set(test_df["year"].unique())
    
    overlap = train_years & test_years
    if overlap:
        raise AssertionError(
            f"Year leakage detected: {overlap} years appear in both train and test sets. "
            "This violates temporal independence for time series forecasting."
        )


def prepare_tft_dataset(
    regional_ts: pd.DataFrame,
    encoder_length: int = ENCODER_LENGTH,
    predictor_length: int = PREDICTOR_LENGTH,
) -> TimeSeriesDataSet:
    """
    Prepare pytorch-forecasting TimeSeriesDataSet for TFT.
    
    Args:
        regional_ts: Regional time series DataFrame
        encoder_length: Look-back window (configured on TimeSeriesDataSet)
        predictor_length: Forecast horizon (configured on TimeSeriesDataSet)
    
    Returns:
        TimeSeriesDataSet configured for TFT
    """
    logger.info("Preparing TFT TimeSeriesDataSet...")
    
    # Define static covariates (region-level features that don't change over time)
    static_categoricals = ["region_id"]
    
    # Define time-varying known categoricals (known in advance for forecast horizon)
    time_varying_known_categoricals = []
    
    # Define time-varying unknown categoricals (not known in advance)
    time_varying_unknown_categoricals = ["enso_phase"]
    
    # Define time-varying known reals (known in advance for forecast horizon)
    # NOTE: Climate covariates (SST, ONI) are NOT known for future years at prediction time.
    # Marking them as 'unknown' to prevent data leakage. In production, these would need
    # to be forecasted separately or treated as lagged features only.
    time_varying_known_reals = []
    
    # Define time-varying unknown reals (not known in advance)
    time_varying_unknown_reals = [
        "basin_sst_celsius_mean",
        "basin_sst_celsius_max",
        "oni_anomaly_celsius",
    ]
    
    # Define target variables
    target = ["frequency", "severity"]
    
    # Create TimeSeriesDataSet
    dataset = TimeSeriesDataSet(
        regional_ts,
        time_idx="year",
        target=target,
        group_ids=["region_id"],
        static_categoricals=static_categoricals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        min_encoder_length=encoder_length // 2,   # allow some flexibility; or set equal to encoder_length for fixed windows
        max_encoder_length=encoder_length,
        min_prediction_length=predictor_length,
        max_prediction_length=predictor_length,
    )
    
    logger.info(f"TFT dataset created: {len(dataset)} samples")
    
    return dataset


def run_dataset_preparation(
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, Path]:
    """
    Run full dataset preparation pipeline with temporal split.
    
    Args:
        test_size: Fraction of years to reserve for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, test_dataset, path to saved regional time series)
    """
    # Load data
    logger.info("Loading input data...")
    gold_features = pd.read_parquet(GOLD_FEATURES_PATH)
    claims = pd.read_parquet(CLAIMS_PATH)
    oni_df = pd.read_parquet(ONI_PATH)
    
    # Compute climate summaries (ERA5 read incrementally)
    climate_summary = compute_regional_climate_summary(ERA5_PATH, oni_df)
    
    # Build regional time series
    regional_ts = build_regional_time_series(gold_features, claims, climate_summary)
    
    # Save regional time series to Bronze
    ts_path = Path("data_pipeline/bronze/tft_climate_trend/regional_time_series.parquet")
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Delete existing file to avoid schema mismatch
    if ts_path.exists():
        ts_path.unlink()
    
    regional_ts.to_parquet(
        ts_path,
        index=False,
        engine="pyarrow",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    
    logger.info(f"Regional time series saved to {ts_path}")
    
    # Perform temporal split to prevent year leakage
    train_ts, test_ts = stratified_year_split_tft(regional_ts, test_size=test_size, seed=seed)
    assert_no_year_leakage_tft(train_ts, test_ts)
    
    # Prepare TFT datasets
    train_dataset = prepare_tft_dataset(train_ts)
    test_dataset = prepare_tft_dataset(test_ts)
    
    return train_dataset, test_dataset, ts_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_dataset, test_dataset, ts_path = run_dataset_preparation()
    logger.info(f"Dataset preparation complete. Time series saved to {ts_path}")
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")
