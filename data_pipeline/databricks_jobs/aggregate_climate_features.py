"""
Aggregate the per-timestamp ERA5 geo-join output (from geo_join_era5.py)
into per-location climate summary features.

DESIGN NOTE — one row per location, not per (location, timestamp):
we deliberately aggregate to one row per location rather than keeping the
full (location, timestamp) grain. At full grain, the Gold table would be
~20K locations x ~720 ERA5 timesteps = ~14.4M rows — too granular for a
per-location MODELING feature table (Phase 4's frequency-severity model
needs one feature vector per location, not a time series). Full-grain
climate data remains available in the Bronze/geo-joined layer if a later
phase needs it (e.g. the TFT in Phase 5, which explicitly wants a time
series).

DESIGN NOTE — SST is a BASIN-WIDE covariate, not a per-location feature:
Sea-surface temperature is, physically, only defined over ocean — ERA5
correctly returns NaN for every land grid cell. Since most of this
project's population-weighted locations are inland (Orlando, inland
Tampa/Miami, etc.), naively joining each location to its nearest grid
cell's SST produces ~70% missing values — discovered as a real bug while
running this against the full dataset (14,275/20,000 locations had no SST
match). The fix is not to patch around the missing values, but to
recognize the original per-location design was methodologically wrong:
SST's real predictive relevance to hurricane risk is as a BASIN-WIDE ocean
heat content signal (a warmer Gulf fuels more intense storms), not a
hyper-local property of any single address — no production cat model uses
"the SST at this house's front door" either. See
compute_basin_wide_sst_summary() below: SST is computed once as a Gulf-wide
regional covariate and applied uniformly to every location.
"""

from __future__ import annotations

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def aggregate_climate_features_per_location(geo_joined: SparkDataFrame) -> SparkDataFrame:
    """
    Per-location atmospheric features (pressure, wind — both genuinely
    defined everywhere, including over land, unlike SST).

    Args:
        geo_joined: output of geo_join_era5.geo_join_era5() — one row per
            (location_id, timestamp) with sst_celsius, mslp_hpa, wind_speed_ms.

    Returns:
        Spark DataFrame with one row per location_id and summary stats for
        mslp_hpa and wind_speed_ms across the full ERA5 time range available.
    """
    return geo_joined.groupBy("location_id").agg(
        F.mean("mslp_hpa").alias("mslp_hpa_mean"),
        F.min("mslp_hpa").alias("mslp_hpa_min"),
        F.mean("wind_speed_ms").alias("era5_wind_speed_ms_mean"),
        F.max("wind_speed_ms").alias("era5_wind_speed_ms_max"),
    )


def compute_basin_wide_sst_summary(era5_sdf: SparkDataFrame) -> dict:
    """
    Compute Gulf-wide SST summary statistics across all ocean grid cells
    with a valid (non-null, non-NaN) SST reading, over the full ERA5 time
    period available. This single summary is applied UNIFORMLY to every
    location in the Gold table (see run_gold_assembly.py) as a regional
    covariate, not a per-location feature.

    IMPORTANT: filters on BOTH isNotNull() AND ~isnan(). Land-cell SST
    values arriving from pandas/pyarrow can surface in Spark as IEEE NaN
    rather than SQL NULL — isNotNull() alone does NOT catch NaN (NaN is a
    valid float value in Spark's type system, not a null), so NaN silently
    poisons F.mean()/F.max() if not filtered explicitly. Confirmed as a
    real failure mode while building this: isNotNull()-only filtering let
    4/4 rows (including 2 NaN "land" rows) through, corrupting the mean to
    NaN instead of the correct ocean-only average.

    Args:
        era5_sdf: the RAW ERA5 Spark DataFrame (pre-geo-join) — operates
            on the full grid, not the location-joined subset, since this
            is a basin-wide statistic independent of any location.

    Returns:
        {"basin_sst_celsius_mean": float, "basin_sst_celsius_max": float}
    """
    valid_sst = era5_sdf.filter(
        F.col("sst_celsius").isNotNull() & ~F.isnan("sst_celsius")
    )
    row = valid_sst.agg(
        F.mean("sst_celsius").alias("basin_sst_celsius_mean"),
        F.max("sst_celsius").alias("basin_sst_celsius_max"),
    ).collect()[0]
    return {
        "basin_sst_celsius_mean": row["basin_sst_celsius_mean"],
        "basin_sst_celsius_max": row["basin_sst_celsius_max"],
    }