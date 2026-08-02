"""
Tests for data_pipeline.databricks_jobs.aggregate_climate_features and
aggregate_hazard_features.
"""

import pandas as pd
import pytest

from data_pipeline.databricks_jobs.aggregate_climate_features import (
    aggregate_climate_features_per_location,
    compute_basin_wide_sst_summary,
)
from data_pipeline.databricks_jobs.aggregate_hazard_features import (
    aggregate_hazard_features_per_location,
)
from data_pipeline.databricks_jobs.geo_join_era5 import get_spark_session


@pytest.fixture(scope="module")
def spark():
    session = get_spark_session(app_name="test-aggregation")
    yield session
    session.stop()


# --- aggregate_climate_features_per_location (mslp/wind only) ---------------


def test_climate_aggregation_produces_one_row_per_location(spark):
    geo_joined = spark.createDataFrame(
        pd.DataFrame(
            {
                "location_id": ["L1", "L1", "L2", "L2"],
                "mslp_hpa": [1008.0, 1006.0, 1012.0, 1010.0],
                "wind_speed_ms": [5.0, 7.0, 3.0, 4.0],
            }
        )
    )
    result = aggregate_climate_features_per_location(geo_joined).toPandas()
    assert len(result) == 2
    assert set(result["location_id"]) == {"L1", "L2"}


def test_climate_aggregation_mean_max_correct(spark):
    """L1's wind values are [5.0, 7.0] -> mean=6.0 — hand-verifiable."""
    geo_joined = spark.createDataFrame(
        pd.DataFrame(
            {
                "location_id": ["L1", "L1"],
                "mslp_hpa": [1008.0, 1006.0],
                "wind_speed_ms": [5.0, 7.0],
            }
        )
    )
    result = aggregate_climate_features_per_location(geo_joined).toPandas()
    row = result.iloc[0]
    assert row["era5_wind_speed_ms_mean"] == pytest.approx(6.0)
    assert row["mslp_hpa_min"] == pytest.approx(1006.0)


def test_climate_aggregation_never_has_missing_atmospheric_values(spark):
    """Regression guard for the real bug: mslp/wind (unlike SST) are
    defined everywhere including over land, so this must never produce
    missing values the way the old per-location SST design did."""
    geo_joined = spark.createDataFrame(
        pd.DataFrame(
            {
                "location_id": ["L1", "L2", "L3"],
                "mslp_hpa": [1008.0, 1012.0, 1005.0],
                "wind_speed_ms": [5.0, 3.0, 8.0],
            }
        )
    )
    result = aggregate_climate_features_per_location(geo_joined).toPandas()
    assert result["mslp_hpa_mean"].isna().sum() == 0
    assert result["era5_wind_speed_ms_mean"].isna().sum() == 0


# --- compute_basin_wide_sst_summary ------------------------------------------


def test_basin_wide_sst_ignores_null_land_cells(spark):
    """Regression test for the real bug: SST is NaN over land grid cells.
    The basin-wide summary must filter these out rather than let them
    corrupt the mean (which pandas/Spark would otherwise silently do
    inconsistently depending on null-handling)."""
    era5_sdf = spark.createDataFrame(
        pd.DataFrame(
            {
                "lat": [25.0, 26.0, 27.0, 28.0],
                "lon": [-82.0, -81.5, -80.0, -81.0],
                "sst_celsius": [28.0, 30.0, None, None],  # last two are "land"
            }
        )
    )
    result = compute_basin_wide_sst_summary(era5_sdf)
    assert result["basin_sst_celsius_mean"] == pytest.approx(29.0)  # mean(28, 30), NOT mean(28,30,0,0)
    assert result["basin_sst_celsius_max"] == pytest.approx(30.0)


def test_basin_wide_sst_is_a_single_scalar_summary(spark):
    """Confirms this returns ONE value for the whole basin, not per-location
    — the whole point of the fix."""
    era5_sdf = spark.createDataFrame(
        pd.DataFrame(
            {
                "lat": [25.0, 25.0, 26.0],
                "lon": [-82.0, -82.0, -81.0],
                "sst_celsius": [28.0, 29.0, 30.0],
            }
        )
    )
    result = compute_basin_wide_sst_summary(era5_sdf)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"basin_sst_celsius_mean", "basin_sst_celsius_max"}

# --- aggregate_hazard_features_per_location ----------------------------------


@pytest.fixture
def locations():
    return pd.DataFrame({"location_id": ["L1", "L2", "L3"]})  # L3 has no claims


@pytest.fixture
def claims():
    return pd.DataFrame(
        {
            "claim_id": ["C1", "C2", "C3"],
            "location_id": ["L1", "L1", "L2"],
            "storm_id": ["S1", "S2", "S1"],
            "max_wind_experienced_kt": [90.0, 120.0, 80.0],
            "damage_ratio": [0.1, 0.3, 0.05],
            "incurred_loss_usd": [10_000.0, 40_000.0, 5_000.0],
        }
    )


def test_hazard_aggregation_includes_all_locations_even_with_zero_claims(locations, claims):
    """L3 has no claims but MUST still appear in the output (with zeroed
    features), not be silently dropped — this is the core correctness
    requirement of the left-join design."""
    result = aggregate_hazard_features_per_location(locations, claims)
    assert set(result["location_id"]) == {"L1", "L2", "L3"}

    l3_row = result[result["location_id"] == "L3"].iloc[0]
    assert l3_row["historical_claim_count"] == 0
    assert l3_row["historical_incurred_loss_usd"] == 0
    assert l3_row["distinct_storms_experienced"] == 0


def test_hazard_aggregation_correctly_sums_and_counts(locations, claims):
    """L1 has 2 claims from 2 distinct storms, total incurred $50,000,
    max wind 120kt — hand-verifiable against the fixture."""
    result = aggregate_hazard_features_per_location(locations, claims)
    l1_row = result[result["location_id"] == "L1"].iloc[0]

    assert l1_row["historical_claim_count"] == 2
    assert l1_row["historical_incurred_loss_usd"] == pytest.approx(50_000.0)
    assert l1_row["historical_max_wind_kt"] == pytest.approx(120.0)
    assert l1_row["distinct_storms_experienced"] == 2
    assert l1_row["historical_mean_damage_ratio"] == pytest.approx(0.2)  # mean(0.1, 0.3)


def test_hazard_aggregation_single_claim_location(locations, claims):
    """L2 has 1 claim from 1 storm."""
    result = aggregate_hazard_features_per_location(locations, claims)
    l2_row = result[result["location_id"] == "L2"].iloc[0]

    assert l2_row["historical_claim_count"] == 1
    assert l2_row["distinct_storms_experienced"] == 1