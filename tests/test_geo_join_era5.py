"""
Tests for data_pipeline.databricks_jobs.geo_join_era5.

Uses a real (local) SparkSession — small enough to run fast, but exercises
the actual join logic rather than mocking Spark away.
"""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.databricks_jobs.geo_join_era5 import (
    _nearest_grid_value,
    geo_join_era5,
    get_spark_session,
    snap_locations_to_era5_grid,
)


@pytest.fixture(scope="module")
def spark():
    session = get_spark_session(app_name="test-geo-join")
    yield session
    session.stop()


# --- _nearest_grid_value (pure numpy logic) ----------------------------------


def test_nearest_grid_value_exact_match():
    grid = np.array([25.0, 25.25, 25.5, 25.75])
    result = _nearest_grid_value(np.array([25.25]), grid)
    assert result[0] == pytest.approx(25.25)


def test_nearest_grid_value_rounds_to_closer_neighbor():
    grid = np.array([25.0, 25.25, 25.5])
    result = _nearest_grid_value(np.array([25.1]), grid)  # closer to 25.0 than 25.25
    assert result[0] == pytest.approx(25.0)

    result2 = _nearest_grid_value(np.array([25.2]), grid)  # closer to 25.25
    assert result2[0] == pytest.approx(25.25)


def test_nearest_grid_value_handles_values_outside_grid_range():
    """A location slightly outside the ERA5 bounding box should still snap
    to the nearest edge grid value, not error."""
    grid = np.array([25.0, 25.25, 25.5])
    result = _nearest_grid_value(np.array([24.5, 26.0]), grid)
    assert result[0] == pytest.approx(25.0)  # clamped to nearest edge
    assert result[1] == pytest.approx(25.5)


# --- snap_locations_to_era5_grid ---------------------------------------------


def test_snap_locations_adds_grid_columns():
    locations = pd.DataFrame({"location_id": ["L1", "L2"], "lat": [25.1, 26.4], "lon": [-81.9, -82.6]})
    grid_lats = np.array([25.0, 25.25, 26.25, 26.5])
    grid_lons = np.array([-82.0, -81.75, -82.5, -82.75])

    result = snap_locations_to_era5_grid(locations, grid_lats, grid_lons)
    assert "grid_lat" in result.columns
    assert "grid_lon" in result.columns
    assert len(result) == 2


# --- geo_join_era5 (real Spark execution) ------------------------------------


def test_geo_join_produces_one_row_per_location_per_timestamp(spark):
    locations = pd.DataFrame(
        {
            "location_id": ["L1", "L2"],
            "lat": [25.05, 26.45],  # close to grid points 25.0 and 26.5
            "lon": [-82.05, -82.55],
        }
    )
    era5_data = spark.createDataFrame(
        pd.DataFrame(
            {
                "lat": [25.0, 25.0, 26.5, 26.5],
                "lon": [-82.0, -82.0, -82.5, -82.5],
                "timestamp": pd.to_datetime(
                    ["2023-08-01", "2023-08-02", "2023-08-01", "2023-08-02"]
                ),
                "sst_celsius": [29.0, 29.5, 28.0, 28.2],
                "mslp_hpa": [1008.0, 1007.5, 1010.0, 1009.8],
                "wind_speed_ms": [8.0, 9.0, 5.0, 5.5],
            }
        )
    )

    result = geo_join_era5(spark, locations, era5_data).toPandas()

    # 2 locations x 2 timestamps each = 4 rows
    assert len(result) == 4
    assert set(result["location_id"].unique()) == {"L1", "L2"}


def test_geo_join_matches_correct_grid_cell_not_a_random_one(spark):
    """L1 is near (25.0, -82.0) — it must be joined to THAT grid cell's
    climate data, not the other, more distant one."""
    locations = pd.DataFrame({"location_id": ["L1"], "lat": [25.02], "lon": [-82.02]})
    era5_data = spark.createDataFrame(
        pd.DataFrame(
            {
                "lat": [25.0, 30.0],
                "lon": [-82.0, -87.0],
                "timestamp": pd.to_datetime(["2023-08-01", "2023-08-01"]),
                "sst_celsius": [29.0, 25.0],
                "mslp_hpa": [1008.0, 1015.0],
                "wind_speed_ms": [8.0, 3.0],
            }
        )
    )

    result = geo_join_era5(spark, locations, era5_data).toPandas()
    assert len(result) == 1
    assert result.iloc[0]["sst_celsius"] == pytest.approx(29.0)  # the NEAR cell's value
    assert result.iloc[0]["grid_lat"] == pytest.approx(25.0)
    assert result.iloc[0]["grid_lon"] == pytest.approx(-82.0)


def test_geo_join_output_schema(spark):
    locations = pd.DataFrame({"location_id": ["L1"], "lat": [25.0], "lon": [-82.0]})
    era5_data = spark.createDataFrame(
        pd.DataFrame(
            {
                "lat": [25.0],
                "lon": [-82.0],
                "timestamp": pd.to_datetime(["2023-08-01"]),
                "sst_celsius": [29.0],
                "mslp_hpa": [1008.0],
                "wind_speed_ms": [8.0],
            }
        )
    )
    result = geo_join_era5(spark, locations, era5_data)
    expected_cols = {
        "location_id",
        "lat",
        "lon",
        "grid_lat",
        "grid_lon",
        "timestamp",
        "sst_celsius",
        "mslp_hpa",
        "wind_speed_ms",
    }
    assert set(result.columns) == expected_cols