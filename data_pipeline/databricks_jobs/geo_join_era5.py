"""
Geo-join: attach the nearest ERA5 grid cell's climate features to each
synthetic insured location, for every ERA5 timestep.

COST/INFRASTRUCTURE NOTE:
Written as a standard PySpark job (runnable unchanged on Azure Databricks)
but executed via LOCAL PySpark (`SparkSession.builder.master("local[*]")`)
rather than an actual Databricks cluster. At this project's data volume
(20K locations x ~11.9M ERA5 rows), a Databricks cluster is genuine
overkill and a real risk to a limited student-credit budget. See
docs/architecture_decisions.md for the full rationale. To port this to a
real Databricks job: change `.master("local[*]")` to the cluster's config
(or remove it entirely — Databricks notebooks provide `spark`
automatically) and point the read/write paths at `abfss://...` instead of
local disk / `az://` blob URIs.

METHODOLOGY NOTE — nearest-grid-cell matching:
Rather than assuming ERA5's grid spacing and origin offset (risky to
hardcode — a wrong assumption would silently produce a slightly-wrong
join), this module snaps each location to the ACTUAL nearest ERA5 grid
coordinate present in the real fetched data. Since ERA5's distinct
lat/lon grid values are few (~100 x ~160 for the Gulf Coast bounding box),
this snap is done efficiently on the driver via numpy binary search
(`np.searchsorted`), then the join itself runs in Spark.
"""

from __future__ import annotations

import logging
import os
import sys

# PySpark 3.5.x internally imports `distutils`, which was removed from the
# Python standard library in 3.12 — this is a real, confirmed upstream
# incompatibility (not fixed as of PySpark 3.5.9), producing
# `ModuleNotFoundError: No module named 'distutils'` on any pyspark
# DataFrame operation. setuptools ships a compatible shim; this activates
# it BEFORE pyspark is imported below. Harmless no-op on Python <3.12
# (this project's pinned version is 3.11, where distutils still exists
# natively — this shim exists purely as defensive robustness in case this
# code ever runs on 3.12+).
if sys.version_info >= (3, 12):
    os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
    import setuptools  # noqa: F401  (import triggers the shim registration)

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def get_spark_session(app_name: str = "climateguard-geo-join") -> SparkSession:
    """
    Local PySpark session — see module docstring for the cost rationale.
    `local[*]` uses all available cores on this machine.

    WINDOWS FIX: PySpark launches worker subprocesses via the bare `python`
    command by default. On many Windows setups, `python`/`python3` on PATH
    resolve to a non-functional Microsoft Store stub rather than a real
    interpreter, causing every worker launch to fail with
    "Python worker failed to connect back" (a socket-accept timeout, not a
    logic bug). Explicitly pointing PYSPARK_PYTHON /
    PYSPARK_DRIVER_PYTHON at `sys.executable` (the interpreter actually
    running this code — i.e. the active venv's real python.exe) fixes this
    reliably and is a no-op on Linux/Mac where this issue doesn't occur.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config(
            "spark.sql.shuffle.partitions", "8"
        )  # low, since this is a small local job
        .getOrCreate()
    )

def _nearest_grid_value(targets: np.ndarray, grid_values: np.ndarray) -> np.ndarray:
    """
    For each value in `targets`, find the nearest value in the sorted array
    `grid_values`, via binary search (O(log n) per target, not a full
    pairwise distance computation).
    """
    grid_sorted = np.sort(np.unique(grid_values))
    idx = np.searchsorted(grid_sorted, targets)
    idx = np.clip(idx, 1, len(grid_sorted) - 1)

    left = grid_sorted[idx - 1]
    right = grid_sorted[idx]
    use_left = np.abs(targets - left) <= np.abs(targets - right)
    return np.where(use_left, left, right)


def snap_locations_to_era5_grid(
    locations: pd.DataFrame,
    era5_grid_lats: np.ndarray,
    era5_grid_lons: np.ndarray,
) -> pd.DataFrame:
    """
    Add grid_lat/grid_lon columns to `locations`, snapped to the nearest
    actual ERA5 grid coordinates.

    Args:
        locations: must have 'lat', 'lon' columns.
        era5_grid_lats, era5_grid_lons: the DISTINCT lat/lon values actually
            present in the ERA5 dataset (get via era5_df[['lat']].distinct()
            collected to the driver — small, ~100-160 values).

    Returns:
        locations with two new columns: grid_lat, grid_lon.
    """
    result = locations.copy()
    result["grid_lat"] = _nearest_grid_value(locations["lat"].to_numpy(), era5_grid_lats)
    result["grid_lon"] = _nearest_grid_value(locations["lon"].to_numpy(), era5_grid_lons)
    return result


def geo_join_era5(
    spark: SparkSession,
    locations: pd.DataFrame,
    era5_sdf: SparkDataFrame,
) -> SparkDataFrame:
    """
    Join every location to its nearest ERA5 grid cell's climate features,
    for every ERA5 timestep present in era5_sdf.

    Args:
        spark: active SparkSession.
        locations: pandas DataFrame with ['location_id', 'lat', 'lon'].
        era5_sdf: Spark DataFrame with
            ['lat', 'lon', 'timestamp', 'sst_celsius', 'mslp_hpa', 'wind_speed_ms']
            (the flattened ERA5 schema from Milestone 1.2).

    Returns:
        Spark DataFrame: one row per (location_id, timestamp), with the
        matched climate features and the snapped grid_lat/grid_lon used for
        auditability.
    """
    distinct_grid = era5_sdf.select("lat", "lon").distinct().toPandas()
    era5_grid_lats = distinct_grid["lat"].to_numpy()
    era5_grid_lons = distinct_grid["lon"].to_numpy()

    n_unique_lat = len(np.unique(era5_grid_lats))
    n_unique_lon = len(np.unique(era5_grid_lons))
    logger.info(
        "ERA5 grid has %d distinct lat values, %d distinct lon values", n_unique_lat, n_unique_lon
    )

    locations_snapped = snap_locations_to_era5_grid(locations, era5_grid_lats, era5_grid_lons)

    locations_sdf = spark.createDataFrame(
        locations_snapped[["location_id", "lat", "lon", "grid_lat", "grid_lon"]]
    )
    # locations is small (thousands of rows) — broadcast it explicitly so
    # Spark doesn't shuffle the much larger ERA5 side of the join.
    from pyspark.sql.functions import broadcast

    era5_renamed = era5_sdf.withColumnRenamed("lat", "grid_lat").withColumnRenamed(
        "lon", "grid_lon"
    )

    joined = broadcast(locations_sdf).join(era5_renamed, on=["grid_lat", "grid_lon"], how="inner")

    n_locations = len(locations)
    matched_locations = joined.select("location_id").distinct().count()
    unmatched_pct = 100.0 * (1 - matched_locations / n_locations)
    logger.info(
        "Geo-join matched %d/%d locations to at least one ERA5 record (%.2f%% unmatched)",
        matched_locations,
        n_locations,
        unmatched_pct,
    )
    if unmatched_pct > 5.0:
        logger.warning(
            "More than 5%% of locations failed to match any ERA5 grid cell — "
            "check that the ERA5 bounding box actually covers all location coordinates."
        )

    return joined.select(
        "location_id",
        "lat",
        "lon",
        "grid_lat",
        "grid_lon",
        "timestamp",
        "sst_celsius",
        "mslp_hpa",
        "wind_speed_ms",
    )