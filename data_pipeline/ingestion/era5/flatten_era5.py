"""
Flatten a gridded ERA5 NetCDF file into a tidy tabular DataFrame suitable
for the Bronze layer and, later, the Phase 3 geo-join to insured locations.

Kept separate from fetch_era5.py so the flattening logic — the part with
real bug risk (unit conversion, variable naming, wind-speed derivation) —
can be unit-tested against a small synthetic xarray Dataset with no network
dependency, per the same pattern used for the HURDAT2 parser.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# ERA5 short variable names as returned by the CDS API's NetCDF format.
_VAR_SST = "sst"
_VAR_MSLP = "msl"
_VAR_U10 = "u10"
_VAR_V10 = "v10"


def flatten_era5(ds: xr.Dataset) -> pd.DataFrame:
    """
    Convert a gridded ERA5 xarray Dataset into a flat DataFrame with columns
    [lat, lon, timestamp, sst_celsius, mslp_hpa, wind_speed_ms].

    Unit conversions applied (ERA5 native units -> project convention):
        - SST: Kelvin -> Celsius
        - Mean sea-level pressure: Pa -> hPa (matches HURDAT2's min_pressure_mb)
        - Wind: u10/v10 components (m/s) -> scalar wind speed (m/s) via
          magnitude, since the downstream vulnerability curve (Phase 2)
          consumes a single wind-speed value, not vector components.
    """
    df = ds.to_dataframe().reset_index()

    rename_map = {}

    if "latitude" in df.columns:
        rename_map["latitude"] = "lat"

    if "longitude" in df.columns:
        rename_map["longitude"] = "lon"

    # ERA5 CDS uses valid_time in recent NetCDF files
    if "valid_time" in df.columns:
        rename_map["valid_time"] = "timestamp"
    elif "time" in df.columns:
        rename_map["time"] = "timestamp"

    df = df.rename(columns=rename_map)
    df["sst_celsius"] = df[_VAR_SST] - 273.15
    df["mslp_hpa"] = df[_VAR_MSLP] / 100.0
    df["wind_speed_ms"] = np.sqrt(df[_VAR_U10] ** 2 + df[_VAR_V10] ** 2)

    out_cols = ["lat", "lon", "timestamp", "sst_celsius", "mslp_hpa", "wind_speed_ms"]
    result = df[out_cols].dropna(
        subset=["sst_celsius", "mslp_hpa", "wind_speed_ms"], how="all"
    )
    return result.reset_index(drop=True)


def flatten_era5_file(path) -> pd.DataFrame:
    from pathlib import Path

    path = Path(path).resolve()

    print("Opening:", path)
    print("Exists:", path.exists())
    print("Size:", path.stat().st_size)
    with xr.open_dataset(str(path), engine="netcdf4", cache=False) as ds:
        return flatten_era5(ds)
