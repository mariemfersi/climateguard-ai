"""
Unit tests for data_pipeline.ingestion.era5.

flatten_era5 is tested against a small synthetic xarray Dataset built
in-memory — no network access or real ERA5 download required, matching the
pattern used for the HURDAT2 parser tests.

fetch_era5_raw's *request construction* (bounding box, variable list) is
tested with a mocked cdsapi.Client so we verify we're asking CDS for the
right, tightly-scoped data without actually calling the live API.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from data_pipeline.ingestion.era5.fetch_era5 import GULF_COAST_BBOX, fetch_era5_raw
from data_pipeline.ingestion.era5.flatten_era5 import flatten_era5


@pytest.fixture
def synthetic_era5_dataset() -> xr.Dataset:
    """A tiny 2x2 lat/lon x 2 time-step synthetic ERA5-shaped dataset."""
    lats = [20.0, 25.0]
    lons = [-90.0, -85.0]
    times = np.array(
        ["2023-08-01T00:00:00", "2023-08-01T06:00:00"], dtype="datetime64[ns]"
    )

    shape = (len(times), len(lats), len(lons))
    sst_kelvin = 301.15 + np.random.rand(*shape)  # ~28C
    mslp_pa = 100800 + np.random.rand(*shape) * 100
    u10 = np.full(shape, 3.0)
    v10 = np.full(shape, 4.0)  # magnitude should come out to 5.0 m/s

    return xr.Dataset(
        {
            "sst": (("time", "latitude", "longitude"), sst_kelvin),
            "msl": (("time", "latitude", "longitude"), mslp_pa),
            "u10": (("time", "latitude", "longitude"), u10),
            "v10": (("time", "latitude", "longitude"), v10),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def test_flatten_era5_produces_expected_row_count(synthetic_era5_dataset):
    df = flatten_era5(synthetic_era5_dataset)
    # 2 timesteps x 2 lats x 2 lons = 8 rows
    assert len(df) == 8


def test_flatten_era5_expected_columns(synthetic_era5_dataset):
    df = flatten_era5(synthetic_era5_dataset)
    assert set(df.columns) == {
        "lat",
        "lon",
        "timestamp",
        "sst_celsius",
        "mslp_hpa",
        "wind_speed_ms",
    }


def test_flatten_era5_unit_conversions_correct(synthetic_era5_dataset):
    df = flatten_era5(synthetic_era5_dataset)

    # SST: 301.15 K (+jitter) -> ~28.0 C (+jitter); should be well within
    # a physically plausible Gulf-of-Mexico summer SST range.
    assert df["sst_celsius"].between(27.0, 30.0).all()

    # MSLP: ~100800-100900 Pa -> ~1008-1009 hPa.
    assert df["mslp_hpa"].between(1007.0, 1010.0).all()

    # Wind: u=3, v=4 -> magnitude 5.0 exactly (classic 3-4-5 triangle).
    assert df["wind_speed_ms"].apply(lambda w: w == pytest.approx(5.0)).all()


def test_fetch_era5_uses_gulf_coast_bbox_not_global():
    """Guards against the roadmap's flagged risk: accidentally requesting
    global-extent ERA5 data instead of the scoped Gulf Coast bounding box."""
    with patch(
        "data_pipeline.ingestion.era5.fetch_era5.cdsapi.Client"
    ) as MockClient, patch(
        "data_pipeline.ingestion.era5.fetch_era5.get_settings"
    ) as mock_settings:
        mock_settings.return_value.cds_api_key = "dummy-key"
        mock_settings.return_value.cds_api_url = "https://dummy"
        mock_settings.return_value.require.return_value = "dummy-key"

        mock_client_instance = MagicMock()
        MockClient.return_value = mock_client_instance

        fetch_era5_raw(
            year="2023",
            months=["08"],
            dest=__import__("pathlib").Path("/tmp/test_era5.nc"),
        )

        _, request_body, _ = mock_client_instance.retrieve.call_args[0]
        assert request_body["area"] == GULF_COAST_BBOX
        assert "sea_surface_temperature" in request_body["variable"]
