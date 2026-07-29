"""
Unit tests for data_pipeline.ingestion.fred.fetch_fred.

Mocks the FRED HTTP response so tests run with no network access and no
API key required, matching the pattern used for HURDAT2/ERA5 tests.
"""

from unittest.mock import MagicMock, patch

import pytest

from data_pipeline.ingestion.fred.fetch_fred import (
    DEFAULT_SERIES,
    fetch_all_default_series,
    fetch_fred_series,
)

FAKE_FRED_RESPONSE = {
    "observations": [
        {"date": "2023-01-01", "value": "300.5"},
        {"date": "2023-02-01", "value": "301.2"},
        {"date": "2023-03-01", "value": "."},  # FRED's missing-value sentinel
    ]
}


def _mock_settings():
    mock = MagicMock()
    mock.require.return_value = "dummy-fred-key"
    return mock


def test_fetch_fred_series_parses_response_correctly():
    with (
        patch("data_pipeline.ingestion.fred.fetch_fred.requests.get") as mock_get,
        patch("data_pipeline.ingestion.fred.fetch_fred.get_settings", return_value=_mock_settings()),
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_FRED_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        df = fetch_fred_series("CPIAUCSL", "cpi_all_urban")

        assert len(df) == 3
        assert list(df.columns) == ["series_id", "series_name", "date", "value"]
        assert (df["series_id"] == "CPIAUCSL").all()
        assert (df["series_name"] == "cpi_all_urban").all()


def test_fetch_fred_series_converts_missing_value_sentinel_to_nan():
    """FRED encodes missing observations as the literal string '.' — this
    must become NaN, not silently become 0.0 or a parse error."""
    with (
        patch("data_pipeline.ingestion.fred.fetch_fred.requests.get") as mock_get,
        patch("data_pipeline.ingestion.fred.fetch_fred.get_settings", return_value=_mock_settings()),
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_FRED_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        df = fetch_fred_series("CPIAUCSL", "cpi_all_urban")

        assert df["value"].isna().sum() == 1
        assert df.loc[df["date"] == "2023-03-01", "value"].isna().all()
        assert df["value"].iloc[0] == pytest.approx(300.5)


def test_fetch_fred_series_raises_on_http_error():
    with (
        patch("data_pipeline.ingestion.fred.fetch_fred.requests.get") as mock_get,
        patch("data_pipeline.ingestion.fred.fetch_fred.get_settings", return_value=_mock_settings()),
    ):
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_resp

        with pytest.raises(requests.HTTPError):
            fetch_fred_series("INVALID_SERIES", "bad_series")


def test_fetch_all_default_series_covers_all_configured_series():
    with (
        patch("data_pipeline.ingestion.fred.fetch_fred.requests.get") as mock_get,
        patch("data_pipeline.ingestion.fred.fetch_fred.get_settings", return_value=_mock_settings()),
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_FRED_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        df = fetch_all_default_series()

        assert set(df["series_name"].unique()) == set(DEFAULT_SERIES.keys())
        assert len(df) == 3 * len(DEFAULT_SERIES)