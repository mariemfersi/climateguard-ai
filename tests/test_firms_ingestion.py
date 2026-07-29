"""
Unit tests for data_pipeline.ingestion.firms.

fetch_firms_raw is tested with a mocked HTTP response (no network, no API
key required). normalize_firms is tested against known raw FIRMS-shaped
rows, including the acq_time zero-padding edge case that's a classic
off-by-a-digit bug source (e.g. acq_time=907 means 09:07, not 90:70 or 9:70).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_pipeline.ingestion.firms.fetch_firms import MAX_DAY_RANGE, fetch_firms_raw
from data_pipeline.ingestion.firms.normalize_firms import normalize_firms

FAKE_FIRMS_CSV = (
    "latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,frp\n"
    "27.5,-82.4,2023-08-01,907,N,VIIRS,high,12.3\n"
    "26.1,-81.8,2023-08-01,1345,N,VIIRS,nominal,45.7\n"
)


def _mock_settings():
    mock = MagicMock()
    mock.require.return_value = "dummy-firms-key"
    return mock


# --- fetch_firms_raw ---------------------------------------------------------


def test_fetch_firms_raw_parses_csv_response():
    with (
        patch("data_pipeline.ingestion.firms.fetch_firms.requests.get") as mock_get,
        patch("data_pipeline.ingestion.firms.fetch_firms.get_settings", return_value=_mock_settings()),
    ):
        mock_resp = MagicMock()
        mock_resp.text = FAKE_FIRMS_CSV
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        df = fetch_firms_raw()

        assert len(df) == 2
        assert "latitude" in df.columns
        assert "acq_time" in df.columns


def test_fetch_firms_raw_rejects_invalid_day_range():
    with pytest.raises(ValueError, match="day_range must be between 1 and"):
        fetch_firms_raw(day_range=0)
    with pytest.raises(ValueError, match="day_range must be between 1 and"):
        fetch_firms_raw(day_range=MAX_DAY_RANGE + 1)


def test_fetch_firms_raw_raises_on_api_error_text():
    """FIRMS returns error messages as 200-OK plain text, not HTTP error
    codes — must be detected explicitly or a bad key silently 'succeeds'
    with zero rows."""
    with (
        patch("data_pipeline.ingestion.firms.fetch_firms.requests.get") as mock_get,
        patch("data_pipeline.ingestion.firms.fetch_firms.get_settings", return_value=_mock_settings()),
    ):
        mock_resp = MagicMock()
        mock_resp.text = "Invalid MAP_KEY"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with pytest.raises(RuntimeError, match="FIRMS API returned an error"):
            fetch_firms_raw()


def test_fetch_firms_raw_handles_empty_response():
    with (
        patch("data_pipeline.ingestion.firms.fetch_firms.requests.get") as mock_get,
        patch("data_pipeline.ingestion.firms.fetch_firms.get_settings", return_value=_mock_settings()),
    ):
        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        df = fetch_firms_raw()
        assert df.empty


# --- normalize_firms ---------------------------------------------------------


def test_normalize_firms_produces_expected_schema():
    raw = pd.read_csv(pd.io.common.StringIO(FAKE_FIRMS_CSV))
    df = normalize_firms(raw)

    assert {"lat", "lon", "timestamp"}.issubset(df.columns)
    assert len(df) == 2


def test_normalize_firms_acq_time_zero_padding_is_correct():
    """acq_time=907 must parse as 09:07, not 90:70 or 9:07-misaligned —
    this is the classic bug this test guards against."""
    raw = pd.read_csv(pd.io.common.StringIO(FAKE_FIRMS_CSV))
    df = normalize_firms(raw)

    first_row = df.iloc[0]
    assert first_row["timestamp"].hour == 9
    assert first_row["timestamp"].minute == 7

    second_row = df.iloc[1]
    assert second_row["timestamp"].hour == 13
    assert second_row["timestamp"].minute == 45


def test_normalize_firms_timestamp_is_utc_localized():
    raw = pd.read_csv(pd.io.common.StringIO(FAKE_FIRMS_CSV))
    df = normalize_firms(raw)
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_normalize_firms_handles_empty_dataframe():
    df = normalize_firms(pd.DataFrame())
    assert df.empty
    assert {"lat", "lon", "timestamp"}.issubset(df.columns)