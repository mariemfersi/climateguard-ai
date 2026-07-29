"""
Unit tests for data_pipeline.ingestion.hurdat2.parse_hurdat2.

Per Roadmap Task 1.1.1: "unit test validates parsing against a known sample
storm's known landfall record." These run against a local, synthetic-but-
format-correct fixture (tests/fixtures/hurdat2_sample.txt) so they require
no network access and are safe to run in CI or restricted-network sandboxes.
"""

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.ingestion.hurdat2.parse_hurdat2 import (
    _parse_latlon,
    parse_hurdat2,
    parse_hurdat2_file,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hurdat2_sample.txt"


@pytest.fixture
def parsed_df() -> pd.DataFrame:
    return parse_hurdat2_file(FIXTURE_PATH)


def test_parses_expected_row_count(parsed_df):
    # TESTSTORM has 4 observations, SECONDSTORM has 2 -> 6 total.
    assert len(parsed_df) == 6


def test_parses_expected_columns(parsed_df):
    expected = {
        "storm_id",
        "name",
        "timestamp",
        "status",
        "lat",
        "lon",
        "max_wind_kt",
        "min_pressure_mb",
    }
    assert expected.issubset(set(parsed_df.columns))


def test_storm_names_and_ids_correctly_associated(parsed_df):
    teststorm_rows = parsed_df[parsed_df["storm_id"] == "AL011999"]
    assert len(teststorm_rows) == 4
    assert (teststorm_rows["name"] == "TESTSTORM").all()

    secondstorm_rows = parsed_df[parsed_df["storm_id"] == "AL021999"]
    assert len(secondstorm_rows) == 2
    assert (secondstorm_rows["name"] == "SECONDSTORM").all()


def test_known_landfall_record_parsed_correctly(parsed_df):
    """
    The fixture's third TESTSTORM record (1999-08-01 12:00) is a landfall
    record ('L' record identifier) with a known status and intensity.
    This is the "known sample storm's known landfall record" check
    required by the roadmap.
    """
    landfall = parsed_df[
        (parsed_df["storm_id"] == "AL011999")
        & (parsed_df["timestamp"] == pd.Timestamp("1999-08-01 12:00", tz="UTC"))
    ]
    assert len(landfall) == 1
    row = landfall.iloc[0]

    assert row["status"] == "HU"
    assert row["max_wind_kt"] == 65
    assert row["min_pressure_mb"] == 985
    assert row["lat"] == pytest.approx(13.0)
    assert row["lon"] == pytest.approx(-52.0)


def test_intensification_trend_is_physically_sensible(parsed_df):
    """Sanity check: TESTSTORM strengthens monotonically from TD to HU across
    its 4 observations in the fixture, and pressure drops as wind increases —
    a basic physical consistency check, not just a schema check."""
    teststorm = parsed_df[parsed_df["storm_id"] == "AL011999"].sort_values("timestamp")
    winds = teststorm["max_wind_kt"].tolist()
    pressures = teststorm["min_pressure_mb"].tolist()

    assert winds == sorted(
        winds
    ), "wind speed should increase monotonically in this fixture"
    assert pressures == sorted(
        pressures, reverse=True
    ), "pressure should decrease monotonically in this fixture"


def test_lat_lon_hemisphere_parsing():
    assert _parse_latlon("20.9N") == pytest.approx(20.9)
    assert _parse_latlon("85.5W") == pytest.approx(-85.5)
    assert _parse_latlon("10.0S") == pytest.approx(-10.0)
    assert _parse_latlon("30.0E") == pytest.approx(30.0)


def test_data_record_before_header_raises():
    malformed = "20120829, 1200,  , TS, 20.9N,  85.5W,  45, 1000,\n"
    with pytest.raises(ValueError, match="data record encountered before"):
        parse_hurdat2(malformed)


def test_empty_input_returns_empty_dataframe():
    df = parse_hurdat2("")
    assert df.empty
