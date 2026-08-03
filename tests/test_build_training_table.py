"""
Tests for ml.frequency_severity.build_training_table.
"""

import pandas as pd
import pytest

from ml.frequency_severity.build_training_table import (
    FEATURE_COLUMNS,
    build_training_table,
)


@pytest.fixture
def gold_features():
    return pd.DataFrame(
        {
            "location_id": ["L1", "L2"],
            "lat": [26.6, 28.5],
            "lon": [-81.9, -81.4],
            "metro_center": ["Lee (Fort Myers)", "Orange (Orlando)"],
            "year_built": [1998, 2015],
            "construction_class": ["frame", "masonry_cbs"],
            "roof_type": ["gable", "hip"],
            "tiv_usd": [250_000.0, 400_000.0],
            "mslp_hpa_mean": [1014.9, 1015.1],
            "mslp_hpa_min": [1014.4, 1014.6],
            "era5_wind_speed_ms_mean": [3.1, 2.9],
            "era5_wind_speed_ms_max": [5.0, 4.8],
            "basin_sst_celsius_mean": [28.9, 28.9],
            "basin_sst_celsius_max": [35.9, 35.9],
        }
    )


@pytest.fixture
def claims():
    return pd.DataFrame(
        {
            "claim_id": ["C1", "C2", "C3"],
            "location_id": ["L1", "L1", "L2"],
            "loss_date": pd.to_datetime(["1992-08-24", "2004-08-13", "2022-09-28"]),
            "incurred_loss_usd": [80_000.0, 20_000.0, 100_000.0],
            "damage_ratio": [0.32, 0.08, 0.25],
        }
    )


def test_build_training_table_shape(gold_features, claims):
    result = build_training_table(gold_features, claims, start_year=2000, end_year=2004)
    # 2 locations x 5 years = 10 rows
    assert len(result) == 10


def test_build_training_table_expected_columns(gold_features, claims):
    result = build_training_table(gold_features, claims, start_year=2000, end_year=2004)
    expected = set(FEATURE_COLUMNS) | {
        "year",
        "had_claim",
        "incurred_loss_usd",
        "max_damage_ratio",
        "claim_count_in_year",
    }
    assert expected.issubset(result.columns)


def test_build_training_table_excludes_out_of_range_claims(gold_features, claims):
    """The 1992 claim is outside the [2000, 2004] range and must not
    contribute to any row's target."""
    result = build_training_table(gold_features, claims, start_year=2000, end_year=2004)
    assert result["had_claim"].sum() == 1  # only the 2004 claim falls in range


def test_build_training_table_correctly_flags_claim_years(gold_features, claims):
    result = build_training_table(gold_features, claims, start_year=1990, end_year=2023)

    l1_1992 = result[(result["location_id"] == "L1") & (result["year"] == 1992)].iloc[0]
    assert l1_1992["had_claim"] == 1
    assert l1_1992["incurred_loss_usd"] == pytest.approx(80_000.0)
    assert l1_1992["max_damage_ratio"] == pytest.approx(0.32)

    l1_1993 = result[(result["location_id"] == "L1") & (result["year"] == 1993)].iloc[0]
    assert l1_1993["had_claim"] == 0
    assert l1_1993["incurred_loss_usd"] == pytest.approx(0.0)
    assert l1_1993["max_damage_ratio"] == pytest.approx(0.0)


def test_build_training_table_aggregates_multiple_claims_same_location_year(
    gold_features,
):
    """If a location has 2 claims in the SAME year (e.g. two storms), they
    must be summed for incurred_loss_usd and max'd for damage_ratio."""
    claims_same_year = pd.DataFrame(
        {
            "claim_id": ["C1", "C2"],
            "location_id": ["L1", "L1"],
            "loss_date": pd.to_datetime(["2005-08-01", "2005-10-15"]),
            "incurred_loss_usd": [30_000.0, 50_000.0],
            "damage_ratio": [0.1, 0.2],
        }
    )
    result = build_training_table(
        gold_features, claims_same_year, start_year=2000, end_year=2010
    )
    l1_2005 = result[(result["location_id"] == "L1") & (result["year"] == 2005)].iloc[0]

    assert l1_2005["claim_count_in_year"] == 2
    assert l1_2005["incurred_loss_usd"] == pytest.approx(80_000.0)  # 30k + 50k
    assert l1_2005["max_damage_ratio"] == pytest.approx(0.2)  # max(0.1, 0.2)


def test_build_training_table_location_with_no_claims_ever(gold_features, claims):
    """L2 has one claim in 2022 only — every other year should be
    had_claim=0, not silently dropped from the table."""
    result = build_training_table(gold_features, claims, start_year=2000, end_year=2023)
    l2_rows = result[result["location_id"] == "L2"]
    assert len(l2_rows) == 24  # 2000-2023 inclusive
    assert l2_rows["had_claim"].sum() == 1  # only 2022


def test_build_training_table_raises_on_missing_feature_columns(claims):
    bad_gold = pd.DataFrame({"location_id": ["L1"]})
    with pytest.raises(ValueError, match="gold_features is missing required columns"):
        build_training_table(bad_gold, claims)


def test_build_training_table_raises_on_missing_claims_columns(gold_features):
    bad_claims = pd.DataFrame({"location_id": ["L1"]})
    with pytest.raises(ValueError, match="claims is missing required columns"):
        build_training_table(gold_features, bad_claims)


def test_build_training_table_feature_columns_identical_across_years_for_same_location(
    gold_features, claims
):
    """Core design property: a location's FEATURES must be identical
    regardless of year (only the target varies) — this is what makes the
    'event-level split by year' methodology valid."""
    result = build_training_table(gold_features, claims, start_year=2000, end_year=2010)
    l1_rows = result[result["location_id"] == "L1"]

    for col in ["tiv_usd", "construction_class", "roof_type", "mslp_hpa_mean"]:
        assert l1_rows[col].nunique() == 1