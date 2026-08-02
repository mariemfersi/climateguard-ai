"""
Tests for data_pipeline.great_expectations.validate_silver.

Per roadmap Task 2.3.1 acceptance criteria: "suite fails loudly on injected
bad data (test with a deliberately corrupted sample) and passes on the real
generated Silver tables." Both directions are tested explicitly below.
"""

import pandas as pd
import pytest

from data_pipeline.great_expectations.validate_silver import (
    build_claims_suite,
    build_locations_suite,
    build_policies_suite,
    validate_dataframe,
)


@pytest.fixture
def valid_locations():
    return pd.DataFrame(
        {
            "location_id": ["LOC0000001", "LOC0000002"],
            "lat": [26.64, 25.76],
            "lon": [-81.87, -80.19],
            "metro_center": ["Lee (Fort Myers)", "Miami-Dade"],
            "year_built": [1998, 2015],
            "construction_class": ["frame", "masonry_cbs"],
            "roof_type": ["gable", "hip"],
            "tiv_usd": [250_000.0, 400_000.0],
        }
    )


@pytest.fixture
def valid_policies():
    return pd.DataFrame(
        {
            "policy_id": ["POL0000001", "POL0000002"],
            "location_id": ["LOC0000001", "LOC0000002"],
            "effective_date": pd.to_datetime(["2020-06-01", "2021-03-15"]),
            "expiry_date": pd.to_datetime(["2021-06-01", "2022-03-15"]),
            "limit_usd": [250_000.0, 400_000.0],
            "deductible_pct": [0.05, 0.02],
            "deductible_usd": [12_500.0, 8_000.0],
        }
    )


@pytest.fixture
def valid_claims():
    return pd.DataFrame(
        {
            "claim_id": ["CLM00000001", "CLM00000002"],
            "location_id": ["LOC0000001", "LOC0000002"],
            "policy_id": ["POL0000001", "POL0000002"],
            "storm_id": ["AL041992", "AL111950"],
            "storm_name": ["ANDREW", "KING"],
            "loss_date": pd.to_datetime(["1992-08-24", "1950-10-18"]),
            "peril_type": ["hurricane_wind", "hurricane_wind"],
            "max_wind_experienced_kt": [130.0, 95.0],
            "damage_ratio": [0.45, 0.15],
            "incurred_loss_usd": [112_500.0, 60_000.0],
            "paid_loss_usd": [100_000.0, 52_000.0],
        }
    )


# --- passes on valid data ----------------------------------------------------


def test_locations_suite_passes_on_valid_data(valid_locations):
    success, _ = validate_dataframe(
        valid_locations, build_locations_suite, "locations_test"
    )
    assert success is True


def test_policies_suite_passes_on_valid_data(valid_policies):
    success, _ = validate_dataframe(
        valid_policies, build_policies_suite, "policies_test"
    )
    assert success is True


def test_claims_suite_passes_on_valid_data(valid_claims):
    success, _ = validate_dataframe(valid_claims, build_claims_suite, "claims_test")
    assert success is True


# --- fails loudly on injected bad data ---------------------------------------


def test_locations_suite_fails_on_duplicate_id(valid_locations):
    corrupted = valid_locations.copy()
    corrupted.loc[1, "location_id"] = corrupted.loc[0, "location_id"]  # inject duplicate
    success, _ = validate_dataframe(
        corrupted, build_locations_suite, "locations_test_bad"
    )
    assert success is False


def test_locations_suite_fails_on_out_of_bounds_lat(valid_locations):
    corrupted = valid_locations.copy()
    corrupted.loc[0, "lat"] = 45.0  # well outside Florida
    success, _ = validate_dataframe(
        corrupted, build_locations_suite, "locations_test_bad2"
    )
    assert success is False


def test_locations_suite_fails_on_invalid_construction_class(valid_locations):
    corrupted = valid_locations.copy()
    corrupted.loc[0, "construction_class"] = "log_cabin"  # not in the valid set
    success, _ = validate_dataframe(
        corrupted, build_locations_suite, "locations_test_bad3"
    )
    assert success is False


def test_policies_suite_fails_on_invalid_deductible_pct(valid_policies):
    corrupted = valid_policies.copy()
    corrupted.loc[0, "deductible_pct"] = 0.25  # not a valid FL hurricane deductible tier
    success, _ = validate_dataframe(
        corrupted, build_policies_suite, "policies_test_bad"
    )
    assert success is False


def test_policies_suite_fails_when_expiry_before_effective(valid_policies):
    corrupted = valid_policies.copy()
    corrupted.loc[0, "expiry_date"] = pd.Timestamp("2019-01-01")  # before effective_date
    success, _ = validate_dataframe(
        corrupted, build_policies_suite, "policies_test_bad2"
    )
    assert success is False


def test_claims_suite_fails_on_damage_ratio_out_of_range(valid_claims):
    corrupted = valid_claims.copy()
    corrupted.loc[0, "damage_ratio"] = 1.5  # impossible — max is 1.0 (total loss)
    success, _ = validate_dataframe(corrupted, build_claims_suite, "claims_test_bad")
    assert success is False


def test_claims_suite_fails_when_paid_exceeds_incurred(valid_claims):
    corrupted = valid_claims.copy()
    corrupted.loc[0, "paid_loss_usd"] = 999_999.0  # exceeds incurred_loss_usd — impossible
    success, _ = validate_dataframe(corrupted, build_claims_suite, "claims_test_bad2")
    assert success is False


def test_claims_suite_fails_on_negative_loss(valid_claims):
    corrupted = valid_claims.copy()
    corrupted.loc[0, "incurred_loss_usd"] = -5000.0  # negative loss is impossible
    success, _ = validate_dataframe(corrupted, build_claims_suite, "claims_test_bad3")
    assert success is False