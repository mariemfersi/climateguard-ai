"""
Unit tests for data_pipeline.synthetic.claims_generator.

Uses small, deliberately-crafted fixtures (a "direct hit" storm, a "miss"
storm, and known deductible/limit values) so every assertion is hand-
verifiable, following the same pattern as the HURDAT2 parser and wind
field model tests.
"""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.synthetic.claims_generator import (
    FLORIDA_INFLUENCE_BBOX,
    filter_storms_near_florida,
    generate_claims,
)


@pytest.fixture
def locations():
    """Two locations: one directly in a strong storm's path (weak frame
    construction -> should generate a large claim), one far away in the
    Atlantic-mid-ocean sense (should generate no claim from a FL-hitting storm)."""
    return pd.DataFrame(
        {
            "location_id": ["LOC0000001", "LOC0000002"],
            "lat": [26.64, 40.0],  # Fort Myers area vs. far north (out of range)
            "lon": [-81.87, -70.0],
            "construction_class": ["frame", "reinforced_concrete"],
            "roof_type": ["gable", "hip"],
            "tiv_usd": [300_000.0, 500_000.0],
        }
    )


@pytest.fixture
def policies():
    return pd.DataFrame(
        {
            "location_id": ["LOC0000001", "LOC0000002"],
            "policy_id": ["POL0000001", "POL0000002"],
            "deductible_usd": [15_000.0, 25_000.0],
            "limit_usd": [300_000.0, 500_000.0],
        }
    )


@pytest.fixture
def hurdat2_tracks():
    """A single strong synthetic storm making a direct approach on Fort
    Myers (26.64, -81.87), analogous in shape (not claimed to be identical
    in magnitude) to a real Southwest-Florida-landfalling hurricane."""
    return pd.DataFrame(
        {
            "storm_id": ["AL999999"] * 4,
            "name": ["TESTIAN"] * 4,
            "timestamp": pd.to_datetime(
                ["2022-09-27 12:00", "2022-09-27 18:00", "2022-09-28 00:00", "2022-09-28 06:00"]
            ),
            "lat": [25.5, 26.0, 26.6, 27.2],
            "lon": [-83.0, -82.5, -81.9, -81.3],
            "max_wind_kt": [120.0, 130.0, 135.0, 100.0],  # peaks near landfall
        }
    )


@pytest.fixture
def far_away_storm_tracks():
    """A storm that never approaches Florida at all — used to verify the
    bounding-box pre-filter actually excludes it."""
    return pd.DataFrame(
        {
            "storm_id": ["AL888888"] * 2,
            "name": ["FARAWAY"] * 2,
            "timestamp": pd.to_datetime(["1999-01-01", "1999-01-02"]),
            "lat": [45.0, 46.0],
            "lon": [-40.0, -41.0],
            "max_wind_kt": [90.0, 95.0],
        }
    )


# --- filter_storms_near_florida ---------------------------------------------


def test_filter_keeps_storms_near_florida(hurdat2_tracks):
    result = filter_storms_near_florida(hurdat2_tracks)
    assert "AL999999" in result


def test_filter_excludes_storms_far_from_florida(far_away_storm_tracks):
    result = filter_storms_near_florida(far_away_storm_tracks)
    assert result == []


def test_filter_bbox_bounds_are_sane():
    assert FLORIDA_INFLUENCE_BBOX["lat_min"] < FLORIDA_INFLUENCE_BBOX["lat_max"]
    assert FLORIDA_INFLUENCE_BBOX["lon_min"] < FLORIDA_INFLUENCE_BBOX["lon_max"]


# --- generate_claims: schema and validation ----------------------------------


def test_generate_claims_requires_location_columns(policies, hurdat2_tracks):
    bad_locations = pd.DataFrame({"location_id": ["LOC1"], "lat": [26.0], "lon": [-81.0]})
    with pytest.raises(ValueError, match="locations is missing required columns"):
        generate_claims(bad_locations, policies, hurdat2_tracks)


def test_generate_claims_requires_policy_columns(locations, hurdat2_tracks):
    bad_policies = pd.DataFrame({"location_id": ["LOC0000001"]})
    with pytest.raises(ValueError, match="policies is missing required columns"):
        generate_claims(locations, bad_policies, hurdat2_tracks)


def test_generate_claims_returns_expected_schema(locations, policies, hurdat2_tracks):
    claims = generate_claims(locations, policies, hurdat2_tracks)
    expected_cols = {
        "claim_id",
        "location_id",
        "policy_id",
        "storm_id",
        "storm_name",
        "loss_date",
        "peril_type",
        "max_wind_experienced_kt",
        "damage_ratio",
        "incurred_loss_usd",
        "paid_loss_usd",
    }
    assert expected_cols.issubset(claims.columns)


def test_generate_claims_empty_when_no_storms_near_book(
    locations, policies, far_away_storm_tracks
):
    claims = generate_claims(locations, policies, far_away_storm_tracks)
    assert claims.empty


# --- generate_claims: core correctness ---------------------------------------


def test_direct_hit_location_generates_a_claim_distant_location_does_not(
    locations, policies, hurdat2_tracks
):
    claims = generate_claims(locations, policies, hurdat2_tracks)
    assert "LOC0000001" in claims["location_id"].to_numpy()
    assert "LOC0000002" not in claims["location_id"].to_numpy()


def test_claim_ids_are_unique(locations, policies, hurdat2_tracks):
    claims = generate_claims(locations, policies, hurdat2_tracks)
    assert claims["claim_id"].is_unique


def test_paid_loss_never_exceeds_limit(locations, policies, hurdat2_tracks):
    claims = generate_claims(locations, policies, hurdat2_tracks)
    merged = claims.merge(policies, on="location_id")
    assert (merged["paid_loss_usd"] <= merged["limit_usd"] + 0.01).all()


def test_paid_loss_correctly_nets_deductible(locations, policies, hurdat2_tracks):
    """paid_loss_usd should equal max(0, incurred - deductible), capped at
    limit — verified by hand against the actual policy values."""
    claims = generate_claims(locations, policies, hurdat2_tracks)
    merged = claims.merge(policies, on="location_id")

    expected_paid = np.minimum(
        np.maximum(merged["incurred_loss_usd"] - merged["deductible_usd"], 0),
        merged["limit_usd"],
    )
    assert np.allclose(merged["paid_loss_usd"], expected_paid, atol=1.0)


def test_incurred_loss_equals_damage_ratio_times_tiv(locations, policies, hurdat2_tracks):
    claims = generate_claims(locations, policies, hurdat2_tracks)
    merged = claims.merge(locations[["location_id", "tiv_usd"]], on="location_id")
    expected = merged["damage_ratio"] * merged["tiv_usd"]
    assert np.allclose(merged["incurred_loss_usd"], expected, atol=1.0)


def test_loss_date_falls_within_storm_track_time_range(locations, policies, hurdat2_tracks):
    claims = generate_claims(locations, policies, hurdat2_tracks)
    assert (claims["loss_date"] >= hurdat2_tracks["timestamp"].min()).all()
    assert (claims["loss_date"] <= hurdat2_tracks["timestamp"].max()).all()


def test_stronger_storm_produces_larger_claims_than_weaker_storm(locations, policies):
    """Comparative sanity check: doubling storm intensity should increase,
    not decrease, the simulated loss for the same book of business."""
    weak_track = pd.DataFrame(
        {
            "storm_id": ["AL111111"] * 2,
            "name": ["WEAK"] * 2,
            "timestamp": pd.to_datetime(["2020-08-01 00:00", "2020-08-01 06:00"]),
            "lat": [26.5, 26.7],
            "lon": [-81.9, -81.8],
            "max_wind_kt": [60.0, 60.0],
        }
    )
    strong_track = pd.DataFrame(
        {
            "storm_id": ["AL222222"] * 2,
            "name": ["STRONG"] * 2,
            "timestamp": pd.to_datetime(["2020-08-01 00:00", "2020-08-01 06:00"]),
            "lat": [26.5, 26.7],
            "lon": [-81.9, -81.8],
            "max_wind_kt": [150.0, 150.0],
        }
    )

    weak_claims = generate_claims(locations, policies, weak_track)
    strong_claims = generate_claims(locations, policies, strong_track)

    weak_total = weak_claims["incurred_loss_usd"].sum() if not weak_claims.empty else 0.0
    strong_total = strong_claims["incurred_loss_usd"].sum() if not strong_claims.empty else 0.0
    assert strong_total > weak_total