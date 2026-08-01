"""
Unit tests for data_pipeline.synthetic (Milestone 2.1).

Per roadmap validation checklist: spatial density plausibility, construction-
class distribution shape, no duplicate IDs, TIV distribution shape.
"""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.synthetic.assign_attributes import (
    _CODE_REFORM_YEAR,
    assign_attributes,
)
from data_pipeline.synthetic.generate_locations import (
    FLORIDA_BBOX,
    generate_locations,
)
from data_pipeline.synthetic.generate_policies import generate_policies


# --- generate_locations -------------------------------------------------


def test_generate_locations_produces_n_rows():
    df = generate_locations(500)
    assert len(df) == 500


def test_generate_locations_unique_ids():
    df = generate_locations(1000)
    assert df["location_id"].is_unique


def test_generate_locations_within_florida_bbox():
    df = generate_locations(2000)
    assert (df["lat"] >= FLORIDA_BBOX["lat_min"]).all()
    assert (df["lat"] <= FLORIDA_BBOX["lat_max"]).all()
    assert (df["lon"] >= FLORIDA_BBOX["lon_min"]).all()
    assert (df["lon"] <= FLORIDA_BBOX["lon_max"]).all()


def test_generate_locations_is_population_weighted_not_uniform():
    """Miami-Dade (pop ~2.7M) should get meaningfully more locations than
    Key West (pop ~83K) — this is the actual point of the module."""
    df = generate_locations(5000)
    counts = df["metro_center"].value_counts()
    assert counts["Miami-Dade"] > counts["Monroe (Key West)"] * 5


def test_generate_locations_is_deterministic_given_seed():
    df1 = generate_locations(100, seed=7)
    df2 = generate_locations(100, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_locations_rejects_non_positive_n():
    with pytest.raises(ValueError, match="n must be positive"):
        generate_locations(0)


# --- assign_attributes ---------------------------------------------------


@pytest.fixture
def sample_locations():
    return generate_locations(3000, seed=1)


def test_assign_attributes_adds_expected_columns(sample_locations):
    df = assign_attributes(sample_locations)
    for col in ["year_built", "construction_class", "roof_type", "tiv_usd"]:
        assert col in df.columns


def test_assign_attributes_tiv_is_right_skewed_and_bounded(sample_locations):
    df = assign_attributes(sample_locations)
    assert (df["tiv_usd"] >= 75_000).all()
    assert (df["tiv_usd"] <= 5_000_000).all()
    # Right-skewed: mean > median is the classic signature.
    assert df["tiv_usd"].mean() > df["tiv_usd"].median()


def test_assign_attributes_post_code_favors_hip_roofs_and_masonry(sample_locations):
    """Verifies the 2002 Florida Building Code breakpoint actually shifts
    the distribution the way the methodology note claims it does — this
    is the core, checkable claim of this module."""
    df = assign_attributes(sample_locations)

    post_code = df[df["year_built"] >= _CODE_REFORM_YEAR]
    pre_code = df[df["year_built"] < _CODE_REFORM_YEAR]

    post_hip_rate = (post_code["roof_type"] == "hip").mean()
    pre_hip_rate = (pre_code["roof_type"] == "hip").mean()
    assert post_hip_rate > pre_hip_rate

    post_masonry_rate = (post_code["construction_class"] == "masonry_cbs").mean()
    pre_masonry_rate = (pre_code["construction_class"] == "masonry_cbs").mean()
    assert post_masonry_rate > pre_masonry_rate


def test_assign_attributes_year_built_within_plausible_range(sample_locations):
    df = assign_attributes(sample_locations)
    assert (df["year_built"] >= 1900).all()
    assert (df["year_built"] <= 2024).all()


# --- generate_policies ----------------------------------------------------


@pytest.fixture
def sample_locations_with_attributes(sample_locations):
    return assign_attributes(sample_locations)


def test_generate_policies_one_per_location(sample_locations_with_attributes):
    policies = generate_policies(sample_locations_with_attributes)
    assert len(policies) == len(sample_locations_with_attributes)
    assert policies["policy_id"].is_unique
    assert set(policies["location_id"]) == set(sample_locations_with_attributes["location_id"])


def test_generate_policies_requires_tiv_column():
    locations_without_tiv = generate_locations(10)
    with pytest.raises(ValueError, match="tiv_usd"):
        generate_policies(locations_without_tiv)


def test_generate_policies_deductible_pct_is_valid_florida_convention(
    sample_locations_with_attributes,
):
    policies = generate_policies(sample_locations_with_attributes)
    assert policies["deductible_pct"].isin([0.02, 0.05, 0.10]).all()
    # deductible_usd should equal limit_usd * deductible_pct (within rounding).
    expected = policies["limit_usd"] * policies["deductible_pct"]
    assert np.allclose(policies["deductible_usd"], expected, atol=100)


def test_generate_policies_expiry_is_one_year_after_effective(
    sample_locations_with_attributes,
):
    policies = generate_policies(sample_locations_with_attributes)
    diffs = (policies["expiry_date"] - policies["effective_date"]).dt.days
    assert diffs.between(363, 367).all()  # ~1 year, allowing leap-year slack


def test_generate_policies_limit_close_to_but_not_exceeding_tiv(
    sample_locations_with_attributes,
):
    policies = generate_policies(sample_locations_with_attributes)
    merged = policies.merge(
        sample_locations_with_attributes[["location_id", "tiv_usd"]], on="location_id"
    )
    assert (merged["limit_usd"] <= merged["tiv_usd"] + 1).all()  # +1 for rounding
    assert (merged["limit_usd"] >= merged["tiv_usd"] * 0.7).all()