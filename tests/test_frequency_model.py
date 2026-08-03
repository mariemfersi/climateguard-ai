"""
Tests for ml.frequency_severity.event_split and train_frequency.

Uses a synthetic training-table-shaped dataset with a KNOWN, learnable
signal (higher basin SST -> higher claim probability) so the model's AUC
being meaningfully above 0.5 is a real, checkable correctness signal, not
just a smoke test.
"""

import numpy as np
import pandas as pd
import pytest

from ml.frequency_severity.event_split import event_level_train_test_split
from ml.frequency_severity.train_frequency import (
    assert_no_year_leakage,
    train_frequency_model,
)


# --- event_level_train_test_split --------------------------------------------


def test_split_never_puts_same_year_in_both_sides():
    df = pd.DataFrame({"year": list(range(1950, 2024)), "value": range(74)})
    _train, test = event_level_train_test_split(df, test_size=0.2, seed=1)

    train_years = set(_train["year"])
    test_years = set(test["year"])
    assert train_years.isdisjoint(test_years)


def test_split_respects_approximate_test_size():
    df = pd.DataFrame({"year": list(range(1950, 2024))})  # 74 distinct years
    train, test = event_level_train_test_split(df, test_size=0.2, seed=1)

    n_test_years = test["year"].nunique()
    # 20% of 74 ~= 15, allow a little rounding slack
    assert 12 <= n_test_years <= 18


def test_split_is_deterministic_given_seed():
    df = pd.DataFrame({"year": list(range(1950, 2024)), "value": range(74)})
    train1, test1 = event_level_train_test_split(df, seed=7)
    train2, test2 = event_level_train_test_split(df, seed=7)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_split_rejects_invalid_test_size():
    df = pd.DataFrame({"year": [2000, 2001, 2002]})
    with pytest.raises(ValueError, match="test_size must be in"):
        event_level_train_test_split(df, test_size=1.5)


def test_split_rejects_too_few_years():
    df = pd.DataFrame({"year": [2000, 2000, 2000]})
    with pytest.raises(ValueError, match="Need at least 2 distinct years"):
        event_level_train_test_split(df)


def test_assert_no_year_leakage_passes_on_clean_split():
    train = pd.DataFrame({"year": [2000, 2001]})
    test = pd.DataFrame({"year": [2002, 2003]})
    assert_no_year_leakage(train, test)  # should not raise


def test_assert_no_year_leakage_catches_overlap():
    train = pd.DataFrame({"year": [2000, 2001, 2002]})
    test = pd.DataFrame({"year": [2002, 2003]})  # 2002 overlaps
    with pytest.raises(AssertionError, match="Event-level split violated"):
        assert_no_year_leakage(train, test)


# --- train_frequency_model ----------------------------------------------------


@pytest.fixture
def synthetic_training_table():
    """
    A synthetic (location, year) table with a KNOWN signal: locations with
    higher basin_sst_celsius_mean have a genuinely higher claim
    probability (injected via the random generation itself), so a
    correctly-working model should achieve AUC meaningfully above 0.5.
    """
    rng = np.random.default_rng(0)
    n_locations = 200
    years = list(range(1950, 2000))  # 50 years

    rows = []
    for loc_idx in range(n_locations):
        # Each location has a fixed "risk level" tied to basin SST exposure —
        # constant across all its years (matches the real design's property).
        sst_mean = rng.uniform(27.0, 30.0)
        true_claim_prob = (sst_mean - 27.0) / 3.0 * 0.5  # 0.0 to 0.5

        for year in years:
            had_claim = int(rng.random() < true_claim_prob)
            rows.append(
                {
                    "location_id": f"L{loc_idx}",
                    "year": year,
                    "lat": 27.0,
                    "lon": -81.0,
                    "metro_center": "Test Metro",
                    "year_built": 2000,
                    "construction_class": "frame",
                    "roof_type": "gable",
                    "tiv_usd": 300_000.0,
                    "mslp_hpa_mean": 1015.0,
                    "mslp_hpa_min": 1014.0,
                    "era5_wind_speed_ms_mean": 3.0,
                    "era5_wind_speed_ms_max": 5.0,
                    "basin_sst_celsius_mean": sst_mean,
                    "basin_sst_celsius_max": sst_mean + 5.0,
                    "had_claim": had_claim,
                }
            )
    return pd.DataFrame(rows)


def test_frequency_model_beats_random_on_synthetic_signal(synthetic_training_table):
    train_df, test_df = event_level_train_test_split(
        synthetic_training_table, test_size=0.3, seed=1
    )
    assert_no_year_leakage(train_df, test_df)

    _model, metrics = train_frequency_model(train_df, test_df)

    # A model that learned nothing would score ~0.5 AUC. Given the
    # genuinely injected SST->claim-probability signal, we expect
    # meaningfully better than random.
    assert metrics["test_auc"] > 0.55


def test_frequency_model_returns_expected_metrics_keys(synthetic_training_table):
    train_df, test_df = event_level_train_test_split(
        synthetic_training_table, test_size=0.3, seed=1
    )
    _model, metrics = train_frequency_model(train_df, test_df)

    expected_keys = {
        "train_auc",
        "test_auc",
        "train_base_rate",
        "test_base_rate",
        "n_train",
        "n_test",
    }
    assert expected_keys.issubset(metrics.keys())


def test_frequency_model_handles_test_set_with_unseen_category(synthetic_training_table):
    """A category present in test but not train (e.g. a rare metro_center)
    must not crash the pipeline — reindex should zero-fill it."""
    train_df, test_df = event_level_train_test_split(
        synthetic_training_table, test_size=0.3, seed=1
    )
    test_df = test_df.copy()
    test_df.loc[test_df.index[0], "metro_center"] = "Brand New Unseen Metro"

    # Should not raise.
    _model, metrics = train_frequency_model(train_df, test_df)
    assert metrics["n_test"] == len(test_df)