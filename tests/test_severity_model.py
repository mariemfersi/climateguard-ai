"""
Tests for ml.frequency_severity.train_severity.
"""

import numpy as np
import pandas as pd
import pytest

from ml.frequency_severity.event_split import stratified_year_split
from ml.frequency_severity.train_frequency import prepare_features
from ml.frequency_severity.train_severity import (
    MONOTONIC_INCREASING_FEATURE,
    train_severity_model,
)


@pytest.fixture
def synthetic_training_table_with_severity_signal():
    """
    A synthetic (location, year) table where, for the subset of rows with
    a claim, severity genuinely increases (with noise) as basin SST
    increases — lets us test both raw predictive signal AND the monotonic
    constraint against a KNOWN ground-truth relationship.
    """
    rng = np.random.default_rng(0)
    n_locations = 200
    years = list(range(1950, 2000))

    rows = []
    for loc_idx in range(n_locations):
        sst_mean = rng.uniform(27.0, 30.0)
        true_claim_prob = (sst_mean - 27.0) / 3.0 * 0.5
        true_severity_base = 0.05 + (sst_mean - 27.0) / 3.0 * 0.3  # 0.05 to 0.35

        for year in years:
            had_claim = int(rng.random() < true_claim_prob)
            damage = (
                float(np.clip(true_severity_base + rng.normal(0, 0.03), 0.01, 0.95))
                if had_claim
                else 0.0
            )
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
                    "max_damage_ratio": damage,
                }
            )
    return pd.DataFrame(rows)


def test_severity_model_trains_and_returns_metrics(
    synthetic_training_table_with_severity_signal,
):
    train_df, test_df = stratified_year_split(
        synthetic_training_table_with_severity_signal, test_size=0.3, seed=1
    )
    _model, metrics = train_severity_model(train_df, test_df)
    assert "test_mae" in metrics
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0


def test_severity_model_raises_if_no_positive_rows_in_train():
    df = pd.DataFrame(
        {
            "location_id": ["L1", "L2"],
            "year": [2000, 2001],
            "lat": [27.0, 27.0],
            "lon": [-81.0, -81.0],
            "metro_center": ["M", "M"],
            "year_built": [2000, 2000],
            "construction_class": ["frame", "frame"],
            "roof_type": ["gable", "gable"],
            "tiv_usd": [300_000.0, 300_000.0],
            "mslp_hpa_mean": [1015.0, 1015.0],
            "mslp_hpa_min": [1014.0, 1014.0],
            "era5_wind_speed_ms_mean": [3.0, 3.0],
            "era5_wind_speed_ms_max": [5.0, 5.0],
            "basin_sst_celsius_mean": [28.0, 28.0],
            "basin_sst_celsius_max": [33.0, 33.0],
            "had_claim": [0, 0],
            "max_damage_ratio": [0.0, 0.0],
        }
    )
    train_df, test_df = df.iloc[[0]], df.iloc[[1]]
    with pytest.raises(ValueError, match="No claims in training data"):
        train_severity_model(train_df, test_df)


def test_severity_model_respects_monotonic_constraint_on_basin_sst(
    synthetic_training_table_with_severity_signal,
):
    """
    Core acceptance criterion (adapted from roadmap Task 4.1.2):
    predictions must be non-decreasing as basin_sst_celsius_mean
    increases, holding all other features fixed — verified via a
    synthetic monotonicity grid, exactly as the roadmap requires.
    """
    train_df, test_df = stratified_year_split(
        synthetic_training_table_with_severity_signal, test_size=0.3, seed=1
    )
    model, _ = train_severity_model(train_df, test_df)

    base_row = train_df[train_df["had_claim"] == 1].iloc[[0]].copy()
    X_base = prepare_features(base_row)

    sst_values = np.linspace(27.0, 30.0, 15)
    preds = []
    for v in sst_values:
        row = X_base.copy()
        row[MONOTONIC_INCREASING_FEATURE] = v
        preds.append(model.predict(row)[0])

    preds = np.array(preds)
    assert (np.diff(preds) >= -1e-9).all()  # non-decreasing (tiny float tolerance)