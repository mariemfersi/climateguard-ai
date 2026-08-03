"""
Tests for ml.frequency_severity.train_catboost and ensemble_blend.
"""

import numpy as np
import pandas as pd
import pytest

from ml.frequency_severity.build_training_table import FEATURE_COLUMNS
from ml.frequency_severity.ensemble_blend import blend_frequency_predictions, evaluate_blend
from ml.frequency_severity.event_split import event_level_train_test_split
from ml.frequency_severity.train_catboost import train_catboost_frequency_model
from ml.frequency_severity.train_frequency import prepare_features, train_frequency_model


@pytest.fixture
def synthetic_training_table():
    rng = np.random.default_rng(3)
    n_locations = 300
    years = list(range(1950, 2000))
    metros = ["Metro A", "Metro B", "Metro C"]
    construction_classes = ["frame", "masonry_cbs"]

    rows = []
    for loc_idx in range(n_locations):
        sst_mean = rng.uniform(27.0, 30.0)
        true_claim_prob = (sst_mean - 27.0) / 3.0 * 0.5
        metro = metros[rng.integers(0, len(metros))]
        construction = construction_classes[rng.integers(0, len(construction_classes))]

        for year in years:
            had_claim = int(rng.random() < true_claim_prob)
            rows.append(
                {
                    "location_id": f"L{loc_idx}",
                    "year": year,
                    "lat": 27.0,
                    "lon": -81.0,
                    "metro_center": metro,
                    "year_built": 2000,
                    "construction_class": construction,
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


def test_catboost_frequency_model_beats_random(synthetic_training_table):
    train_df, test_df = event_level_train_test_split(
        synthetic_training_table, test_size=0.3, seed=1
    )
    _model, metrics = train_catboost_frequency_model(train_df, test_df)
    assert metrics["test_auc"] > 0.55


def test_catboost_handles_categoricals_without_manual_encoding(synthetic_training_table):
    """CatBoost should train successfully directly on raw string categorical
    columns (no pd.get_dummies) — this is the whole point of using it."""
    train_df, test_df = event_level_train_test_split(
        synthetic_training_table, test_size=0.3, seed=1
    )
    feature_cols = [c for c in FEATURE_COLUMNS if c != "location_id"]
    assert train_df[feature_cols]["metro_center"].dtype == object  # still raw strings
    model, _ = train_catboost_frequency_model(train_df, test_df)
    assert model is not None


def test_blend_combines_both_models_predictions(synthetic_training_table):
    train_df, test_df = event_level_train_test_split(
        synthetic_training_table, test_size=0.3, seed=1
    )

    xgb_model, _xgb_metrics = train_frequency_model(train_df, test_df)
    xgb_columns = prepare_features(train_df).columns

    catboost_model, _catboost_metrics = train_catboost_frequency_model(train_df, test_df)
    feature_cols = [c for c in FEATURE_COLUMNS if c != "location_id"]

    blended, xgb_proba, cat_proba = blend_frequency_predictions(
        xgb_model, xgb_columns, catboost_model, feature_cols, test_df
    )
    result = evaluate_blend(test_df["had_claim"], blended, xgb_proba, cat_proba)

    # Acceptance criterion (roadmap Task 4.1.3): blended should be
    # competitive with (not meaningfully worse than) the weaker of the two
    # individual models — small tolerance since blending isn't GUARANTEED
    # to strictly beat both on every random draw, only to not hurt
    # performance materially while adding robustness/diversity.
    assert result["blended_auc"] >= min(result["xgb_auc"], result["catboost_auc"]) - 0.02

    # Sanity: blended predictions are a genuine combination, not a copy of
    # either individual model's output.
    assert not np.allclose(blended, xgb_proba)
    assert not np.allclose(blended, cat_proba)
    assert np.allclose(blended, (xgb_proba + cat_proba) / 2.0)