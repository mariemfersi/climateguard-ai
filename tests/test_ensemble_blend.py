"""
Tests for ml.frequency_severity.ensemble_blend.

Acceptance criterion (Task 4.1.3): blended ensemble should outperform or match
any single model on held-out calibration metrics.
"""

import numpy as np
import pandas as pd
import pytest

from ml.frequency_severity.ensemble_blend import blend_frequency_predictions, evaluate_blend
from ml.frequency_severity.event_split import stratified_year_split
from ml.frequency_severity.train_catboost import train_catboost_frequency_model
from ml.frequency_severity.train_frequency import (
    FEATURE_COLUMNS,
    prepare_features,
    train_frequency_model,
)


@pytest.fixture
def synthetic_training_table_for_ensemble():
    """
    A synthetic (location, year) table with enough signal that both XGBoost
    and CatBoost can learn something, but with enough variance that their
    predictions differ enough for blending to provide meaningful benefit.
    """
    rng = np.random.default_rng(42)
    n_locations = 200
    years = list(range(1950, 2000))

    rows = []
    for loc_idx in range(n_locations):
        # Vary risk by location characteristics
        sst_mean = rng.uniform(27.0, 30.0)
        construction_risk = 1.0 if rng.random() > 0.5 else 0.5  # frame vs masonry
        true_claim_prob = ((sst_mean - 27.0) / 3.0 * 0.3 + construction_risk * 0.1) * 0.5

        for year in years:
            had_claim = int(rng.random() < true_claim_prob)
            construction_class = "frame" if construction_risk == 1.0 else "masonry"
            rows.append(
                {
                    "location_id": f"L{loc_idx}",
                    "year": year,
                    "lat": 27.0,
                    "lon": -81.0,
                    "metro_center": "Test Metro",
                    "year_built": 2000,
                    "construction_class": construction_class,
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


def test_blend_function_returns_correct_shape(synthetic_training_table_for_ensemble):
    """Verify blend_frequency_predictions returns three arrays of correct length."""
    train_df, test_df = stratified_year_split(
        synthetic_training_table_for_ensemble, test_size=0.3, seed=1
    )

    xgb_model, _ = train_frequency_model(train_df, test_df)
    catboost_model, _ = train_catboost_frequency_model(train_df, test_df)

    xgb_columns = prepare_features(train_df).columns
    feature_cols = [c for c in FEATURE_COLUMNS if c != "location_id"]

    blended, xgb_proba, cat_proba = blend_frequency_predictions(
        xgb_model, xgb_columns, catboost_model, feature_cols, test_df
    )

    assert len(blended) == len(test_df)
    assert len(xgb_proba) == len(test_df)
    assert len(cat_proba) == len(test_df)
    assert blended.shape == (len(test_df),)
    assert xgb_proba.shape == (len(test_df),)
    assert cat_proba.shape == (len(test_df),)


def test_blend_is_simple_average(synthetic_training_table_for_ensemble):
    """Verify blended prediction is exactly the arithmetic mean of the two models."""
    train_df, test_df = stratified_year_split(
        synthetic_training_table_for_ensemble, test_size=0.3, seed=1
    )

    xgb_model, _ = train_frequency_model(train_df, test_df)
    catboost_model, _ = train_catboost_frequency_model(train_df, test_df)

    xgb_columns = prepare_features(train_df).columns
    feature_cols = [c for c in FEATURE_COLUMNS if c != "location_id"]

    blended, xgb_proba, cat_proba = blend_frequency_predictions(
        xgb_model, xgb_columns, catboost_model, feature_cols, test_df
    )

    # Check that blended is exactly the average
    expected_blend = (xgb_proba + cat_proba) / 2.0
    np.testing.assert_array_almost_equal(blended, expected_blend)


def test_evaluate_blend_returns_all_metrics(synthetic_training_table_for_ensemble):
    """Verify evaluate_blend returns AUC for all three models."""
    train_df, test_df = stratified_year_split(
        synthetic_training_table_for_ensemble, test_size=0.3, seed=1
    )

    xgb_model, _ = train_frequency_model(train_df, test_df)
    catboost_model, _ = train_catboost_frequency_model(train_df, test_df)

    xgb_columns = prepare_features(train_df).columns
    feature_cols = [c for c in FEATURE_COLUMNS if c != "location_id"]

    blended, xgb_proba, cat_proba = blend_frequency_predictions(
        xgb_model, xgb_columns, catboost_model, feature_cols, test_df
    )

    metrics = evaluate_blend(test_df["had_claim"], blended, xgb_proba, cat_proba)

    assert "xgb_auc" in metrics
    assert "catboost_auc" in metrics
    assert "blended_auc" in metrics
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_ensemble_outperforms_or_matches_single_models(synthetic_training_table_for_ensemble):
    """
    Core acceptance criterion (Task 4.1.3): blended ensemble should outperform
    or match any single model on held-out calibration metrics.

    In practice, simple averaging of two diverse models typically provides
    variance reduction without significantly degrading bias, so the blend should
    at least match the better of the two base models.
    """
    train_df, test_df = stratified_year_split(
        synthetic_training_table_for_ensemble, test_size=0.3, seed=1
    )

    xgb_model, _ = train_frequency_model(train_df, test_df)
    catboost_model, _ = train_catboost_frequency_model(train_df, test_df)

    xgb_columns = prepare_features(train_df).columns
    feature_cols = [c for c in FEATURE_COLUMNS if c != "location_id"]

    blended, xgb_proba, cat_proba = blend_frequency_predictions(
        xgb_model, xgb_columns, catboost_model, feature_cols, test_df
    )

    metrics = evaluate_blend(test_df["had_claim"], blended, xgb_proba, cat_proba)

    # Acceptance criterion: blend should be >= max of single models
    # (allow tiny numerical tolerance for floating point)
    max_single_auc = max(metrics["xgb_auc"], metrics["catboost_auc"])
    assert metrics["blended_auc"] >= max_single_auc - 1e-6, (
        f"Ensemble AUC ({metrics['blended_auc']:.4f}) should be >= "
        f"max single model AUC ({max_single_auc:.4f})"
    )


def test_ensemble_handles_test_set_with_unseen_category(synthetic_training_table_for_ensemble):
    """Verify ensemble doesn't crash when test set has unseen categorical values."""
    train_df, test_df = stratified_year_split(
        synthetic_training_table_for_ensemble, test_size=0.3, seed=1
    )
    
    # Inject an unseen category in test set
    test_df = test_df.copy()
    test_df.loc[test_df.index[0], "metro_center"] = "Brand New Unseen Metro"

    xgb_model, _ = train_frequency_model(train_df, test_df)
    catboost_model, _ = train_catboost_frequency_model(train_df, test_df)

    xgb_columns = prepare_features(train_df).columns
    feature_cols = [c for c in FEATURE_COLUMNS if c != "location_id"]

    # Should not raise
    blended, xgb_proba, cat_proba = blend_frequency_predictions(
        xgb_model, xgb_columns, catboost_model, feature_cols, test_df
    )

    assert len(blended) == len(test_df)
