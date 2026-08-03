"""
Milestone 4.1 entrypoint: build the real training table from Gold features
+ claims, train frequency (XGBoost + CatBoost, blended) and severity
(LightGBM) models, and report real metrics.

Usage:
    python -m ml.frequency_severity.run_milestone_4_1
"""

from __future__ import annotations

import logging

from ml.frequency_severity.build_training_table import load_and_build
from ml.frequency_severity.ensemble_blend import blend_frequency_predictions, evaluate_blend
from ml.frequency_severity.event_split import event_level_train_test_split
from ml.frequency_severity.train_catboost import train_catboost_frequency_model
from ml.frequency_severity.train_frequency import (
    FEATURE_COLUMNS,
    assert_no_year_leakage,
    prepare_features,
    train_frequency_model,
)
from ml.frequency_severity.train_severity import train_severity_model

logger = logging.getLogger(__name__)


def run() -> None:
    logger.info("Building training table from real Gold features + claims...")
    training_table = load_and_build()

    logger.info("Splitting by year (event-level split)...")
    train_df, test_df = event_level_train_test_split(training_table, test_size=0.2, seed=42)
    assert_no_year_leakage(train_df, test_df)
    logger.info(
        "Train: %d rows across %d years | Test: %d rows across %d years",
        len(train_df),
        train_df["year"].nunique(),
        len(test_df),
        test_df["year"].nunique(),
    )

    print("\n=== Frequency Model (XGBoost) ===")
    xgb_model, xgb_metrics = train_frequency_model(train_df, test_df)
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Frequency Model (CatBoost, native categoricals) ===")
    catboost_model, catboost_metrics = train_catboost_frequency_model(train_df, test_df)
    for k, v in catboost_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Ensemble Blend ===")
    xgb_columns = prepare_features(train_df).columns
    feature_cols = [c for c in FEATURE_COLUMNS if c != "location_id"]
    blended, xgb_proba, cat_proba = blend_frequency_predictions(
        xgb_model, xgb_columns, catboost_model, feature_cols, test_df
    )
    blend_metrics = evaluate_blend(test_df["had_claim"], blended, xgb_proba, cat_proba)
    for k, v in blend_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Severity Model (LightGBM, monotonic constraint on basin SST) ===")
    severity_model, severity_metrics = train_severity_model(train_df, test_df)
    _ = severity_model  # kept for potential later use (e.g. saving/registering); silences lint
    for k, v in severity_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()