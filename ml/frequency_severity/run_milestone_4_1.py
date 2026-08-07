"""
Milestone 4.1 entrypoint: build the real training table from Gold features
+ claims, train frequency (XGBoost + CatBoost, blended) and severity
(LightGBM) models, and report real metrics.

Usage:
    python -m ml.frequency_severity.run_milestone_4_1

Options:
    --use-regional-encoding: Replace lat/lon with regional clusters to reduce memorization
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np

import mlflow

from ml.mlflow_config import configure_mlflow

from ml.frequency_severity.build_training_table import (
    FEATURE_COLUMNS,
    USE_REGIONAL_ENCODING,
    load_and_build,
)
from ml.frequency_severity.ensemble_blend import (
    blend_frequency_predictions,
    evaluate_blend,
)
from ml.validation import (
    run_validation_gate,
    validate_frequency_model,
    validate_severity_model,
    validate_ensemble,
)
from ml.frequency_severity.train_catboost import (
    train_catboost_frequency_model,
)
from ml.frequency_severity.event_split import (
    assert_no_year_leakage,
    stratified_year_split,
)
from ml.frequency_severity.train_frequency import (
    prepare_features,
    train_frequency_model,
)
from ml.frequency_severity.train_severity import (
    train_severity_model,
)

logger = logging.getLogger(__name__)


# ============================================================
# Temporal split
# ============================================================

def temporal_year_split(
    df,
    year_col="year",
    cutoff_year=2010
):
    """
    Train on historical years.
    Test on future unseen years.

    More realistic for climate risk prediction.
    """

    train_df = df[df[year_col] <= cutoff_year].copy()
    test_df = df[df[year_col] > cutoff_year].copy()

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# Feature diagnostics
# ============================================================

def feature_diagnostics(df):

    print("\n=== Feature correlation with claim occurrence ===")

    numeric_cols = (
        df
        .select_dtypes(include=["number"])
        .columns
    )

    corr = (
        df[numeric_cols]
        .corr()["had_claim"]
        .sort_values(
            ascending=False
        )
    )

    print(corr)


# ============================================================
# Main pipeline
# ============================================================

def run(
    use_regional_encoding: bool = False,
    enforce_validation_gate: bool = False,
) -> None:
    # Configure MLflow tracking (Azure ML or local fallback)
    configure_mlflow()
    
    logger.info(
        "Building training table from real Gold features + claims..."
    )
    
    if use_regional_encoding:
        logger.info("Using regional encoding to replace lat/lon")
    
    training_table = load_and_build(
        use_regional_encoding=use_regional_encoding,
    )

    print("\nTraining table:")
    print(training_table.head())

    print("\nColumns:")
    print(training_table.columns.tolist())


    # ========================================================
    # Diagnostics
    # ========================================================

    feature_diagnostics(training_table)
    print("\n=== distance_to_coast_km vs severity (had_claim==1 only) ===")
    hit_rows = training_table[training_table["had_claim"] == 1]
    print(hit_rows["distance_to_coast_km"].corr(hit_rows["max_damage_ratio"]))
    print(hit_rows["distance_to_coast_km"].corr(hit_rows["incurred_loss_usd"]))



    # ========================================================
    # Remove geographic memorization
    # ========================================================

    if "metro_center" in training_table.columns:

        logger.info(
            "Removing metro_center to avoid geographic memorization"
        )

        training_table = training_table.drop(
            columns=[
                "metro_center"
            ]
        )



    # ========================================================
    # Temporal validation
    # ========================================================

    logger.info(
        "Performing stratified year split for frequency model..."
    )

    # Use stratified year split for frequency model to confirm AUC ~0.50
    train_df, test_df = stratified_year_split(
        training_table,
        test_size=0.2,
        seed=42,
    )

    assert_no_year_leakage(
        train_df,
        test_df
    )

    logger.info(
        "Train: %d rows (%d years)",
        len(train_df),
        train_df["year"].nunique()
    )

    logger.info(
        "Test: %d rows (%d years)",
        len(test_df),
        test_df["year"].nunique()
    )



    # ========================================================
    # MLflow
    # ========================================================

    with mlflow.start_run(
        run_name=
        "milestone_4_1_frequency_severity_temporal"
    ):


        mlflow.log_param(
            "split_type",
            "stratified_year"
        )

        mlflow.log_param(
            "test_size",
            0.2
        )
        
        mlflow.log_param("use_regional_encoding", use_regional_encoding)



        # ====================================================
        # Frequency XGBoost
        # ====================================================

        print(
            "\n=== Frequency Model (XGBoost) ==="
        )


        xgb_model, xgb_metrics = (
            train_frequency_model(
                train_df,
                test_df
            )
        )


        for k,v in xgb_metrics.items():
            print(
                f"{k}: {v}"
            )
        
        # Validate XGBoost frequency model
        xgb_validation = validate_frequency_model(
            train_auc=xgb_metrics["train_auc"],
            test_auc=xgb_metrics["test_auc"]
        )
        print(f"\nXGBoost Validation: {xgb_validation}")



        # ====================================================
        # Feature importance
        # ====================================================

        print(
            "\n=== XGBoost Feature Importance ==="
        )


        booster = (
            xgb_model
            .get_booster()
        )


        importance = (
            booster
            .get_score(
                importance_type="gain"
            )
        )


        for feat,score in sorted(
            importance.items(),
            key=lambda x:-x[1]
        )[:20]:

            print(
                f"{feat}: {score:.3f}"
            )



        # ====================================================
        # CatBoost
        # ====================================================

        print(
            "\n=== Frequency Model (CatBoost) ==="
        )


        catboost_model, catboost_metrics = (
            train_catboost_frequency_model(
                train_df,
                test_df
            )
        )


        for k,v in catboost_metrics.items():
            print(
                f"{k}: {v}"
            )
        
        # Validate CatBoost frequency model
        catboost_validation = validate_frequency_model(
            train_auc=catboost_metrics["train_auc"],
            test_auc=catboost_metrics["test_auc"]
        )
        print(f"\nCatBoost Validation: {catboost_validation}")



        # ====================================================
        # Ensemble
        # ====================================================

        print(
            "\n=== Ensemble ==="
        )


        xgb_columns = (
            prepare_features(train_df)
            .columns
        )


        feature_cols = [
            c for c in FEATURE_COLUMNS
            if c != "location_id"
        ]


        blended, xgb_prob, cat_prob = (
            blend_frequency_predictions(
                xgb_model,
                xgb_columns,
                catboost_model,
                feature_cols,
                test_df
            )
        )


        blend_metrics = (
            evaluate_blend(
                test_df["had_claim"],
                blended,
                xgb_prob,
                cat_prob
            )
        )


        for k,v in blend_metrics.items():
            print(
                f"{k}: {v}"
            )

        mlflow.log_params({
            "ensemble_blend": "average",
            "ensemble_base_models": "xgboost,catboost",
        })
        mlflow.log_metrics(blend_metrics)
        
        # Validate ensemble
        ensemble_validation = validate_ensemble(
            xgb_test_auc=xgb_metrics["test_auc"],
            catboost_test_auc=catboost_metrics["test_auc"],
            blended_test_auc=blend_metrics["blended_auc"]
        )
        print(f"\nEnsemble Validation: {ensemble_validation}")



        # ====================================================
        # Severity
        # ====================================================

        print(
            "\n=== Severity Model ==="
        )


        # log transform heavy-tail losses

        train_df = train_df.copy()
        test_df = test_df.copy()


        train_df["incurred_loss_usd"] = np.log1p(
            train_df["incurred_loss_usd"]
        )

        test_df["incurred_loss_usd"] = np.log1p(
            test_df["incurred_loss_usd"]
        )


        severity_model, severity_metrics = (
            train_severity_model(
                train_df,
                test_df
            )
        )


        for k,v in severity_metrics.items():
            print(
                f"{k}: {v}"
            )
        
        # Validate severity model
        severity_validation = validate_severity_model(
            train_r2=severity_metrics["train_r2"],
            test_r2=severity_metrics["test_r2"]
        )
        print(f"\nSeverity Validation: {severity_validation}")
        
        # Run validation gate - blocks registration if any validation fails
        # Set enforce=False for development (logs warnings but doesn't crash)
        # Set enforce=True for production to actually block registration
        run_validation_gate(
            frequency_result=xgb_validation,
            severity_result=severity_validation,
            ensemble_result=ensemble_validation,
            enforce=enforce_validation_gate,
        )



        # ====================================================
        # Summary
        # ====================================================

        print(
            "\n=============================="
        )

        print(
            "Milestone 4.1 Summary"
        )

        print(
            "=============================="
        )


        print(
            f"Frequency AUC: "
            f"{blend_metrics['blended_auc']:.4f}"
        )


        print(
            f"Severity R2: "
            f"{severity_metrics['test_r2']:.4f}"
        )


        print(
            "Train years:",
            sorted(train_df.year.unique())
        )

        print(
            "Test years:",
            sorted(test_df.year.unique())
        )
        
        print(
            f"Feature configuration: regional_encoding={use_regional_encoding}"
        )



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(
        description="Train frequency-severity models with improved climate features"
    )
    parser.add_argument(
        "--use-regional-encoding",
        action="store_true",
        help="Replace lat/lon with regional clusters to reduce memorization"
    )
    parser.add_argument(
        "--enforce-validation-gate",
        action="store_true",
        help="Raise an exception if any validation gate check fails"
    )
    
    args = parser.parse_args()
    
    run(
        use_regional_encoding=args.use_regional_encoding,
        enforce_validation_gate=args.enforce_validation_gate,
    )