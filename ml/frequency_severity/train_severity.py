"""
Severity model: LightGBM regressor predicting damage ratio given a claim.

Two-part actuarial model:
1) Frequency:
   P(claim occurs)

2) Severity:
   E(damage ratio | claim occurs)

Uses only pricing-time features.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import mlflow
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score

from ml.frequency_severity.train_frequency import prepare_features

logger = logging.getLogger(__name__)


MONOTONIC_INCREASING_FEATURE = "basin_sst_celsius_mean"



def build_monotone_constraints(
    columns: pd.Index
) -> list[int]:
    """
    Apply monotonic constraint:
    
    Higher basin SST should not reduce expected severity.

    1 = increasing
    0 = unconstrained
    """

    return [
        1 if col == MONOTONIC_INCREASING_FEATURE else 0
        for col in columns
    ]



def train_severity_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "max_damage_ratio",
    seed: int = 42,
):

    """
    Train severity model only on locations-years where a claim happened.
    """


    train_claims = train_df[
        train_df["had_claim"] == 1
    ]

    test_claims = test_df[
        test_df["had_claim"] == 1
    ]


    if len(train_claims) == 0:
        raise ValueError(
            "No claims in training data."
        )

    if len(test_claims) == 0:
        raise ValueError(
            "No claims in test data."
        )



    # -------------------------------
    # Feature preparation
    # -------------------------------

    X_train = prepare_features(
        train_claims
    )

    X_test = prepare_features(
        test_claims
    )


    # align columns
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )


    y_train = train_claims[target_col]
    y_test = test_claims[target_col]



    constraints = build_monotone_constraints(
        X_train.columns
    )



    model = lgb.LGBMRegressor(

        n_estimators=300,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        monotone_constraints=constraints,

        random_state=seed,

        verbosity=-1
    )



    model.fit(
        X_train,
        y_train
    )



    train_pred = model.predict(
        X_train
    )

    test_pred = model.predict(
        X_test
    )



    metrics = {

        "train_mae":
            mean_absolute_error(
                y_train,
                train_pred
            ),

        "test_mae":
            mean_absolute_error(
                y_test,
                test_pred
            ),

        "train_r2":
            r2_score(
                y_train,
                train_pred
            ),

        "test_r2":
            r2_score(
                y_test,
                test_pred
            ),

        "n_train":
            len(train_claims),

        "n_test":
            len(test_claims)

    }



    logger.info(
        "Severity model metrics: %s",
        metrics
    )



    # -------------------------------
    # MLflow
    # -------------------------------

    with mlflow.start_run(
        nested=True
    ):

        mlflow.log_params(
            {

            "n_estimators":300,

            "max_depth":5,

            "learning_rate":0.05,

            "model_type":
                "LightGBM",

            "task":
                "severity_regression",

            "monotonic_feature":
                MONOTONIC_INCREASING_FEATURE

            }
        )


        mlflow.log_metrics(
            metrics
        )


        mlflow.lightgbm.log_model(
            model,
            "lightgbm_severity_model"
        )



    return model, metrics