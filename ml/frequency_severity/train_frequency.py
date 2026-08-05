"""
Frequency model: XGBoost classifier predicting probability of at least one
claim occurring at a location during a hurricane season.

Uses only pricing-time features.
"""

from __future__ import annotations

import logging

import mlflow
import pandas as pd
import xgboost as xgb

from sklearn.metrics import log_loss, roc_auc_score

from ml.frequency_severity.build_training_table import FEATURE_COLUMNS


logger = logging.getLogger(__name__)


# Candidate categorical features
# Some may be removed upstream (example: metro_center)
CATEGORICAL_FEATURES = [
    "metro_center",
    "construction_class",
    "roof_type",
]


NUMERIC_FEATURES = [
    c
    for c in FEATURE_COLUMNS
    if c not in CATEGORICAL_FEATURES
    and c != "location_id"
]


def prepare_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare model features.

    - Keeps numerical variables unchanged.
    - One-hot encodes available categorical variables.
    - Handles removed columns gracefully.
    """

    # Keep only categorical columns that exist
    available_categorical = [
        c
        for c in CATEGORICAL_FEATURES
        if c in df.columns
    ]


    # Keep only numerical columns that exist
    available_numeric = [
        c
        for c in NUMERIC_FEATURES
        if c in df.columns
    ]


    X_numeric = (
        df[available_numeric]
        .copy()
        .reset_index(drop=True)
    )


    if available_categorical:

        X_categorical = pd.get_dummies(
            df[available_categorical],
            prefix=available_categorical,
            dtype=int
        ).reset_index(drop=True)


        X = pd.concat(
            [
                X_numeric,
                X_categorical
            ],
            axis=1
        )

    else:

        X = X_numeric


    # Replace missing values
    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    X = X.fillna(0)


    return X



def train_frequency_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "had_claim",
    seed: int = 42,
) -> tuple[xgb.XGBClassifier, dict]:


    X_train = prepare_features(train_df)

    X_test = prepare_features(test_df)


    # Ensure identical feature space
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )


    y_train = train_df[target_col]
    y_test = test_df[target_col]


    model = xgb.XGBClassifier(

        n_estimators=200,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        eval_metric="auc",

        random_state=seed,
    )


    model.fit(
        X_train,
        y_train
    )


    train_pred = model.predict_proba(
        X_train
    )[:,1]


    test_pred = model.predict_proba(
        X_test
    )[:,1]



    metrics = {

        "train_auc":
            roc_auc_score(
                y_train,
                train_pred
            ),

        "test_auc":
            roc_auc_score(
                y_test,
                test_pred
            ),


        "train_logloss":
            log_loss(
                y_train,
                train_pred
            ),


        "test_logloss":
            log_loss(
                y_test,
                test_pred
            ),


        "train_base_rate":
            float(
                y_train.mean()
            ),


        "test_base_rate":
            float(
                y_test.mean()
            ),


        "n_train":
            len(train_df),


        "n_test":
            len(test_df),
    }


    logger.info(
        "Frequency model metrics: %s",
        metrics
    )


    with mlflow.start_run(
        nested=True
    ):

        mlflow.log_params({

            "model_type":
                "XGBoost",

            "task":
                "frequency_classification",

            "n_estimators":
                200,

            "max_depth":
                4,

            "learning_rate":
                0.05,

            "subsample":
                0.8,

            "colsample_bytree":
                0.8,

        })


        mlflow.log_metrics(
            metrics
        )


        mlflow.xgboost.log_model(
            model,
            "xgboost_frequency_model"
        )


    return model, metrics



def assert_no_year_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    year_col: str = "year"
) -> None:


    train_years = set(
        train_df[year_col].unique()
    )

    test_years = set(
        test_df[year_col].unique()
    )


    overlap = (
        train_years
        &
        test_years
    )


    if overlap:

        raise AssertionError(
            f"Event leakage detected: {sorted(overlap)}"
        )