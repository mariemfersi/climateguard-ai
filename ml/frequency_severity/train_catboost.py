"""
CatBoost frequency classifier — ensemble diversifier against XGBoost.

CatBoost handles categorical features natively.
"""

from __future__ import annotations

import logging

import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from ml.frequency_severity.build_training_table import FEATURE_COLUMNS
from ml.frequency_severity.train_frequency import CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


def train_catboost_frequency_model(
    train_df,
    test_df,
    target_col: str = "had_claim",
    seed: int = 42,
) -> tuple[CatBoostClassifier, dict]:
    """
    Train CatBoost frequency model.

    Uses only columns available after preprocessing.
    Supports removal of metro_center for anti-memorization experiments.
    """

    # Keep only available features
    feature_cols = [
        c for c in FEATURE_COLUMNS
        if c != "location_id" and c in train_df.columns
    ]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df[target_col]
    y_test = test_df[target_col]


    # Only categorical features that still exist
    cat_features = [
        c for c in CATEGORICAL_FEATURES
        if c in feature_cols
    ]

    cat_feature_idx = [
        feature_cols.index(c)
        for c in cat_features
    ]


    model = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        cat_features=cat_feature_idx,
        verbose=False,
    )


    model.fit(
        X_train,
        y_train
    )


    train_pred = model.predict_proba(X_train)[:,1]
    test_pred = model.predict_proba(X_test)[:,1]


    metrics = {
        "train_auc": roc_auc_score(
            y_train,
            train_pred
        ),

        "test_auc": roc_auc_score(
            y_test,
            test_pred
        ),

        "train_logloss": log_loss(
            y_train,
            train_pred
        ),

        "test_logloss": log_loss(
            y_test,
            test_pred
        ),

        "n_features": len(feature_cols),
        "n_categorical_features": len(cat_features),
    }


    logger.info(
        "CatBoost frequency model metrics: %s",
        metrics
    )


    with mlflow.start_run(
        nested=True
    ):

        mlflow.log_params(
            {
                "iterations":300,
                "depth":5,
                "learning_rate":0.05,
                "cat_features":len(cat_features),
                "model_type":"CatBoost",
                "task":"frequency_classification",
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.catboost.log_model(
            model,
            "catboost_frequency_model"
        )


    return model, metrics