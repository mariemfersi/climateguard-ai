"""
Frequency model: XGBoost classifier predicting the probability that a
given location has 1+ claim in a given hurricane season, using only
features knowable at pricing time (see build_training_table.py's
methodology note for why storm-specific features are deliberately
excluded).

This is the "frequency" half of the classical actuarial two-part
frequency-severity model — see train_severity.py for the "severity" half
(expected damage GIVEN a claim occurred).
"""

from __future__ import annotations

import logging

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from ml.frequency_severity.build_training_table import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ["metro_center", "construction_class", "roof_type"]
NUMERIC_FEATURES = [
    c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES and c != "location_id"
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features, keep numeric features as-is.
    Shared between frequency and severity models so both operate on an
    identical feature representation.
    """
    encoded = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    return pd.concat(
        [df[NUMERIC_FEATURES].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1
    )


def train_frequency_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "had_claim",
    seed: int = 42,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Train an XGBoost classifier for claim frequency.

    Returns:
        (fitted model, metrics dict with at least 'train_auc', 'test_auc')
    """
    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)
    # Align columns (test set's one-hot encoding might be missing a rare
    # category present in train, or vice versa) — reindex to train's schema.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

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
    model.fit(X_train, y_train)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_auc": roc_auc_score(y_train, train_pred),
        "test_auc": roc_auc_score(y_test, test_pred),
        "train_base_rate": float(y_train.mean()),
        "test_base_rate": float(y_test.mean()),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    logger.info("Frequency model metrics: %s", metrics)
    return model, metrics


def assert_no_year_leakage(
    train_df: pd.DataFrame, test_df: pd.DataFrame, year_col: str = "year"
) -> None:
    """
    Hard runtime guard: raises if any year appears in BOTH train and test.
    Intended to be called before training in any pipeline entrypoint, so a
    future refactor that accidentally reintroduces row-level splitting
    fails loudly and immediately rather than silently inflating metrics.
    """
    train_years = set(train_df[year_col].unique())
    test_years = set(test_df[year_col].unique())
    overlap = train_years & test_years
    if overlap:
        raise AssertionError(
            f"Event-level split violated — {len(overlap)} year(s) appear in "
            f"BOTH train and test: {sorted(overlap)}"
        )