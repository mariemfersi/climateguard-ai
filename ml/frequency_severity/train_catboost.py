"""
CatBoost frequency classifier — ensemble diversifier against the XGBoost
frequency model (train_frequency.py).

DESIGN RATIONALE (per design doc §5): CatBoost handles high-cardinality
categorical features (here: metro_center, construction_class, roof_type)
NATIVELY, without the manual one-hot encoding that train_frequency.py's
XGBoost model requires. This reduces target-leakage/overfitting risk from
manual categorical encoding and gives genuine model diversity for
ensembling — not just two copies of the same algorithm with different
hyperparameters.
"""

from __future__ import annotations

import logging

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from ml.frequency_severity.build_training_table import FEATURE_COLUMNS
from ml.frequency_severity.train_frequency import CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)

_FEATURE_COLS = [c for c in FEATURE_COLUMNS if c != "location_id"]


def train_catboost_frequency_model(
    train_df,
    test_df,
    target_col: str = "had_claim",
    seed: int = 42,
) -> tuple[CatBoostClassifier, dict]:
    """
    Train a CatBoost classifier for claim frequency, using raw categorical
    columns directly (CatBoost's native categorical handling) rather than
    one-hot encoding.
    """
    X_train = train_df[_FEATURE_COLS].copy()
    X_test = test_df[_FEATURE_COLS].copy()
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    cat_feature_idx = [_FEATURE_COLS.index(c) for c in CATEGORICAL_FEATURES]

    model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        cat_features=cat_feature_idx,
        random_seed=seed,
        verbose=False,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_auc": roc_auc_score(y_train, train_pred),
        "test_auc": roc_auc_score(y_test, test_pred),
    }
    logger.info("CatBoost frequency model metrics: %s", metrics)
    return model, metrics