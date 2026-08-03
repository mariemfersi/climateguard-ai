"""
Blend the XGBoost (train_frequency.py) and CatBoost (train_catboost.py)
frequency models' predictions.

Simple average blending — appropriate here since both models solve the
identical prediction task on the identical target, and the goal is
variance reduction / diversity capture, not a learned meta-model (which
would need its own held-out validation fold to avoid overfitting the
blend weights, unnecessary complexity for a two-model blend at this scale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.frequency_severity.train_frequency import prepare_features


def blend_frequency_predictions(
    xgb_model,
    xgb_train_columns: pd.Index,
    catboost_model,
    catboost_feature_cols: list[str],
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        xgb_model: fitted XGBClassifier from train_frequency_model().
        xgb_train_columns: the exact column order/schema the XGBoost model
            was trained on (from X_train.columns during training) — needed
            to correctly reindex df's one-hot encoding at inference time.
        catboost_model: fitted CatBoostClassifier from
            train_catboost_frequency_model().
        catboost_feature_cols: the raw feature column list CatBoost was
            trained on (build_training_table.FEATURE_COLUMNS minus
            location_id).
        df: data to predict on.

    Returns:
        (blended_proba, xgb_proba, catboost_proba) — each shape (len(df),)
    """
    X_xgb = prepare_features(df).reindex(columns=xgb_train_columns, fill_value=0)
    xgb_proba = xgb_model.predict_proba(X_xgb)[:, 1]

    X_cat = df[catboost_feature_cols]
    catboost_proba = catboost_model.predict_proba(X_cat)[:, 1]

    blended_proba = (xgb_proba + catboost_proba) / 2.0
    return blended_proba, xgb_proba, catboost_proba


def evaluate_blend(
    y_true: pd.Series, blended: np.ndarray, xgb_proba: np.ndarray, catboost_proba: np.ndarray
) -> dict:
    return {
        "xgb_auc": roc_auc_score(y_true, xgb_proba),
        "catboost_auc": roc_auc_score(y_true, catboost_proba),
        "blended_auc": roc_auc_score(y_true, blended),
    }