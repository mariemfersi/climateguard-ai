"""
Quick diagnostics (sampled) for faster iteration.

Produces files in `mlruns/training_diagnostics_quick/`:
- train_test_years.json
- feature_drift.csv
- cv_results.json
- calibration_metrics.json
- calibration_plot.png

Run:
    python scripts/training_diagnostics_quick.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import xgboost as xgb

from ml.frequency_severity.build_training_table import load_and_build, FEATURE_COLUMNS
from ml.frequency_severity.event_split import event_level_train_test_split
from ml.frequency_severity.train_frequency import prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training_diagnostics_quick")

OUT_DIR = Path("mlruns/training_diagnostics_quick")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def year_folds(years: List[int], n_folds: int = 3) -> List[List[int]]:
    years = sorted(years)
    folds = []
    for i in range(n_folds):
        folds.append([y for idx, y in enumerate(years) if idx % n_folds == i])
    return folds


def main():
    df = load_and_build()
    logger.info("Loaded training table rows=%d", len(df))

    # Sample locations to speed up diagnostics
    sample_locations = df['location_id'].unique()[:5000]
    df = df[df['location_id'].isin(sample_locations)].reset_index(drop=True)
    logger.info("Sampled rows=%d (locations=%d)", len(df), len(sample_locations))

    train_df, test_df = event_level_train_test_split(df, test_size=0.2, seed=42)

    train_years = sorted(train_df['year'].unique().tolist())
    test_years = sorted(test_df['year'].unique().tolist())
    save_json(OUT_DIR / "train_test_years.json", {"train_years": train_years, "test_years": test_years})

    # Feature drift
    feature_list = [c for c in FEATURE_COLUMNS if c != 'location_id']
    drift_rows = []
    import pandas as _pd
    from pandas.api.types import is_numeric_dtype

    for c in feature_list:
        if c in train_df.columns and c in test_df.columns:
            # Only compute drift for numeric columns
            try:
                if not is_numeric_dtype(train_df[c]) or not is_numeric_dtype(test_df[c]):
                    continue
                train_mean = float(_pd.to_numeric(train_df[c], errors='coerce').mean())
                test_mean = float(_pd.to_numeric(test_df[c], errors='coerce').mean())
                drift_rows.append({"feature": c, "train_mean": train_mean, "test_mean": test_mean, "diff": test_mean - train_mean})
            except Exception:
                # Skip any problematic column
                continue
    drift_df = pd.DataFrame(drift_rows).sort_values(by='diff', key=lambda s: s.abs(), ascending=False)
    drift_df.to_csv(OUT_DIR / "feature_drift.csv", index=False)
    logger.info("Wrote feature drift")

    years = sorted(df['year'].unique())
    folds = year_folds(years, n_folds=3)

    X_cols = [c for c in FEATURE_COLUMNS if c != 'location_id' and c in df.columns]

    cv_results = {"baseline": [], "regularized": []}

    baseline_params = dict(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    reg_params = dict(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, random_state=42)

    for i, val_years in enumerate(folds):
        train_years_fold = [y for y in years if y not in val_years]
        train_fold = df[df['year'].isin(train_years_fold)].reset_index(drop=True)
        val_fold = df[df['year'].isin(val_years)].reset_index(drop=True)

        X_train = prepare_features(train_fold)
        X_val = prepare_features(val_fold)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        y_train = train_fold['had_claim']
        y_val = val_fold['had_claim']

        model_baseline = xgb.XGBClassifier(**baseline_params, use_label_encoder=False, eval_metric='auc')
        model_baseline.fit(X_train, y_train)
        pred_val = model_baseline.predict_proba(X_val)[:,1]
        auc = float(roc_auc_score(y_val, pred_val))
        cv_results['baseline'].append(auc)

        model_reg = xgb.XGBClassifier(**reg_params, use_label_encoder=False, eval_metric='auc')
        model_reg.fit(X_train, y_train)
        pred_val_reg = model_reg.predict_proba(X_val)[:,1]
        auc_reg = float(roc_auc_score(y_val, pred_val_reg))
        cv_results['regularized'].append(auc_reg)

        logger.info("Fold %d baseline AUC=%.4f regularized AUC=%.4f", i+1, auc, auc_reg)

    cv_summary = {
        "baseline_mean_auc": float(np.mean(cv_results['baseline'])),
        "regularized_mean_auc": float(np.mean(cv_results['regularized'])),
        "baseline_per_fold": cv_results['baseline'],
        "regularized_per_fold": cv_results['regularized'],
    }

    save_json(OUT_DIR / "cv_results.json", cv_summary)
    logger.info("Saved CV summary")

    # Final regularized model on train/test
    X_train_full = prepare_features(train_df)
    X_test_full = prepare_features(test_df)
    X_test_full = X_test_full.reindex(columns=X_train_full.columns, fill_value=0)
    y_train_full = train_df['had_claim']
    y_test_full = test_df['had_claim']

    final_model = xgb.XGBClassifier(**reg_params, use_label_encoder=False, eval_metric='auc')
    final_model.fit(X_train_full, y_train_full)
    test_pred = final_model.predict_proba(X_test_full)[:,1]
    final_metrics = {"test_auc": float(roc_auc_score(y_test_full, test_pred)), "test_logloss": float(log_loss(y_test_full, test_pred)), "test_brier": float(brier_score_loss(y_test_full, test_pred))}

    # Calibration (Platt)
    # Use a small fraction of train years as calibration
    calib_years = np.random.default_rng(123).choice(sorted(train_df['year'].unique()), size=max(1,int(0.15*len(train_df['year'].unique()))), replace=False)
    calib_df = train_df[train_df['year'].isin(calib_years)].reset_index(drop=True)
    train_sub_df = train_df[~train_df['year'].isin(calib_years)].reset_index(drop=True)

    X_train_sub = prepare_features(train_sub_df)
    X_calib = prepare_features(calib_df)
    X_calib = X_calib.reindex(columns=X_train_sub.columns, fill_value=0)
    y_train_sub = train_sub_df['had_claim']
    y_calib = calib_df['had_claim']

    base_model = xgb.XGBClassifier(**reg_params, use_label_encoder=False, eval_metric='auc')
    base_model.fit(X_train_sub, y_train_sub)

    # Use logistic regression on model scores for Platt scaling (works across sklearn versions)
    from sklearn.linear_model import LogisticRegression
    calib_scores = base_model.predict_proba(X_calib)[:, 1].reshape(-1, 1)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(calib_scores, y_calib)

    uncal_pred = base_model.predict_proba(X_test_full)[:, 1]
    cal_pred = lr.predict_proba(uncal_pred.reshape(-1, 1))[:, 1]

    brier_uncal = float(brier_score_loss(y_test_full, uncal_pred))
    brier_cal = float(brier_score_loss(y_test_full, cal_pred))
    auc_uncal = float(roc_auc_score(y_test_full, uncal_pred))
    auc_cal = float(roc_auc_score(y_test_full, cal_pred))

    prob_true, prob_pred = calibration_curve(y_test_full, cal_pred, n_bins=10)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibrated')
    plt.plot([0,1],[0,1], linestyle='--', label='Perfect')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.legend()
    plt.title('Calibration curve (Platt scaling)')
    plt.savefig(OUT_DIR / 'calibration_plot.png')
    plt.close()

    calib_metrics = {"brier_uncal": brier_uncal, "brier_cal": brier_cal, "auc_uncal": auc_uncal, "auc_cal": auc_cal, "calibration_curve": {"prob_pred": prob_pred.tolist(), "prob_true": prob_true.tolist()}}
    save_json(OUT_DIR / "calibration_metrics.json", calib_metrics)

    summary = {"final_metrics": final_metrics, "cv_summary_file": "cv_results.json", "drift_file": "feature_drift.csv", "calibration_file": "calibration_metrics.json"}
    save_json(OUT_DIR / "summary.json", summary)
    logger.info("Quick diagnostics complete; outputs in %s", OUT_DIR)


if __name__ == '__main__':
    main()
