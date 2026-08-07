"""
Run full-scale frequency model training pipeline.

Steps:
- Load gold + claims + yearly_sst via build_training_table.load_and_build
- Split by event-year using event_level_train_test_split
- Train XGBoost and CatBoost frequency models
- Save metrics summary to `mlruns/full_training_summary.json`

Run:
    python scripts/run_full_training.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ml.frequency_severity.build_training_table import load_and_build
from ml.frequency_severity.event_split import event_level_train_test_split
from ml.frequency_severity.train_frequency import train_frequency_model
from ml.frequency_severity.train_catboost import train_catboost_frequency_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_full_training")

OUT_DIR = Path("mlruns")
OUT_DIR.mkdir(exist_ok=True)


def main():
    logger.info("Loading and building training table (this may take a while)...")
    training = load_and_build()
    logger.info("Built training table: %d rows", len(training))

    logger.info("Splitting train/test by event-year...")
    train_df, test_df = event_level_train_test_split(training, test_size=0.2, seed=42)
    logger.info("Train rows=%d test rows=%d", len(train_df), len(test_df))

    logger.info("Training XGBoost frequency model...")
    xgb_model, xgb_metrics = train_frequency_model(train_df, test_df, seed=42)

    logger.info("Training CatBoost frequency model...")
    cat_model, cat_metrics = train_catboost_frequency_model(train_df, test_df, seed=42)

    summary = {
        "xgb_metrics": xgb_metrics,
        "cat_metrics": cat_metrics,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    out_path = OUT_DIR / "full_training_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote metrics summary to %s", out_path)


if __name__ == "__main__":
    main()
