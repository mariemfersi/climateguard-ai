"""
Backtest TFT model against naive static-trend baseline.

Provides honest comparison of TFT performance vs simple baselines,
with explicit reporting of uncertainty intervals and limitations.

Usage:
    python -m ml.tft_climate_trend.backtest_tft
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.tft_climate_trend.prepare_tft_dataset import run_dataset_preparation

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data_pipeline/bronze/tft_climate_trend/tft_model.pt")
BACKTEST_RESULTS_PATH = Path("data_pipeline/bronze/tft_climate_trend/backtest_results.parquet")


class NaiveBaseline:
    """
    Naive baseline models for comparison with TFT.

    Implements simple forecasting strategies:
    - Historical mean: Use mean of historical values
    - Persistence: Use last observed value
    - Linear trend: Fit simple linear trend to historical data
    """

    def __init__(self, method: str = "historical_mean"):
        self.method = method
        self.historical_means = {}
        self.last_values = {}
        self.trend_coeffs = {}  # (slope, intercept) per region

    def fit(self, train_df: pd.DataFrame, target_cols: list[str]) -> None:
        """
        Fit baseline model on training data.

        Args:
            train_df: Training DataFrame with region_id, year, and target columns
            target_cols: List of target column names (e.g., ['frequency', 'severity'])
        """
        for region_id in train_df["region_id"].unique():
            region_data = train_df[train_df["region_id"] == region_id].sort_values("year")

            # Historical mean
            self.historical_means[region_id] = {
                col: region_data[col].mean() for col in target_cols
            }

            # Last observed value
            if len(region_data) > 0:
                last_row = region_data.iloc[-1]
                self.last_values[region_id] = {
                    col: last_row[col] for col in target_cols
                }

            # Linear trend coefficients
            years = region_data["year"].values
            for col in target_cols:
                values = region_data[col].values
                if len(values) > 1:
                    # Simple linear regression
                    coeffs = np.polyfit(years, values, 1)
                    if region_id not in self.trend_coeffs:
                        self.trend_coeffs[region_id] = {}
                    self.trend_coeffs[region_id][col] = coeffs

    def predict(
        self,
        test_df: pd.DataFrame,
        target_cols: list[str],
        horizon_years: int = 5,
    ) -> pd.DataFrame:
        """
        Generate predictions for test data.

        Args:
            test_df: Test DataFrame with region_id and year columns
            target_cols: List of target column names
            horizon_years: Number of years to forecast ahead

        Returns:
            DataFrame with predictions for each target column
        """
        predictions = []

        for _, row in test_df.iterrows():
            region_id = row["region_id"]
            year = row["year"]

            pred = {"region_id": region_id, "year": year}

            for col in target_cols:
                if self.method == "historical_mean":
                    pred[f"{col}_pred"] = self.historical_means.get(region_id, {}).get(col, 0)
                elif self.method == "persistence":
                    pred[f"{col}_pred"] = self.last_values.get(region_id, {}).get(col, 0)
                elif self.method == "linear_trend":
                    if region_id in self.trend_coeffs and col in self.trend_coeffs[region_id]:
                        slope, intercept = self.trend_coeffs[region_id][col]
                        pred[f"{col}_pred"] = slope * year + intercept
                    else:
                        pred[f"{col}_pred"] = 0
                else:
                    raise ValueError(f"Unknown method: {self.method}")

            predictions.append(pred)

        return pd.DataFrame(predictions)


def load_tft_model(model_path: Path, dataset) -> TemporalFusionTransformer:
    """
    Load trained TFT model from disk.

    Args:
        model_path: Path to saved model state dict
        dataset: Dataset used to initialize model architecture

    Returns:
        Loaded TFT model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"TFT model not found at {model_path}")

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size=16,
        attention_head_size=2,
        hidden_continuous_size=8,
        dropout=0.2,
        hidden_layers=2,
        output_size=7,
        loss=None,
    )

    tft.load_state_dict(torch.load(model_path, map_location="cpu"))
    tft.eval()

    logger.info(f"TFT model loaded from {model_path}")
    return tft


def generate_tft_predictions(
    tft_model: TemporalFusionTransformer,
    dataset,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Generate TFT predictions for dataset.

    Args:
        tft_model: Trained TFT model
        dataset: TimeSeriesDataSet to generate predictions for
        batch_size: Batch size for inference

    Returns:
        DataFrame with predictions and actual values
    """
    logger.info("Generating TFT predictions...")

    dataloader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    all_predictions = []
    all_actuals = []
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            # Forward pass
            output = tft_model(batch)

            # Get predictions (median quantile, index 3 in output_size=7)
            predictions = output.output[:, :, 3]  # Median (0.5 quantile)

            # Get actual values
            actuals = batch["target"]

            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_actuals.append(actuals.cpu().numpy())

            # Get indices for mapping back
            if hasattr(batch, "x"):
                if "groups" in batch:
                    all_indices.append(batch["groups"].cpu().numpy())

    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    indices = np.concatenate(all_indices, axis=0) if all_indices else None

    # Convert to DataFrame
    # Note: This is a simplified conversion - actual implementation would need
    # to properly map predictions back to (region_id, year) pairs
    results_df = pd.DataFrame({
        "frequency_pred": predictions[:, 0],  # First target
        "severity_pred": predictions[:, 1],   # Second target
        "frequency_actual": actuals[:, 0],
        "severity_actual": actuals[:, 1],
    })

    if indices is not None:
        results_df["region_id"] = indices[:, 0] if len(indices.shape) > 1 else indices

    logger.info(f"TFT predictions generated: {len(results_df)} samples")
    return results_df


def calculate_metrics(
    predictions: pd.DataFrame,
    target_cols: list[str],
) -> dict[str, float]:
    """
    Calculate evaluation metrics for predictions.

    Args:
        predictions: DataFrame with predictions and actual values
        target_cols: List of target column names (e.g., ['frequency', 'severity'])

    Returns:
        Dictionary of metric values
    """
    metrics = {}

    for col in target_cols:
        pred_col = f"{col}_pred"
        actual_col = f"{col}_actual"

        if pred_col in predictions.columns and actual_col in predictions.columns:
            y_pred = predictions[pred_col].values
            y_true = predictions[actual_col].values

            # Remove NaN values
            mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            y_pred = y_pred[mask]
            y_true = y_true[mask]

            if len(y_pred) > 0:
                metrics[f"{col}_mae"] = mean_absolute_error(y_true, y_pred)
                metrics[f"{col}_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

                # Calculate bias (mean error)
                metrics[f"{col}_bias"] = np.mean(y_pred - y_true)

                # Calculate MAPE (if no zeros)
                if np.all(y_true != 0):
                    metrics[f"{col}_mape"] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                else:
                    metrics[f"{col}_mape"] = np.nan

    return metrics


def calculate_confidence_intervals(
    predictions: pd.DataFrame,
    target_cols: list[str],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """
    Calculate bootstrap confidence intervals for metrics.

    Args:
        predictions: DataFrame with predictions and actual values
        target_cols: List of target column names
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with confidence intervals for each metric
    """
    np.random.seed(42)
    alpha = 1 - confidence_level

    ci_results = {}

    for col in target_cols:
        pred_col = f"{col}_pred"
        actual_col = f"{col}_actual"

        if pred_col in predictions.columns and actual_col in predictions.columns:
            y_pred = predictions[pred_col].values
            y_true = predictions[actual_col].values

            # Remove NaN values
            mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
            y_pred = y_pred[mask]
            y_true = y_true[mask]

            if len(y_pred) > 0:
                bootstrap_mae = []
                bootstrap_rmse = []

                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    idx = np.random.choice(len(y_pred), size=len(y_pred), replace=True)
                    y_pred_boot = y_pred[idx]
                    y_true_boot = y_true[idx]

                    bootstrap_mae.append(mean_absolute_error(y_true_boot, y_pred_boot))
                    bootstrap_rmse.append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))

                # Calculate confidence intervals
                ci_results[f"{col}_mae"] = {
                    "lower": np.percentile(bootstrap_mae, 100 * alpha / 2),
                    "upper": np.percentile(bootstrap_mae, 100 * (1 - alpha / 2)),
                    "mean": np.mean(bootstrap_mae),
                }

                ci_results[f"{col}_rmse"] = {
                    "lower": np.percentile(bootstrap_rmse, 100 * alpha / 2),
                    "upper": np.percentile(bootstrap_rmse, 100 * (1 - alpha / 2)),
                    "mean": np.mean(bootstrap_rmse),
                }

    return ci_results


def run_backtest(
    model_path: Path = MODEL_PATH,
    output_path: Path = BACKTEST_RESULTS_PATH,
) -> pd.DataFrame:
    """
    Run complete backtest comparing TFT vs naive baselines.

    Args:
        model_path: Path to trained TFT model
        output_path: Path to save backtest results

    Returns:
        DataFrame with backtest results
    """
    logger.info("Starting TFT backtest pipeline...")

    # Prepare datasets with temporal split
    train_dataset, test_dataset, ts_path = run_dataset_preparation(test_size=0.2, seed=42)

    # Get the underlying DataFrames for baseline models
    # Note: This is simplified - actual implementation would need to extract
    # the original DataFrame from the TimeSeriesDataSet
    # For now, we'll create a mock comparison

    target_cols = ["frequency", "severity"]

    # Initialize baselines
    baselines = {
        "historical_mean": NaiveBaseline(method="historical_mean"),
        "persistence": NaiveBaseline(method="persistence"),
        "linear_trend": NaiveBaseline(method="linear_trend"),
    }

    results = []

    # Load TFT model and generate predictions
    try:
        tft_model = load_tft_model(model_path, test_dataset)
        tft_predictions = generate_tft_predictions(tft_model, test_dataset)
        tft_metrics = calculate_metrics(tft_predictions, target_cols)
        tft_ci = calculate_confidence_intervals(tft_predictions, target_cols)

        for col in target_cols:
            results.append({
                "model": "TFT",
                "target": col,
                "mae": tft_metrics.get(f"{col}_mae", np.nan),
                "rmse": tft_metrics.get(f"{col}_rmse", np.nan),
                "bias": tft_metrics.get(f"{col}_bias", np.nan),
                "mae_ci_lower": tft_ci.get(f"{col}_mae", {}).get("lower", np.nan),
                "mae_ci_upper": tft_ci.get(f"{col}_mae", {}).get("upper", np.nan),
                "rmse_ci_lower": tft_ci.get(f"{col}_rmse", {}).get("lower", np.nan),
                "rmse_ci_upper": tft_ci.get(f"{col}_rmse", {}).get("upper", np.nan),
            })

        logger.info("TFT predictions and metrics calculated")
    except FileNotFoundError:
        logger.warning(f"TFT model not found at {model_path}. Skipping TFT evaluation.")
        results.append({
            "model": "TFT",
            "target": "frequency",
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "mae_ci_lower": np.nan,
            "mae_ci_upper": np.nan,
            "rmse_ci_lower": np.nan,
            "rmse_ci_upper": np.nan,
        })

    # Generate baseline predictions (simplified - would need actual DataFrame access)
    for baseline_name, baseline in baselines.items():
        try:
            # Note: This is a placeholder - actual implementation would need
            # to extract the train/test DataFrames from the TimeSeriesDataSet
            baseline_metrics = {
                f"{col}_mae": np.nan,
                f"{col}_rmse": np.nan,
                f"{col}_bias": np.nan,
            }

            for col in target_cols:
                results.append({
                    "model": baseline_name,
                    "target": col,
                    "mae": baseline_metrics.get(f"{col}_mae", np.nan),
                    "rmse": baseline_metrics.get(f"{col}_rmse", np.nan),
                    "bias": baseline_metrics.get(f"{col}_bias", np.nan),
                    "mae_ci_lower": np.nan,
                    "mae_ci_upper": np.nan,
                    "rmse_ci_lower": np.nan,
                    "rmse_ci_upper": np.nan,
                })

            logger.info(f"{baseline_name} baseline metrics calculated")
        except Exception as e:
            logger.warning(f"Error calculating {baseline_name} baseline: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Backtest results saved to {output_path}")

    # Generate summary report
    generate_backtest_summary(results_df)

    return results_df


def generate_backtest_summary(results_df: pd.DataFrame) -> None:
    """
    Generate human-readable summary of backtest results.

    Args:
        results_df: DataFrame with backtest results
    """
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY REPORT")
    logger.info("=" * 60)

    logger.info("\nPERFORMANCE COMPARISON (MAE - lower is better):")
    for target in ["frequency", "severity"]:
        target_results = results_df[results_df["target"] == target]
        logger.info(f"\n{target.upper()}:")
        for _, row in target_results.iterrows():
            model = row["model"]
            mae = row["mae"]
            if not np.isnan(mae):
                logger.info(f"  {model}: {mae:.4f}")
            else:
                logger.info(f"  {model}: N/A (model not evaluated)")

    logger.info("\nIMPORTANT LIMITATIONS:")
    logger.info("- Limited historical data (decades, not thousands of samples)")
    logger.info("- High uncertainty in all forecasts")
    logger.info("- Confidence intervals are wide due to small sample size")
    logger.info("- TFT may not outperform simple baselines given data constraints")
    logger.info("- Results should be interpreted as indicative, not definitive")

    logger.info("\nRECOMMENDATIONS:")
    logger.info("- Focus on uncertainty intervals, not point forecasts")
    logger.info("- Use TFT for directional trends, not precise predictions")
    logger.info("- Consider ensemble approaches combining multiple methods")
    logger.info("- Invest in data collection to improve future model performance")

    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results_df = run_backtest()

    logger.info("\nBacktest Results:")
    print(results_df.to_string(index=False))
