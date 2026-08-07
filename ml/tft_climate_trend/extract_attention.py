"""
Extract attention weights from trained TFT model for explainability.

Provides insights into which time steps and features are most important
for predictions, feeding into Phase 8's explainability layer.

Usage:
    python -m ml.tft_climate_trend.extract_attention
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from ml.tft_climate_trend.prepare_tft_dataset import run_dataset_preparation

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data_pipeline/bronze/tft_climate_trend/tft_model.pt")
ATTENTION_OUTPUT_PATH = Path("data_pipeline/bronze/tft_climate_trend/attention_weights.parquet")


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

    # Reinitialize model with same architecture as training
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size=16,  # Must match training parameters
        attention_head_size=2,
        hidden_continuous_size=8,
        dropout=0.2,
        hidden_layers=2,
        output_size=7,
        loss=None,  # Not needed for inference
    )

    # Load trained weights
    tft.load_state_dict(torch.load(model_path, map_location="cpu"))
    tft.eval()

    logger.info(f"TFT model loaded from {model_path}")
    return tft


def extract_attention_weights(
    tft_model: TemporalFusionTransformer,
    dataset,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """
    Extract attention weights from TFT model for all samples in dataset.

    Args:
        tft_model: Trained TFT model
        dataset: TimeSeriesDataSet to extract attention for
        batch_size: Batch size for inference

    Returns:
        Dictionary containing attention arrays:
        - 'static_attention': Attention weights for static covariates
        - 'variable_attention': Attention weights for time-varying covariates
        - 'temporal_attention': Attention weights across time steps
    """
    logger.info("Extracting attention weights...")

    dataloader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    all_static_attention = []
    all_variable_attention = []
    all_temporal_attention = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass to get attention weights
            output = tft_model(batch)

            # Extract attention weights from model internals
            # TFT stores attention in output.interpretation dict
            interpretation = output.interpretation

            if interpretation is None:
                logger.warning(f"Batch {batch_idx}: No interpretation data available")
                continue

            # Collect different attention types
            if "static" in interpretation:
                all_static_attention.append(interpretation["static"].cpu().numpy())
            if "variable" in interpretation:
                all_variable_attention.append(interpretation["variable"].cpu().numpy())
            if "temporal" in interpretation:
                all_temporal_attention.append(interpretation["temporal"].cpu().numpy())

            # Track sample indices for mapping back to original data
            if hasattr(batch, "x"):
                # Get group_ids (region_id) and time_idx (year)
                if "groups" in batch:
                    all_indices.append(batch["groups"].cpu().numpy())

    # Concatenate all batches
    attention_data = {}

    if all_static_attention:
        attention_data["static_attention"] = np.concatenate(all_static_attention, axis=0)
        logger.info(f"Static attention shape: {attention_data['static_attention'].shape}")

    if all_variable_attention:
        attention_data["variable_attention"] = np.concatenate(all_variable_attention, axis=0)
        logger.info(f"Variable attention shape: {attention_data['variable_attention'].shape}")

    if all_temporal_attention:
        attention_data["temporal_attention"] = np.concatenate(all_temporal_attention, axis=0)
        logger.info(f"Temporal attention shape: {attention_data['temporal_attention'].shape}")

    if all_indices:
        attention_data["indices"] = np.concatenate(all_indices, axis=0)

    logger.info(f"Attention extraction complete. {len(all_indices)} batches processed.")
    return attention_data


def summarize_attention_importance(
    attention_data: dict[str, np.ndarray],
    dataset,
) -> pd.DataFrame:
    """
    Summarize attention weights into feature importance scores.

    Args:
        attention_data: Dictionary of attention arrays from extract_attention_weights
        dataset: TimeSeriesDataSet for feature names

    Returns:
        DataFrame with feature importance scores
    """
    logger.info("Summarizing attention importance...")

    importance_records = []

    # Process variable attention (which features are most important)
    if "variable_attention" in attention_data:
        var_attention = attention_data["variable_attention"]
        # Average across samples and time steps
        mean_var_attention = var_attention.mean(axis=(0, 1))

        # Get feature names from dataset
        feature_names = dataset.reals + dataset.categoricals

        for i, importance in enumerate(mean_var_attention):
            if i < len(feature_names):
                importance_records.append({
                    "feature_type": "time_varying",
                    "feature_name": feature_names[i],
                    "importance_score": float(importance),
                })

    # Process static attention (which static covariates are most important)
    if "static_attention" in attention_data:
        static_attention = attention_data["static_attention"]
        mean_static_attention = static_attention.mean(axis=0)

        static_features = dataset.static_reals + dataset.static_categoricals
        for i, importance in enumerate(mean_static_attention):
            if i < len(static_features):
                importance_records.append({
                    "feature_type": "static",
                    "feature_name": static_features[i],
                    "importance_score": float(importance),
                })

    importance_df = pd.DataFrame(importance_records)
    importance_df = importance_df.sort_values("importance_score", ascending=False)

    logger.info(f"Feature importance summary: {len(importance_df)} features ranked")
    return importance_df


def save_attention_analysis(
    attention_data: dict[str, np.ndarray],
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save attention analysis results to disk.

    Args:
        attention_data: Raw attention arrays
        importance_df: Feature importance summary
        output_path: Path to save results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save feature importance as main output
    importance_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Attention importance saved to {output_path}")

    # Save raw attention arrays as separate files for detailed analysis
    raw_dir = output_path.parent / "raw_attention"
    raw_dir.mkdir(exist_ok=True)

    for key, array in attention_data.items():
        if isinstance(array, np.ndarray):
            np.save(raw_dir / f"{key}.npy", array)
            logger.info(f"Raw attention '{key}' saved to {raw_dir / f'{key}.npy'}")


def run_attention_extraction(
    model_path: Path = MODEL_PATH,
    output_path: Path = ATTENTION_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Run complete attention extraction pipeline.

    Args:
        model_path: Path to trained TFT model
        output_path: Path to save attention analysis results

    Returns:
        Tuple of (importance DataFrame, raw attention data)
    """
    logger.info("Starting attention extraction pipeline...")

    # Load dataset (use full dataset for attention extraction)
    _, _, ts_path = run_dataset_preparation(test_size=0.0, seed=42)

    # Re-load dataset without split for full attention extraction
    from ml.tft_climate_trend.prepare_tft_dataset import build_regional_time_series, compute_regional_climate_summary
    from ml.tft_climate_trend.prepare_tft_dataset import prepare_tft_dataset
    import pandas as pd

    # Quick reload for full dataset
    gold_features = pd.read_parquet("data_pipeline/gold/gold_features.parquet")
    claims = pd.read_parquet("data_pipeline/silver/claims.parquet")
    oni_df = pd.read_parquet("data_pipeline/bronze/climate_indices/oni.parquet")

    climate_summary = compute_regional_climate_summary(
        Path("data_pipeline/bronze/era5/gulf_coast_reanalysis.parquet"),
        oni_df
    )
    regional_ts = build_regional_time_series(gold_features, claims, climate_summary)
    full_dataset = prepare_tft_dataset(regional_ts)

    # Load trained model
    tft_model = load_tft_model(model_path, full_dataset)

    # Extract attention weights
    attention_data = extract_attention_weights(tft_model, full_dataset)

    # Summarize importance
    importance_df = summarize_attention_importance(attention_data, full_dataset)

    # Save results
    save_attention_analysis(attention_data, importance_df, output_path)

    logger.info("Attention extraction pipeline complete.")
    return importance_df, attention_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    importance_df, attention_data = run_attention_extraction()

    logger.info("Top 10 most important features:")
    print(importance_df.head(10).to_string(index=False))
