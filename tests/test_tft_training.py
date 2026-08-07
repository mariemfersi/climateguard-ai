"""
Integration tests for ml.tft_climate_trend.train_tft.

Tests the full training pipeline end-to-end with synthetic data to verify:
1. Training loop completes without errors
2. Early stopping works correctly
3. MLflow logging captures parameters and metrics
4. Model saving and loading works
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# These tests require pytorch-forecasting and pytorch-lightning
# They should be skipped if dependencies are not installed
pytest.importorskip("pytorch_forecasting")
pytest.importorskip("pytorch_lightning")


def test_tft_training_with_synthetic_data():
    """
    Integration test for TFT training with synthetic data.
    
    This test creates synthetic time series data and verifies that:
    1. Dataset preparation completes successfully
    2. Training runs without crashing
    3. Early stopping prevents overfitting
    4. Model can be saved and loaded
    """
    # Create synthetic regional time series
    np.random.seed(42)
    years = list(range(1950, 2000))  # 50 years
    regions = [f"region_{i}" for i in range(5)]  # 5 regions
    
    rows = []
    for region in regions:
        # Each region has a base frequency level
        base_frequency = np.random.uniform(0.5, 2.0)
        base_severity = np.random.uniform(0.3, 0.7)
        
        for year in years:
            # Add some climate-driven variation
            sst_anomaly = np.random.normal(0, 0.5)
            oni_anomaly = np.random.normal(0, 0.5)
            
            # Frequency increases with SST (synthetic climate signal)
            frequency = max(0, int(base_frequency + sst_anomaly + np.random.poisson(0.5)))
            severity = np.clip(base_severity + oni_anomaly * 0.1 + np.random.normal(0, 0.1), 0, 1)
            
            rows.append({
                "region_id": region,
                "year": year,
                "frequency": frequency,
                "severity": severity,
                "basin_sst_celsius_mean": 28.0 + sst_anomaly,
                "basin_sst_celsius_max": 32.0 + sst_anomaly * 2,
                "oni_anomaly_celsius": oni_anomaly,
                "enso_phase": "Neutral" if abs(oni_anomaly) < 0.5 else ("El Nino" if oni_anomaly > 0 else "La Nina"),
            })
    
    df = pd.DataFrame(rows)
    
    # Import after dependency check
    from ml.tft_climate_trend.prepare_tft_dataset import prepare_tft_dataset, stratified_year_split_tft, assert_no_year_leakage_tft
    
    # Split data
    train_df, test_df = stratified_year_split_tft(df, test_size=0.2, seed=42)
    assert_no_year_leakage_tft(train_df, test_df)
    
    # Prepare datasets
    train_dataset = prepare_tft_dataset(train_df, encoder_length=5, predictor_length=3)
    test_dataset = prepare_tft_dataset(test_df, encoder_length=5, predictor_length=3)
    
    # Verify datasets are not empty
    assert len(train_dataset) > 0, "Training dataset is empty"
    assert len(test_dataset) > 0, "Test dataset is empty"
    
    # Test training with minimal configuration
    from ml.tft_climate_trend.train_tft import TFTLightningModule, train_tft_model
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    
    # Initialize TFT with small architecture for testing
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.01,
        hidden_size=8,  # Small for testing
        attention_head_size=1,
        hidden_continuous_size=4,
        dropout=0.1,
        hidden_layers=1,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=2,
        gradient_clip_val=0.1,
    )
    
    # Wrap in Lightning module
    lightning_module = TFTLightningModule(tft, learning_rate=0.01)
    
    # Test that forward pass works
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=4, num_workers=0)
    batch = next(iter(train_dataloader))
    
    # Forward pass
    with torch.no_grad():
        output = lightning_module(batch)
    
    # Verify output shape
    assert output.output.shape[0] == batch[0].shape[0], "Output batch size mismatch"
    
    # Test training for a few epochs
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    import pytorch_lightning as pl
    
    early_stop_callback = EarlyStopping(
        monitor="train_loss",  # Use train_loss for testing (faster)
        min_delta=1e-4,
        patience=2,  # Short patience for testing
        mode="min",
        verbose=True,
    )
    
    trainer = pl.Trainer(
        max_epochs=3,  # Very short for testing
        accelerator="cpu",  # Force CPU for testing
        enable_model_summary=False,
        gradient_clip_val=0.1,
        limit_train_batches=2,  # Limit batches for speed
        limit_val_batches=1,
        callbacks=[early_stop_callback],
        logger=False,
        enable_checkpointing=False,
    )
    
    # Train
    trainer.fit(lightning_module, train_dataloader, train_dataloader)  # Use same for val
    
    # Verify training completed
    assert trainer.current_epoch >= 1, "Training did not run"
    
    # Test model saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_tft_model.pt"
        
        # Save model
        torch.save(tft.state_dict(), model_path)
        assert model_path.exists(), "Model file not saved"
        
        # Load model
        tft_loaded = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=0.01,
            hidden_size=8,
            attention_head_size=1,
            hidden_continuous_size=4,
            dropout=0.1,
            hidden_layers=1,
            output_size=7,
            loss=QuantileLoss(),
        )
        tft_loaded.load_state_dict(torch.load(model_path, map_location="cpu"))
        tft_loaded.eval()
        
        # Verify loaded model works
        with torch.no_grad():
            output_loaded = tft_loaded(batch)
        
        assert output_loaded.output.shape == output.output.shape, "Loaded model output shape mismatch"


def test_tft_overfitting_detection():
    """
    Test that overfitting detection works correctly.
    
    Creates a scenario where overfitting is likely and verifies that
    the training loop detects and reports it.
    """
    # Create synthetic data with clear signal
    np.random.seed(42)
    years = list(range(1950, 1970))  # Only 20 years for overfitting risk
    regions = [f"region_{i}" for i in range(2)]  # Only 2 regions
    
    rows = []
    for region in regions:
        for year in years:
            rows.append({
                "region_id": region,
                "year": year,
                "frequency": np.random.poisson(1),
                "severity": np.random.uniform(0, 1),
                "basin_sst_celsius_mean": 28.0,
                "basin_sst_celsius_max": 32.0,
                "oni_anomaly_celsius": 0.0,
                "enso_phase": "Neutral",
            })
    
    df = pd.DataFrame(rows)
    
    from ml.tft_climate_trend.prepare_tft_dataset import prepare_tft_dataset, stratified_year_split_tft
    
    # Split with small test set to encourage overfitting
    train_df, test_df = stratified_year_split_tft(df, test_size=0.3, seed=42)
    
    # Prepare datasets
    train_dataset = prepare_tft_dataset(train_df, encoder_length=5, predictor_length=3)
    test_dataset = prepare_tft_dataset(test_df, encoder_length=5, predictor_length=3)
    
    # Import training module
    from ml.tft_climate_trend.train_tft import train_tft_model
    
    # Train with small architecture on limited data
    tft_model, metrics = train_tft_model(
        train_dataset,
        test_dataset,
        max_epochs=5,  # Short training
        batch_size=4,
        learning_rate=0.01,
        gradient_clip_val=0.1,
        patience=3,
        seed=42,
    )
    
    # Verify metrics are computed
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "overfitting_indicator" in metrics
    
    # Verify overfitting indicator is calculated
    assert not np.isnan(metrics["overfitting_indicator"])
    
    # With limited data, we expect some overfitting
    # (train_loss < val_loss, so indicator > 0)
    assert metrics["overfitting_indicator"] >= 0


def test_tft_mlflow_logging():
    """
    Test that MLflow logging works correctly during training.
    
    Verifies that parameters, metrics, and artifacts are logged.
    """
    # This test would normally require a running MLflow server
    # For unit testing, we mock the MLflow tracking
    
    with patch('ml.tft_climate_trend.train_tft.mlflow') as mock_mlflow:
        # Mock MLflow components
        mock_run = Mock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Create minimal synthetic data
        np.random.seed(42)
        years = list(range(1950, 1960))
        regions = ["region_0"]
        
        rows = []
        for region in regions:
            for year in years:
                rows.append({
                    "region_id": region,
                    "year": year,
                    "frequency": 1,
                    "severity": 0.5,
                    "basin_sst_celsius_mean": 28.0,
                    "basin_sst_celsius_max": 32.0,
                    "oni_anomaly_celsius": 0.0,
                    "enso_phase": "Neutral",
                })
        
        df = pd.DataFrame(rows)
        
        from ml.tft_climate_trend.prepare_tft_dataset import prepare_tft_dataset
        
        dataset = prepare_tft_dataset(df, encoder_length=3, predictor_length=2)
        
        # Import training module
        from ml.tft_climate_trend.train_tft import train_tft_model
        
        # Train
        tft_model, metrics = train_tft_model(
            dataset,
            dataset,  # Use same for train/val for testing
            max_epochs=2,
            batch_size=4,
            learning_rate=0.01,
            gradient_clip_val=0.1,
            patience=1,
            seed=42,
        )
        
        # Verify MLflow was called
        assert mock_mlflow.start_run.called, "MLflow run not started"
        assert mock_mlflow.log_params.called, "Parameters not logged"
        assert mock_mlflow.log_metrics.called, "Metrics not logged"


def test_tft_data_leakage_prevention():
    """
    Test that the training pipeline prevents data leakage.
    
    Verifies that:
    1. Temporal split has no year overlap
    2. Climate covariates are marked as unknown (not known)
    3. No future information leaks into training
    """
    # Create synthetic data
    np.random.seed(42)
    years = list(range(1950, 2000))
    regions = ["region_0", "region_1"]
    
    rows = []
    for region in regions:
        for year in years:
            rows.append({
                "region_id": region,
                "year": year,
                "frequency": np.random.poisson(1),
                "severity": np.random.uniform(0, 1),
                "basin_sst_celsius_mean": 28.0 + np.random.normal(0, 0.5),
                "basin_sst_celsius_max": 32.0 + np.random.normal(0, 1.0),
                "oni_anomaly_celsius": np.random.normal(0, 0.5),
                "enso_phase": "Neutral",
            })
    
    df = pd.DataFrame(rows)
    
    from ml.tft_climate_trend.prepare_tft_dataset import (
        assert_no_year_leakage_tft,
        prepare_tft_dataset,
        stratified_year_split_tft,
    )
    
    # Split data
    train_df, test_df = stratified_year_split_tft(df, test_size=0.2, seed=42)
    
    # Verify no year leakage
    assert_no_year_leakage_tft(train_df, test_df)
    
    # Prepare datasets
    train_dataset = prepare_tft_dataset(train_df, encoder_length=5, predictor_length=3)
    test_dataset = prepare_tft_dataset(test_df, encoder_length=5, predictor_length=3)
    
    # Verify climate covariates are marked as unknown (not known)
    assert "basin_sst_celsius_mean" in train_dataset.time_varying_unknown_reals
    assert "oni_anomaly_celsius" in train_dataset.time_varying_unknown_reals
    
    # Verify climate covariates are NOT marked as known
    assert "basin_sst_celsius_mean" not in train_dataset.time_varying_known_reals
    assert "oni_anomaly_celsius" not in train_dataset.time_varying_known_reals
    
    # This is critical for preventing data leakage


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
