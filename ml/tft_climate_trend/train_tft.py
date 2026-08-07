"""
TFT training loop with early stopping and regularization.

Trains a Temporal Fusion Transformer to forecast multi-horizon climate trends
while preventing overfitting on limited historical data.

Usage:
    python -m ml.tft_climate_trend.train_tft
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ml.tft_climate_trend.prepare_tft_dataset import (
    ENCODER_LENGTH,
    PREDICTOR_LENGTH,
    run_dataset_preparation,
)

logger = logging.getLogger(__name__)

# TFT hyperparameters - conservative for limited data
TFT_HIDDEN_SIZE = 16  # Reduced from default 64 to prevent overfitting
TFT_ATTENTION_HEADS = 2  # Reduced from default 4
TFT_HIDDEN_LAYERS = 2  # Reduced from default 4
TFT_DROPOUT = 0.2  # Increased dropout for regularization
TFT_LEARNING_RATE = 0.01  # Conservative learning rate
TFT_GRADIENT_CLIP_VAL = 0.1  # Strong gradient clipping
TFT_MAX_EPOCHS = 50  # Early stopping will likely stop earlier
TFT_BATCH_SIZE = 32  # Smaller batches for better generalization

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs
MIN_DELTA = 1e-4  # Minimum change to qualify as improvement


class TFTLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for TFT with custom training logic.
    """

    def __init__(self, tft_model: TemporalFusionTransformer, learning_rate: float = TFT_LEARNING_RATE):
        super().__init__()
        self.tft_model = tft_model
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.tft_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.tft_model.loss(x, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.tft_model.loss(x, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_tft_model(
    train_dataset,
    val_dataset,
    max_epochs: int = TFT_MAX_EPOCHS,
    batch_size: int = TFT_BATCH_SIZE,
    learning_rate: float = TFT_LEARNING_RATE,
    gradient_clip_val: float = TFT_GRADIENT_CLIP_VAL,
    patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = 42,
) -> tuple[TemporalFusionTransformer, dict]:
    """
    Train TFT model with early stopping and regularization.

    Args:
        train_dataset: Training TimeSeriesDataSet
        val_dataset: Validation TimeSeriesDataSet
        max_epochs: Maximum training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        gradient_clip_val: Gradient clipping value
        patience: Early stopping patience
        seed: Random seed for reproducibility

    Returns:
        Tuple of (trained TFT model, training metrics)
    """
    # Set random seeds for reproducibility
    pl.seed_everything(seed, workers=True)

    # Create data loaders
    train_dataloader = train_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    # Initialize TFT model with conservative architecture
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=TFT_HIDDEN_SIZE,
        attention_head_size=TFT_ATTENTION_HEADS,
        hidden_continuous_size=TFT_HIDDEN_SIZE // 2,
        dropout=TFT_DROPOUT,
        hidden_layers=TFT_HIDDEN_LAYERS,
        output_size=7,  # Quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        gradient_clip_val=gradient_clip_val,
    )

    # Wrap in Lightning module
    lightning_module = TFTLightningModule(tft, learning_rate=learning_rate)

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=MIN_DELTA,
        patience=patience,
        mode="min",
        verbose=True,
    )

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # Auto-detect GPU if available
        enable_model_summary=True,
        gradient_clip_val=gradient_clip_val,
        limit_train_batches=30,  # Limit batches for faster training during development
        callbacks=[early_stop_callback],
        logger=False,  # We'll use MLflow instead
        enable_checkpointing=False,  # We'll save manually with MLflow
    )

    # Train the model
    logger.info("Starting TFT training...")
    trainer.fit(lightning_module, train_dataloader, val_dataloader)

    # Extract training metrics
    train_loss = trainer.callback_metrics.get("train_loss", float("nan"))
    val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
    epochs_trained = trainer.current_epoch + 1

    metrics = {
        "train_loss": float(train_loss.cpu().item()) if isinstance(train_loss, torch.Tensor) else float(train_loss),
        "val_loss": float(val_loss.cpu().item()) if isinstance(val_loss, torch.Tensor) else float(val_loss),
        "epochs_trained": epochs_trained,
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "overfitting_indicator": float(train_loss - val_loss) if not np.isnan(train_loss) and not np.isnan(val_loss) else float("nan"),
    }

    logger.info(f"Training complete. Final train_loss: {metrics['train_loss']:.4f}, val_loss: {metrics['val_loss']:.4f}")
    logger.info(f"Epochs trained: {epochs_trained} (early stopped at {epochs_trained}/{max_epochs})")

    # Log overfitting warning if gap is large
    if not np.isnan(metrics["overfitting_indicator"]) and metrics["overfitting_indicator"] > 0.1:
        logger.warning(
            f"Large train/val loss gap ({metrics['overfitting_indicator']:.4f}) suggests overfitting. "
            "Consider increasing dropout, reducing model size, or adding more regularization."
        )

    return tft, metrics


def save_tft_model(tft_model: TemporalFusionTransformer, save_path: Path) -> None:
    """
    Save trained TFT model to disk.

    Args:
        tft_model: Trained TFT model
        save_path: Path to save the model
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tft_model.state_dict(), save_path)
    logger.info(f"TFT model saved to {save_path}")


def run_tft_training(
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[TemporalFusionTransformer, dict]:
    """
    Run complete TFT training pipeline.

    Args:
        test_size: Fraction of years to reserve for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (trained TFT model, training metrics)
    """
    logger.info("Starting TFT training pipeline...")

    # Prepare datasets with temporal split
    train_dataset, val_dataset, ts_path = run_dataset_preparation(
        test_size=test_size,
        seed=seed,
    )

    # Train TFT model
    tft_model, metrics = train_tft_model(
        train_dataset,
        val_dataset,
        seed=seed,
    )

    # Save model
    model_save_path = Path("data_pipeline/bronze/tft_climate_trend/tft_model.pt")
    save_tft_model(tft_model, model_save_path)

    # Log to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "model_type": "TemporalFusionTransformer",
            "task": "climate_trend_forecasting",
            "hidden_size": TFT_HIDDEN_SIZE,
            "attention_heads": TFT_ATTENTION_HEADS,
            "hidden_layers": TFT_HIDDEN_LAYERS,
            "dropout": TFT_DROPOUT,
            "learning_rate": TFT_LEARNING_RATE,
            "gradient_clip_val": TFT_GRADIENT_CLIP_VAL,
            "encoder_length": ENCODER_LENGTH,
            "predictor_length": PREDICTOR_LENGTH,
            "batch_size": TFT_BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": seed,
        })

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(ts_path), "data/regional_time_series.parquet")
        mlflow.pytorch.log_model(tft_model, "tft_model")

    logger.info("TFT training pipeline complete.")
    return tft_model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tft_model, metrics = run_tft_training()
    logger.info(f"Final metrics: {metrics}")
