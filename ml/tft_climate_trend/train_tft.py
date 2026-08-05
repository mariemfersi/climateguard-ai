
"""
Train Temporal Fusion Transformer (TFT) for climate trend forecasting.

Features:
- Quantile regression (P10, P50, P90)
- Early stopping
- MLflow tracking
- TensorBoard logging

Usage:
    python -m ml.tft_climate_trend.train_tft
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


from ml.mlflow_config import configure_mlflow

from ml.tft_climate_trend.prepare_tft_dataset import (
    run_dataset_preparation,
    ENCODER_LENGTH,
    PREDICTOR_LENGTH,
    BATCH_SIZE,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Hyperparameters
# ==========================================================

HIDDEN_SIZE = 16
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 8

LEARNING_RATE = 0.03

MAX_EPOCHS = 50

GRADIENT_CLIP_VAL = 0.1

QUANTILES = [0.1, 0.5, 0.9]


# ==========================================================
# Training
# ==========================================================


def train_tft(
    dataset,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    quantiles=QUANTILES,
):

    logger.info("Setting up TFT training...")


    train_loader = dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,
    )


    val_loader = dataset.to_dataloader(
        train=False,
        batch_size=batch_size * 10,
        num_workers=0,
    )


    configure_mlflow()


    logger.info(
        "Creating Temporal Fusion Transformer..."
    )


    #
    # IMPORTANT:
    # output_size MUST be a list for QuantileLoss
    #
    tft = TemporalFusionTransformer.from_dataset(
        dataset,

        learning_rate=learning_rate,

        hidden_size=HIDDEN_SIZE,

        attention_head_size=ATTENTION_HEAD_SIZE,

        dropout=DROPOUT,

        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,


        output_size=[
            len(quantiles)
        ],


        loss=QuantileLoss(
            quantiles=quantiles
        ),


        log_interval=10,

        reduce_on_plateau_patience=4,
    )



    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-4,
        mode="min",
        verbose=True,
    )


    checkpoint = ModelCheckpoint(
        dirpath="ml/tft_climate_trend/checkpoints",
        filename="tft-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )


    tb_logger = TensorBoardLogger(
        save_dir="ml/tft_climate_trend/tb_logs",
        name="tft_climate_trend",
    )


    trainer = pl.Trainer(

        max_epochs=max_epochs,

        accelerator="auto",

        gradient_clip_val=GRADIENT_CLIP_VAL,

        callbacks=[
            early_stop,
            checkpoint,
        ],

        logger=tb_logger,

        enable_checkpointing=True,

        enable_progress_bar=True,
    )



    logger.info(
        "Starting training..."
    )


    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


    logger.info(
        "Training finished"
    )


    logger.info(
        f"Best checkpoint: {checkpoint.best_model_path}"
    )


    return tft



# ==========================================================
# Pipeline
# ==========================================================


def run_tft_training():


    dataset, _ = run_dataset_preparation()


    model = train_tft(dataset)


    logger.info(
        "TFT training pipeline complete."
    )


    return model



# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )


    run_tft_training()
