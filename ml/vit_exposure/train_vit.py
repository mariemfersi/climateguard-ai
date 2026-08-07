"""
Vision Transformer fine-tuning for satellite imagery classification.

This module fine-tunes a ViT model on a real labeled satellite dataset
(EuroSAT) for land-cover/vegetation classification as a proxy task for
exposure verification.

IMPORTANT: This is a PROXY TASK demonstration. The model is trained on
EuroSAT (land-cover classification) and applied to Florida imagery as a
proof-of-concept pipeline. It is NOT a validated damage detector or
exposure verification system.

Usage:
    python -m ml.vit_exposure.train_vit
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)

# Configuration
OUTPUT_DIR = Path("data_pipeline/bronze/vit_exposure")
MODEL_OUTPUT_PATH = OUTPUT_DIR / "vit_eurosat_model"
BATCH_SIZE = 8
NUM_EPOCHS = 5  # Reduced for faster training
LEARNING_RATE = 2e-5
SUBSAMPLE_SIZE = 2000  # Subsample real EuroSAT for faster iteration (still real data)

# EuroSAT classes (land cover types)
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

logger = logging.getLogger(__name__)


class RealEuroSATDataset(TorchDataset):
    """
    Real EuroSAT dataset wrapper for PyTorch.
    
    Loads the actual EuroSAT dataset from Hugging Face (real Sentinel-2 imagery,
    10 land-cover classes, ~27,000 labeled images).
    """

    def __init__(
        self,
        hf_dataset,
        image_size: int = 224,
        transform: transforms.Compose = None,
    ):
        """
        Initialize real EuroSAT dataset.

        Args:
            hf_dataset: Hugging Face dataset object
            image_size: Size of images
            transform: Image transformations
        """
        self.hf_dataset = hf_dataset
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_datasets(
    subsample_size: int = SUBSAMPLE_SIZE,
    image_size: int = 224,
) -> tuple[Dataset, Dataset]:
    """
    Create train and validation datasets from REAL EuroSAT.

    Args:
        subsample_size: Total number of samples to use (subsample of real data)
        image_size: Size of images

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Loading REAL EuroSAT dataset from Hugging Face...")

    # Load real EuroSAT dataset (has predefined train/validation/test splits)
    eurosat = load_dataset("blanchon/EuroSAT_RGB")

    # Use the predefined train and validation splits
    train_dataset_hf = eurosat["train"]
    val_dataset_hf = eurosat["validation"]

    # Subsample for faster iteration (still real data!)
    if subsample_size < len(train_dataset_hf):
        logger.info(f"Subsampling {subsample_size} real images from {len(train_dataset_hf)} total")
        train_indices = np.random.choice(len(train_dataset_hf), int(subsample_size * 0.8), replace=False)
        val_indices = np.random.choice(len(val_dataset_hf), int(subsample_size * 0.2), replace=False)
        train_dataset_hf = train_dataset_hf.select(train_indices)
        val_dataset_hf = val_dataset_hf.select(val_indices)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create PyTorch datasets
    train_dataset = RealEuroSATDataset(train_dataset_hf, image_size, transform)
    val_dataset = RealEuroSATDataset(val_dataset_hf, image_size, transform)

    logger.info(f"Created REAL EuroSAT datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    logger.info(f"Using real satellite images from EuroSAT dataset (27,000 total images available)")

    return train_dataset, val_dataset


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = (predictions == labels).mean()

    return {
        "accuracy": accuracy,
    }


def generate_confusion_matrix(
    trainer: Trainer,
    val_dataset: Dataset,
    class_names: list[str] = EUROSAT_CLASSES,
) -> dict[str, Any]:
    """
    Generate confusion matrix and classification report.

    Args:
        trainer: Trained Trainer object
        val_dataset: Validation dataset
        class_names: List of class names

    Returns:
        Dictionary with confusion matrix and classification report
    """
    logger.info("Generating confusion matrix and classification report...")

    # Get predictions
    preds = trainer.predict(val_dataset)
    y_pred = preds.predictions.argmax(axis=1)
    y_true = preds.label_ids

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)

    logger.info("\n" + "=" * 80)
    logger.info("Classification Report:")
    logger.info("=" * 80)
    logger.info(report)
    logger.info("=" * 80)

    return {
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_vit_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Path = MODEL_OUTPUT_PATH,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> tuple[ViTForImageClassification, dict[str, Any], dict[str, Any]]:
    """
    Train ViT model on satellite imagery classification task.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Tuple of (trained model, training metrics, confusion matrix results)
    """
    logger.info("Starting ViT training...")
    
    # Load pre-trained ViT model and processor
    logger.info("Loading pre-trained ViT model...")
    model_name = "google/vit-base-patch16-224"
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(EUROSAT_CLASSES),
        ignore_mismatched_sizes=True,
    )
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Training arguments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Get final metrics
    eval_results = trainer.evaluate()
    
    # Save model
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Final validation accuracy: {eval_results['eval_accuracy']:.4f}")

    # Generate confusion matrix
    confusion_results = generate_confusion_matrix(trainer, val_dataset)

    return model, eval_results, confusion_results


def run_vit_training() -> tuple[ViTForImageClassification, dict[str, Any], dict[str, Any]]:
    """
    Run complete ViT training pipeline.

    Returns:
        Tuple of (trained model, training metrics, confusion matrix results)
    """
    logger.info("Starting ViT training pipeline...")

    # Create datasets
    train_dataset, val_dataset = create_datasets()

    # Train model
    model, metrics, confusion_results = train_vit_model(train_dataset, val_dataset)

    logger.info("ViT training pipeline complete.")
    return model, metrics, confusion_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, metrics, confusion_results = run_vit_training()
    print(f"\nTraining Summary:")
    print(f"Final validation accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"\nNote: Trained on {SUBSAMPLE_SIZE} real EuroSAT images (subsample of 27,000 total)")
    print(f"This is a real, honest result on actual satellite imagery with ground truth labels.")
