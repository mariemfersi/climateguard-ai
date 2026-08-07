"""
Apply trained ViT model to Florida satellite imagery tiles.

This module applies the fine-tuned ViT model to Florida satellite imagery
to demonstrate the exposure verification pipeline.

IMPORTANT: This is a PROOF-OF-CONCEPT demonstration. The model was trained on
EuroSAT (land-cover classification) and is being applied to Florida imagery
as a demonstration of the pipeline, not as a validated damage detection system.

Usage:
    python -m ml.vit_exposure.apply_to_florida_tiles
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# Configuration
OUTPUT_DIR = Path("data_pipeline/bronze/vit_exposure")
MODEL_OUTPUT_PATH = OUTPUT_DIR / "vit_eurosat_model"
FLORIDA_TILES_DIR = OUTPUT_DIR
RESULTS_OUTPUT_PATH = OUTPUT_DIR / "florida_classification_results.parquet"

# EuroSAT classes (land cover types)
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

logger = logging.getLogger(__name__)


class FloridaTileDataset(TorchDataset):
    """Dataset for Florida satellite imagery tiles."""
    
    def __init__(
        self,
        tiles_dir: Path,
        processor: ViTImageProcessor,
    ):
        """
        Initialize Florida tile dataset.
        
        Args:
            tiles_dir: Directory containing tile images
            processor: ViT image processor
        """
        self.tiles_dir = tiles_dir
        self.processor = processor
        
        # Find all tile images
        self.tile_paths = list(tiles_dir.glob("*_tile.png"))
        logger.info(f"Found {len(self.tile_paths)} tile images")
    
    def __len__(self) -> int:
        return len(self.tile_paths)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        tile_path = self.tile_paths[idx]
        
        # Load image
        image = Image.open(tile_path).convert("RGB")
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "location_id": tile_path.stem.replace("_tile", ""),
            "tile_path": str(tile_path),
        }


def apply_vit_to_florida_tiles(
    model_path: Path = MODEL_OUTPUT_PATH,
    tiles_dir: Path = FLORIDA_TILES_DIR,
    output_path: Path = RESULTS_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Apply trained ViT model to Florida satellite imagery tiles.
    
    Args:
        model_path: Path to trained ViT model
        tiles_dir: Directory containing Florida tile images
        output_path: Path to save classification results
    
    Returns:
        DataFrame with classification results
    """
    logger.info("Applying ViT model to Florida tiles...")
    
    # Load model and processor
    logger.info(f"Loading model from {model_path}...")
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)
    
    model.eval()
    
    # Create dataset
    tile_dataset = FloridaTileDataset(tiles_dir, processor)
    
    if len(tile_dataset) == 0:
        logger.warning("No Florida tiles found. Skipping classification.")
        return pd.DataFrame()
    
    # Create dataloader
    dataloader = DataLoader(tile_dataset, batch_size=4, shuffle=False)
    
    # Run inference
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"]
            location_ids = batch["location_id"]
            tile_paths = batch["tile_path"]
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            
            # Store results
            for i in range(len(location_ids)):
                pred_class_idx = predictions[i].item()
                pred_class = EUROSAT_CLASSES[pred_class_idx]
                confidence = probabilities[i][pred_class_idx].item()
                
                results.append({
                    "location_id": location_ids[i],
                    "tile_path": tile_paths[i],
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "all_probabilities": probabilities[i].cpu().numpy().tolist(),
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    
    logger.info(f"Classification results saved to {output_path}")
    logger.info(f"Classified {len(results_df)} tiles")
    
    # Log class distribution
    if not results_df.empty:
        class_counts = results_df["predicted_class"].value_counts()
        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count}")
    
    return results_df


def run_florida_classification() -> pd.DataFrame:
    """
    Run complete Florida tile classification pipeline.
    
    Returns:
        DataFrame with classification results
    """
    logger.info("Starting Florida tile classification pipeline...")
    
    # Apply model
    results_df = apply_vit_to_florida_tiles()
    
    logger.info("Florida tile classification pipeline complete.")
    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_florida_classification()
    
    if not results.empty:
        print(f"\nClassification Summary:")
        print(f"Total tiles classified: {len(results)}")
        print(f"\nClass distribution:")
        print(results["predicted_class"].value_counts())
        print(f"\nAverage confidence: {results['confidence'].mean():.4f}")
    else:
        print("No tiles were classified. Ensure Florida tiles are available.")
