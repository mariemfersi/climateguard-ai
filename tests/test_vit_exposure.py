"""
Unit tests for ViT exposure verification components.

Tests cover:
1. Sentinel-2 tile generation
2. ViT dataset creation
3. ViT training pipeline
4. Florida tile classification
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml.vit_exposure.fetch_sentinel2 import (
    generate_synthetic_sentinel2_tile,
    fetch_sentinel2_for_locations,
)
from ml.vit_exposure.train_vit import (
    RealEuroSATDataset,
    create_datasets,
    compute_metrics,
)
from ml.vit_exposure.apply_to_florida_tiles import (
    FloridaTileDataset,
    apply_vit_to_florida_tiles,
)


def test_generate_synthetic_sentinel2_tile():
    """Test synthetic satellite tile generation."""
    tile = generate_synthetic_sentinel2_tile(27.0, -81.0, tile_size=64, num_bands=3)
    
    assert tile.shape == (64, 64, 3), f"Expected shape (64, 64, 3), got {tile.shape}"
    assert tile.dtype == np.uint8, f"Expected dtype uint8, got {tile.dtype}"
    assert tile.min() >= 0 and tile.max() <= 255, "Pixel values should be in [0, 255]"


def test_fetch_sentinel2_for_locations():
    """Test Sentinel-2 fetch for sample locations."""
    # Create sample locations
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1", "loc_2"],
        "lat": [27.0, 26.5, 28.0],
        "lon": [-81.0, -80.5, -82.0],
    })
    
    # Fetch tiles (small sample)
    results = fetch_sentinel2_for_locations(
        locations,
        sample_size=2,
        tile_size=32,  # Smaller for faster test
    )
    
    assert results["successful_fetches"] == 2, f"Expected 2 successful fetches, got {results['successful_fetches']}"
    assert results["sample_size"] == 2, f"Expected sample_size 2, got {results['sample_size']}"
    assert len(results["fetched_tiles"]) == 2, f"Expected 2 fetched tiles, got {len(results['fetched_tiles'])}"


def test_real_eurosat_dataset():
    """Test real EuroSAT dataset creation (requires Hugging Face download)."""
    # Skip this test if we don't want to download real data in unit tests
    pytest.skip("Skipping real EuroSAT dataset test (requires download)")


def test_create_datasets():
    """Test train/validation dataset creation (with real EuroSAT)."""
    # Skip this test if we don't want to download real data in unit tests
    pytest.skip("Skipping real EuroSAT dataset test (requires download)")


def test_compute_metrics():
    """Test metric computation."""
    # Create dummy predictions and labels
    predictions = np.array([
        [0.1, 0.7, 0.2],  # Predict class 1
        [0.8, 0.1, 0.1],  # Predict class 0
        [0.3, 0.3, 0.4],  # Predict class 2
    ])
    labels = np.array([1, 0, 2])  # Correct labels
    
    metrics = compute_metrics((predictions, labels))
    
    assert "accuracy" in metrics, "Expected 'accuracy' in metrics"
    assert metrics["accuracy"] == 1.0, f"Expected accuracy 1.0, got {metrics['accuracy']}"


def test_compute_metrics_incorrect():
    """Test metric computation with incorrect predictions."""
    predictions = np.array([
        [0.1, 0.7, 0.2],  # Predict class 1
        [0.8, 0.1, 0.1],  # Predict class 0
        [0.3, 0.3, 0.4],  # Predict class 2
    ])
    labels = np.array([0, 1, 1])  # Incorrect labels
    
    metrics = compute_metrics((predictions, labels))
    
    assert metrics["accuracy"] == 0.0, f"Expected accuracy 0.0, got {metrics['accuracy']}"


def test_florida_tile_dataset_empty():
    """Test Florida tile dataset with no tiles."""
    # Create empty directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        tmpdir_path = Path(tmpdir)
        
        # Mock processor
        class MockProcessor:
            def __call__(self, images, return_tensors=None):
                return {"pixel_values": None}
        
        dataset = FloridaTileDataset(tmpdir_path, MockProcessor())
        
        assert len(dataset) == 0, f"Expected 0 tiles, got {len(dataset)}"


def test_fetch_sentinel2_no_locations():
    """Test fetch with empty locations DataFrame."""
    locations = pd.DataFrame({
        "location_id": [],
        "lat": [],
        "lon": [],
    })
    
    results = fetch_sentinel2_for_locations(locations, sample_size=5)
    
    assert results["successful_fetches"] == 0, f"Expected 0 successful fetches, got {results['successful_fetches']}"
    # Sample size is requested but actual should be 0 since no locations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
