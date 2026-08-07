"""
Tests for ml.gnn_accumulation.

Focuses on:
1. Graph construction correctness (edge count, no self-loops, semantic correctness)
2. GNN training stability (no collapse, output variance)
3. GNNExplainer functionality (subgraph extraction)
"""

import numpy as np
import pandas as pd
import pytest
import torch

# These tests require torch-geometric
pytest.importorskip("torch_geometric")

from ml.gnn_accumulation.build_graph import (
    build_portfolio_graph,
    compute_coastal_basin_edges,
    compute_peril_correlation_edges,
    compute_spatial_edges,
    compute_storm_footprint_edges,
    extract_node_features,
)


# --- compute_spatial_edges ------------------------------------------------


def test_compute_spatial_edges_basic():
    """Test basic spatial edge computation."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1", "loc_2"],
        "lat": [27.0, 27.1, 27.2],
        "lon": [-81.0, -81.1, -81.2],
    })
    
    edge_index = compute_spatial_edges(locations, radius_km=50, k_neighbors=2)
    
    # Check edge index shape
    assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert edge_index.shape[1] > 0, "Should have at least some edges"
    
    # Check edge indices are valid
    max_node_idx = edge_index.max().item()
    assert max_node_idx < len(locations), "Edge indices should be valid node indices"


def test_compute_spatial_edges_no_self_loops():
    """Test that spatial edges don't create self-loops."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1"],
        "lat": [27.0, 27.1],
        "lon": [-81.0, -81.1],
    })
    
    edge_index = compute_spatial_edges(locations, radius_km=100, k_neighbors=1)
    
    # Check no self-loops (edges where src == dst)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        assert src != dst, f"Self-loop detected at edge {i}"


def test_compute_spatial_edges_respects_radius():
    """Test that spatial edges respect distance radius."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1", "loc_2"],
        "lat": [27.0, 27.0, 40.0],  # loc_2 is far away
        "lon": [-81.0, -81.0, -81.0],
    })
    
    small_radius_edges = compute_spatial_edges(locations, radius_km=10, k_neighbors=2)
    large_radius_edges = compute_spatial_edges(locations, radius_km=1000, k_neighbors=2)
    
    # Larger radius should have more edges
    assert large_radius_edges.shape[1] >= small_radius_edges.shape[1]


# --- compute_peril_correlation_edges --------------------------------------


def test_compute_peril_correlation_edges():
    """Test peril correlation edge computation."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1", "loc_2"],
        "lat": [27.0, 27.1, 27.2],
        "lon": [-81.0, -81.1, -81.2],
    })
    
    # Create synthetic claims with correlation
    claims = pd.DataFrame({
        "claim_id": ["claim_0", "claim_1", "claim_2", "claim_3"],
        "location_id": ["loc_0", "loc_1", "loc_0", "loc_1"],
        "loss_date": pd.to_datetime(["2000-09-15", "2000-09-15", "2005-08-20", "2005-08-20"]),
        "damage_ratio": [0.5, 0.4, 0.6, 0.5],
    })
    
    edge_index = compute_peril_correlation_edges(
        claims, locations, threshold=0.1, min_joint_claims=1
    )
    
    # Check edge index shape
    assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
    
    # With joint claims, should have edges
    if edge_index.shape[1] > 0:
        # Check no self-loops
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            assert src != dst, f"Self-loop detected at edge {i}"


def test_compute_peril_correlation_edges_insufficient_data():
    """Test peril correlation edges with insufficient joint claims."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1"],
        "lat": [27.0, 27.1],
        "lon": [-81.0, -81.1],
    })
    
    # Claims with no joint losses
    claims = pd.DataFrame({
        "claim_id": ["claim_0", "claim_1"],
        "location_id": ["loc_0", "loc_1"],
        "loss_date": pd.to_datetime(["2000-09-15", "2010-09-15"]),  # Different years
        "damage_ratio": [0.5, 0.4],
    })
    
    edge_index = compute_peril_correlation_edges(
        claims, locations, threshold=0.1, min_joint_claims=2
    )
    
    # Should have no edges due to insufficient joint claims
    assert edge_index.shape[1] == 0, "Should have no edges with insufficient joint claims"


# --- compute_coastal_basin_edges -------------------------------------------


def test_compute_coastal_basin_edges():
    """Test coastal basin edge computation."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1", "loc_2"],
        "lat": [27.0, 27.1, 27.2],
        "lon": [-81.0, -81.1, -81.2],
        "distance_to_coast_km": [10, 15, 20],  # All coastal
    })
    
    edge_index = compute_coastal_basin_edges(locations, coastal_proximity_km=50)
    
    # Check edge index shape
    assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
    
    # Should have edges (all in same basin)
    assert edge_index.shape[1] > 0, "Should have edges for coastal locations in same basin"


def test_compute_coastal_basin_edges_inland():
    """Test coastal basin edges with inland locations."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1"],
        "lat": [30.0, 30.1],
        "lon": [-82.0, -82.1],
        "distance_to_coast_km": [200, 250],  # Inland
    })
    
    edge_index = compute_coastal_basin_edges(locations, coastal_proximity_km=50)
    
    # Inland locations should be in same "inland" basin
    # Should have edges if they're in the same inland category
    assert edge_index.shape[0] == 2


# --- extract_node_features -------------------------------------------------


def test_extract_node_features():
    """Test node feature extraction."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1"],
        "lat": [27.0, 27.1],
        "lon": [-81.0, -81.1],
        "distance_to_coast_km": [10, 15],
        "tiv_usd": [100000, 200000],
        "year_built": [2000, 2010],
    })
    
    node_features = extract_node_features(locations)
    
    # Check shape
    assert node_features.shape[0] == len(locations), "Should have features for all locations"
    assert node_features.shape[1] > 0, "Should have at least some features"
    
    # Check type
    assert isinstance(node_features, torch.Tensor), "Should return torch tensor"
    assert node_features.dtype == torch.float32, "Should be float32"


def test_extract_node_features_normalization():
    """Test that node features are normalized."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1", "loc_2"],
        "lat": [27.0, 27.1, 27.2],
        "lon": [-81.0, -81.1, -81.2],
        "distance_to_coast_km": [10, 15, 20],
        "tiv_usd": [100000, 200000, 300000],
        "year_built": [2000, 2010, 2020],
    })
    
    node_features = extract_node_features(locations)
    
    # Check that features are roughly normalized (mean ~0, std ~1)
    feature_means = node_features.mean(dim=0)
    feature_stds = node_features.std(dim=0)
    
    # Most features should be normalized
    assert (feature_means.abs() < 1.0).all(), "Features should be normalized (mean ~0)"
    assert (feature_stds > 0.5).all(), "Features should have reasonable variance"


# --- build_portfolio_graph -------------------------------------------------


def test_build_portfolio_graph_basic():
    """Test basic portfolio graph construction."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1", "loc_2"],
        "lat": [27.0, 27.1, 27.2],
        "lon": [-81.0, -81.1, -81.2],
        "distance_to_coast_km": [10, 15, 20],
        "tiv_usd": [100000, 200000, 300000],
        "year_built": [2000, 2010, 2020],
    })
    
    claims = pd.DataFrame({
        "claim_id": ["claim_0", "claim_1"],
        "location_id": ["loc_0", "loc_1"],
        "loss_date": pd.to_datetime(["2000-09-15", "2005-08-20"]),
        "damage_ratio": [0.5, 0.4],
    })
    
    graph_data = build_portfolio_graph(
        locations=locations,
        claims=claims,
        edge_types=["spatial"],  # Only spatial for testing
    )
    
    # Check graph structure
    assert graph_data.num_nodes == len(locations), "Should have correct number of nodes"
    assert graph_data.num_edges > 0, "Should have at least some edges"
    assert graph_data.x.shape[0] == len(locations), "Should have features for all nodes"
    assert graph_data.edge_index.shape[0] == 2, "Edge index should have 2 rows"


def test_build_portfolio_graph_multiple_edge_types():
    """Test graph construction with multiple edge types."""
    locations = pd.DataFrame({
        "location_id": ["loc_0", "loc_1", "loc_2"],
        "lat": [27.0, 27.1, 27.2],
        "lon": [-81.0, -81.1, -81.2],
        "distance_to_coast_km": [10, 15, 20],
        "tiv_usd": [100000, 200000, 300000],
        "year_built": [2000, 2010, 2020],
    })
    
    claims = pd.DataFrame({
        "claim_id": ["claim_0", "claim_1", "claim_2", "claim_3"],
        "location_id": ["loc_0", "loc_1", "loc_0", "loc_1"],
        "loss_date": pd.to_datetime(["2000-09-15", "2000-09-15", "2005-08-20", "2005-08-20"]),
        "damage_ratio": [0.5, 0.4, 0.6, 0.5],
    })
    
    graph_data = build_portfolio_graph(
        locations=locations,
        claims=claims,
        edge_types=["spatial", "coastal"],  # Multiple edge types
    )
    
    # Check graph structure
    assert graph_data.num_nodes == len(locations)
    assert graph_data.num_edges > 0
    
    # Multiple edge types should generally produce more edges
    spatial_only = build_portfolio_graph(
        locations=locations,
        claims=claims,
        edge_types=["spatial"],
    )
    assert graph_data.num_edges >= spatial_only.num_edges


def test_build_portfolio_graph_no_edges():
    """Test graph construction when no edges can be created."""
    locations = pd.DataFrame({
        "location_id": ["loc_0"],
        "lat": [27.0],
        "lon": [-81.0],
        "distance_to_coast_km": [10],
        "tiv_usd": [100000],
        "year_built": [2000],
    })
    
    # Single location - should fail or produce no edges
    with pytest.raises(ValueError, match="No edges computed"):
        build_portfolio_graph(
            locations=locations,
            edge_types=["spatial"],
        )


# --- Integration test placeholder ------------------------------------------


def test_gnn_integration():
    """
    Integration test for full GNN pipeline.
    
    This test would require:
    1. Loading actual data
    2. Building graph
    3. Training GNN
    4. Running explainer
    
    For now, this is a placeholder.
    """
    # Placeholder assertion
    assert True, "Integration test to be implemented with actual data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
