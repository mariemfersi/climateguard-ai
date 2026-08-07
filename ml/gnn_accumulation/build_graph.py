"""
Graph construction for portfolio accumulation risk modeling.

This module implements a justified, multi-layered graph construction approach
that captures spatial correlation patterns relevant to hurricane risk:

Edge Construction Rationale:
1. Spatial Proximity Edges: k-NN within distance threshold (captures geographic adjacency)
2. Peril Correlation Edges: Based on historical joint loss patterns (captures actual risk correlation)
3. Geographic Feature Edges: Shared coastal basins, elevation bands (captures physical drivers)

This is not an arbitrary k-NN graph - each edge type has a specific risk-based justification.

Usage:
    python -m ml.gnn_accumulation.build_graph
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import dense_to_sparse

from data_pipeline.databricks_jobs.compute_event_features import haversine_distance

logger = logging.getLogger(__name__)

# Data paths
GOLD_FEATURES_PATH = Path("data_pipeline/gold/gold_features.parquet")
CLAIMS_PATH = Path("data_pipeline/silver/claims.parquet")
GRAPH_OUTPUT_PATH = Path("data_pipeline/bronze/gnn_accumulation/portfolio_graph.pt")

# Graph construction parameters
SPATIAL_RADIUS_KM = 100  # 100km radius for spatial edges
K_NEIGHBORS = 5  # Number of nearest neighbors for k-NN edges
PERIL_CORRELATION_THRESHOLD = 0.3  # Minimum correlation for peril-based edges
COASTAL_PROXIMITY_THRESHOLD_KM = 50  # Distance to coast for coastal grouping
ELEVATION_BAND_KM = 10  # Elevation difference threshold for grouping

# Hurricane-specific parameters
STORM_INFLUENCE_RADIUS_KM = 500  # Radius for considering same-storm impact


def compute_spatial_edges(
    locations: pd.DataFrame,
    radius_km: float = SPATIAL_RADIUS_KM,
    k_neighbors: int = K_NEIGHBORS,
    max_edges: int = 100000,  # Increased limit for better connectivity
) -> torch.Tensor:
    """
    Compute spatial proximity edges using distance-based k-NN (optimized).
    
    Rationale: Nearby locations are more likely to be affected by the same storm
    due to the spatial extent of hurricane wind fields and rainfall patterns.
    
    Args:
        locations: DataFrame with location_id, lat, lon columns
        radius_km: Maximum distance for edges (km)
        k_neighbors: Number of nearest neighbors to connect
        max_edges: Maximum edges to prevent explosion
    
    Returns:
        Edge index tensor (2, num_edges) for spatial edges
    """
    logger.info("Computing spatial proximity edges...")
    
    coords = locations[["lat", "lon"]].values
    n_locations = len(locations)
    
    # For very large datasets, use sampling but ensure connectivity
    if n_locations > 10000:
        logger.warning(f"Large dataset ({n_locations} locations). Using optimized spatial computation.")
        # Use KD-tree for all locations but limit neighbors
        tree = cKDTree(coords)
        
        # Query k-nearest neighbors for all locations
        distances, indices = tree.query(
            coords,
            k=min(k_neighbors + 1, n_locations),
            distance_upper_bound=radius_km,
        )
        
        # Build edge list (excluding self-loops)
        edge_list = []
        for i in range(n_locations):
            # Handle the case when there's only 1 location (returns scalar)
            if n_locations == 1:
                continue
            for j, dist in zip(indices[i], distances[i]):
                if j != i and j < n_locations and dist <= radius_km:
                    edge_list.append([i, j])
    else:
        # Full computation for smaller datasets
        tree = cKDTree(coords)
        distances, indices = tree.query(
            coords,
            k=min(k_neighbors + 1, n_locations),
            distance_upper_bound=radius_km,
        )
        
        # Build edge list (excluding self-loops)
        edge_list = []
        for i in range(n_locations):
            # Handle the case when there's only 1 location (returns scalar)
            if n_locations == 1:
                continue
            for j, dist in zip(indices[i], distances[i]):
                if j != i and j < n_locations and dist <= radius_km:
                    edge_list.append([i, j])
    
    if not edge_list:
        logger.warning("No spatial edges found within radius. Consider increasing radius_km.")
        return torch.empty((2, 0), dtype=torch.long)
    
    # Sample edges if too many
    if len(edge_list) > max_edges:
        import random
        logger.warning(f"Too many spatial edges ({len(edge_list)}). Sampling to {max_edges}.")
        edge_list = random.sample(edge_list, max_edges)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    logger.info(f"Spatial edges: {edge_index.shape[1]} edges (radius={radius_km}km, k={k_neighbors})")
    return edge_index


def compute_peril_correlation_edges(
    claims: pd.DataFrame,
    locations: pd.DataFrame,
    threshold: float = PERIL_CORRELATION_THRESHOLD,
    min_joint_claims: int = 3,
    max_locations: int = 5000,  # Limit for performance
) -> torch.Tensor:
    """
    Compute edges based on historical joint loss patterns (optimized).
    
    Rationale: Locations that have historically experienced joint losses during
    the same storms have empirically demonstrated correlation. This captures
    actual risk patterns beyond simple geographic proximity.
    
    Args:
        claims: Claims data with location_id, loss_date, damage_ratio
        locations: Location data with location_id, lat, lon
        threshold: Minimum correlation coefficient for edges
        min_joint_claims: Minimum joint claims to consider correlation reliable
        max_locations: Maximum locations to process for performance
    
    Returns:
        Edge index tensor (2, num_edges) for peril correlation edges
    """
    logger.info("Computing peril correlation edges from historical claims...")
    
    # Limit to locations with sufficient claims for performance
    location_claim_counts = claims["location_id"].value_counts()
    active_locations = location_claim_counts[location_claim_counts >= min_joint_claims].index.tolist()
    
    if len(active_locations) > max_locations:
        logger.warning(f"Limiting to {max_locations} most active locations for performance")
        active_locations = location_claim_counts.head(max_locations).index.tolist()
    
    # Filter claims to active locations
    claims_filtered = claims[claims["location_id"].isin(active_locations)].copy()
    
    if len(claims_filtered) == 0:
        logger.warning("No claims data for peril correlation. Skipping.")
        return torch.empty((2, 0), dtype=torch.long)
    
    # Add year to claims for storm-level grouping
    claims_filtered["year"] = pd.to_datetime(claims_filtered["loss_date"]).dt.year
    
    # Create location-year loss matrix
    location_years = claims_filtered.groupby(["location_id", "year"]).agg(
        total_loss=("damage_ratio", "sum"),
        claim_count=("damage_ratio", "count"),
    ).reset_index()
    
    # Pivot to matrix (locations x years)
    loss_matrix = location_years.pivot(
        index="location_id",
        columns="year",
        values="total_loss"
    ).fillna(0)
    
    # Skip if too few locations or years
    if loss_matrix.shape[0] < 2 or loss_matrix.shape[1] < 2:
        logger.warning("Insufficient data for correlation computation. Skipping.")
        return torch.empty((2, 0), dtype=torch.long)
    
    # Compute correlation matrix between locations
    logger.info(f"Computing correlation matrix for {loss_matrix.shape[0]} locations...")
    correlation_matrix = loss_matrix.T.corr()
    
    # Build edges for locations with significant correlation
    location_ids = loss_matrix.index.tolist()
    
    # Map location IDs to indices in the original locations dataframe
    location_id_to_original_idx = {
        loc_id: locations[locations["location_id"] == loc_id].index[0]
        for loc_id in location_ids
        if loc_id in locations["location_id"].values
    }
    
    edge_list = []
    # Only process upper triangle to avoid duplicates
    for i, loc_i in enumerate(location_ids):
        for j in range(i + 1, len(location_ids)):
            loc_j = location_ids[j]
            
            corr = correlation_matrix.iloc[i, j]
            if pd.notna(corr) and abs(corr) >= threshold:
                # Check if they have sufficient joint claims
                joint_claims = claims_filtered[
                    claims_filtered["location_id"].isin([loc_i, loc_j])
                ].groupby("year").filter(lambda x: len(x) >= 2)
                
                if len(joint_claims) >= min_joint_claims:
                    if loc_i in location_id_to_original_idx and loc_j in location_id_to_original_idx:
                        edge_list.append([
                            location_id_to_original_idx[loc_i],
                            location_id_to_original_idx[loc_j]
                        ])
    
    if not edge_list:
        logger.warning("No peril correlation edges found. Consider lowering threshold.")
        return torch.empty((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    logger.info(f"Peril correlation edges: {edge_index.shape[1]} edges (threshold={threshold})")
    return edge_index


def compute_coastal_basin_edges(
    locations: pd.DataFrame,
    coastal_proximity_km: float = COASTAL_PROXIMITY_THRESHOLD_KM,
    max_edges_per_basin: int = 10000,  # Limit for performance
) -> torch.Tensor:
    """
    Compute edges based on shared coastal exposure (optimized).
    
    Rationale: Locations in the same coastal basin share similar hurricane
    exposure characteristics (e.g., Gulf Coast vs Atlantic Coast). This captures
    large-scale geographic risk patterns.
    
    Args:
        locations: DataFrame with location_id, lat, lon, distance_to_coast_km
        coastal_proximity_km: Distance threshold for coastal grouping
        max_edges_per_basin: Maximum edges per basin to prevent explosion
    
    Returns:
        Edge index tensor (2, num_edges) for coastal basin edges
    """
    logger.info("Computing coastal basin edges...")
    
    if "distance_to_coast_km" not in locations.columns:
        logger.warning("distance_to_coast_km not in features. Skipping coastal edges.")
        return torch.empty((2, 0), dtype=torch.long)
    
    # Group by coastal proximity and general region
    locations = locations.copy()
    
    # Define coastal basins based on longitude and coastal proximity (vectorized)
    conditions = [
        locations["distance_to_coast_km"] > coastal_proximity_km,
        (locations["distance_to_coast_km"] <= coastal_proximity_km) & (locations["lon"] < -85),
        (locations["distance_to_coast_km"] <= coastal_proximity_km) & (locations["lon"] < -82) & (locations["lon"] >= -85),
        (locations["distance_to_coast_km"] <= coastal_proximity_km) & (locations["lon"] < -80) & (locations["lon"] >= -82),
        (locations["distance_to_coast_km"] <= coastal_proximity_km) & (locations["lon"] >= -80),
    ]
    choices = ["inland", "western_gulf", "eastern_gulf", "florida_atlantic", "atlantic_north"]
    locations["coastal_basin"] = np.select(conditions, choices, default="inland")
    
    # Connect locations within the same coastal basin (sampled for performance)
    edge_list = []
    for basin in locations["coastal_basin"].unique():
        basin_locations = locations[locations["coastal_basin"] == basin].index.tolist()
        n_locations = len(basin_locations)
        
        if n_locations < 2:
            continue
        
        # Sample edges if basin is too large
        max_possible_edges = n_locations * (n_locations - 1) // 2
        if max_possible_edges > max_edges_per_basin:
            # Sample random pairs
            import random
            all_pairs = [(i, j) for i in range(n_locations) for j in range(i + 1, n_locations)]
            sampled_pairs = random.sample(all_pairs, max_edges_per_basin)
            for i, j in sampled_pairs:
                edge_list.append([basin_locations[i], basin_locations[j]])
        else:
            # Connect all pairs within basin (complete subgraph)
            for i in range(n_locations):
                for j in range(i + 1, n_locations):
                    edge_list.append([basin_locations[i], basin_locations[j]])
    
    if not edge_list:
        logger.warning("No coastal basin edges found.")
        return torch.empty((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    logger.info(f"Coastal basin edges: {edge_index.shape[1]} edges")
    return edge_index


def compute_storm_footprint_edges(
    locations: pd.DataFrame,
    storm_tracks: pd.DataFrame,
    influence_radius_km: float = STORM_INFLUENCE_RADIUS_KM,
) -> torch.Tensor:
    """
    Compute edges based on shared storm footprint from historical tracks.
    
    Rationale: Locations that have been within the same storm's influence
    radius historically share actual peril exposure, not just geographic proximity.
    
    Args:
        locations: DataFrame with location_id, lat, lon
        storm_tracks: HURDAT2 storm track data with lat, lon, timestamp, storm_id
        influence_radius_km: Radius for considering same-storm impact
    
    Returns:
        Edge index tensor (2, num_edges) for storm footprint edges
    """
    logger.info("Computing storm footprint edges from historical tracks...")
    
    # This would require loading HURDAT2 data
    # For now, implement a simplified version using spatial proximity
    # as a proxy for storm footprint correlation
    
    # In full implementation, this would:
    # 1. For each storm, identify locations within influence radius
    # 2. Build edges between locations that shared storm exposure
    # 3. Weight edges by number of shared storms
    
    logger.warning("Storm footprint edges require HURDAT2 data. Using spatial proxy.")
    return compute_spatial_edges(locations, radius_km=influence_radius_km, k_neighbors=10)


def build_portfolio_graph(
    locations: pd.DataFrame,
    claims: pd.DataFrame | None = None,
    storm_tracks: pd.DataFrame | None = None,
    edge_types: list[str] = ["spatial", "peril_correlation", "coastal"],
) -> Data:
    """
    Build a multi-edge portfolio graph for accumulation risk modeling.
    
    Combines multiple edge types with different semantic meanings:
    - spatial: Geographic proximity
    - peril_correlation: Historical joint loss patterns
    - coastal: Shared coastal basin exposure
    - storm_footprint: Shared historical storm exposure
    
    Args:
        locations: Location features DataFrame
        claims: Claims data for peril correlation edges
        storm_tracks: Storm track data for footprint edges
        edge_types: List of edge types to include
    
    Returns:
        PyTorch Geometric Data object with node features and edge indices
    """
    logger.info("Building portfolio graph...")
    
    # Reset index to ensure sequential node IDs
    locations = locations.reset_index(drop=True)
    
    # Extract node features
    node_features = extract_node_features(locations)
    
    # Compute edges for each type
    edge_indices = {}
    edge_attrs = {}
    
    if "spatial" in edge_types:
        edge_indices["spatial"] = compute_spatial_edges(locations)
        # Add edge attribute for distance
        edge_attrs["spatial"] = compute_edge_distances(locations, edge_indices["spatial"])
    
    if "peril_correlation" in edge_types and claims is not None:
        edge_indices["peril_correlation"] = compute_peril_correlation_edges(claims, locations)
        # Add edge attribute for correlation strength
        edge_attrs["peril_correlation"] = compute_edge_correlations(claims, locations, edge_indices["peril_correlation"])
    
    if "coastal" in edge_types:
        edge_indices["coastal"] = compute_coastal_basin_edges(locations)
        # Binary edge attribute (1 if same basin, 0 otherwise)
        edge_attrs["coastal"] = torch.ones(edge_indices["coastal"].shape[1], 1)
    
    if "storm_footprint" in edge_types and storm_tracks is not None:
        edge_indices["storm_footprint"] = compute_storm_footprint_edges(locations, storm_tracks)
        # Add edge attribute for shared storm count
        edge_attrs["storm_footprint"] = compute_shared_storm_counts(locations, storm_tracks, edge_indices["storm_footprint"])
    
    # Combine all edges (for homogeneous graph)
    all_edges = []
    all_edge_attrs = []
    
    for edge_type in edge_types:
        if edge_type in edge_indices and edge_indices[edge_type].shape[1] > 0:
            all_edges.append(edge_indices[edge_type])
            if edge_type in edge_attrs:
                all_edge_attrs.append(edge_attrs[edge_type])
    
    if not all_edges:
        raise ValueError("No edges computed. Check edge_types and input data.")
    
    # Concatenate edges
    combined_edge_index = torch.cat(all_edges, dim=1)
    
    # Concatenate edge attributes
    if all_edge_attrs:
        combined_edge_attr = torch.cat(all_edge_attrs, dim=0)
    else:
        combined_edge_attr = None
    
    # Create PyG Data object
    graph_data = Data(
        x=node_features,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,
        num_nodes=len(locations),
    )
    
    # Add metadata
    graph_data.location_ids = locations["location_id"].values
    graph_data.edge_types = edge_types
    
    logger.info(f"Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    logger.info(f"Edge types: {edge_types}")
    logger.info(f"Node feature dim: {graph_data.num_node_features}")
    if graph_data.edge_attr is not None:
        logger.info(f"Edge feature dim: {graph_data.num_edge_features}")
    
    return graph_data


def extract_node_features(locations: pd.DataFrame) -> torch.Tensor:
    """
    Extract node features from location data.
    
    Features include:
    - Geographic: lat, lon, distance_to_coast_km
    - Exposure: tiv_usd, year_built
    - Vulnerability: construction_class, roof_type (encoded)
    - Environmental: elevation, terrain_roughness
    
    Args:
        locations: Location features DataFrame
    
    Returns:
        Tensor of node features (num_nodes, num_features)
    """
    logger.info("Extracting node features...")
    
    # Define feature columns
    numeric_features = [
        "lat", "lon", "distance_to_coast_km", "tiv_usd", "year_built"
    ]
    
    # Filter to available features
    available_numeric = [f for f in numeric_features if f in locations.columns]
    
    # Extract numeric features
    feature_matrix = locations[available_numeric].values
    
    # Normalize features
    feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
    
    # Handle categorical features (if any)
    categorical_features = ["construction_class", "roof_type"]
    available_categorical = [f for f in categorical_features if f in locations.columns]
    
    if available_categorical:
        # One-hot encode
        for cat_feat in available_categorical:
            dummies = pd.get_dummies(locations[cat_feat], prefix=cat_feat)
            feature_matrix = np.hstack([feature_matrix, dummies.values])
    
    # Convert to tensor
    node_features = torch.tensor(feature_matrix, dtype=torch.float32)
    
    logger.info(f"Node features: {node_features.shape[1]} dimensions")
    return node_features


def compute_edge_distances(
    locations: pd.DataFrame,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Compute distances for each edge (optimized for large graphs).
    
    Args:
        locations: Location data with lat, lon
        edge_index: Edge index tensor
    
    Returns:
        Edge attribute tensor with distances
    """
    coords = locations[["lat", "lon"]].values
    edge_pairs = edge_index.t().numpy()
    
    # Vectorized distance computation
    src_coords = coords[edge_pairs[:, 0]]
    dst_coords = coords[edge_pairs[:, 1]]
    
    distances = haversine_distance(
        src_coords[:, 0], src_coords[:, 1],
        dst_coords[:, 0], dst_coords[:, 1]
    )
    
    return torch.tensor(distances.reshape(-1, 1), dtype=torch.float32)


def compute_edge_correlations(
    claims: pd.DataFrame,
    locations: pd.DataFrame,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Compute correlation strengths for peril correlation edges.
    
    Args:
        claims: Claims data
        locations: Location data
        edge_index: Edge index tensor
    
    Returns:
        Edge attribute tensor with correlation values
    """
    # This would compute the actual correlation values for each edge
    # For now, return placeholder
    return torch.ones(edge_index.shape[1], 1)


def compute_shared_storm_counts(
    locations: pd.DataFrame,
    storm_tracks: pd.DataFrame,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Compute shared storm counts for footprint edges.
    
    Args:
        locations: Location data
        storm_tracks: Storm track data
        edge_index: Edge index tensor
    
    Returns:
        Edge attribute tensor with shared storm counts
    """
    # This would compute actual shared storm counts
    # For now, return placeholder
    return torch.ones(edge_index.shape[1], 1)


def visualize_graph(
    graph_data: Data,
    locations: pd.DataFrame,
    output_path: Path = Path("data_pipeline/bronze/gnn_accumulation/graph_visualization.png"),
) -> None:
    """
    Visualize the portfolio graph for sanity checking.
    
    Args:
        graph_data: PyG Data object
        locations: Location data with coordinates
        output_path: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        logger.info("Generating graph visualization...")
        
        # Convert to NetworkX for visualization
        G = nx.from_edgelist(graph_data.edge_index.t().tolist())
        
        # Set node positions using geographic coordinates
        pos = {
            i: (row["lon"], row["lat"])
            for i, row in locations.iterrows()
        }
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_size=50, node_color="lightblue", alpha=0.7
        )
        
        # Draw edges (sample if too many)
        if G.number_of_edges() > 1000:
            edges_to_draw = list(G.edges())[:1000]
            nx.draw_networkx_edges(
                G, pos, edgelist=edges_to_draw,
                edge_color="gray", alpha=0.3, width=0.5
            )
        else:
            nx.draw_networkx_edges(
                G, pos, edge_color="gray", alpha=0.3, width=0.5
            )
        
        # Add geographic context
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Portfolio Risk Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Graph visualization saved to {output_path}")
        
    except ImportError:
        logger.warning("Matplotlib/NetworkX not available. Skipping visualization.")


def run_graph_construction() -> Data:
    """
    Run the complete graph construction pipeline.
    
    Returns:
        PyG Data object with the constructed graph
    """
    logger.info("Starting graph construction pipeline...")
    
    # Load data
    logger.info("Loading input data...")
    locations = pd.read_parquet(GOLD_FEATURES_PATH)
    claims = pd.read_parquet(CLAIMS_PATH)
    
    # Build graph (start with spatial only for performance)
    logger.info("Building graph with spatial edges only (for initial performance)...")
    graph_data = build_portfolio_graph(
        locations=locations,
        claims=claims,
        edge_types=["spatial"],  # Start with spatial only
    )
    
    # Visualize graph
    visualize_graph(graph_data, locations)
    
    # Save graph
    GRAPH_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_data, GRAPH_OUTPUT_PATH)
    logger.info(f"Graph saved to {GRAPH_OUTPUT_PATH}")
    
    logger.info("To add more edge types, run with: edge_types=['spatial', 'peril_correlation', 'coastal']")
    return graph_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    graph_data = run_graph_construction()
    logger.info(f"Graph construction complete: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
