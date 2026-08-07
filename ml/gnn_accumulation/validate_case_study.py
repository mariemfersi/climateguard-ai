"""
Historical case study validation for GNN accumulation risk model.

Validates the GNN model against known historical hurricane events to ensure
it captures real correlated-loss patterns. Focuses on Hurricane Ian (2022)
as a primary case study.

Usage:
    python -m ml.gnn_accumulation.validate_case_study
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch_geometric.data import Data

from ml.gnn_accumulation.build_graph import GRAPH_OUTPUT_PATH
from ml.gnn_accumulation.train_gnn import (
    AccumulationRiskGNN,
    MODEL_OUTPUT_PATH,
    load_gnn_model,
)

logger = logging.getLogger(__name__)

# Case study parameters
HURRICANE_IAN_DATE = "2022-09-28"
HURRICANE_IAN_LANDFALL_LOC = (26.2, -81.8)  # Fort Myers, FL
VALIDATION_OUTPUT_PATH = Path("data_pipeline/bronze/gnn_accumulation/case_study_validation.parquet")


def load_historical_storm_data(storm_name: str = "Ian") -> pd.DataFrame:
    """
    Load historical storm data for case study validation.
    
    Args:
        storm_name: Name of hurricane to analyze
    
    Returns:
        DataFrame with storm track and impact data
    """
    # This would load from HURDAT2 or similar database
    # For now, create synthetic data for Hurricane Ian
    logger.info(f"Loading historical data for Hurricane {storm_name}...")
    
    # Simple synthetic storm track data
    storm_track = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-09-23", "2022-09-24", "2022-09-25", "2022-09-26", "2022-09-27"]),
        "lat": [22.0, 23.5, 25.0, 26.5, 28.0],
        "lon": [-82.0, -81.5, -81.0, -80.5, -80.0],
        "max_wind_kt": [65, 80, 100, 140, 120],
        "min_pressure_mb": [970, 960, 950, 940, 950],
    })
    
    logger.info(f"Loaded {len(storm_track)} storm track points")
    return storm_track


def compute_actual_joint_losses(
    claims: pd.DataFrame,
    storm_date: str,
    influence_radius_km: float = 500,
) -> dict[str, Any]:
    """
    Compute actual joint loss pattern for a historical storm.

    Args:
        claims: Historical claims data
        storm_date: Date of storm to analyze
        influence_radius_km: Radius for considering storm impact

    Returns:
        Dictionary with joint loss statistics
    """
    logger.info(f"Computing actual joint losses for storm on {storm_date}...")

    # ------------------------------------------------------------------
    # Convert ALL timestamps to UTC to avoid tz-aware / tz-naive issues
    # ------------------------------------------------------------------
    claims = claims.copy()

    claims["loss_date_parsed"] = pd.to_datetime(
        claims["loss_date"],
        utc=True,
        errors="coerce",
    )

    storm_date_parsed = pd.to_datetime(
        storm_date,
        utc=True,
        errors="coerce",
    )

    # Remove rows with invalid dates
    claims = claims.dropna(subset=["loss_date_parsed"])

    # ------------------------------------------------------------------
    # Filter claims within ±7 days of the storm
    # ------------------------------------------------------------------
    date_window = (
        (claims["loss_date_parsed"] >= storm_date_parsed - pd.Timedelta(days=7))
        &
        (claims["loss_date_parsed"] <= storm_date_parsed + pd.Timedelta(days=7))
    )

    storm_claims = claims.loc[date_window].copy()

    if storm_claims.empty:
        logger.warning(f"No claims found for storm on {storm_date}")
        return {
            "storm_date": storm_date,
            "affected_locations": [],
            "total_loss": 0.0,
            "location_losses": {},
            "correlation_matrix": None,
        }

    # ------------------------------------------------------------------
    # Aggregate losses by location
    # ------------------------------------------------------------------
    location_losses = (
        storm_claims.groupby("location_id")
        .agg(
            total_loss=("damage_ratio", "sum"),
            claim_count=("claim_id", "count"),
            mean_severity=("damage_ratio", "mean"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Build correlation matrix
    # ------------------------------------------------------------------
    if len(location_losses) > 1:

        storm_claims["day"] = (
            storm_claims["loss_date_parsed"]
            .dt.tz_convert("UTC")
            .dt.dayofyear
        )

        location_day_matrix = (
            storm_claims.pivot_table(
                index="location_id",
                columns="day",
                values="damage_ratio",
                aggfunc="sum",
                fill_value=0,
            )
        )

        if location_day_matrix.shape[0] > 1:
            correlation_matrix = location_day_matrix.T.corr()
        else:
            correlation_matrix = None

    else:
        correlation_matrix = None

    joint_loss_data = {
        "storm_date": storm_date,
        "affected_locations": location_losses["location_id"].tolist(),
        "total_loss": float(location_losses["total_loss"].sum()),
        "location_losses": location_losses.set_index("location_id").to_dict("index"),
        "correlation_matrix": correlation_matrix,
    }

    logger.info(
        "Found %d affected locations, total loss %.2f",
        len(location_losses),
        joint_loss_data["total_loss"],
    )

    return joint_loss_data


def compute_gnn_predicted_correlations(
    model: AccumulationRiskGNN,
    graph_data: Data,
    locations: pd.DataFrame,
    location_ids: list[str],
) -> dict[str, float]:
    """
    Compute GNN-predicted correlations between locations.
    
    Args:
        model: Trained GNN model
        graph_data: PyG Data object
        locations: Location data
        location_ids: List of location IDs to analyze
    
    Returns:
        Dictionary of pairwise correlation predictions
    """
    logger.info("Computing GNN-predicted correlations...")
    
    # Get location indices
    location_id_to_idx = {
        loc_id: idx for idx, loc_id in enumerate(locations["location_id"].values)
    }
    
    # Get embeddings for all locations
    model.eval()
    with torch.no_grad():
        # Use the model's input projection first
        x = model.input_proj(graph_data.x)
        x = torch.nn.functional.relu(x)
        
        # Forward pass through GNN layers (skip output projection)
        embeddings = x
        for i, conv in enumerate(model.convs):
            embeddings = conv(embeddings, graph_data.edge_index)
            embeddings = torch.nn.functional.relu(embeddings)
    
    # Compute pairwise similarity as proxy for correlation
    embeddings_np = embeddings.cpu().numpy()
    predicted_correlations = {}
    
    for i, loc_i in enumerate(location_ids):
        for j, loc_j in enumerate(location_ids):
            if i >= j:  # Avoid duplicates
                continue
            
            if loc_i in location_id_to_idx and loc_j in location_id_to_idx:
                idx_i = location_id_to_idx[loc_i]
                idx_j = location_id_to_idx[loc_j]
                
                # Cosine similarity as correlation proxy
                emb_i = embeddings_np[idx_i]
                emb_j = embeddings_np[idx_j]
                
                similarity = np.dot(emb_i, emb_j) / (
                    np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8
                )
                
                predicted_correlations[f"{loc_i}_{loc_j}"] = similarity
    
    logger.info(f"Computed {len(predicted_correlations)} pairwise correlations")
    return predicted_correlations


def compare_correlations(
    actual_correlations: pd.DataFrame,
    predicted_correlations: dict[str, float],
) -> dict[str, Any]:
    """
    Compare actual vs predicted correlations.
    
    Args:
        actual_correlations: Actual correlation matrix from historical data
        predicted_correlations: GNN-predicted correlations
    
    Returns:
        Dictionary with comparison metrics
    """
    logger.info("Comparing actual vs predicted correlations...")
    
    # Extract comparable pairs
    actual_values = []
    predicted_values = []
    location_pairs = []
    
    for pair, pred_corr in predicted_correlations.items():
        loc_i, loc_j = pair.split("_")
        
        if loc_i in actual_correlations.index and loc_j in actual_correlations.index:
            actual_corr = actual_correlations.loc[loc_i, loc_j]
            if pd.notna(actual_corr):
                actual_values.append(actual_corr)
                predicted_values.append(pred_corr)
                location_pairs.append(pair)
    
    if not actual_values:
        logger.warning("No comparable correlation pairs found")
        return {
            "correlation": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "num_pairs": 0,
        }
    
    # Compute metrics
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    correlation, _ = pearsonr(actual_values, predicted_values)
    mae = np.mean(np.abs(actual_values - predicted_values))
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    
    comparison_metrics = {
        "correlation": correlation,
        "mae": mae,
        "rmse": rmse,
        "num_pairs": len(actual_values),
    }
    
    logger.info(
        f"Correlation comparison: r={correlation:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}"
    )
    
    return comparison_metrics


def validate_risk_clusters(
    model: AccumulationRiskGNN,
    graph_data: Data,
    locations: pd.DataFrame,
    joint_loss_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate that GNN identifies correct risk clusters.
    
    Args:
        model: Trained GNN model
        graph_data: PyG Data object
        locations: Location data
        joint_loss_data: Historical joint loss data
    
    Returns:
        Dictionary with cluster validation results
    """
    logger.info("Validating risk cluster identification...")
    
    # Get risk scores for all locations
    model.eval()
    with torch.no_grad():
        risk_scores = model(graph_data.x, graph_data.edge_index).cpu().numpy().flatten()
    
    # Add risk scores to locations
    locations_with_risk = locations.copy()
    locations_with_risk["risk_score"] = risk_scores
    
    # Check if affected locations have higher risk scores
    affected_ids = joint_loss_data["affected_locations"]
    affected_risk = locations_with_risk[
        locations_with_risk["location_id"].isin(affected_ids)
    ]["risk_score"].values
    
    non_affected_risk = locations_with_risk[
        ~locations_with_risk["location_id"].isin(affected_ids)
    ]["risk_score"].values
    
    # Statistical test
    from scipy.stats import mannwhitneyu
    
    statistic, p_value = mannwhitneyu(affected_risk, non_affected_risk, alternative="greater")
    
    cluster_validation = {
        "mean_affected_risk": np.mean(affected_risk),
        "mean_non_affected_risk": np.mean(non_affected_risk),
        "risk_difference": np.mean(affected_risk) - np.mean(non_affected_risk),
        "mann_whitney_u_statistic": statistic,
        "mann_whitney_p_value": p_value,
        "significant": p_value < 0.05,
    }
    
    logger.info(
        f"Risk cluster validation: affected mean={cluster_validation['mean_affected_risk']:.3f}, "
        f"non-affected mean={cluster_validation['mean_non_affected_risk']:.3f}, "
        f"p-value={p_value:.3f}"
    )
    
    return cluster_validation


def run_case_study_validation(
    storm_name: str = "Ian",
    storm_date: str = HURRICANE_IAN_DATE,
    model_path: Path = MODEL_OUTPUT_PATH,
    graph_path: Path = GRAPH_OUTPUT_PATH,
    output_path: Path = VALIDATION_OUTPUT_PATH,
) -> dict[str, Any]:
    """
    Run complete historical case study validation.
    
    Args:
        storm_name: Name of hurricane to validate
        storm_date: Date of storm
        model_path: Path to trained GNN model
        graph_path: Path to graph data
        output_path: Path to save validation results
    
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Starting case study validation for Hurricane {storm_name}...")
    
    # Load data
    logger.info("Loading model and data...")
    
    # Add safe globals for torch_geometric classes
    try:
        from torch_geometric.data.data import DataEdgeAttr
        from torch_geometric.data.data import Data
        torch.serialization.add_safe_globals([DataEdgeAttr, Data])
    except ImportError:
        pass  # Older torch_geometric versions may not have this
    
    graph_data = torch.load(graph_path, weights_only=False)
    locations = pd.read_parquet("data_pipeline/gold/gold_features.parquet")
    claims = pd.read_parquet("data_pipeline/silver/claims.parquet")
    
    # Load model
    model = load_gnn_model(model_path, graph_data)
    
    # Load historical storm data
    storm_track = load_historical_storm_data(storm_name)
    
    # Compute actual joint losses
    joint_loss_data = compute_actual_joint_losses(claims, storm_date)
    
    if not joint_loss_data["affected_locations"]:
        logger.warning("No affected locations found. Skipping validation.")
        return {"error": "No affected locations found"}
    
    # Compute GNN-predicted correlations
    predicted_correlations = compute_gnn_predicted_correlations(
        model, graph_data, locations, joint_loss_data["affected_locations"]
    )
    
    # Compare correlations
    if joint_loss_data["correlation_matrix"] is not None:
        correlation_comparison = compare_correlations(
            joint_loss_data["correlation_matrix"],
            predicted_correlations,
        )
    else:
        correlation_comparison = {"error": "Insufficient data for correlation comparison"}
    
    # Validate risk clusters
    cluster_validation = validate_risk_clusters(
        model, graph_data, locations, joint_loss_data
    )
    
    # Compile results
    validation_results = {
        "storm_name": storm_name,
        "storm_date": storm_date,
        "num_affected_locations": len(joint_loss_data["affected_locations"]),
        "total_actual_loss": joint_loss_data["total_loss"],
        "correlation_comparison": correlation_comparison,
        "cluster_validation": cluster_validation,
        "success": (
            correlation_comparison.get("correlation", 0) > 0.3 and
            cluster_validation.get("significant", False)
        ),
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([validation_results])
    results_df.to_parquet(output_path, index=False)
    logger.info(f"Validation results saved to {output_path}")
    
    # Log summary
    logger.info("=" * 60)
    logger.info("CASE STUDY VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Storm: {storm_name} ({storm_date})")
    logger.info(f"Affected locations: {validation_results['num_affected_locations']}")
    logger.info(f"Total actual loss: {validation_results['total_actual_loss']:.2f}")
    
    if "correlation" in correlation_comparison:
        logger.info(f"Correlation match: r={correlation_comparison['correlation']:.3f}")
    
    logger.info(f"Risk cluster significance: p={cluster_validation['mann_whitney_p_value']:.3f}")
    logger.info(f"Validation successful: {validation_results['success']}")
    logger.info("=" * 60)
    
    return validation_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_case_study_validation()
    
    if results.get("success"):
        logger.info("✓ Case study validation PASSED")
    else:
        logger.warning("✗ Case study validation FAILED")
        logger.info("This may indicate the model needs refinement or data issues exist.")
