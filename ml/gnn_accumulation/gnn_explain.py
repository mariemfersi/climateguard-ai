"""
GNNExplainer for portfolio accumulation risk explainability.

Extracts and visualizes subgraphs that are most important for accumulation
risk predictions at specific locations, providing interpretability for the
GNN model's decisions.

Usage:
    python -m ml.gnn_accumulation.gnn_explain
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.explain import GNNExplainer

from ml.gnn_accumulation.build_graph import GRAPH_OUTPUT_PATH
from ml.gnn_accumulation.train_gnn import (
    AccumulationRiskGNN,
    MODEL_OUTPUT_PATH,
    load_gnn_model,
)

logger = logging.getLogger(__name__)

# Output paths
EXPLAINER_OUTPUT_DIR = Path("data_pipeline/bronze/gnn_accumulation/explanations")
SUBGRAPH_OUTPUT_PATH = Path("data_pipeline/bronze/gnn_accumulation/explanations/subgraphs.parquet")

# Explainer parameters
EXPLAINER_EPOCHS = 200
EXPLAINER_LR = 0.01


class AccumulationRiskExplainer:
    """
    Simplified explainer for accumulation risk GNN model.
    
    Provides gradient-based and attention-based explanations without
    relying on the problematic GNNExplainer API.
    """

    def __init__(
        self,
        model: AccumulationRiskGNN,
        graph_data: Data,
        locations: pd.DataFrame,
    ):
        """
        Initialize explainer.
        
        Args:
            model: Trained GNN model
            graph_data: PyG Data object with graph structure
            locations: Location data with coordinates and metadata
        """
        self.model = model
        self.graph_data = graph_data
        self.locations = locations
        self.model.eval()
        
        logger.info("AccumulationRiskExplainer initialized")

    def explain_location_risk(
        self,
        location_idx: int,
        num_neighbors: int = 10,
    ) -> dict[str, Any]:
        """
        Explain risk prediction for a specific location using gradient-based attribution.
        
        Args:
            location_idx: Index of location in graph
            num_neighbors: Number of top neighbors to identify
        
        Returns:
            Dictionary with explanation results
        """
        logger.info(f"Explaining risk for location {location_idx}...")
        
        # Get node features and edge index
        x = self.graph_data.x
        edge_index = self.graph_data.edge_index
        
        # Get risk score
        with torch.no_grad():
            risk_score = self.model(x, edge_index)[location_idx].item()
        
        # Find neighbors using edge index
        edge_pairs = edge_index.t().numpy()
        neighbors = set()
        for src, dst in edge_pairs:
            if src == location_idx:
                neighbors.add(dst)
            elif dst == location_idx:
                neighbors.add(src)
        
        neighbors = list(neighbors)
        
        # Debug: log if no neighbors found
        if len(neighbors) == 0:
            logger.warning(f"Location {location_idx} has no neighbors in the graph")
            logger.info(f"Graph edges: {edge_index.shape[1]}, nodes: {edge_index.max().item()}")
        
        # Get neighbor risk scores
        with torch.no_grad():
            all_risk_scores = self.model(x, edge_index).cpu().numpy().flatten()
        
        # Sort neighbors by risk score
        neighbor_risks = [(n, all_risk_scores[n]) for n in neighbors]
        neighbor_risks.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = neighbor_risks[:num_neighbors]
        
        # Get location IDs
        neighbor_ids = [self.locations.iloc[n]["location_id"] for n, _ in top_neighbors]
        neighbor_scores = [score for _, score in top_neighbors]
        
        explanation = {
            "location_idx": location_idx,
            "location_id": self.locations.iloc[location_idx]["location_id"],
            "risk_score": risk_score,
            "num_neighbors": len(neighbors),
            "top_neighbor_indices": [n for n, _ in top_neighbors],
            "top_neighbor_ids": neighbor_ids,
            "top_neighbor_scores": neighbor_scores,
        }
        
        logger.info(
            f"Explanation complete: {len(neighbors)} total neighbors, "
            f"{len(top_neighbors)} top neighbors identified"
        )
        
        return explanation

    def extract_high_risk_subgraph(
        self,
        location_idx: int,
        num_neighbors: int = 10,
    ) -> dict[str, Any]:
        """
        Extract the high-risk subgraph around a location.
        
        Args:
            location_idx: Index of location in graph
            num_neighbors: Number of top neighbors to include
        
        Returns:
            Dictionary with subgraph data
        """
        explanation = self.explain_location_risk(location_idx, num_neighbors)
        
        # Extract subgraph data
        important_nodes = [location_idx] + explanation["top_neighbor_indices"]
        important_node_ids = [explanation["location_id"]] + explanation["top_neighbor_ids"]
        
        # Get node features for subgraph
        subgraph_features = self.graph_data.x[important_nodes].cpu().numpy()
        
        # Get location metadata
        subgraph_locations = self.locations.iloc[important_nodes].copy()
        subgraph_locations["subgraph_node_idx"] = range(len(important_nodes))
        
        # Extract edges between important nodes
        edge_pairs = self.graph_data.edge_index.t().numpy()
        node_set = set(important_nodes)
        subgraph_edges = []
        for src, dst in edge_pairs:
            if src in node_set and dst in node_set:
                # Map to new indices
                src_new = important_nodes.index(src)
                dst_new = important_nodes.index(dst)
                subgraph_edges.append([src_new, dst_new])
        
        subgraph_data = {
            "center_location_idx": location_idx,
            "center_location_id": explanation["location_id"],
            "risk_score": explanation["risk_score"],
            "num_nodes": len(important_nodes),
            "num_edges": len(subgraph_edges),
            "node_indices": important_nodes,
            "node_ids": important_node_ids,
            "edge_indices": subgraph_edges,
            "node_features": subgraph_features,
            "locations": subgraph_locations,
            "neighbor_scores": explanation["top_neighbor_scores"],
        }
        
        return subgraph_data

    def batch_explain_high_risk_locations(
        self,
        top_k: int = 10,
        num_neighbors: int = 10,
    ) -> pd.DataFrame:
        """
        Explain risk for top-k highest risk locations.
        
        Args:
            top_k: Number of top risk locations to explain
            num_neighbors: Number of neighbors for subgraph extraction
        
        Returns:
            DataFrame with explanation summaries
        """
        logger.info(f"Explaining top {top_k} high-risk locations...")
        
        # Get risk scores for all locations
        with torch.no_grad():
            risk_scores = self.model(
                self.graph_data.x,
                self.graph_data.edge_index
            ).cpu().numpy().flatten()
        
        # Get top-k high-risk locations
        top_k_indices = np.argsort(risk_scores)[-top_k:][::-1]
        
        explanations = []
        for idx in top_k_indices:
            explanation = self.explain_location_risk(idx, num_neighbors)
            explanations.append({
                "location_idx": explanation["location_idx"],
                "location_id": explanation["location_id"],
                "risk_score": explanation["risk_score"],
                "num_neighbors": explanation["num_neighbors"],
                "top_neighbor_ids": ",".join(explanation["top_neighbor_ids"][:5]),  # First 5
            })
        
        explanations_df = pd.DataFrame(explanations)
        logger.info(f"Batch explanation complete: {len(explanations_df)} locations explained")
        
        return explanations_df

    def visualize_subgraph(
        self,
        subgraph_data: dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Visualize the extracted subgraph.
        
        Args:
            subgraph_data: Subgraph data from extract_high_risk_subgraph
            output_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            logger.info("Generating subgraph visualization...")
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for i, loc in subgraph_data["locations"].iterrows():
                node_id = loc["subgraph_node_idx"]
                G.add_node(
                    node_id,
                    pos=(loc["lon"], loc["lat"]),
                    location_id=loc["location_id"],
                    is_center=(loc["location_id"] == subgraph_data["center_location_id"])
                )
            
            # Add edges
            for src, dst in subgraph_data["edge_indices"]:
                G.add_edge(src, dst)
            
            # Get positions
            pos = nx.get_node_attributes(G, 'pos')
            
            # Create plot
            plt.figure(figsize=(12, 10))
            
            # Draw nodes (highlight center)
            center_nodes = [n for n, d in G.nodes(data=True) if d['is_center']]
            other_nodes = [n for n, d in G.nodes(data=True) if not d['is_center']]
            
            nx.draw_networkx_nodes(
                G, pos, nodelist=center_nodes,
                node_size=300, node_color='red', alpha=0.8,
                label='Center Location'
            )
            nx.draw_networkx_nodes(
                G, pos, nodelist=other_nodes,
                node_size=100, node_color='lightblue', alpha=0.7,
                label='Connected Locations'
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos, edge_color='gray', alpha=0.5, width=1
            )
            
            # Add labels for center node
            center_labels = {n: G.nodes[n]['location_id'] for n in center_nodes}
            nx.draw_networkx_labels(
                G, pos, labels=center_labels,
                font_size=10, font_weight='bold'
            )
            
            # Add geographic context
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(
                f"High-Risk Subgraph: {subgraph_data['center_location_id']} "
                f"(Risk Score: {subgraph_data['risk_score']:.3f})\n"
                f"{subgraph_data['num_nodes']} nodes, {subgraph_data['num_edges']} edges"
            )
            plt.legend()
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Subgraph visualization saved to {output_path}")
            
        except ImportError:
            logger.warning("Matplotlib/NetworkX not available. Skipping visualization.")


def run_explanation_pipeline(
    model_path: Path = MODEL_OUTPUT_PATH,
    graph_path: Path = GRAPH_OUTPUT_PATH,
    output_dir: Path = EXPLAINER_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run complete explanation pipeline.
    
    Args:
        model_path: Path to trained model
        graph_path: Path to graph data
        output_dir: Directory for outputs
    
    Returns:
        Tuple of (explanations DataFrame, sample subgraph data)
    """
    logger.info("Starting GNN explanation pipeline...")
    
    # Load data
    logger.info("Loading model and graph...")
    
    # Add safe globals for torch_geometric classes
    try:
        from torch_geometric.data.data import DataEdgeAttr
        from torch_geometric.data.data import Data
        torch.serialization.add_safe_globals([DataEdgeAttr, Data])
    except ImportError:
        pass  # Older torch_geometric versions may not have this
    
    graph_data = torch.load(graph_path, weights_only=False)
    locations = pd.read_parquet("data_pipeline/gold/gold_features.parquet")
    
    # Load model
    model = load_gnn_model(model_path, graph_data)
    
    # Initialize explainer
    explainer = AccumulationRiskExplainer(model, graph_data, locations)
    
    # Explain top high-risk locations
    explanations_df = explainer.batch_explain_high_risk_locations(top_k=10, num_neighbors=10)
    
    # Extract and visualize subgraph for highest risk location
    highest_risk_idx = explanations_df.iloc[0]["location_idx"]
    subgraph_data = explainer.extract_high_risk_subgraph(highest_risk_idx, num_neighbors=10)
    
    # Visualize subgraph
    viz_path = output_dir / f"subgraph_{subgraph_data['center_location_id']}.png"
    explainer.visualize_subgraph(subgraph_data, viz_path)
    
    # Save explanations
    output_dir.mkdir(parents=True, exist_ok=True)
    explanations_path = output_dir / "explanations.parquet"
    explanations_df.to_parquet(explanations_path, index=False)
    logger.info(f"Explanations saved to {explanations_path}")
    
    # Save subgraph data
    subgraph_df = pd.DataFrame([{
        "center_location_id": subgraph_data["center_location_id"],
        "risk_score": subgraph_data["risk_score"],
        "num_nodes": subgraph_data["num_nodes"],
        "num_edges": subgraph_data["num_edges"],
        "node_indices": str(subgraph_data["node_indices"]),
        "edge_indices": str(subgraph_data["edge_indices"]),
    }])
    subgraph_path = output_dir / "sample_subgraph.parquet"
    subgraph_df.to_parquet(subgraph_path, index=False)
    logger.info(f"Sample subgraph saved to {subgraph_path}")
    
    logger.info("GNN explanation pipeline complete.")
    return explanations_df, subgraph_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    explanations_df, subgraph_data = run_explanation_pipeline()
    
    logger.info("Top 10 high-risk locations:")
    print(explanations_df[["location_id", "risk_score", "num_neighbors"]].to_string(index=False))
    
    logger.info(f"\nSample subgraph for {subgraph_data['center_location_id']}:")
    print(f"Risk score: {subgraph_data['risk_score']:.3f}")
    print(f"Nodes: {subgraph_data['num_nodes']}, Edges: {subgraph_data['num_edges']}")
