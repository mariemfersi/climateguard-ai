"""
GraphSAGE/GAT training for portfolio accumulation risk scoring.

Trains a Graph Neural Network to predict accumulation risk scores that capture
spatial correlation patterns in hurricane losses across the portfolio.

Usage:
    python -m ml.gnn_accumulation.train_gnn
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GraphSAGE, global_mean_pool
from torch_geometric.loader import NeighborLoader

from ml.gnn_accumulation.build_graph import GRAPH_OUTPUT_PATH

logger = logging.getLogger(__name__)

# Model paths
MODEL_OUTPUT_PATH = Path("data_pipeline/bronze/gnn_accumulation/gnn_model.pt")
TRAINING_RESULTS_PATH = Path("data_pipeline/bronze/gnn_accumulation/training_results.parquet")

# GNN hyperparameters
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 1024

# GraphSAGE specific parameters
NUM_NEIGHBORS = [10, 5, 5]  # Neighbor sampling for each layer


class AccumulationRiskGNN(torch.nn.Module):
    """
    Graph Neural Network for accumulation risk scoring.
    
    Uses GraphSAGE for message passing with attention mechanisms to capture
    spatial correlation patterns in hurricane risk across the portfolio.
    """

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = HIDDEN_CHANNELS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention

        # Input projection
        self.input_proj = torch.nn.Linear(num_features, hidden_channels)

        # GraphSAGE layers with optional attention
        self.convs = torch.nn.ModuleList()
        if use_attention:
            # Use GAT (Graph Attention Network) layers
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=4, dropout=dropout)
            )
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
                )
        else:
            # Use standard GraphSAGE
            self.convs.append(
                GraphSAGE(hidden_channels, hidden_channels)
            )
            for _ in range(num_layers - 1):
                self.convs.append(
                    GraphSAGE(hidden_channels, hidden_channels)
                )

        # Output projection for accumulation risk score
        self.output_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * (4 if use_attention else 1), hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for accumulation risk scoring.
        
        Args:
            x: Node features (num_nodes, num_features)
            edge_index: Edge indices (2, num_edges)
            batch: Batch vector for graph-level tasks
        
        Returns:
            Node-level risk scores or graph-level risk score
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output projection
        risk_scores = self.output_proj(x)

        # If batch provided, return graph-level score (mean of node scores)
        if batch is not None:
            risk_scores = global_mean_pool(risk_scores, batch)

        return risk_scores.squeeze()  # Ensure (N,) shape instead of (N,1)

    def predict_accumulation_risk(self, graph_data: Data) -> np.ndarray:
        """
        Predict accumulation risk scores for all nodes.
        
        Args:
            graph_data: PyG Data object
        
        Returns:
            Array of risk scores (num_nodes,)
        """
        self.eval()
        with torch.no_grad():
            risk_scores = self.forward(
                graph_data.x,
                graph_data.edge_index
            )
        return risk_scores.cpu().numpy().flatten()


def create_training_targets(
    claims: pd.DataFrame,
    locations: pd.DataFrame,
) -> torch.Tensor:
    """
    Create training targets for accumulation risk scoring.
    
    Target definition: A location's accumulation risk is defined by:
    1. Historical loss frequency
    2. Historical loss severity
    3. Spatial correlation with nearby losses (joint loss probability)
    
    Args:
        claims: Claims data with location_id, loss_date, damage_ratio
        locations: Location data with location_id
    
    Returns:
        Tensor of target values (num_locations,)
    """
    logger.info("Creating training targets...")
    
    # Aggregate claims by location
    location_claims = claims.groupby("location_id").agg(
        claim_count=("claim_id", "count"),
        total_loss=("damage_ratio", "sum"),
        mean_severity=("damage_ratio", "mean"),
    ).reset_index()
    
    # Merge with locations
    locations_with_claims = locations.merge(
        location_claims,
        on="location_id",
        how="left"
    )
    
    # Fill missing with zeros
    locations_with_claims["claim_count"] = locations_with_claims["claim_count"].fillna(0)
    locations_with_claims["total_loss"] = locations_with_claims["total_loss"].fillna(0)
    locations_with_claims["mean_severity"] = locations_with_claims["mean_severity"].fillna(0)
    
    # Normalize features to 0-1 range for risk score
    max_claims = locations_with_claims["claim_count"].max()
    max_loss = locations_with_claims["total_loss"].max()
    max_severity = locations_with_claims["mean_severity"].max()
    
    # Composite risk score (weighted combination)
    risk_scores = (
        0.4 * (locations_with_claims["claim_count"] / (max_claims + 1e-8)) +
        0.4 * (locations_with_claims["total_loss"] / (max_loss + 1e-8)) +
        0.2 * (locations_with_claims["mean_severity"] / (max_severity + 1e-8))
    )
    
    # Clip to [0, 1]
    risk_scores = np.clip(risk_scores, 0, 1)
    
    # Convert to tensor
    targets = torch.tensor(risk_scores, dtype=torch.float32)
    
    logger.info(f"Training targets: mean={targets.mean():.3f}, std={targets.std():.3f}")
    return targets


def train_gnn_model(
    graph_data: Data,
    targets: torch.Tensor,
    validation_split: float = 0.2,
    hidden_channels: int = HIDDEN_CHANNELS,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    epochs: int = EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = 42,
) -> tuple[AccumulationRiskGNN, dict]:
    """
    Train GNN model for accumulation risk scoring.
    
    Args:
        graph_data: PyG Data object with graph structure
        targets: Target values for training
        validation_split: Fraction of nodes for validation
        hidden_channels: Number of hidden channels
        num_layers: Number of GNN layers
        dropout: Dropout rate
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        epochs: Maximum training epochs
        patience: Early stopping patience
        seed: Random seed
    
    Returns:
        Tuple of (trained model, training metrics)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info("Starting GNN training...")
    
    # Split nodes into train/validation
    num_nodes = graph_data.num_nodes
    num_val = int(num_nodes * validation_split)
    
    # Random split
    perm = torch.randperm(num_nodes)
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    # Add masks to graph data
    graph_data.train_mask = train_mask
    graph_data.val_mask = val_mask
    graph_data.y = targets
    
    # Initialize model
    model = AccumulationRiskGNN(
        num_features=graph_data.num_node_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=True,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_out = model(graph_data.x, graph_data.edge_index).squeeze()
        train_loss = F.mse_loss(train_out[train_mask], targets[train_mask])
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(graph_data.x, graph_data.edge_index).squeeze()
            val_loss = F.mse_loss(val_out[val_mask], targets[val_mask])
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
            )
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_out = model(graph_data.x, graph_data.edge_index).squeeze()
        train_mse = F.mse_loss(final_out[train_mask], targets[train_mask])
        val_mse = F.mse_loss(final_out[val_mask], targets[val_mask])
        train_mae = F.l1_loss(final_out[train_mask], targets[train_mask])
        val_mae = F.l1_loss(final_out[val_mask], targets[val_mask])
    
    metrics = {
        "train_mse": train_mse.item(),
        "val_mse": val_mse.item(),
        "train_mae": train_mae.item(),
        "val_mae": val_mae.item(),
        "epochs_trained": epoch + 1,
        "best_val_loss": best_val_loss.item(),
        "overfitting_indicator": (train_mse - val_mse).item(),
    }
    
    logger.info(f"Training complete. Final metrics: {metrics}")
    
    return model, metrics


def load_gnn_model(model_path: Path, graph_data: Data) -> AccumulationRiskGNN:
    """
    Load trained GNN model from disk.
    
    Args:
        model_path: Path to saved model
        graph_data: Graph data for model initialization
    
    Returns:
        Loaded GNN model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"GNN model not found at {model_path}")
    
    # Add safe globals for torch_geometric classes
    try:
        from torch_geometric.data.data import DataEdgeAttr
        torch.serialization.add_safe_globals([DataEdgeAttr])
    except ImportError:
        pass  # Older torch_geometric versions may not have this
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint['model_config']
    
    model = AccumulationRiskGNN(
        num_features=config['num_features'],
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_attention=config['use_attention'],
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"GNN model loaded from {model_path}")
    return model


def save_gnn_model(model: AccumulationRiskGNN, save_path: Path) -> None:
    """
    Save trained GNN model to disk.
    
    Args:
        model: Trained GNN model
        save_path: Path to save model
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_features': model.num_features,
            'hidden_channels': model.hidden_channels,
            'num_layers': model.num_layers,
            'dropout': model.dropout,
            'use_attention': model.use_attention,
        }
    }, save_path)
    logger.info(f"GNN model saved to {save_path}")


def run_gnn_training(
    graph_path: Path = GRAPH_OUTPUT_PATH,
    model_path: Path = MODEL_OUTPUT_PATH,
    results_path: Path = TRAINING_RESULTS_PATH,
) -> tuple[AccumulationRiskGNN, dict]:
    """
    Run complete GNN training pipeline.
    
    Args:
        graph_path: Path to saved graph
        model_path: Path to save trained model
        results_path: Path to save training results
    
    Returns:
        Tuple of (trained model, training metrics)
    """
    logger.info("Starting GNN training pipeline...")
    
    # Load or build graph
    if graph_path.exists():
        logger.info(f"Loading graph from {graph_path}")
        # Add safe globals for torch_geometric classes
        try:
            from torch_geometric.data.data import DataEdgeAttr
            from torch_geometric.data.data import Data
            torch.serialization.add_safe_globals([DataEdgeAttr, Data])
        except ImportError:
            pass  # Older torch_geometric versions may not have this
        
        graph_data = torch.load(graph_path, weights_only=False)
    else:
        logger.info("Building new graph")
        graph_data = run_graph_construction()
    
    # Load claims data for targets
    claims = pd.read_parquet("data_pipeline/silver/claims.parquet")
    locations = pd.read_parquet("data_pipeline/gold/gold_features.parquet")
    
    # Create training targets
    targets = create_training_targets(claims, locations)
    
    # Train model
    model, metrics = train_gnn_model(
        graph_data,
        targets,
        seed=42,
    )
    
    # Save model
    save_gnn_model(model, model_path)
    
    # Save training results
    results_df = pd.DataFrame([metrics])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(results_path, index=False)
    logger.info(f"Training results saved to {results_path}")
    
    # Log to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "model_type": "GraphAttentionNetwork",
            "task": "accumulation_risk_scoring",
            "hidden_channels": HIDDEN_CHANNELS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        })
        
        mlflow.log_metrics(metrics)
        
        # Log model with input example
        input_example = (graph_data.x[:1], graph_data.edge_index)
        mlflow.pytorch.log_model(
            model, 
            "gnn_accumulation_model",
            input_example=input_example,
            serialization_format="pickle"  # Use pickle format
        )
        mlflow.log_artifact(str(graph_path), "data/portfolio_graph.pt")
    
    logger.info("GNN training pipeline complete.")
    return model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, metrics = run_gnn_training()
    logger.info(f"Final metrics: {metrics}")
