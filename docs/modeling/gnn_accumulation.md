# Graph Neural Network (Portfolio Accumulation Risk) — Phase 6 Methodology & Results

## Summary

Phase 6 implements a Graph Neural Network (GNN) to model portfolio accumulation risk as a spatial correlation problem. Unlike traditional per-location independent risk models, the GNN captures the joint probability of correlated losses across nearby locations — a critical component of hurricane risk that simple models miss.

**Key Innovation:** Multi-layered graph construction with semantically justified edge types, not arbitrary k-NN connections. Each edge type captures a specific risk correlation mechanism relevant to hurricane peril.

## Graph Construction Rationale

The graph construction is the most critical design decision in this phase. We use a **multi-layered edge approach** where each edge type has a specific, domain-justified purpose:

### 1. Spatial Proximity Edges

**Rationale:** Nearby locations are more likely to be affected by the same storm due to the spatial extent of hurricane wind fields (typically 100-500km radius) and rainfall patterns.

**Implementation:**
- Distance-based k-NN with radius threshold (100km default)
- Uses Haversine distance for accurate geographic calculation
- k=5 neighbors to balance connectivity vs. computational efficiency

**Physical Justification:** Hurricane wind fields decay with distance but can cause significant damage over large areas. Two locations 50km apart are likely to experience similar wind speeds from the same storm.

### 2. Peril Correlation Edges

**Rationale:** Locations that have historically experienced joint losses during the same storms have empirically demonstrated correlation. This captures actual risk patterns beyond simple geographic proximity.

**Implementation:**
- Computed from historical claims data
- Correlation threshold of 0.3 (minimum meaningful correlation)
- Minimum 3 joint claim events for reliability
- Based on year-level loss correlation matrix

**Physical Justification:** Some location pairs may have high correlation despite geographic distance due to:
- Similar coastal exposure (e.g., both on same peninsula)
- Similar vulnerability (e.g., same construction types)
- Shared microclimate effects

### 3. Coastal Basin Edges

**Rationale:** Locations in the same coastal basin share similar hurricane exposure characteristics. Large-scale geographic patterns (Gulf Coast vs Atlantic Coast) drive systematic risk differences.

**Implementation:**
- Groups locations by coastal proximity and longitude
- 5 basin types: inland, western_gulf, eastern_gulf, florida_atlantic, atlantic_north
- Complete subgraph within each basin (all pairs connected)
- 50km coastal proximity threshold

**Physical Justification:** Hurricanes tend to follow specific tracks based on large-scale climate patterns. The Gulf Coast has different storm tracks, wind speed distributions, and exposure than the Atlantic Coast.

### 4. Storm Footprint Edges

**Rationale:** Locations that have been within the same storm's influence radius historically share actual peril exposure, not just geographic proximity.

**Implementation:**
- Based on HURDAT2 storm track data
- 500km influence radius per storm
- Edges weighted by number of shared storm exposures
- Captures actual historical co-exposure

**Physical Justification:** Two locations might be geographically close but rarely affected by the same storms due to local geography (barrier islands, elevation). Storm footprint edges capture the realized co-exposure.

## Architecture & Model Design

### Graph Construction Parameters

```python
SPATIAL_RADIUS_KM = 100  # 100km radius for spatial edges
K_NEIGHBORS = 5  # Number of nearest neighbors
PERIL_CORRELATION_THRESHOLD = 0.3  # Minimum correlation
COASTAL_PROXIMITY_THRESHOLD_KM = 50  # Coastal grouping
STORM_INFLUENCE_RADIUS_KM = 500  # Storm impact radius
```

### GNN Architecture

We use a **Graph Attention Network (GAT)** for its ability to learn edge importance weights:

- **Hidden channels:** 64 (balances capacity vs. overfitting)
- **Layers:** 3 (sufficient for multi-hop dependencies)
- **Attention heads:** 4 per layer (captures different relationship types)
- **Dropout:** 0.3 (regularization for limited data)
- **Output:** Sigmoid activation for risk score [0,1]

**Why GAT over GraphSAGE:**
- Attention mechanisms provide interpretability (edge importance)
- Better handles heterogeneous edge types
- Attention weights can be visualized for explainability

### Training Setup

**Target Definition:**
Composite risk score combining:
- 40% historical claim frequency
- 40% historical total loss
- 20% historical mean severity

**Training Parameters:**
- Learning rate: 0.001 (conservative)
- Weight decay: 1e-5 (L2 regularization)
- Early stopping: Patience=10 epochs
- Validation split: 20% of nodes
- Optimizer: Adam

**Loss Function:** MSE between predicted and target risk scores

## Node Features

Node features capture location-specific risk drivers:

**Geographic:**
- `lat`, `lon`: Geographic coordinates
- `distance_to_coast_km`: Coastal exposure

**Exposure:**
- `tiv_usd`: Total insured value
- `year_built`: Construction age

**Vulnerability:**
- `construction_class`: Building type (one-hot encoded)
- `roof_type`: Roof material (one-hot encoded)

**Features are normalized to zero mean, unit variance.**

## Edge Attributes

Each edge type includes semantic attributes:

**Spatial edges:** Distance (km)
**Peril correlation edges:** Correlation coefficient
**Coastal basin edges:** Binary (1 if same basin)
**Storm footprint edges:** Shared storm count

These attributes help the GNN distinguish between edge types and learn their relative importance.

## Training Pipeline

### Data Preparation

1. Load location features from Gold layer
2. Load historical claims data
3. Compute graph edges (multi-type)
4. Extract node features
5. Create training targets from claims
6. Split nodes into train/validation (20% holdout)

### Training Loop

1. Initialize GAT model with random weights
2. For each epoch:
   - Forward pass through all GNN layers
   - Compute MSE loss on training nodes
   - Backpropagate and update weights
   - Evaluate on validation nodes
   - Track best model based on validation loss
3. Early stopping if no improvement for 10 epochs
4. Save best model checkpoint

### Overfitting Prevention

- Node-level train/validation split (not graph-level)
- Dropout regularization (0.3)
- L2 weight decay (1e-5)
- Early stopping (patience=10)
- Conservative architecture (64 hidden channels)

## Explainability with GNNExplainer

### Subgraph Extraction

GNNExplainer identifies the most important subgraph for individual location predictions:

**Process:**
1. Mask out edges and node features
2. Learn optimal mask via gradient descent
3. Extract high-importance nodes and edges
4. Visualize subgraph with geographic context

**Output:**
- Important neighboring locations
- Critical edge connections
- Feature importance scores
- Visual subgraph plot

### Interpretation

The explainer reveals:
- **Spatial risk clusters:** Which locations jointly drive risk
- **Critical connections:** Which edges matter most for prediction
- **Feature drivers:** Which location features are most important

This feeds into Phase 8's explainability layer and Phase 9's LLM reasoning.

## Historical Case Study Validation

### Validation Approach

We validate against historical hurricane events to ensure the GNN captures real correlation patterns:

**Case Study: Hurricane Ian (2022)**
- Identify locations affected by Ian
- Compute actual joint loss pattern
- Compare with GNN-predicted correlation
- Verify high-risk subgraph matches actual impact zone

**Success Criteria:**
- GNN identifies high-risk clusters in actual impact zones
- Predicted correlations match historical joint loss patterns
- Subgraph explanations align with storm physics

### Expected Findings

Based on hurricane physics, we expect:
- Coastal locations show higher correlation than inland
- Nearby locations (<100km) show higher correlation
- Same-basin locations show systematic correlation patterns
- Storm footprint edges capture historical co-exposure

## Integration with Downstream Phases

### Phase 5 (TFT)

TFT regional trend forecasts inform GNN node features:
- Regional climate trends as node features
- Time-varying risk adjustment factors
- Uncertainty quantification from TFT

### Phase 8 (Monte Carlo Engine)

GNN correlation matrix informs joint loss simulation:
- Correlation matrix for Monte Carlo sampling
- Spatial dependency structure
- Portfolio-level aggregation risk

### Phase 9 (Multi-agent LLM)

GNN explainability provides:
- Subgraph visualizations for LLM reasoning
- Risk cluster explanations
- Scenario analysis support

## Known Limitations

### Graph Construction

**Edge definition sensitivity:** Results depend on edge type selection and parameters. Different thresholds may produce different risk clusters.

**Data requirements:** Peril correlation edges require sufficient historical claims. New locations may have limited correlation data.

**Geographic scope:** Graph is currently limited to specific regions. Expansion to new geographies requires re-computing edges.

### Model Limitations

**Static graph:** Current graph is static (doesn't change over time). Future work could incorporate temporal graph structure.

**Training data size:** Limited historical hurricane events may constrain learning. Transfer learning from other catastrophe perils could help.

**Interpretability trade-off:** GNNs are more interpretable than some deep learning models but still less transparent than linear models.

## Performance Expectations

Given the innovation and complexity, we expect:

1. **Improved risk clustering:** GNN should identify meaningful risk clusters that align with geography and historical patterns.

2. **Better correlation capture:** GNN correlation estimates should match historical joint loss patterns better than simple distance-based correlations.

3. **Explainable insights:** Subgraph explanations should be interpretable and align with domain knowledge about hurricane risk.

4. **Computational efficiency:** Graph operations should scale to portfolio sizes relevant for insurance applications (thousands of locations).

**Success criteria:** GNN-derived accumulation risk scores demonstrably capture known correlated-loss patterns from historical storms.

## Reproducibility

- **Graph construction:** `ml/gnn_accumulation/build_graph.py`
- **GNN training:** `ml/gnn_accumulation/train_gnn.py`
- **Explainability:** `ml/gnn_accumulation/gnn_explain.py`
- **MLflow:** All runs logged with parameters, metrics, and artifacts
- **Random seeds:** Fixed (seed=42) for reproducibility
- **Node split:** Random but stratified by risk level

## Usage

### Build portfolio graph
```bash
python -m ml.gnn_accumulation.build_graph
```

### Train GNN model
```bash
python -m ml.gnn_accumulation.train_gnn
```

### Extract explanations
```bash
python -m ml.gnn_accumulation.gnn_explain
```

### Full pipeline
```bash
python -m ml.gnn_accumulation.build_graph
python -m ml.gnn_accumulation.train_gnn
python -m ml.gnn_accumulation.gnn_explain
```

### Run tests
```bash
pytest tests/test_gnn_accumulation.py -v
```

## Future Improvements

1. **Temporal graphs:** Incorporate time-varying edge weights based on climate trends
2. **Heterogeneous graphs:** Use different message passing for different edge types
3. **Transfer learning:** Pre-train on global hurricane data, fine-tune on portfolio
4. **Alternative architectures:** Explore Graph Transformers, Graph Isomorphism Networks
5. **Uncertainty quantification:** Bayesian GNNs for confidence intervals

## Conclusion

The GNN accumulation risk model represents the most technically differentiated component of the ClimateGuard AI stack. By modeling portfolio risk as a graph learning problem with semantically justified edge construction, we capture spatial correlation patterns that traditional independent location models cannot miss.

The multi-layered edge approach (spatial, peril correlation, coastal basin, storm footprint) ensures the graph reflects actual hurricane risk physics, not arbitrary geometric relationships. This provides a genuine innovation over standard catastrophe modeling approaches and should be emphasized in technical discussions.

The model's explainability through GNNExplainer provides transparency into risk cluster formation, supporting both regulatory requirements and stakeholder communication. This integration of deep learning with domain-specific graph design represents the cutting edge of climate risk modeling.
