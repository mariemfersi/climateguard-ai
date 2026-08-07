# Monte Carlo Portfolio Loss Simulation Engine

## Overview
The **Monte Carlo Loss Simulation Engine** (`ml/monte_carlo_engine/simulate_portfolio_losses.py`) fuses the four core predictive models (Phases 4–7) into a unified, stochastic simulation of $N$ synthetic storm seasons. Rather than relying on single point predictions or static vendor cat model lookups, it constructs a complete, climate-conditioned annual loss distribution for the portfolio.

```
       ┌────────────────────────────────────────────────────────┐
       │             Phase 4: GBM Frequency & Severity          │
       └───────────────────────────┬────────────────────────────┘
                                   │
       ┌───────────────────────────┼────────────────────────────┐
       │             Phase 5: Temporal Fusion Transformer       │
       │                  (Climate Trend Multiplier)            │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │   Poisson/Bernoulli Frequency × Beta Severity Sampling  │
       └───────────────────────────┬────────────────────────────┘
                                   │
       ┌───────────────────────────┼────────────────────────────┐
       │             Phase 6: Graph Neural Network              │
       │            (Spatial Accumulation Multiplier)           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │        Portfolio Loss Distribution (VaR & TVaR 99.5%)   │
       └────────────────────────────────────────────────────────┘
```

## Model Fusion Architecture & Anti-Double-Counting Methodology
1. **Frequency Model (GBM + TFT)**:
   - Baseline annual probability $p_{\text{base}, i}$ per location $i$ comes from static exposure/climate features (XGBoost/CatBoost).
   - Climate trend multiplier $\tau_{\text{climate}}$ comes from the Temporal Fusion Transformer (TFT) conditioned on sea-surface temperature (SST) and macro-climate indices.
   - Per-location event occurrence: Bernoulli trial with $p_i = \min(p_{\text{base}, i} \cdot \tau_{\text{climate}}, 1.0)$.
   - *Rationale*: GBM captures cross-sectional location risk; TFT captures non-stationary temporal climate trends. Multiplying them preserves orthogonality.

2. **Severity Model (GBM + Beta Distribution)**:
   - Expected damage ratio $\mu_i = \mathbb{E}[\text{damage\_ratio} \mid \text{claim}]$ comes from the LightGBM severity model.
   - Individual claim severity $S_i$ is sampled from a $\text{Beta}(\alpha_i, \beta_i)$ distribution re-parameterized around mean $\mu_i$ with coefficient of variation $\text{CV} = 0.5$.

3. **Spatial Accumulation Model (GNN Multiplier)**:
   - Spatial accumulation factor $\alpha_{\text{GNN}, i}$ is derived from the Graph Neural Network (GraphSAGE / GAT) adjacency matrix.
   - Correlated loss amplification: $S_{i, \text{correlated}} = S_i \cdot \alpha_{\text{GNN}, i}$, capturing cascading peril contagion without double-counting per-location severity means.

4. **Loss Calculation**:
   - Location Loss: $L_{i, s} = S_{i, s, \text{correlated}} \cdot \text{TIV}_i$.
   - Annual Portfolio Loss: $L_s = \sum_{i=1}^{M} L_{i, s}$ for season $s \in \{1, \dots, N\}$.

## Capital Risk Metrics (Solvency II SCR Alignment)
- **Value-at-Risk (VaR at 99.5%)**:
  $$\text{VaR}_{0.995} = Q_{0.995}(\{L_1, L_2, \dots, L_N\})$$
- **Tail Value-at-Risk (TVaR / Expected Shortfall at 99.5%)**:
  $$\text{TVaR}_{0.995} = \mathbb{E}[L \mid L \ge \text{VaR}_{0.995}]$$

## Predefined Stress Scenario Library
The `ml/monte_carlo_engine/stress_scenarios.py` module supports deterministic regulatory and climate stress tests:
- `baseline`: Current climate baseline ($1.0\times$ frequency, $1.0\times$ severity).
- `p100_1x`: 1-in-100 year event loading ($1.25\times$ frequency, $1.35\times$ severity).
- `p200_1x`: 1-in-200 year Solvency II SCR stress ($1.50\times$ frequency, $1.60\times$ severity).
- `climate_change_1c`: +1°C SST warming ($1.20\times$ frequency trend).
- `climate_change_2c`: +2°C severe warming ($1.45\times$ frequency trend, $1.30\times$ severity shift).
- `concentrated_coastal`: Coastal surge accumulation ($1.50\times$ severity loading on near-coast assets).

## Outputs & Persistence
- Loss distribution parquet: `data_pipeline/bronze/monte_carlo/portfolio_loss_distribution.parquet`
- Metrics JSON: `data_pipeline/bronze/monte_carlo/risk_metrics.json`
- Stress comparison parquet: `data_pipeline/bronze/monte_carlo/stress_scenario_comparison.parquet`
