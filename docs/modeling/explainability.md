# Explainability Layer Architecture (SHAP & Actionable Counterfactuals)

## Overview
The **Explainability Layer** (`ml/explainability/`) ensures that every prediction, risk score, and generated treaty pricing memo produced by ClimateGuard AI is fully auditable, explainable, and grounded in quantitative evidence.

```
       ┌────────────────────────────────────────────────────────┐
       │               GBM Frequency / Severity Model           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌──────────────────────────┐  ┌──────────────────────────┐
       │   TreeSHAP Explainer     │  │ Counterfactual Explainer │
       │ (Feature Attributions)   │  │  (Risk Mitigation Engine)│
       └────────────┬─────────────┘  └────────────┬─────────────┘
                    │                             │
                    ▼                             ▼
       ┌────────────────────────────────────────────────────────┐
       │ DB / Parquet Storage Schema matching Design Doc        │
       │ explanations(prediction_id, location_id, feature, ...) │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │       Grounding Context for Multi-Agent LLM Memos      │
       └────────────────────────────────────────────────────────┘
```

## 1. TreeSHAP Feature Attribution (`shap_explainer.py`)
Uses `shap.TreeExplainer` on LightGBM/XGBoost ensembles to extract exact Shapley value attributions for each feature.

### Database Schema Alignment
Attributions are formatted and saved according to the database schema defined in Section 12 of the design specification:
```sql
CREATE TABLE explanations (
    prediction_id VARCHAR(64) PRIMARY KEY,
    location_id   VARCHAR(32) NOT NULL,
    feature_name  VARCHAR(64) NOT NULL,
    feature_value FLOAT,
    shap_value    FLOAT NOT NULL,
    rank          INT NOT NULL
);
```

### Key Capabilities
- `explain_instance(feature_vector, location_id)`: Explains individual location predictions with top 5 risk drivers.
- `explain_batch(X_matrix, location_ids)`: Computes batch attributions saved to `data_pipeline/bronze/explainability/shap_explanations.parquet`.

## 2. Actionable Counterfactual Engine (`counterfactual_explainer.py`)
Generates "what-if" risk improvement recommendations by evaluating actionable property physical modifications against actuarial vulnerability curves.

### Supported Actionable Modifications
1. **Roof Reinforcement**:
   - `flat` $\rightarrow$ `hip`: -35% damage ratio reduction.
   - `gable` $\rightarrow$ `hip`: -25% damage ratio reduction.
   - `flat` $\rightarrow$ `reinforced_hip`: -55% damage ratio reduction.
2. **Construction Class Upgrade**:
   - `frame` $\rightarrow$ `masonry_cbs`: -40% damage ratio reduction.
   - `frame` $\rightarrow$ `reinforced_concrete`: -60% damage ratio reduction.
3. **Comprehensive Fortification**:
   - Combined roof + construction upgrade: -65% total expected loss reduction.

### Sample Underwriting Recommendation Output
```json
{
  "location_id": "LOC0002042",
  "scenario_type": "roof_reinforcement",
  "baseline_annual_loss": 2128.40,
  "new_annual_loss": 1596.30,
  "annual_loss_reduction_usd": 532.10,
  "pct_loss_reduction": 25.0,
  "recommendation": "Upgrade roof from 'gable' to 'hip' to reduce expected annual loss by $532 (-25.0%)."
}
```

## Grounding Context for Downstream LLM Agents
Every generated sentence in pricing memos (Phase 9) cites these SHAP values and counterfactual recommendations. This guarantees zero numeric hallucination during automated report writing.
