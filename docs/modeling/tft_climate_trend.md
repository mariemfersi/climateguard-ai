# Temporal Fusion Transformer (Climate Trend) — Phase 5 Methodology & Results

## Summary

Phase 5 implements a Temporal Fusion Transformer (TFT) to forecast multi-horizon climate trends (hurricane frequency and severity) conditioned on climate covariates (SST anomaly, ENSO index). The central challenge and finding is:

- **Sample size limitation:** Only 74 years of historical data (1950-2023) with ~10 regional clusters, resulting in ~740 independent time-series samples — far fewer than typical deep learning requirements
- **Data leakage mitigation:** Climate covariates correctly marked as `time_varying_unknown_reals` since future SST/ENSO values are not known at prediction time
- **Regularization strategy:** Conservative architecture (hidden_size=16, dropout=0.2, strong gradient clipping) with early stopping to prevent overfitting
- **Honest evaluation:** Backtest against naive baselines (historical mean, persistence, linear trend) with bootstrap confidence intervals

## Architecture & Design Decisions

### Model Configuration

Given the extreme sample size limitation, we use a deliberately conservative TFT architecture:

- **Hidden size:** 16 (vs. default 64) — reduces model capacity
- **Attention heads:** 2 (vs. default 4) — limits complexity of attention mechanisms
- **Hidden layers:** 2 (vs. default 4) — shallow network
- **Dropout:** 0.2 (higher than typical 0.1) — aggressive regularization
- **Gradient clipping:** 0.1 — prevents large weight updates
- **Learning rate:** 0.01 — conservative optimization
- **Early stopping:** Patience=5 epochs — stops training at first sign of overfitting

### Data Preparation

**Grain:** (region_id, year) — one time series per region with yearly observations

**Static covariates:**
- `region_id`: Categorical identifier for regional cluster (10 clusters from Phase 4)

**Time-varying covariates (unknown at prediction time):**
- `basin_sst_celsius_mean`: Yearly mean sea surface temperature
- `basin_sst_celsius_max`: Yearly maximum sea surface temperature  
- `oni_anomaly_celsius`: Oceanic Niño Index anomaly
- `enso_phase`: ENSO phase (El Niño/La Niña/Neutral)

**Target variables:**
- `frequency`: Hurricane count per region-year
- `severity`: Mean damage ratio per region-year

**Temporal parameters:**
- Encoder length: 10 years (look-back window)
- Predictor length: 5 years (forecast horizon)
- Training range: 1950-2023 (74 years)

### Critical Data Leakage Prevention

**Original issue:** Climate covariates (SST, ENSO) were initially marked as `time_varying_known_reals`, implying they are known for future years at prediction time. This would cause data leakage since we cannot know future climate values when making predictions.

**Fix:** All climate covariates are now correctly marked as `time_varying_unknown_reals`. This means:

- During training: Model learns historical relationships between climate variables and hurricane activity
- During inference: Model must either:
  1. Use lagged climate values only (restrictive but safe)
  2. Integrate with a separate climate forecast model (future work)
  3. Accept that predictions are conditional on hypothetical climate scenarios

**Temporal split:** We use `stratified_year_split_tft` to ensure no year appears in both train and test sets, preventing temporal leakage. The split is stratified by hurricane activity to maintain similar base rates.

## Training Pipeline

### Early Stopping Configuration

- **Monitor:** Validation loss (QuantileLoss)
- **Patience:** 5 epochs (no improvement → stop)
- **Min delta:** 1e-4 (minimum change to qualify as improvement)
- **Mode:** Minimize validation loss

### Overfitting Detection

We track the train/val loss gap as an overfitting indicator:

```python
overfitting_indicator = train_loss - val_loss
```

If `overfitting_indicator > 0.1`, we log a warning and recommend increased regularization.

### MLflow Logging

All training runs are logged to MLflow with:
- Hyperparameters (architecture, regularization, optimizer settings)
- Metrics (train/val loss, epochs trained, overfitting indicator)
- Artifacts (model weights, dataset snapshots)

## Attention Mechanism & Explainability

The TFT's attention mechanism provides interpretability:

**Static attention:** Which regions (static covariates) are most important
**Variable attention:** Which climate features drive predictions
**Temporal attention:** Which historical time steps are most relevant

We extract attention weights using `extract_attention.py` and:
1. Compute feature importance scores
2. Save raw attention arrays for detailed analysis
3. Feed into Phase 8's explainability layer

## Backtesting & Evaluation

### Baseline Models

We compare TFT against three naive baselines:

1. **Historical mean:** Use mean of historical values for each region
2. **Persistence:** Use last observed value (naive "no change" forecast)
3. **Linear trend:** Fit simple linear trend to historical data

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Primary metric for forecast accuracy
- **RMSE (Root Mean Squared Error):** Penalizes large errors more heavily
- **Bias:** Mean prediction error (positive = overprediction)
- **MAPE (Mean Absolute Percentage Error):** Relative error when non-zero

### Confidence Intervals

We use bootstrap resampling (n=1000) to compute 95% confidence intervals for all metrics, explicitly showing the uncertainty due to limited sample size.

### Honest Reporting

The backtest report explicitly includes:

1. **Performance comparison:** TFT vs. baselines with confidence intervals
2. **Limitations section:** Acknowledges sample size constraints
3. **Recommendations:** How to interpret and use the results appropriately

## Known Limitations & Sample Size Constraints

### Limited Historical Data

**Problem:** Only 74 years of data (1950-2023) with ~10 regions = ~740 independent samples

**Impact:**
- Deep learning models are prone to overfitting
- High uncertainty in all forecasts
- Wide confidence intervals
- Risk of learning spurious correlations

**Mitigation:**
- Conservative architecture (reduced capacity)
- Aggressive regularization (dropout, gradient clipping)
- Early stopping (prevent overfitting)
- Ensemble approaches (future work)

### Climate Covariate Availability

**Problem:** ERA5 SST data has limited coverage (only 14 of 74 years in original dataset)

**Impact:**
- Missing climate values require interpolation/forward-fill
- Potential bias from filled values
- Reduced effective sample size for climate relationships

**Mitigation:**
- Incremental ERA5 backfill (ongoing)
- Interpolation with explicit logging of filled years
- Consider external SST datasets (NOAA ERSST) for full coverage

### Temporal Autocorrelation

**Problem:** Hurricane activity shows strong temporal autocorrelation (active vs. quiet periods)

**Impact:**
- Risk of overfitting to period-specific patterns
- Difficulty distinguishing climate trends from natural variability
- Challenges for out-of-sample generalization

**Mitigation:**
- Long encoder/predictor windows (10/5 years)
- Stratified temporal split by activity level
- Focus on regional aggregates vs. individual locations

## Expected Performance

Given the sample size limitations, we expect:

1. **TFT may not outperform simple baselines** — The signal-to-noise ratio may be too low for deep learning advantages to manifest
2. **High uncertainty** — Confidence intervals will be wide, reflecting genuine uncertainty
3. **Period-specific performance** — Model may perform better in certain climate regimes (e.g., active Atlantic periods)
4. **Regional variability** — Some regions may have more learnable patterns than others

**Success criteria:** TFT is competitive with baselines (not necessarily better) and provides interpretable attention weights that align with domain knowledge (e.g., SST attention for regions where SST is known to influence hurricane activity).

## Integration with Downstream Phases

### Phase 6 (GNN)

TFT trend forecasts provide regional risk trends that can inform:
- Spatial correlation modeling in GNN
- Portfolio accumulation risk assessment
- Climate-conditioned spatial dependency structures

### Phase 8 (Monte Carlo Engine)

TFT outputs feed into the Monte Carlo simulation as:
- Trend adjustment factors for frequency/severity
- Scenario-based climate conditioning (e.g., warming scenarios)
- Uncertainty quantification via prediction intervals

### Phase 9 (Multi-agent LLM)

Attention weights and feature importance provide:
- Explainable AI inputs for LLM reasoning
- Natural language explanations of model decisions
- Scenario analysis capabilities

## Reproducibility

- **Dataset:** `ml/tft_climate_trend/prepare_tft_dataset.py`
- **Training:** `ml/tft_climate_trend/train_tft.py`
- **Attention extraction:** `ml/tft_climate_trend/extract_attention.py`
- **Backtesting:** `ml/tft_climate_trend/backtest_tft.py`
- **MLflow:** All runs logged with parameters, metrics, and artifacts
- **Random seeds:** Fixed (seed=42) for reproducibility
- **Temporal split:** Stratified by hurricane activity, no year leakage

## Usage

### Train TFT model
```bash
python -m ml.tft_climate_trend.train_tft
```

### Extract attention weights
```bash
python -m ml.tft_climate_trend.extract_attention
```

### Run backtest vs baselines
```bash
python -m ml.tft_climate_trend.backtest_tft
```

### Full pipeline
```bash
python -m ml.tft_climate_trend.prepare_tft_dataset
python -m ml.tft_climate_trend.train_tft
python -m ml.tft_climate_trend.extract_attention
python -m ml.tft_climate_trend.backtest_tft
```

## Future Improvements

1. **Data augmentation:** Synthesize additional samples via bootstrapping or climate model outputs
2. **Transfer learning:** Pre-train on larger climate datasets, fine-tune on hurricane data
3. **Ensemble methods:** Combine TFT with baselines for robustness
4. **Climate forecasts:** Integrate with dedicated climate prediction models for future covariates
5. **Alternative architectures:** Consider simpler time-series models (Prophet, N-BEATS) that may handle small samples better

## Conclusion

The TFT implementation represents a cautious, regularization-heavy approach to climate trend forecasting given extreme sample size constraints. The primary value is not necessarily outperforming simple baselines, but rather:

1. **Providing a framework** for climate-conditioned risk modeling
2. **Enabling explainability** through attention mechanisms
3. **Supporting scenario analysis** for climate change impacts
4. **Feeding uncertainty quantification** into downstream Monte Carlo simulations

Honest reporting of limitations and uncertainty is central to this approach — the model should be used for directional insights and trend analysis, not precise point predictions.
