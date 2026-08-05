# Frequency-Severity Modeling — Milestone 4.1 Results & Methodology

## Summary

This milestone trained baseline frequency (claim occurrence) and severity
(damage given a claim) models using only features knowable at pricing time
— i.e., before a given hurricane season's storms exist. The central,
intentional finding is:

- **Frequency (per-location, per-year claim occurrence): AUC ~0.48–0.59**,
  consistent and stable across two independent train/test split strategies
  (strict temporal 1950–2010/2011–2023, and stratified-by-activity year
  split). This is close to random and should be read as a genuine empirical
  result, not a bug.
- **Severity (damage ratio given a claim occurred): R² ranges from -0.04 to
  0.39** depending on split, with weak but real, non-leaky signal —
  `distance_to_coast_km` shows a consistent, physically sensible negative
  correlation with damage severity (-0.09).

## Why frequency AUC is near-random, and why that's expected

Hurricane landfall at a specific location in a specific year is driven
almost entirely by storm-track realization — a process that is close to
exogenous with respect to any single property's static characteristics
(construction class, TIV, year built). A location's long-run *hazard rate*
is a property of its region and coastline exposure, observable only over
many years/many storms — not something a per-year classifier can recover
from static features alone, and not something real cat models attempt
either. This is the standard actuarial framing: frequency-severity models
exist to estimate a rate, which is then fed into a simulated loss
distribution (Monte Carlo, Phase 8) — not to classify whether any single
future year will produce a claim at any single location.

## Known Limitations for Phase 5 (TFT)

**SST time-varying covariate (FIXED):** The original Gold table applied
`basin_sst_celsius_mean` and `basin_sst_celsius_max` uniformly across all
locations and all years (single value: 28.1°C mean, 35.9°C max). This was a
bug that would have blocked Phase 5's TFT model, which requires time-varying
climate covariates to capture SST-driven hurricane frequency trends.

**Root cause:** The `compute_basin_wide_sst_summary()` function in
`aggregate_climate_features.py` was computing a single basin-wide average
across all years instead of yearly summaries. This caused pandas `.corr()`
to return NaN for SST columns (correlation undefined with zero variance).

**Fix:** 
1. Updated `compute_basin_wide_sst_summary()` to return yearly SST
   summaries (14 years: 1955, 1961, 1969, 1975, 1983, 1989, 1995, 1998, 2001,
   2004, 2005, 2008, 2012, 2023) with mean range 27.73-28.95°C and max range
   33.15-35.95°C
2. Saved yearly SST to `yearly_sst.parquet` 
3. Joined SST at the correct (location_id, year) grain in `build_training_table.py`
   (not in the static Gold table)
4. Added SST columns back to `FEATURE_COLUMNS` for model input

**Verification:** Training table now shows 14 unique SST values (was 1 before),
with 280,000 non-null rows (14 years × 20,000 locations), confirming genuine
interannual variability.

**Coverage note:** Only 14 of 74 years have ERA5 SST coverage. The remaining 60
years will have NaN SST values in the training table. XGBoost/LightGBM handle
NaN natively, so this won't crash models. For Phase 5 TFT, consider either:
- Restricting training to the 14 years with SST coverage
- Interpolating/forward-filling SST values
- Using external SST datasets with full 74-year coverage (e.g., NOAA ERSST)

**Status:** Resolved. SST now provides genuine interannual variability for
Phase 5's TFT model to learn SST-frequency relationships.

## SST Overfitting Failure Mode (Frequency-Severity Models)

**Issue:** An attempt to add basin SST as a frequency feature was reverted after
discovering that 19% year coverage (14 of 74 years) caused severe overfitting
to spurious per-year artifacts.

**Failure details:**
- Train AUC: 0.70 (highest ever observed)
- Test AUC: 0.34 (below random, catastrophic collapse)
- SST feature importance: basin_sst_celsius_max = 921 (dominant, 2x next feature)
- Train-test AUC gap: 0.36 (severe overfitting)

**Root cause:** The stratified year split placed nearly all 14 SST-covered years
into train (1955, 1961, 1969, 1975, 1983, 1989, 1995, 1998, 2001, 2004, 2005,
2008, 2012, 2023) while test had mostly NaN SST values (only 2001 and 2004
overlap). With only 14 real SST values, XGBoost learned spurious per-year
quirks disguised as an "SST relationship" — effectively memorizing which
specific years correlated with claims rather than a genuine climate signal.

**Resolution:** SST features (`basin_sst_celsius_mean`, `basin_sst_celsius_max`)
were removed from the frequency-severity model feature set. The yearly SST
merge remains in `build_training_table.py` for future use after full ERA5
backfill, but SST is excluded from model input.

**Lesson:** Insufficient temporal coverage (19% years) for a time-varying
covariate can cause catastrophic overfitting that inverts model performance.
SST will be reintroduced only after completing the full 74-year ERA5 backfill
(Phase 5 Milestone 5.1 prerequisite).

**Expected impact:** Removing SST should restore AUC to the established baseline
range of 0.48-0.59 with legitimate features.

## What was ruled out before reaching this conclusion

1. **Split-induced base-rate mismatch.** An initial 59/15 year-level
   random split happened to concentrate several of the most extreme
   hurricane seasons on record (1992, 2004, 2005, 2017) into the test
   fold, producing a 12.4% vs 34.1% train/test base-rate mismatch and an
   artificially depressed AUC. Fixed by moving to `stratified_year_split`
   (stratifies years by activity tercile before sampling). After the fix,
   train/test base rates converge to 15–17% on both sides, and AUC
   settles into the stable ~0.48–0.59 band reported above — confirming
   the near-random result is not a split artifact.

2. **Target leakage via same-year storm-event features.** An alternate
   feature set (`event_features.parquet`, containing `days_above_64kt`,
   `min_distance_to_track_km`, `max_wind_nearby_kt`, etc. — all computed
   from *that year's actual storms*) produced AUC ~0.92. This was
   correctly rejected: those features are derived from the same
   deterministic wind-field → vulnerability-curve pipeline used to
   generate the claims themselves, so the model was reverse-engineering
   the claims generator, not learning transferable risk signal. A real
   pricing model cannot condition on a future storm's realized wind
   speed, since at pricing time the storm hasn't happened. The
   near-perfect train AUC (0.998+) on that feature set is itself
   diagnostic of this leakage.

## Severity findings

`distance_to_coast_km` correlates at -0.092 with `max_damage_ratio` and
-0.030 with `incurred_loss_usd` among claim-positive rows — directionally
correct (closer to coast → worse damage) and consistent with hurricane
wind-decay physics. This is retained as a legitimate severity feature.
Severity R² is more split-sensitive than frequency (ranging -0.04 to 0.39)
due to the much smaller positive-claim sample size (~200K rows) being
concentrated in a handful of high-loss storm-years; this is a known
limitation expected to stabilize as later phases (TFT trend-adjustment,
Monte Carlo aggregation) add structure the two-part GBM baseline can't
capture alone.

## Implication for downstream architecture

This milestone's near-random frequency AUC is the empirical motivation for
not stopping at a point-classification model:

- **Phase 5 (TFT):** climate-conditioned regional/trend-level frequency
  forecasting operates at a level (multi-year, regional) where signal is
  expected to actually exist, unlike per-location-per-year classification.
- **Phase 6 (GNN):** portfolio accumulation risk is a spatial-correlation
  problem, not a per-location independent-prediction problem.
- **Phase 8 (Monte Carlo engine):** the correct way to use a near-random
  per-location frequency model is as one input to a simulated loss
  distribution, not as a standalone predictive tool.

## Reproducibility

- Feature set: `location_id, lat, lon, metro_center, distance_to_coast_km,
  year_built, construction_class, roof_type, tiv_usd, mslp_hpa_mean,
  mslp_hpa_min, era5_wind_speed_ms_mean, era5_wind_speed_ms_max,
  basin_sst_celsius_mean, basin_sst_celsius_max`
- Split: `stratified_year_split` (74 years, 59 train / 15 test, stratified
  by yearly claim-activity tercile)
- Models: XGBoost + CatBoost (frequency, blended), LightGBM with monotonic
  SST constraint (severity)
- MLflow run: `milestone_4_1_frequency_severity`