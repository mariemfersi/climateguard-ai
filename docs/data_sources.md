# Data Sources and Coverage

## Overview

This document documents the data sources used in the ClimateGuard AI project, their coverage, and known limitations.

## ERA5 Reanalysis Data

### Source
- **Provider**: ECMWF (European Centre for Medium-Range Weather Forecasts)
- **Product**: ERA5 reanalysis
- **Path**: `data_pipeline/bronze/era5/gulf_coast_reanalysis.parquet`

### Coverage
- **Years with data**: 1955, 1961, 1969, 1975, 1983, 1989, 1995, 1998, 2001, 2004, 2005, 2008, 2012, 2023
- **Total years**: 14 of 74 (1950-2023 training range)
- **Coverage percentage**: ~19%
- **Grid**: Gulf Coast bounding box (101 lat × 161 lon grid cells)
- **Records**: 166,642,728 rows

### Variables
- `sst_celsius`: Sea surface temperature (ocean cells only)
- `mslp_hpa`: Mean sea-level pressure
- `wind_speed_ms`: Wind speed at 10m
- `lat`, `lon`: Grid coordinates
- `timestamp`: Timestamp

### Coverage Pattern Analysis
The 14 years show **uneven spacing**, indicating interrupted backfill rather than deliberate sampling:
- Early period (1955-1995): ~6-8 year gaps
- Recent period (1998-2012): 1-3 year gaps (denser - likely a successful backfill session)
- Final gap: 11 years (2012-2023 - likely a later one-off fetch)

**Assessment**: This is an **unfinished ingestion** (interrupted backfill), not a deliberate representative sample. The irregular spacing pattern suggests data was added in multiple sessions with different sampling strategies, consistent with known bronze-write MemoryErrors during Phase 1 ERA5 ingestion.

### Implications

**For Milestone 4.1 (Frequency-Severity):**
- Not a blocker - XGBoost/LightGBM handle NaN natively
- SST had near-zero importance even when constant
- Frequency/severity models can train on the full 74-year range with NaN SST for 60 years
- Results unchanged by SST coverage gap

**For Phase 5 (TFT):**
- **BLOCKER**: The TFT model requires time-varying climate covariates to learn SST-frequency trends
- 14 years is insufficient for robust trend detection
- Uneven spacing makes trend analysis unreliable

**Required action before Phase 5:**
Complete ERA5 backfill for the remaining 60 years (1950-2023 range) to provide full temporal resolution for TFT trend learning. This is a Phase 1/ingestion-layer task that should be completed as part of Phase 5 Milestone 5.1 (data preparation) before TFT dataset construction.

## HURDAT2 Storm Tracks

### Source
- **Provider**: NOAA National Hurricane Center
- **Product**: HURDAT2 Atlantic basin hurricane database
- **Path**: `data_pipeline/bronze/hurdat2/tracks.parquet`

### Coverage
- **Years**: 1851-present (full historical record)
- **Basin**: Atlantic/Gulf of Mexico
- **Storm types**: Tropical cyclones, hurricanes

### Variables
- Storm identification (storm_id, name, year)
- Track positions (lat, lon, timestamp)
- Intensity (max_wind_kt, min_pressure_mb)
- Storm status

## Synthetic Claims Data

### Source
- **Generator**: Phase 2 claims generation pipeline
- **Path**: `data_pipeline/silver/claims.parquet`

### Coverage
- **Years**: 1950-2023 (74 years)
- **Locations**: 20,000 locations
- **Total claims**: ~250,000 claim events

### Generation Method
- Uses HURDAT2 storm tracks + vulnerability curves
- Generates claims based on wind speed exposure
- See `docs/synthetic_data_methodology.md` for details

## Silver Locations

### Source
- **Path**: `data_pipeline/silver/locations.parquet`

### Coverage
- **Locations**: 20,000 locations
- **Geography**: Gulf Coast region
- **Attributes**: Construction class, roof type, TIV, year built

## Gold Features

### Source
- **Path**: `data_pipeline/gold/gold_features.parquet`

### Coverage
- **Grain**: One row per location (20,000 rows)
- **Static features**: Geographic, exposure, long-term climate averages
- **Note**: SST is time-varying and joined separately at (location, year) grain

### Yearly SST
- **Path**: `data_pipeline/gold/yearly_sst.parquet`
- **Grain**: One row per year (14 years)
- **Coverage**: Same 14 years as ERA5 bronze data

## Known Gaps and Limitations

1. **ERA5 SST coverage**: Only 14 of 74 years (see above)
2. **SST spatial resolution**: Basin-wide average, not per-location
3. **Claims generation**: Synthetic data, not real insurance claims
4. **Geographic scope**: Limited to Gulf Coast region
