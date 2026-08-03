# ClimateGuard AI

> An agentic, explainable reinsurance pricing & portfolio-steering platform for
> climate-adjusted catastrophe risk.

**Status:** 🚧 Phase 2-3 — Data Pipeline & Infrastructure (Foundations Complete)

ClimateGuard AI transforms static catastrophe models into a continuously updating, AI-driven system that fuses physical climate data with actuarial science for property & casualty reinsurance.

## Current Capabilities

**Data Pipeline (Production-Ready):**
- Ingestion from NOAA HURDAT2 (hurricane tracks 1851-2023), ERA5 reanalysis climate data, NASA FIRMS wildfire risk, and FRED economic indicators
- Synthetic Florida book of business: 20,000 locations with realistic population-weighted distribution and construction attributes
- Claims generation using historical HURDAT2 catalog (892,079 claim records from 216 storms)
- Validated results: ~3.5% average annual expected loss, top loss-driving storms match historical reality
- Bronze/Silver/Gold medallion architecture with Great Expectations data quality validation
- Azure Data Lake Storage Gen2 integration with Bicep infrastructure-as-code

**Infrastructure:**
- Azure-native deployment with ADLS Gen2, Key Vault, and RBAC
- PySpark-based geo-joining and feature engineering
- Basin-wide SST regional covariate design (methodologically sound)
- Comprehensive test suite (14 test files) covering ingestion, synthetic data, and feature engineering

**Scaffolded Components:**
- ML models: Frequency-severity (XGBoost/LightGBM/CatBoost), Temporal Fusion Transformer, Graph Neural Network, Monte Carlo engine, Vision Transformer
- Multi-agent LLM system: Pricing, Regulatory, Scenario, and Report-Writer agents
- FastAPI serving layer with endpoint structure
- MLOps: MLflow tracking, GitHub Actions, monitoring framework
- Frontend: Power BI and React/Next.js directories

## Quickstart

```bash
git clone <this-repo>
cd climateguard-ai
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env   # fill in API keys as you reach the phases that need them
pytest
```

## Generate Synthetic Data

```bash
# Generate 20,000 synthetic locations and policies
python -m data_pipeline.synthetic.run_generation --n 20000

# Generate claims using historical hurricane catalog
python -m data_pipeline.synthetic.run_claims_generation

# Assemble Gold feature table
python -m data_pipeline.databricks_jobs.run_gold_assembly
```

## Project Structure

- `data_pipeline/` — ETL/ELT pipelines: ingestion connectors, synthetic data generation, Databricks feature-engineering jobs, data-quality validation
- `ml/` — Model training code: frequency-severity, TFT climate trends, GNN accumulation, Monte Carlo engine, ViT exposure verification, explainability
- `agents/` — Multi-agent LLM system: pricing, regulatory, scenario, and report-writer agents with orchestrator
- `infra/` — Infrastructure as Code (Bicep) for Azure resources: ADLS Gen2, Key Vault, Databricks, Azure ML, AKS, API Management
- `mlops/` — MLOps/LLMOps tooling: MLflow tracking, CI/CD workflows, drift/monitoring jobs
- `serving_api/` — FastAPI application exposing all model, simulation, and agent endpoints
- `frontend/` — React/Next.js web application and Power BI dashboards
- `tests/` — Unit, integration, and end-to-end tests
- `docs/` — Architecture, data sources, methodology, and guides

## Documentation

- `docs/synthetic_data_methodology.md` — synthetic exposure/claims methodology with validation results
- `docs/implementation_roadmap.md` — full implementation roadmap (referenced in design doc)
- Each module has its own README.md with detailed documentation

## Architecture Overview

ClimateGuard AI fuses physical climate data (satellite, reanalysis, weather-station) with exposure, claims, and financial-market data to produce a dynamically updating, climate-adjusted view of catastrophe risk. The system uses an ensemble of specialized ML models wrapped in a multi-agent LLM system that autonomously drafts treaty pricing memos, answers regulatory questions, runs natural-language "what-if" stress tests, and produces board-ready reports with full explainability.

See `ClimateGuard AI.txt` for the complete technical specification and architecture.
