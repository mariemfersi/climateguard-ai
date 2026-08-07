"""
Unit and integration tests for Phase 8: Monte Carlo Loss Engine & Explainability Layer.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.monte_carlo_engine.simulate_portfolio_losses import (
    MonteCarloLossSimulator,
    load_frequency_baseline,
    load_severity_baseline,
    run_monte_carlo_simulation,
)
from ml.monte_carlo_engine.stress_scenarios import (
    StressScenarioLibrary,
    run_stress_analysis,
)
from ml.explainability.shap_explainer import (
    GOLD_FEATURES_PATH,
    SHAPExplainer,
    run_shap_analysis,
    train_quick_explainable_model,
)
from ml.explainability.counterfactual_explainer import (
    CounterfactualExplainer,
    run_counterfactual_analysis,
)


@pytest.fixture
def dummy_portfolio_and_claims():
    """Create lightweight dummy portfolio and claims for unit tests."""
    locations = []
    for i in range(1, 101):
        locations.append({
            "location_id": f"LOC{i:07d}",
            "lat": 26.0 + (i % 30) * 0.1,
            "lon": -81.0 - (i % 30) * 0.1,
            "tiv_usd": 250000.0 + (i * 1000),
            "distance_to_coast_km": 5.0 + (i * 2.0),
            "year_built": 1980 + (i % 40),
            "construction_class": "frame" if i % 2 == 0 else "masonry_cbs",
            "roof_type": "flat" if i % 3 == 0 else "hip",
            "mslp_hpa_mean": 1015.0,
            "era5_wind_speed_ms_mean": 3.0,
            "era5_wind_speed_ms_max": 20.0,
        })
    gold_df = pd.DataFrame(locations)

    claims = []
    for i in range(1, 51):
        claims.append({
            "claim_id": f"CLM{i:08d}",
            "location_id": f"LOC{(i * 2):07d}",
            "loss_date": f"{1990 + (i % 30)}-08-15",
            "damage_ratio": 0.05 + (i % 10) * 0.03,
            "incurred_loss_usd": 15000.0 * (i % 5 + 1),
        })
    claims_df = pd.DataFrame(claims)

    return gold_df, claims_df


def test_monte_carlo_simulator_reproducibility(dummy_portfolio_and_claims):
    gold_df, claims_df = dummy_portfolio_and_claims

    sim1 = MonteCarloLossSimulator(gold_df, claims_df, num_seasons=100, seed=42)
    res1 = sim1.simulate(trend_multiplier=1.0)

    sim2 = MonteCarloLossSimulator(gold_df, claims_df, num_seasons=100, seed=42)
    res2 = sim2.simulate(trend_multiplier=1.0)

    assert np.allclose(res1["annual_losses"], res2["annual_losses"])
    assert res1["var_995"] == res2["var_995"]
    assert res1["tvar_995"] == res2["tvar_995"]


def test_var_tvar_mathematical_properties(dummy_portfolio_and_claims):
    gold_df, claims_df = dummy_portfolio_and_claims
    sim = MonteCarloLossSimulator(gold_df, claims_df, num_seasons=200, seed=123)
    res = sim.simulate(trend_multiplier=1.1)

    assert res["tvar_995"] >= res["var_995"], "TVaR must be >= VaR at the same quantile"
    assert res["mean_annual_loss"] > 0
    assert res["expected_loss_ratio"] > 0
    assert len(res["annual_losses"]) == 200


def test_stress_scenarios_amplification(dummy_portfolio_and_claims):
    gold_df, claims_df = dummy_portfolio_and_claims
    library = StressScenarioLibrary()

    base_res = library.apply_scenario(gold_df, claims_df, "baseline", num_seasons=100, seed=42)
    p200_res = library.apply_scenario(gold_df, claims_df, "p200_1x", num_seasons=100, seed=42)

    assert p200_res["mean_annual_loss"] > base_res["mean_annual_loss"]
    assert p200_res["var_995"] > base_res["var_995"]


def test_shap_explainer_instance_and_batch():
    gold_df = pd.read_parquet(GOLD_FEATURES_PATH) if GOLD_FEATURES_PATH.exists() else pd.DataFrame()
    model, _, feature_cols = train_quick_explainable_model()

    if "roof_type" in gold_df.columns:
        roof_dummies = pd.get_dummies(gold_df["roof_type"], prefix="roof", dtype=float)
        gold_df = pd.concat([gold_df, roof_dummies], axis=1)
    if "construction_class" in gold_df.columns:
        const_dummies = pd.get_dummies(gold_df["construction_class"], prefix="construction", dtype=float)
        gold_df = pd.concat([gold_df, const_dummies], axis=1)

    df_features = gold_df.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)
    X_sample = df_features.head(10).values
    loc_ids = gold_df["location_id"].head(10).tolist()

    explainer = SHAPExplainer(model, feature_cols)

    # Test single instance explanation
    single_res = explainer.explain_instance(X_sample[0], location_id=loc_ids[0])
    assert "prediction_id" in single_res
    assert len(single_res["top_drivers"]) <= 5
    assert len(single_res["attributions"]) == len(feature_cols)

    # Test batch explanation
    batch_df = explainer.explain_batch(X_sample[:10], loc_ids[:10])
    assert set(["prediction_id", "location_id", "feature_name", "shap_value", "rank"]).issubset(batch_df.columns)
    assert len(batch_df) == 10 * len(feature_cols)


def test_counterfactual_explainer(dummy_portfolio_and_claims):
    gold_df, _ = dummy_portfolio_and_claims
    explainer = CounterfactualExplainer()

    loc_row = gold_df.iloc[0]
    res = explainer.generate_location_counterfactuals(loc_row)

    assert res["location_id"] == loc_row["location_id"]
    assert len(res["counterfactual_scenarios"]) > 0

    scenarios = res["counterfactual_scenarios"]
    for sc in scenarios:
        assert sc["annual_loss_reduction_usd"] >= 0
        assert 0 < sc["pct_loss_reduction"] <= 100
        assert isinstance(sc["recommendation"], str)


def test_end_to_end_phase8_pipelines():
    # Test end to end execution with small season/sample counts
    sim_res = run_monte_carlo_simulation(num_seasons=100, seed=42)
    assert sim_res["var_995"] > 0

    stress_res = run_stress_analysis(scenario_ids=["baseline", "climate_change_1c"], num_seasons=100, seed=42)
    assert len(stress_res) == 2

    shap_df = run_shap_analysis(num_samples=20)
    assert len(shap_df) > 0

    cf_df = run_counterfactual_analysis(num_samples=10)
    assert len(cf_df) > 0
