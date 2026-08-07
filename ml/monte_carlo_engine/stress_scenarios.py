"""
Stress-test scenario library for portfolio catastrophe loss engine.

Provides predefined regulatory and climate stress scenarios (e.g. 1-in-100yr event,
1-in-200yr event, +1°C/+2°C warming, concentrated geographic exposure) and runs them
through the Monte Carlo simulation engine.

Usage:
    python -m ml.monte_carlo_engine.stress_scenarios [--scenario SCENARIO_ID] [--num-seasons N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.monte_carlo_engine.simulate_portfolio_losses import (
    DEFAULT_NUM_SEASONS,
    MonteCarloLossSimulator,
    PORTFOLIO_DATA_PATH,
    CLAIMS_DATA_PATH,
    save_results,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data_pipeline/bronze/monte_carlo")


class StressScenarioLibrary:
    """
    Predefined stress scenarios for climate risk and capital adequacy testing.
    """

    def __init__(self):
        self.scenarios = self._define_scenarios()
        logger.info("Initialized %d stress scenarios in library", len(self.scenarios))

    def _define_scenarios(self) -> dict[str, dict[str, Any]]:
        return {
            "baseline": {
                "name": "Baseline (No Stress)",
                "description": "Current climate baseline without additional trend or shock.",
                "frequency_multiplier": 1.0,
                "severity_multiplier": 1.0,
                "trend_multiplier": 1.0,
            },
            "p100_1x": {
                "name": "1-in-100 Year Climate Stress",
                "description": "Extreme annual season with 1.25x frequency and 1.35x severity loading.",
                "frequency_multiplier": 1.25,
                "severity_multiplier": 1.35,
                "trend_multiplier": 1.20,
            },
            "p200_1x": {
                "name": "1-in-200 Year Extreme Solvency II Stress",
                "description": "Severe 1-in-200 year capital stress scenario with 1.50x frequency and 1.60x severity loading.",
                "frequency_multiplier": 1.50,
                "severity_multiplier": 1.60,
                "trend_multiplier": 1.45,
            },
            "climate_change_1c": {
                "name": "Climate Change +1°C Warming",
                "description": "Moderate climate warming (+1°C SST anomaly) leading to +20% peril frequency trend.",
                "frequency_multiplier": 1.20,
                "severity_multiplier": 1.15,
                "trend_multiplier": 1.20,
            },
            "climate_change_2c": {
                "name": "Climate Change +2°C Severe Warming",
                "description": "High-emissions warming scenario (+2°C SST anomaly) with +45% frequency and +30% severity shift.",
                "frequency_multiplier": 1.45,
                "severity_multiplier": 1.30,
                "trend_multiplier": 1.45,
            },
            "concentrated_coastal": {
                "name": "Severe Coastal Surge Concentration",
                "description": "Concentrated coastal impact event with +50% severity for near-coast assets.",
                "frequency_multiplier": 1.10,
                "severity_multiplier": 1.50,
                "trend_multiplier": 1.15,
            },
        }

    def get_scenario(self, scenario_id: str) -> dict[str, Any]:
        if scenario_id not in self.scenarios:
            raise ValueError(f"Unknown scenario_id: '{scenario_id}'. Available: {list(self.scenarios.keys())}")
        return self.scenarios[scenario_id]

    def list_scenarios(self) -> pd.DataFrame:
        rows = []
        for sid, params in self.scenarios.items():
            rows.append({
                "scenario_id": sid,
                "name": params["name"],
                "frequency_mult": params["frequency_multiplier"],
                "severity_mult": params["severity_multiplier"],
                "trend_mult": params["trend_multiplier"],
                "description": params["description"],
            })
        return pd.DataFrame(rows)

    def apply_scenario(
        self,
        portfolio_data: pd.DataFrame,
        claims_data: pd.DataFrame,
        scenario_id: str,
        num_seasons: int = DEFAULT_NUM_SEASONS,
        seed: int | None = 42,
    ) -> dict[str, Any]:
        scenario = self.get_scenario(scenario_id)
        logger.info("Applying stress scenario '%s': %s", scenario_id, scenario["name"])

        simulator = MonteCarloLossSimulator(
            portfolio_data=portfolio_data,
            claims_data=claims_data,
            num_seasons=num_seasons,
            seed=seed,
        )

        results = simulator.simulate(
            trend_multiplier=scenario["trend_multiplier"],
        )

        # Apply severity multiplier scaling if specified
        if scenario["severity_multiplier"] != 1.0:
            mult = scenario["severity_multiplier"]
            results["annual_losses"] = results["annual_losses"] * mult
            results["var_995"] *= mult
            results["tvar_995"] *= mult
            results["mean_annual_loss"] *= mult
            results["median_annual_loss"] *= mult
            results["std_annual_loss"] *= mult
            results["max_annual_loss"] *= mult
            results["expected_loss_ratio"] *= mult
            results["var_995_as_pct_tiv"] *= mult
            results["tvar_995_as_pct_tiv"] *= mult

        results["scenario_id"] = scenario_id
        results["scenario_name"] = scenario["name"]
        results["scenario_description"] = scenario["description"]
        results["severity_multiplier"] = scenario["severity_multiplier"]

        return results


def run_stress_analysis(
    scenario_ids: list[str] | None = None,
    num_seasons: int = DEFAULT_NUM_SEASONS,
    seed: int | None = 42,
) -> dict[str, dict[str, Any]]:
    logger.info("Loading portfolio and claims data for stress analysis...")
    portfolio_data = pd.read_parquet(PORTFOLIO_DATA_PATH)
    claims_data = pd.read_parquet(CLAIMS_DATA_PATH)

    library = StressScenarioLibrary()

    if scenario_ids is None:
        scenario_ids = list(library.scenarios.keys())

    all_results = {}
    comparison_rows = []

    for sid in scenario_ids:
        res = library.apply_scenario(
            portfolio_data=portfolio_data,
            claims_data=claims_data,
            scenario_id=sid,
            num_seasons=num_seasons,
            seed=seed,
        )
        all_results[sid] = res

        comparison_rows.append({
            "scenario_id": sid,
            "scenario_name": res["scenario_name"],
            "mean_annual_loss": res["mean_annual_loss"],
            "var_995": res["var_995"],
            "tvar_995": res["tvar_995"],
            "expected_loss_ratio": res["expected_loss_ratio"],
            "var_995_pct_tiv": res["var_995_as_pct_tiv"],
            "tvar_995_pct_tiv": res["tvar_995_as_pct_tiv"],
        })

    comparison_df = pd.DataFrame(comparison_rows)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_parquet(output_dir / "stress_scenario_comparison.parquet", index=False)

    summary_json = {
        sid: {
            "name": res["scenario_name"],
            "mean_annual_loss": res["mean_annual_loss"],
            "var_995": res["var_995"],
            "tvar_995": res["tvar_995"],
            "expected_loss_ratio": res["expected_loss_ratio"],
        }
        for sid, res in all_results.items()
    }
    with open(output_dir / "stress_scenario_comparison.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    logger.info("Stress scenario analysis completed and saved to %s", output_dir)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Catastrophe Loss Stress Scenario Analysis")
    parser.add_argument("--scenario", type=str, default=None, help="Specific scenario ID to run (default: run all)")
    parser.add_argument("--num-seasons", type=int, default=1000, help="Number of seasons to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    scenarios_to_run = [args.scenario] if args.scenario else None
    results = run_stress_analysis(scenario_ids=scenarios_to_run, num_seasons=args.num_seasons, seed=args.seed)

    print("\n" + "=" * 70)
    print("STRESS SCENARIO COMPARISON SUMMARY")
    print("=" * 70)
    for sid, res in results.items():
        print(f"[{sid}] {res['scenario_name']}")
        print(f"  Mean Loss: ${res['mean_annual_loss']:,.0f}")
        print(f"  VaR 99.5%: ${res['var_995']:,.0f} ({res['var_995_as_pct_tiv']:.2f}% TIV)")
        print(f"  TVaR 99.5%: ${res['tvar_995']:,.0f} ({res['tvar_995_as_pct_tiv']:.2f}% TIV)")
        print("-" * 70)
