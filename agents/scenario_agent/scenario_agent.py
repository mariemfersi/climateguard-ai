"""
Scenario Agent for natural-language what-if stress test parsing & live Monte Carlo re-runs.

Translates natural-language climate/economic what-if requests (e.g., "what if Cat 5 hurricane
landfall frequency rises by 20% and inflation is 8%?") into structured parameters,
triggers live Monte Carlo re-runs via the simulation engine, and narrates the stressed vs baseline loss metrics.

Usage:
    python -m agents.scenario_agent.scenario_agent [--query "What if hurricane frequency rises by 25%?"]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any

import pandas as pd

from ml.monte_carlo_engine.simulate_portfolio_losses import (
    PORTFOLIO_DATA_PATH,
    CLAIMS_DATA_PATH,
    MonteCarloLossSimulator,
)

logger = logging.getLogger(__name__)


class ScenarioAgent:
    """
    Agent responsible for natural-language what-if scenario parsing and live Monte Carlo simulation execution.
    """

    def __init__(self):
        logger.info("Initialized ScenarioAgent.")

    def parse_what_if_query(self, query: str) -> dict[str, Any]:
        """
        Parse natural-language what-if query into structured parameter overrides.

        Args:
            query: Prompt text containing scenario conditions.

        Returns:
            Dictionary with parsed parameters: trend_multiplier, severity_multiplier, frequency_multiplier.
        """
        query_lower = query.lower()

        trend_mult = 1.0
        sev_mult = 1.0
        freq_mult = 1.0

        # Try LLM extraction if keys exist
        openai_key = os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                response = openai.ChatCompletion.create(
                    model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a reinsurance scenario parser. Extract climate/economic overrides from prompt. "
                                "Return JSON with keys: 'trend_multiplier' (float, default 1.0), "
                                "'severity_multiplier' (float, default 1.0), 'frequency_multiplier' (float, default 1.0)."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.0,
                )
                parsed_json = json.loads(response.choices[0].message.content)
                logger.info("LLM parsed what-if parameters: %s", parsed_json)
                return {
                    "trend_multiplier": float(parsed_json.get("trend_multiplier", 1.0)),
                    "severity_multiplier": float(parsed_json.get("severity_multiplier", 1.0)),
                    "frequency_multiplier": float(parsed_json.get("frequency_multiplier", 1.0)),
                }
            except Exception as e:
                logger.warning("LLM what-if parsing failed (%s), using regex parser.", e)

        # Regex heuristic pattern matching
        # Frequency % increase e.g. "frequency rises 20%", "20% more frequent"
        match_freq = re.search(r"(\d+(?:\.\d+)?)\%\s*(?:increase|rises?|higher|more\s+frequent)", query_lower)
        if match_freq:
            pct = float(match_freq.group(1))
            freq_mult = 1.0 + (pct / 100.0)
            trend_mult = freq_mult

        # Severity / Inflation % increase e.g. "inflation is 10%", "10% inflation", "severity rises 15%"
        match_sev = re.search(r"(?:inflation|severity|cost)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\%|(\d+(?:\.\d+)?)\%\s*(?:severity|inflation|cost|more\s+severe)", query_lower)
        if match_sev:
            pct_str = match_sev.group(1) or match_sev.group(2)
            if pct_str:
                pct = float(pct_str)
                sev_mult = 1.0 + (pct / 100.0)

        # Category 5 landfall frequency keywords
        if "cat 5" in query_lower or "category 5" in query_lower:
            freq_mult = max(freq_mult, 1.20)
            trend_mult = max(trend_mult, 1.20)

        params = {
            "trend_multiplier": trend_mult,
            "severity_multiplier": sev_mult,
            "frequency_multiplier": freq_mult,
        }
        logger.info("Regex parsed what-if parameters: %s", params)
        return params

    def handle_query(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Parse natural language what-if prompt, trigger live Monte Carlo simulation re-run,
        and generate a narrated comparison report.
        """
        context = context or {}
        num_seasons = int(context.get("num_seasons", 500))
        seed = int(context.get("seed", 42))

        params = self.parse_what_if_query(query)

        portfolio_data = pd.read_parquet(PORTFOLIO_DATA_PATH)
        claims_data = pd.read_parquet(CLAIMS_DATA_PATH)

        # Baseline run
        sim_base = MonteCarloLossSimulator(portfolio_data, claims_data, num_seasons=num_seasons, seed=seed)
        base_res = sim_base.simulate(trend_multiplier=1.0)

        # Stressed live re-run
        sim_stress = MonteCarloLossSimulator(portfolio_data, claims_data, num_seasons=num_seasons, seed=seed)
        stress_res = sim_stress.simulate(trend_multiplier=params["trend_multiplier"])

        # Apply severity multiplier if present
        if params["severity_multiplier"] != 1.0:
            sm = params["severity_multiplier"]
            stress_res["mean_annual_loss"] *= sm
            stress_res["var_995"] *= sm
            stress_res["tvar_995"] *= sm
            stress_res["expected_loss_ratio"] *= sm

        # Delta metrics
        delta_mean = stress_res["mean_annual_loss"] - base_res["mean_annual_loss"]
        pct_mean = (delta_mean / base_res["mean_annual_loss"]) * 100.0 if base_res["mean_annual_loss"] > 0 else 0

        delta_var = stress_res["var_995"] - base_res["var_995"]
        pct_var = (delta_var / base_res["var_995"]) * 100.0 if base_res["var_995"] > 0 else 0

        delta_tvar = stress_res["tvar_995"] - base_res["tvar_995"]
        pct_tvar = (delta_tvar / base_res["tvar_995"]) * 100.0 if base_res["tvar_995"] > 0 else 0

        narration = f"""# What-If Stress Scenario Live Simulation Results

**Scenario Prompt:** "{query}"
**Extracted Parameters:** Frequency Multiplier: {params['frequency_multiplier']:.2f}x, Severity Multiplier: {params['severity_multiplier']:.2f}x

## 1. Executive Summary & Impact Analysis
Under the requested stress conditions, expected annual portfolio losses increase by **${delta_mean:,.0f} (+{pct_mean:.1f}%)**, bringing the 1-in-200 year Solvency II SCR VaR requirement from **${base_res['var_995']:,.0f}** to **${stress_res['var_995']:,.0f} (+{pct_var:.1f}%)**.

## 2. Comparative Risk Metrics Table
| Metric | Baseline Portfolio | Stressed Scenario | Absolute Delta | Percentage Shift |
|---|---|---|---|---|
| **Mean Annual Loss** | ${base_res['mean_annual_loss']:,.0f} | ${stress_res['mean_annual_loss']:,.0f} | +${delta_mean:,.0f} | +{pct_mean:.1f}% |
| **99.5% VaR (1-in-200yr)** | ${base_res['var_995']:,.0f} | ${stress_res['var_995']:,.0f} | +${delta_var:,.0f} | +{pct_var:.1f}% |
| **99.5% TVaR (Tail Expected Shortfall)** | ${base_res['tvar_995']:,.0f} | ${stress_res['tvar_995']:,.0f} | +${delta_tvar:,.0f} | +{pct_tvar:.1f}% |
| **Expected Loss Ratio** | {base_res['expected_loss_ratio']:.4f} | {stress_res['expected_loss_ratio']:.4f} | +{(stress_res['expected_loss_ratio'] - base_res['expected_loss_ratio']):.4f} | +{((stress_res['expected_loss_ratio'] - base_res['expected_loss_ratio'])/base_res['expected_loss_ratio'])*100:.1f}% |

*This live simulation was re-run across {num_seasons:,} synthetic storm seasons by ClimateGuard AI Scenario Agent.*
"""

        return {
            "intent": "scenario",
            "response": narration,
            "parsed_params": params,
            "baseline_metrics": base_res,
            "stressed_metrics": stress_res,
            "deltas": {
                "mean_annual_loss": delta_mean,
                "var_995": delta_var,
                "tvar_995": delta_tvar,
            },
            "citations": [],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scenario Agent")
    parser.add_argument("--query", type=str, default="What happens if Category 5 landfall frequency rises 20% and construction inflation is 8%?", help="What-if prompt")
    args = parser.parse_args()

    agent = ScenarioAgent()
    res = agent.handle_query(args.query)

    print("\n" + "=" * 70)
    print("SCENARIO AGENT OUTPUT")
    print("=" * 70)
    print(res["response"])
    print("=" * 70)
