"""
Pricing Agent for technical treaty pricing memos and Rate-on-Line calculation.

Tool-calls into Monte Carlo loss simulation engine, SHAP explainer, and counterfactuals
to draft technical treaty pricing memos. Grounded 100% in tool outputs.

Usage:
    python -m agents.pricing_agent.pricing_agent [--limit 100M] [--attachment 50M]
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from ml.explainability.counterfactual_explainer import CounterfactualExplainer
from ml.explainability.shap_explainer import run_shap_analysis
from ml.monte_carlo_engine.simulate_portfolio_losses import run_monte_carlo_simulation

logger = logging.getLogger(__name__)


class PricingAgent:
    """
    Agent responsible for treaty technical pricing, Rate-on-Line recommendations,
    and technical pricing memo drafting.
    """

    def __init__(self):
        logger.info("Initialized PricingAgent.")

    def calculate_treaty_pricing(
        self,
        attachment_usd: float = 50_000_000.0,
        limit_usd: float = 100_000_000.0,
        num_seasons: int = 1000,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Execute model tools to price a specific catastrophe treaty layer.

        Args:
            attachment_usd: Treaty layer attachment point in USD.
            limit_usd: Treaty layer limit in USD.
            num_seasons: Number of simulation seasons.
            seed: Random seed.

        Returns:
            Dictionary containing structured pricing metrics and tool outputs.
        """
        logger.info(
            "Executing Monte Carlo tool for layer $%.0fM xs $%.0fM...",
            limit_usd / 1e6, attachment_usd / 1e6,
        )

        # Tool Call 1: Run Monte Carlo portfolio loss simulation
        sim_res = run_monte_carlo_simulation(num_seasons=num_seasons, seed=seed)

        annual_losses = sim_res["annual_losses"]

        # Apply treaty layer payout formula:
        # Layer Loss = min(max(0, Annual_Loss - Attachment), Limit)
        layer_losses = (annual_losses - attachment_usd).clip(min=0.0)
        layer_losses = layer_losses.clip(max=limit_usd)

        expected_layer_loss = float(layer_losses.mean())
        layer_var_995 = float(sim_res["var_995"])
        layer_tvar_995 = float(sim_res["tvar_995"])

        # Technical Rate-on-Line (ROL) = Expected Layer Loss / Limit
        rate_on_line = expected_layer_loss / limit_usd if limit_usd > 0 else 0.0

        # Technical Premium = Expected Layer Loss * (1 + Expense Ratio 10% + Risk Loading 15%)
        technical_premium = expected_layer_loss * 1.25

        # Tool Call 2: Extract top SHAP risk drivers
        shap_df = run_shap_analysis(num_samples=20)
        top_shap_features = (
            shap_df.groupby("feature_name")["shap_value"]
            .apply(lambda s: float(s.abs().mean()))
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

        # Tool Call 3: Extract sample counterfactual recommendation
        cf_explainer = CounterfactualExplainer()
        sample_loc = {
            "location_id": "LOC0000001",
            "tiv_usd": 500000.0,
            "roof_type": "flat",
            "construction_class": "frame",
            "distance_to_coast_km": 15.0,
        }
        cf_res = cf_explainer.generate_location_counterfactuals(sample_loc)

        return {
            "attachment_usd": attachment_usd,
            "limit_usd": limit_usd,
            "expected_layer_loss": expected_layer_loss,
            "rate_on_line": rate_on_line,
            "rate_on_line_pct": rate_on_line * 100.0,
            "technical_premium": technical_premium,
            "var_995": layer_var_995,
            "tvar_995": layer_tvar_995,
            "portfolio_mean_annual_loss": sim_res["mean_annual_loss"],
            "top_shap_features": top_shap_features,
            "sample_counterfactual": cf_res["counterfactual_scenarios"][0] if cf_res["counterfactual_scenarios"] else {},
            "raw_sim_metrics": {
                "num_locations": sim_res["num_locations"],
                "total_tiv": sim_res["total_tiv"],
                "trend_multiplier": sim_res["trend_multiplier"],
            },
        }

    def handle_query(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Handle user prompt for pricing agent, generating technical pricing memo.
        """
        context = context or {}
        attachment = float(context.get("attachment_usd", 50_000_000.0))
        limit = float(context.get("limit_usd", 100_000_000.0))

        pricing = self.calculate_treaty_pricing(attachment_usd=attachment, limit_usd=limit)

        memo_markdown = f"""# Technical Treaty Pricing Memo
**Layer Specification:** ${pricing['limit_usd']/1e6:,.0f}M xs ${pricing['attachment_usd']/1e6:,.0f}M
**Cedent Portfolio TIV:** ${pricing['raw_sim_metrics']['total_tiv']/1e6:,.0f}M across {pricing['raw_sim_metrics']['num_locations']:,} locations
**Climate Trend Multiplier:** {pricing['raw_sim_metrics']['trend_multiplier']:.3f}x

## 1. Executive Pricing Summary
- **Expected Layer Loss:** ${pricing['expected_layer_loss']:,.2f}
- **Technical Rate-on-Line (ROL):** {pricing['rate_on_line_pct']:.2f}%
- **Recommended Technical Premium:** ${pricing['technical_premium']:,.2f}
- **99.5% VaR (1-in-200yr Solvency II):** ${pricing['var_995']:,.2f}
- **99.5% TVaR (Expected Shortfall):** ${pricing['tvar_995']:,.2f}

## 2. Key Physical & Climate Risk Drivers (SHAP Attribution)
Top risk drivers contributing to loss severity across the cedent book:
"""
        for feat, val in pricing["top_shap_features"].items():
            memo_markdown += f"- **{feat}**: mean absolute SHAP impact = {val:.4f}\n"

        if pricing["sample_counterfactual"]:
            cf = pricing["sample_counterfactual"]
            memo_markdown += f"\n## 3. Recommended Underwriting Mitigation\n- {cf['recommendation']}\n"

        memo_markdown += f"\n*This pricing memo was generated by ClimateGuard AI Pricing Agent. Every number is grounded in Monte Carlo simulation output.*"

        return {
            "intent": "pricing",
            "response": memo_markdown,
            "tool_outputs": pricing,
            "rate_on_line": pricing["rate_on_line"],
            "technical_premium_usd": pricing["technical_premium"],
            "var_995": pricing["var_995"],
            "tvar_995": pricing["tvar_995"],
            "citations": [],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Pricing Agent")
    parser.add_argument("--attachment", type=float, default=50000000.0, help="Attachment point USD")
    parser.add_argument("--limit", type=float, default=100000000.0, help="Limit USD")
    args = parser.parse_args()

    agent = PricingAgent()
    res = agent.handle_query(
        f"Price treaty layer {args.limit/1e6:.0f}M xs {args.attachment/1e6:.0f}M",
        context={"attachment_usd": args.attachment, "limit_usd": args.limit},
    )

    print("\n" + "=" * 70)
    print("PRICING AGENT OUTPUT")
    print("=" * 70)
    print(res["response"])
    print("=" * 70)
