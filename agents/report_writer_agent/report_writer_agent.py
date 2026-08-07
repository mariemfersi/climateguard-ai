"""
Task 9.2.5 — Report-Writer Agent + numeric-fidelity validation

Assembles a board-ready / SFCR-style narrative by orchestrating outputs
from the other agents (or accepting pre-computed tool results).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agents.base import AgentResponse, build_system_prompt
from agents.llm_client import get_llm
from agents.numeric_fidelity_validator import validate_numeric_fidelity
from agents.pricing_agent.pricing_agent import PricingAgent
from agents.regulatory_agent.regulatory_agent import RegulatoryAgent
from agents.scenario_agent.scenario_agent import ScenarioAgent

logger = logging.getLogger(__name__)


class ReportWriterAgent:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.pricing = PricingAgent()
        self.regulatory = RegulatoryAgent()
        self.scenario = ScenarioAgent()

    def run(self, user_message: str, context: Optional[Dict] = None) -> AgentResponse:
        context = context or {}

        pricing_resp = self.pricing.run(
            context.get("pricing_query")
            or "Provide a technical pricing view for the current Florida hurricane layer.",
            context=context,
        )
        regulatory_resp = self.regulatory.run(
            context.get("regulatory_query")
            or "Summarise the Solvency II SCR requirements relevant to non-life catastrophe risk and model governance.",
            context=context,
        )

        scenario_resp: Optional[AgentResponse] = None
        if any(w in user_message.lower() for w in ("what if", "what-if", "stress", "scenario", "increase")):
            scenario_resp = self.scenario.run(user_message, context=context)

        all_tool_outputs: List[Dict[str, Any]] = []
        all_tool_outputs.extend(pricing_resp.raw_tool_outputs)
        all_tool_outputs.extend(regulatory_resp.raw_tool_outputs)
        if scenario_resp:
            all_tool_outputs.extend(scenario_resp.raw_tool_outputs)

        system = build_system_prompt(
            "Report-Writer Agent",
            extra_rules=(
                "Produce a structured board-ready memo with these sections:\n"
                "1. Executive Summary\n"
                "2. Portfolio / Treaty Snapshot\n"
                "3. Technical Pricing View\n"
                "4. Key Risk Drivers (from explainability)\n"
                "5. Regulatory Considerations (Solvency II)\n"
                "6. Scenario / Stress Insight (if provided)\n"
                "7. Recommendation & Next Steps\n\n"
                "Every number must appear in the provided agent/tool material. "
                "Cite sources inline as [Pricing Model], [SHAP], [Solvency II – source file], etc."
            ),
        )

        material = {
            "pricing_memo": pricing_resp.text,
            "regulatory_answer": regulatory_resp.text,
            "scenario_narrative": scenario_resp.text if scenario_resp else None,
            "pricing_tool_outputs": pricing_resp.raw_tool_outputs,
            "regulatory_citations": regulatory_resp.citations,
        }

        user_content = (
            f"User request for the full report:\n{user_message}\n\n"
            f"Grounded material from specialist agents (use only these facts and numbers):\n"
            f"```json\n{json.dumps(material, indent=2, default=str)}\n```"
        )

        draft = self.llm.chat_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.15,
            max_tokens=2500,
        )

        fidelity = validate_numeric_fidelity(draft, all_tool_outputs)

        citations = (
            pricing_resp.citations
            + regulatory_resp.citations
            + (scenario_resp.citations if scenario_resp else [])
        )

        return AgentResponse(
            text=draft,
            citations=citations,
            raw_tool_outputs=all_tool_outputs,
            fidelity_passed=fidelity.passed,
            fidelity_message=fidelity.message,
        )