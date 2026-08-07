"""
Task 9.2.1 — Intent classification + routing to the correct agent.
"""

from __future__ import annotations

import logging
from typing import Literal

from agents.base import AgentResponse
from agents.llm_client import get_llm
from agents.pricing_agent.pricing_agent import PricingAgent
from agents.regulatory_agent.regulatory_agent import RegulatoryAgent
from agents.report_writer_agent.report_writer_agent import ReportWriterAgent
from agents.scenario_agent.scenario_agent import ScenarioAgent

logger = logging.getLogger(__name__)

Intent = Literal["pricing", "regulatory", "scenario", "report", "unknown"]


class Orchestrator:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.pricing = PricingAgent()
        self.regulatory = RegulatoryAgent()
        self.scenario = ScenarioAgent()
        self.report_writer = ReportWriterAgent()

    def classify_intent(self, user_message: str) -> Intent:
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the user request into exactly one of: "
                    "pricing | regulatory | scenario | report | unknown.\n"
                    "Reply with only the single word label, nothing else.\n\n"
                    "pricing   = treaty pricing, rate-on-line, technical premium, layer pricing\n"
                    "regulatory = Solvency II, SCR, ORSA, articles, regulatory questions\n"
                    "scenario  = what-if, stress test, change frequency/severity, simulate\n"
                    "report    = draft a full memo, board report, SFCR-style narrative\n"
                ),
            },
            {"role": "user", "content": user_message},
        ]
        label = self.llm.chat_text(messages, temperature=0.0, max_tokens=10).strip().lower()
        if label not in {"pricing", "regulatory", "scenario", "report"}:
            return "unknown"
        return label  # type: ignore

    def route(self, user_message: str, context: dict | None = None) -> AgentResponse:
        intent = self.classify_intent(user_message)
        logger.info("Orchestrator classified intent=%s", intent)

        if intent == "pricing":
            return self.pricing.run(user_message, context=context)
        if intent == "regulatory":
            return self.regulatory.run(user_message, context=context)
        if intent == "scenario":
            return self.scenario.run(user_message, context=context)
        if intent == "report":
            return self.report_writer.run(user_message, context=context)

        return AgentResponse(
            text=(
                "I could not confidently classify your request. "
                "Please rephrase as a pricing question, a regulatory question, "
                "a what-if scenario, or a request to generate a report."
            )
        )