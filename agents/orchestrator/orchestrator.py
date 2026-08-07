"""
Agentic Orchestrator for intent classification, state tracking, and routing.

Classifies incoming natural-language requests into specific agent intents:
1. `pricing` -> Pricing Agent (technical pricing memos, Rate-on-Line, layer pricing)
2. `regulatory` -> Regulatory Agent (Solvency II SCR, EIOPA guidance, treaty clauses)
3. `scenario` -> Scenario Agent (natural-language what-if stress simulations)
4. `report_writing` -> Report-Writer Agent (board-ready SFCR / renewal memos)

Supports Azure OpenAI / OpenAI intent classification with deterministic keyword fallback.

Usage:
    python -m agents.orchestrator.orchestrator [--query "Draft a pricing memo for layer 50M xs 50M"]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

VALID_INTENTS = ["pricing", "regulatory", "scenario", "report_writing"]

# Keyword pattern maps for rule-based classification fallback
INTENT_KEYWORDS = {
    "report_writing": [
        r"\bgenerate\s+report\b", r"\bboard\s+report\b", r"\bfull\s+report\b",
        r"\breport\b", r"\bmemo\b", r"\bsummary document\b", r"\bexecutive summary\b",
        r"\bsfcr report\b", r"\bsfcr summary\b"
    ],
    "pricing": [
        r"\bpricing\b", r"\brate[\s\-]on[\s\-]line\b", r"\brol\b", r"\blayer\b",
        r"\battachment\b", r"\blimit\b", r"\bpremium\b", r"\btechnical rate\b",
        r"\bcedent\b", r"\breinsurance pricing\b"
    ],
    "regulatory": [
        r"\bsolvency\b", r"\barticle\b", r"\bscr\b", r"\beiopa\b", r"\bnaic\b",
        r"\borsa\b", r"\bregulation\b", r"\bcompliance\b", r"\b72[\s\-]hour\b", r"\bclause\b"
    ],
    "scenario": [
        r"\bwhat[\s\-]if\b", r"\bstress\b", r"\bsimulat\w*\b", r"\bscenario\b",
        r"\bif frequency\b", r"\bif severity\b", r"\bwarming\b", r"\bcat 5\b",
        r"\blandfall\b", r"\binflation\b"
    ],
}


class AgentOrchestrator:
    """
    Orchestrator for classifying user intent and routing to specialized agents.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.history: list[dict[str, Any]] = []

    def classify_intent(self, query: str) -> str:
        """
        Classify user query into one of the 4 valid agent intents.

        Args:
            query: User prompt text.

        Returns:
            Intent string: 'pricing' | 'regulatory' | 'scenario' | 'report_writing'.
        """
        query_lower = query.lower()

        # Check Azure OpenAI / OpenAI API availability
        openai_key = os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
        if self.use_llm and openai_key:
            try:
                import openai
                # Fast LLM intent classification call
                response = openai.ChatCompletion.create(
                    model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an intent classifier for a reinsurance AI platform. "
                                "Classify the user prompt into exactly ONE category: 'pricing', 'regulatory', 'scenario', or 'report_writing'. "
                                "Respond with only the category string in lowercase."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
                intent_raw = response.choices[0].message.content.strip().lower()
                if intent_raw in VALID_INTENTS:
                    logger.info("LLM intent classified as '%s' for query: '%s'", intent_raw, query)
                    return intent_raw
            except Exception as e:
                logger.warning("LLM intent classification failed (%s), using rule-based classifier.", e)

        # Keyword pattern matching fallback
        scores = {intent: 0 for intent in VALID_INTENTS}
        for intent, patterns in INTENT_KEYWORDS.items():
            for pat in patterns:
                if re.search(pat, query_lower):
                    scores[intent] += 1

        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            # Default to pricing if ambiguous
            best_intent = "pricing"

        logger.info("Rule-based intent classified as '%s' (scores: %s) for query: '%s'", best_intent, scores, query)
        return best_intent

    def route(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Route request to the target specialized agent.

        Args:
            query: User query string.
            context: Additional structured state or payload.

        Returns:
            Dictionary with agent response, intent, and routing metadata.
        """
        context = context or {}
        intent = self.classify_intent(query)

        entry = {
            "query": query,
            "intent": intent,
            "context": context,
        }
        self.history.append(entry)

        # Route to downstream agent handler
        if intent == "pricing":
            from agents.pricing_agent.pricing_agent import PricingAgent
            agent = PricingAgent()
            result = agent.handle_query(query, context)

        elif intent == "regulatory":
            from agents.regulatory_agent.regulatory_agent import RegulatoryAgent
            agent = RegulatoryAgent()
            result = agent.handle_query(query, context)

        elif intent == "scenario":
            from agents.scenario_agent.scenario_agent import ScenarioAgent
            agent = ScenarioAgent()
            result = agent.handle_query(query, context)

        elif intent == "report_writing":
            from agents.report_writer_agent.report_writer_agent import ReportWriterAgent
            agent = ReportWriterAgent()
            result = agent.handle_query(query, context)

        else:
            result = {
                "intent": "unknown",
                "response": "Could not determine intent.",
                "citations": [],
            }

        result["classified_intent"] = intent
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Agent Orchestrator Intent Classifier")
    parser.add_argument("--query", type=str, default="What is the 1-in-200 year SCR VaR requirement under Solvency II Article 105?", help="User query")
    args = parser.parse_args()

    orchestrator = AgentOrchestrator()
    res = orchestrator.route(args.query)

    print("\n" + "=" * 70)
    print("ORCHESTRATOR ROUTING RESULT")
    print("=" * 70)
    print(f"Query:            '{args.query}'")
    print(f"Classified Intent: {res.get('classified_intent')}")
    print(f"Agent Response:\n{res.get('response')}")
    print("=" * 70)
