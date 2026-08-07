"""
Unit and integration tests for Phase 9: Multi-Agent LLM Layer & RAG Index.
Includes adversarial numeric hallucination detection tests for the NumericFidelityValidator.
"""

import json
import os
from pathlib import Path

import pytest

from rag_index.chunk_and_embed import (
    RegulatoryRAGIndex,
    TextChunk,
    build_and_save_index,
    get_rag_index,
)
from agents.orchestrator.orchestrator import AgentOrchestrator
from agents.pricing_agent.pricing_agent import PricingAgent
from agents.regulatory_agent.regulatory_agent import RegulatoryAgent
from agents.scenario_agent.scenario_agent import ScenarioAgent
from agents.numeric_fidelity_validator import (
    NumericFidelityValidator,
    extract_numbers_from_text,
)
from agents.numeric_fidelity_validator import ReportWriterAgent


def test_rag_index_construction_and_search():
    index = get_rag_index()
    assert len(index.chunks) > 0

    # Search for Solvency II Article 105
    results = index.search("Article 105 Solvency Capital Requirement 1-in-200 year VaR", top_k=3)
    assert len(results) > 0
    assert any("105" in r["article_id"] for r in results)

    # Search for out-of-domain query (should return low confidence / empty if threshold high)
    out_domain = index.search("Quantum mechanics particle entanglement equation", top_k=3, min_score_threshold=0.80)
    assert len(out_domain) == 0


def test_orchestrator_intent_classification():
    orchestrator = AgentOrchestrator(use_llm=False)

    test_queries = [
        ("Draft a technical pricing memo for layer 50M xs 50M", "pricing"),
        ("What is Rate-on-Line for cedent renewal?", "pricing"),
        ("What are the Solvency II Article 105 SCR catastrophe requirements?", "regulatory"),
        ("Explain the 72-hour hours clause in Article 118", "regulatory"),
        ("What if Category 5 hurricane frequency increases by 20%?", "scenario"),
        ("Simulate a stress test scenario with 15% inflation", "scenario"),
        ("Generate full executive board report and SFCR summary", "report_writing"),
    ]

    for q, expected in test_queries:
        intent = orchestrator.classify_intent(q)
        assert intent == expected, f"Failed for query '{q}': expected '{expected}', got '{intent}'"


def test_pricing_agent_layer_calculation():
    agent = PricingAgent()
    res = agent.handle_query(
        "Price treaty layer 100M xs 50M",
        context={"attachment_usd": 50_000_000.0, "limit_usd": 100_000_000.0},
    )

    assert res["intent"] == "pricing"
    assert "response" in res
    assert res["rate_on_line"] > 0
    assert res["technical_premium_usd"] > 0
    assert "Technical Treaty Pricing Memo" in res["response"]


def test_regulatory_agent_qa_and_refusal():
    agent = RegulatoryAgent()

    # Valid regulatory prompt
    res_valid = agent.handle_query("Article 105 natural catastrophe SCR VaR")
    assert res_valid["intent"] == "regulatory"
    assert not res_valid["refused_to_answer"]
    assert len(res_valid["citations"]) > 0

    # Out-of-domain query -> low-confidence refusal
    res_out = agent.handle_query("What is the personal income tax rate in Tokyo?")
    assert res_out["intent"] == "regulatory"
    assert res_out["refused_to_answer"]
    assert "cannot answer" in res_out["response"].lower()


def test_scenario_agent_what_if_parsing_and_run():
    agent = ScenarioAgent()

    # What-if query
    query = "What happens if Category 5 hurricane frequency increases by 25% and inflation is 10%?"
    params = agent.parse_what_if_query(query)

    assert params["frequency_multiplier"] >= 1.20
    assert params["severity_multiplier"] >= 1.05

    res = agent.handle_query(query, context={"num_seasons": 100})
    assert res["intent"] == "scenario"
    assert "Live Simulation Results" in res["response"]
    assert "deltas" in res


def test_numeric_fidelity_validator_grounded_and_adversarial():
    validator = NumericFidelityValidator()

    tool_data = {
        "expected_loss": 177291446.0,
        "var_995": 188712120.0,
        "tvar_995": 207443500.0,
        "rate_on_line": 0.0254,
    }

    # 1. Grounded report text -> must pass 100%
    grounded_text = (
        "Expected annual loss is $177,291,446. "
        "The 1-in-200 year VaR is $188,712,120 and TVaR is $207,443,500."
    )
    val_grounded = validator.validate(grounded_text, tool_data)
    assert val_grounded["passed"]
    assert val_grounded["pass_rate"] >= 99.0

    # 2. Adversarial hallucinated text -> MUST BE CAUGHT and failed
    hallucinated_text = (
        "Expected annual loss is $999,999,999. "  # Injected hallucination!
        "The 1-in-200 year VaR is $188,712,120."
    )
    val_hallucinated = validator.validate(hallucinated_text, tool_data)
    assert not val_hallucinated["passed"]
    assert len(val_hallucinated["invalid_numbers"]) > 0
    assert any(inv["extracted_value"] == 999999999.0 for inv in val_hallucinated["invalid_numbers"])


def test_report_writer_agent_end_to_end():
    agent = ReportWriterAgent()
    res = agent.handle_query("Generate full Solvency II SFCR report")

    assert res["intent"] == "report_writing"
    assert "response" in res
    assert "validation_result" in res
    assert res["validation_result"]["pass_rate"] >= 95.0
    assert Path(res["file_path"]).exists()
