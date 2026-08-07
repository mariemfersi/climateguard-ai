"""
Task 9.2.3 — Regulatory Agent (RAG-grounded)

Answers Solvency II / ORSA / model-governance questions using only
retrieved regulation text. Refuses when retrieval confidence is low.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from agents.base import AgentResponse, build_system_prompt
from agents.llm_client import get_llm
from agents.numeric_fidelity_validator import validate_numeric_fidelity
from rag_index.retriever import get_retriever

logger = logging.getLogger(__name__)


class RegulatoryAgent:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.retriever = get_retriever()

    def run(self, user_message: str, context: Optional[Dict] = None) -> AgentResponse:
        chunks = self.retriever.retrieve(user_message, top_k=6)
        context_str = self.retriever.format_context(chunks)

        best_score = max((c.score for c in chunks), default=0.0)
        if best_score < 0.25 and chunks:
            return AgentResponse(
                text=(
                    "I could not retrieve sufficiently relevant regulation text "
                    "to answer this question with confidence. "
                    "Please rephrase or provide more context (e.g. specific Article number)."
                ),
                citations=[],
                fidelity_passed=True,
                fidelity_message="No numbers generated — refusal path",
            )

        system = build_system_prompt(
            "Regulatory Agent",
            extra_rules=(
                "Answer using ONLY the retrieved regulation excerpts provided below.\n"
                "Cite the source filename and, where possible, the Article number.\n"
                "If the retrieved text does not contain the answer, say so explicitly.\n"
                "Never invent Article numbers or regulatory requirements."
            ),
        )

        user_content = (
            f"User question:\n{user_message}\n\n"
            f"Retrieved regulation excerpts:\n{context_str}"
        )

        draft = self.llm.chat_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1000,
        )

        tool_outputs = [
            {
                "source": c.source,
                "title": c.title,
                "score": c.score,
                "text": c.text,
            }
            for c in chunks
        ]
        fidelity = validate_numeric_fidelity(draft, tool_outputs)

        citations = [f"{c.source} (score={c.score:.2f})" for c in chunks]

        return AgentResponse(
            text=draft,
            citations=citations,
            raw_tool_outputs=tool_outputs,
            fidelity_passed=fidelity.passed,
            fidelity_message=fidelity.message,
        )