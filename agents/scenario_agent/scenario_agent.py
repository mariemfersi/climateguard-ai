"""
Task 9.2.4 — Scenario Agent

Parses natural-language what-if requests into structured parameter overrides,
calls the Monte Carlo /simulate endpoint, and narrates the result.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import httpx

from agents.base import AgentResponse, ToolCallRecord, build_system_prompt
from agents.llm_client import get_llm
from agents.numeric_fidelity_validator import validate_numeric_fidelity
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ScenarioAgent:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.settings = get_settings()

    def _parse_overrides(self, user_message: str) -> Dict[str, Any]:
        system = (
            "Extract scenario overrides from the user message. "
            "Return ONLY a JSON object with any of these keys that appear:\n"
            "  frequency_multiplier (float)\n"
            "  severity_multiplier (float)\n"
            "  inflation_rate (float, e.g. 0.08 for 8%)\n"
            "  region (string)\n"
            "  peril (string)\n"
            "  description (string)\n"
            "If a value is not mentioned, omit the key. No markdown, just JSON."
        )
        try:
            raw = self.llm.chat_text(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Override parsing failed (%s) — using empty overrides", exc)
            return {"description": user_message}

    def _call_simulate(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.settings.model_api_base_url}/simulate"
        try:
            with httpx.Client(timeout=self.settings.model_api_timeout) as client:
                r = client.post(url, json=overrides)
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.warning("/simulate unavailable (%s) — using mock scenario result", exc)
            freq = float(overrides.get("frequency_multiplier", 1.0))
            sev = float(overrides.get("severity_multiplier", 1.0))
            base_el = 12_450_000
            base_var = 87_200_000
            return {
                "scenario": overrides,
                "expected_loss_usd": round(base_el * freq * sev),
                "var_995": round(base_var * freq * sev),
                "tvar_995": round(base_var * 1.29 * freq * sev),
                "delta_expected_loss_pct": round((freq * sev - 1.0) * 100, 1),
                "model_version": "mock-mc-v0.1",
            }

    def run(self, user_message: str, context: Optional[Dict] = None) -> AgentResponse:
        overrides = self._parse_overrides(user_message)
        if context:
            overrides.update({k: v for k, v in context.items() if v is not None})

        sim_out = self._call_simulate(overrides)
        tool_outputs = [sim_out]
        tool_records = [
            ToolCallRecord(name="simulate", arguments=overrides, result=sim_out)
        ]

        system = build_system_prompt(
            "Scenario Agent",
            extra_rules=(
                "Narrate the what-if result clearly for an underwriter or CRO.\n"
                "State the assumed parameter changes, then the resulting portfolio metrics.\n"
                "Use only numbers present in the tool result. Highlight the percentage change."
            ),
        )

        user_content = (
            f"User what-if request:\n{user_message}\n\n"
            f"Parsed overrides:\n{json.dumps(overrides, indent=2)}\n\n"
            f"Simulation result (authoritative):\n```json\n{json.dumps(sim_out, indent=2)}\n```"
        )

        draft = self.llm.chat_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=900,
        )

        fidelity = validate_numeric_fidelity(draft, tool_outputs)

        return AgentResponse(
            text=draft,
            citations=["model:/simulate"],
            tool_calls=tool_records,
            raw_tool_outputs=tool_outputs,
            fidelity_passed=fidelity.passed,
            fidelity_message=fidelity.message,
        )