"""
Task 9.2.2 — Pricing Agent

Given a treaty description (layer, attachment, limit, region/peril),
calls the model-serving API (/predict, /explain) and drafts a first-pass
technical pricing memo grounded entirely in tool outputs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

from agents.base import AgentResponse, ToolCallRecord, build_system_prompt
from agents.llm_client import get_llm
from agents.numeric_fidelity_validator import validate_numeric_fidelity
from config.settings import get_settings

logger = logging.getLogger(__name__)


class PricingAgent:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.settings = get_settings()

    def _call_predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Phase-8/10 model-serving /predict endpoint."""
        url = f"{self.settings.model_api_base_url}/predict"
        try:
            with httpx.Client(timeout=self.settings.model_api_timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.warning("/predict unavailable (%s) — using mock response for local demo", exc)
            return {
                "expected_loss_usd": 12_450_000,
                "var_995": 87_200_000,
                "tvar_995": 112_500_000,
                "rate_on_line_technical": 0.078,
                "peril": payload.get("peril", "hurricane"),
                "region": payload.get("region", "florida"),
                "model_version": "mock-freqsev-v0.1",
            }

    def _call_explain(self, prediction_id: str = "mock") -> Dict[str, Any]:
        url = f"{self.settings.model_api_base_url}/explain/{prediction_id}"
        try:
            with httpx.Client(timeout=self.settings.model_api_timeout) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.warning("/explain unavailable (%s) — using mock SHAP", exc)
            return {
                "prediction_id": prediction_id,
                "shap_values": [
                    {"feature": "distance_to_coast_km", "shap_value": -0.12, "rank": 1},
                    {"feature": "construction_class", "shap_value": 0.09, "rank": 2},
                    {"feature": "roof_age_years", "shap_value": 0.07, "rank": 3},
                    {"feature": "sst_anomaly", "shap_value": 0.05, "rank": 4},
                    {"feature": "elevation_m", "shap_value": -0.04, "rank": 5},
                ],
            }

    def run(self, user_message: str, context: Optional[Dict] = None) -> AgentResponse:
        context = context or {}

        payload = {
            "region": context.get("region", "florida"),
            "peril": context.get("peril", "hurricane"),
            "attachment_usd": context.get("attachment_usd", 50_000_000),
            "limit_usd": context.get("limit_usd", 100_000_000),
            "description": user_message,
        }

        predict_out = self._call_predict(payload)
        explain_out = self._call_explain(predict_out.get("prediction_id", "mock"))

        tool_outputs = [predict_out, explain_out]
        tool_records = [
            ToolCallRecord(name="predict", arguments=payload, result=predict_out),
            ToolCallRecord(name="explain", arguments={"prediction_id": "mock"}, result=explain_out),
        ]

        system = build_system_prompt(
            "Pricing Agent",
            extra_rules=(
                "Produce a short technical pricing memo containing:\n"
                "- Treaty summary (layer, attachment, limit)\n"
                "- Technical rate-on-line recommendation grounded in model output\n"
                "- Key risk drivers from the SHAP explanation\n"
                "- One paragraph of qualitative rationale\n"
                "Do not add numbers that are not present in the tool results below."
            ),
        )

        user_content = (
            f"User request:\n{user_message}\n\n"
            f"Tool results (authoritative — use only these numbers):\n"
            f"```json\n{json.dumps(tool_outputs, indent=2)}\n```"
        )

        draft = self.llm.chat_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        fidelity = validate_numeric_fidelity(draft, tool_outputs)

        return AgentResponse(
            text=draft,
            citations=["model:/predict", "model:/explain"],
            tool_calls=tool_records,
            raw_tool_outputs=tool_outputs,
            fidelity_passed=fidelity.passed,
            fidelity_message=fidelity.message,
        )