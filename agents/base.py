from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ToolCallRecord:
    name: str
    arguments: Dict[str, Any]
    result: Any

@dataclass
class AgentResponse:
    text: str
    citations: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    raw_tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    fidelity_passed: Optional[bool] = None
    fidelity_message: str = ""

def build_system_prompt(role: str, extra_rules: str = "") -> str:
    return f"""You are the ClimateGuard AI {role}.

STRICT RULES:
1. Every numeric claim MUST come from a tool result or retrieved document.
2. Never invent loss numbers, VaR, rates, SHAP values or article citations.
3. If a tool fails or data is insufficient, say so clearly.
4. Use concise professional actuarial language.
5. Cite sources when you use them.

{extra_rules}
""".strip()