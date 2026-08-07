from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:USD\s*)?(?:\$)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)(?:\s*%|\s*percent)?(?![\w.])",
    re.IGNORECASE,
)

@dataclass
class ValidationResult:
    passed: bool
    extracted_numbers: List[str] = field(default_factory=list)
    unsupported_numbers: List[str] = field(default_factory=list)
    message: str = ""

def _normalize(num_str: str) -> str:
    s = num_str.upper().replace("USD", "").replace("$", "").replace(",", "").replace("%", "").strip()
    if re.match(r"^\d+\.0+$", s):
        s = s.split(".")[0]
    return s

def extract_numbers(text: str) -> List[str]:
    return [m.group(0).strip() for m in _NUMBER_RE.finditer(text)]

def validate_numeric_fidelity(generated_text: str, tool_outputs: List[Dict[str, Any]], *, tolerance_extra: int = 2) -> ValidationResult:
    extracted = extract_numbers(generated_text)
    if not extracted:
        return ValidationResult(passed=True, message="No numbers found")
    allowed: Set[str] = set()
    for out in tool_outputs:
        _collect_numbers(out, allowed)
    unsupported = []
    for raw in extracted:
        norm = _normalize(raw)
        if _is_structural(norm):
            continue
        if norm not in allowed and not _fuzzy_in(norm, allowed):
            unsupported.append(raw)
    if len(unsupported) <= tolerance_extra:
        return ValidationResult(passed=True, extracted_numbers=extracted, unsupported_numbers=unsupported, message=f"Passed ({len(unsupported)} extras allowed)")
    return ValidationResult(passed=False, extracted_numbers=extracted, unsupported_numbers=unsupported, message=f"Failed: {unsupported}")

def _collect_numbers(obj: Any, sink: Set[str]) -> None:
    if isinstance(obj, (int, float)):
        sink.add(_normalize(str(obj)))
    elif isinstance(obj, str):
        for n in extract_numbers(obj):
            sink.add(_normalize(n))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, sink)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, sink)

def _is_structural(norm: str) -> bool:
    try:
        v = float(norm)
    except ValueError:
        return False
    return (v == int(v) and 0 <= v <= 30) or (1900 <= v <= 2100)

def _fuzzy_in(norm: str, allowed: Set[str]) -> bool:
    try:
        v = float(norm)
    except ValueError:
        return False
    for a in allowed:
        try:
            av = float(a)
            if abs(v - av) < 1e-6 or abs(v - av * 100) < 1e-4 or abs(v * 100 - av) < 1e-4:
                return True
        except ValueError:
            continue
    return False