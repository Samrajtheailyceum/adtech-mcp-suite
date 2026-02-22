"""
Shared Safety Agents — API Security Layer

Used by all MCPs to validate inputs, detect prompt injection, and
enforce rate limiting. These agents sit in front of every tool call.
"""

import re
import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, Optional


# ── Rate limiter ───────────────────────────────────────────────────────────
_CALL_LOG: Dict[str, list] = defaultdict(list)
RATE_LIMIT = 60  # calls per minute


def check_rate_limit(client_id: str = "default") -> Dict:
    now = time.time()
    window = [t for t in _CALL_LOG[client_id] if now - t < 60]
    _CALL_LOG[client_id] = window
    if len(window) >= RATE_LIMIT:
        return {"allowed": False, "reason": f"Rate limit exceeded ({RATE_LIMIT}/min)"}
    _CALL_LOG[client_id].append(now)
    return {"allowed": True, "calls_this_minute": len(window) + 1}


# ── Input sanitization ─────────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"disregard.*system.*prompt",
    r"you are now",
    r"act as",
    r"jailbreak",
    r"<script",
    r"javascript:",
    r"DROP TABLE",
    r"SELECT.*FROM",
    r"'; --",
    r"\$\{.*\}",   # template injection
    r"__import__",  # Python injection
]

_injection_re = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_input(text: str, field_name: str = "input") -> Dict:
    """Detect prompt injection and SQL injection in string inputs."""
    if not isinstance(text, str):
        return {"safe": True, "sanitized": str(text)}

    if len(text) > 10_000:
        return {"safe": False, "reason": f"{field_name} exceeds 10,000 character limit"}

    match = _injection_re.search(text)
    if match:
        return {
            "safe": False,
            "reason": f"Potential injection detected in {field_name}: '{match.group()[:30]}'",
            "pattern": match.group(),
        }

    # Strip null bytes and control characters
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return {"safe": True, "sanitized": sanitized}


def validate_tool_inputs(inputs: Dict[str, Any]) -> Dict:
    """Validate all string inputs to a tool call."""
    issues = []
    for field, value in inputs.items():
        if isinstance(value, str):
            result = sanitize_input(value, field)
            if not result["safe"]:
                issues.append(result["reason"])
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "blocked": len(issues) > 0,
    }


# ── PII guard ─────────────────────────────────────────────────────────────
PII_PATTERNS_GUARD = {
    "email":       re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "ssn":         re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
}


def redact_pii(text: str) -> Dict:
    """Redact PII from text before logging or storing."""
    redacted = text
    found = []
    for pii_type, pattern in PII_PATTERNS_GUARD.items():
        matches = pattern.findall(text)
        if matches:
            found.append(pii_type)
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
    return {"redacted_text": redacted, "pii_types_found": found, "had_pii": len(found) > 0}


# ── Audit logger ───────────────────────────────────────────────────────────
_AUDIT_LOG = []


def log_tool_call(tool_name: str, inputs: Dict, client_id: str = "default") -> None:
    """Append sanitized audit log entry."""
    _AUDIT_LOG.append({
        "timestamp": time.time(),
        "tool": tool_name,
        "client_id": hashlib.sha256(client_id.encode()).hexdigest()[:8],
        "input_hash": hashlib.sha256(str(inputs).encode()).hexdigest()[:16],
    })


def get_audit_log(last_n: int = 20) -> list:
    return _AUDIT_LOG[-last_n:]
