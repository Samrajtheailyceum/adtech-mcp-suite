"""
MCP 7 — Privacy & Compliance MCP

GDPR/CCPA compliance checking, consent signal validation,
and data residency auditing for ad tech stacks.

Agents:
- ConsentValidationAgent   — Validates IAB TCF consent strings
- DataResidencyAgent       — Checks data flows against residency rules
- PIIDetectionAgent        — Detects PII in data pipelines
- ComplianceCritic         — Independent legal risk assessment
"""

import json
import re
import hashlib
from datetime import datetime
from mcp.server.fastmcp import FastMCP
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent

mcp = FastMCP("Privacy Compliance MCP")

# GDPR legal basis options
GDPR_LEGAL_BASIS = {
    "consent": "User explicitly opted in",
    "legitimate_interest": "Balancing test required",
    "contract": "Processing necessary for contract",
    "legal_obligation": "Required by law",
    "vital_interests": "Life or death situation",
    "public_task": "Official authority task",
}

# Data categories risk levels
DATA_RISK_LEVELS = {
    "email": "high", "phone": "high", "ssn": "critical", "ip_address": "medium",
    "device_id": "medium", "cookie_id": "low", "age": "low", "gender": "low",
    "location_precise": "high", "location_city": "low", "browsing_history": "high",
    "purchase_history": "medium", "health_data": "critical", "biometric": "critical",
}

# PII detection patterns
PII_PATTERNS = {
    "email":    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone":    re.compile(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "ip_v4":   re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    "ssn":      re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
}


def detect_pii(text: str) -> dict:
    """Detect PII patterns in text/data samples."""
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Hash found values for safety — never return raw PII
            found[pii_type] = {
                "count": len(matches),
                "hashed_sample": hashlib.sha256(matches[0].encode()).hexdigest()[:16],
                "risk": DATA_RISK_LEVELS.get(pii_type, "medium"),
            }
    return found


def validate_consent_signal(consent_data: dict) -> dict:
    """Validate a consent signal against IAB TCF requirements."""
    issues = []

    if not consent_data.get("user_id"):
        issues.append("Missing user identifier")
    if not consent_data.get("timestamp"):
        issues.append("Missing consent timestamp")
    elif consent_data.get("timestamp"):
        try:
            ts = datetime.fromisoformat(consent_data["timestamp"])
            age_days = (datetime.now() - ts).days
            if age_days > 365:
                issues.append(f"Consent is {age_days} days old — re-consent recommended")
        except ValueError:
            issues.append("Invalid timestamp format")

    purposes = consent_data.get("purposes_consented", [])
    if 3 not in purposes:  # IAB purpose 3 = personalised advertising
        issues.append("Purpose 3 (personalised ads) not consented — contextual only")

    legal_basis = consent_data.get("legal_basis", "consent")
    if legal_basis == "legitimate_interest" and not consent_data.get("li_objection_offered"):
        issues.append("Legitimate interest requires objection mechanism")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "legal_basis": GDPR_LEGAL_BASIS.get(legal_basis, "unknown"),
        "purposes": purposes,
        "compliance_score": round(max(0, 1 - len(issues) * 0.2), 2),
    }


_CONSENT_SYSTEM = """You are a Privacy Compliance Agent for ad tech.
Evaluate consent signals, data flows, and PII risks against GDPR/CCPA requirements.
Provide specific remediation steps for compliance gaps.
Output JSON: compliance_status, critical_issues, remediation_steps, risk_level (low/medium/high/critical),
confidence (0-1), signals."""

_CRITIC_SYSTEM = """You are a Privacy Law Critic.
Challenge compliance assessments. Flag: consent withdrawal mechanisms, children's data (COPPA),
cross-border transfers (SCCs), data processor agreements, and breach notification readiness.
Note: This is not legal advice. Output JSON: legal_gaps, critique_score (0-10), caveats."""


class ConsentComplianceAgent(BaseAgent):
    def __init__(self):
        super().__init__("ConsentComplianceAgent", _CONSENT_SYSTEM)

    def audit(self, consent_data: dict, data_fields: list) -> dict:
        validation = validate_consent_signal(consent_data)
        risk_fields = {f: DATA_RISK_LEVELS.get(f, "unknown") for f in data_fields}
        critical_fields = [f for f, r in risk_fields.items() if r == "critical"]

        prompt = f"""Audit this advertising consent and data collection setup:

Consent Validation Result:
{json.dumps(validation, indent=2)}

Data Fields Collected + Risk Levels:
{json.dumps(risk_fields, indent=2)}

Critical Risk Fields: {critical_fields}

Provide compliance assessment with specific GDPR/CCPA remediation steps."""
        result = self.structured_output(prompt)
        result["agent_type"] = "privacy_compliance"
        result["consent_validation"] = validation
        result["data_risks"] = risk_fields
        return result


class ComplianceCritic(BaseAgent):
    def __init__(self):
        super().__init__("ComplianceCritic", _CRITIC_SYSTEM)

    def critique(self, audit: dict) -> dict:
        prompt = f"""Critically review this compliance audit:
{json.dumps(audit, indent=2)}
What legal risks were missed? (Note: not legal advice)"""
        return self.structured_output(prompt)


_agent = ConsentComplianceAgent()
_critic = ComplianceCritic()


@mcp.tool()
def audit_consent_compliance(
    user_id: str,
    consent_timestamp: str,
    purposes_consented: list[int],
    legal_basis: str,
    data_fields_collected: list[str],
) -> str:
    """
    Audit advertising consent signals and data collection for GDPR/CCPA compliance.

    Args:
        user_id: Anonymous user identifier
        consent_timestamp: ISO 8601 timestamp of consent
        purposes_consented: IAB TCF purpose IDs (e.g. [1, 3, 4])
        legal_basis: One of: consent, legitimate_interest, contract
        data_fields_collected: List of data fields (e.g. ["email", "ip_address"])
    """
    consent_data = {"user_id": user_id, "timestamp": consent_timestamp,
                    "purposes_consented": purposes_consented, "legal_basis": legal_basis}
    result = _agent.audit(consent_data, data_fields_collected)
    critique = _critic.critique(result)
    return json.dumps({"audit": result, "legal_critique": critique}, indent=2, default=str)


@mcp.tool()
def scan_data_for_pii(data_sample: str) -> str:
    """
    Scan a data sample for PII patterns (email, phone, IP, SSN, credit card).
    Returns hashed references only — never raw PII.

    Args:
        data_sample: String representation of data to scan
    """
    found = detect_pii(data_sample)
    risk = "critical" if any(v["risk"] == "critical" for v in found.values()) else \
           "high" if any(v["risk"] == "high" for v in found.values()) else "low"
    return json.dumps({"pii_detected": found, "overall_risk": risk,
                       "action_required": len(found) > 0}, indent=2)


@mcp.tool()
def get_gdpr_requirements() -> str:
    """Return GDPR legal basis options and data category risk levels."""
    return json.dumps({"legal_basis": GDPR_LEGAL_BASIS, "data_risks": DATA_RISK_LEVELS}, indent=2)


if __name__ == "__main__":
    mcp.run()
