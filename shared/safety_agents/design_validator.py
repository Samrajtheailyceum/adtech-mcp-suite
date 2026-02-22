"""
Shared Safety Agents — Design Safety Validator

Validates that MCP tool outputs meet design safety standards:
- No dark patterns in ad recommendations
- WCAG accessibility compliance checks
- Deceptive ad copy detection
- Budget guardrails
"""

import re
from typing import Any, Dict, List


# ── Dark pattern detection ─────────────────────────────────────────────────
DARK_PATTERNS = {
    "false_urgency":    ["limited time", "only today", "expires in", "last chance", "hurry"],
    "misleading_cta":   ["subscribe to cancel", "free*", "no credit card*"],
    "hidden_costs":     ["plus applicable fees", "terms apply", "see details"],
    "forced_continuity": ["auto-renew", "unless you cancel", "will be charged"],
    "trick_questions":  ["do not uncheck", "leave unchecked to"],
    "confirmshaming":   ["no thanks, i prefer to fail", "no, i hate saving money"],
}


def detect_dark_patterns(copy: str) -> Dict:
    """Detect potentially deceptive advertising patterns in copy."""
    copy_lower = copy.lower()
    found = {}
    for pattern_type, phrases in DARK_PATTERNS.items():
        matches = [p for p in phrases if p in copy_lower]
        if matches:
            found[pattern_type] = matches

    risk = "high" if len(found) >= 2 else "medium" if found else "low"
    return {
        "dark_patterns_found": found,
        "risk_level": risk,
        "compliant": risk == "low",
        "recommendation": "Revise copy to remove deceptive elements." if found else "Copy passes dark pattern check.",
    }


# ── Accessibility checks ───────────────────────────────────────────────────
def check_ad_accessibility(ad_spec: Dict) -> Dict:
    """Basic WCAG 2.1 compliance checks for ad creatives."""
    issues = []

    # Alt text requirement
    if ad_spec.get("format") in ["image", "banner"] and not ad_spec.get("alt_text"):
        issues.append("Missing alt text for image creative (WCAG 1.1.1)")

    # Flashing content (epilepsy risk)
    fps = ad_spec.get("animation_fps", 0)
    if fps > 3:
        issues.append(f"Animation FPS {fps} may trigger photosensitive seizures (WCAG 2.3.1)")

    # Autoplay audio
    if ad_spec.get("autoplay_audio"):
        issues.append("Autoplay audio violates WCAG 1.4.2 — require user control")

    # Minimum font size
    font_size = ad_spec.get("min_font_size", 16)
    if font_size < 12:
        issues.append(f"Font size {font_size}px below 12px minimum for legibility")

    # Contrast ratio (simplified check)
    has_contrast_check = ad_spec.get("contrast_ratio_validated", False)
    if not has_contrast_check:
        issues.append("Contrast ratio not validated — WCAG 1.4.3 requires 4.5:1 minimum")

    return {
        "wcag_compliant": len(issues) == 0,
        "issues": issues,
        "severity": "critical" if len(issues) >= 3 else "medium" if issues else "none",
    }


# ── Budget guardrails ─────────────────────────────────────────────────────
MAX_SINGLE_BID_CPM = 100.0     # $100 CPM hard cap
MAX_DAILY_BUDGET_CHANGE = 0.50  # 50% max daily budget adjustment
MAX_CAMPAIGN_SPEND = 1_000_000  # $1M campaign spend flag


def validate_budget_recommendation(rec: Dict) -> Dict:
    """Guard against runaway spend recommendations."""
    warnings = []

    bid = rec.get("recommended_bid_cpm", 0)
    if bid > MAX_SINGLE_BID_CPM:
        warnings.append(f"Bid ${bid} exceeds safe cap of ${MAX_SINGLE_BID_CPM}")

    budget_change = abs(rec.get("budget_change_pct", 0))
    if budget_change > MAX_DAILY_BUDGET_CHANGE:
        warnings.append(f"Budget change {budget_change:.0%} exceeds {MAX_DAILY_BUDGET_CHANGE:.0%} guardrail")

    total_spend = rec.get("total_campaign_spend", 0)
    if total_spend > MAX_CAMPAIGN_SPEND:
        warnings.append(f"Campaign spend ${total_spend:,.0f} flagged for human approval")

    return {
        "passes_guardrails": len(warnings) == 0,
        "warnings": warnings,
        "requires_human_approval": len(warnings) > 0,
    }


def full_design_safety_check(
    copy: str = "",
    ad_spec: Dict = None,
    budget_rec: Dict = None,
) -> Dict:
    """Run all design safety checks and return aggregated report."""
    results = {}

    if copy:
        results["dark_patterns"] = detect_dark_patterns(copy)

    if ad_spec:
        results["accessibility"] = check_ad_accessibility(ad_spec)

    if budget_rec:
        results["budget_guardrails"] = validate_budget_recommendation(budget_rec)

    overall_safe = all(
        r.get("compliant", True) and r.get("passes_guardrails", True)
        and r.get("wcag_compliant", True)
        for r in results.values()
    )

    return {
        "overall_safe": overall_safe,
        "checks": results,
        "action": "proceed" if overall_safe else "review_required",
    }
