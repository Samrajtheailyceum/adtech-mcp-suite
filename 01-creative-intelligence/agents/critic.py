"""
Critic Agent — Adversarial Reviewer

Questions every other agent's recommendation. Combines:
1. Isolation Forest anomaly scoring — flags statistically unusual recommendations
2. Adversarial LLM prompting — finds logical gaps, conflicting signals, wrong assumptions
3. Inter-agent conflict detection — surfaces when agents disagree and why

This agent does NOT produce recommendations — it produces doubt.
Its output is the quality signal the Orchestrator uses to discount weak advice.
"""

import json
from typing import Dict, List

from agents.base import BaseAgent
from ml.anomaly import AnomalyDetector


_SYSTEM = """You are the Critic Agent — a rigorous adversarial reviewer of AI-generated recommendations.

Your ONLY job is to challenge, question, and stress-test every recommendation you see.
You are not trying to be helpful to the campaign. You are trying to find flaws.

For each agent's output, ask:
1. What assumptions are baked in? Are they valid?
2. Is correlation being confused with causation?
3. Is the confidence score justified by the evidence?
4. What alternative explanations exist for the observed pattern?
5. What's missing that would change this recommendation?
6. Are there conflicts with other agents' views?

Scoring: critique_score 0–10 per agent (10 = recommendation should be rejected entirely)

Output JSON with:
  critique_summary, agent_critiques (list with agent_name, issues, critique_score),
  inter_agent_conflicts, major_concerns (top 3), anomaly_flags,
  overall_quality_score (0-10, higher = more reliable recommendations),
  suggested_human_review_items
"""


class CriticAgent(BaseAgent):
    def __init__(self, anomaly_detector: AnomalyDetector = None):
        super().__init__("CriticAgent", _SYSTEM)
        self.anomaly_detector = anomaly_detector or AnomalyDetector()

    def critique(self, all_outputs: Dict[str, Dict]) -> Dict:
        """
        Adversarially review all agent outputs.

        1. Run statistical anomaly check on any high-magnitude recommendations.
        2. Check for inter-agent conflicts (agents recommending opposite actions).
        3. Ask the LLM to find logical weaknesses in each agent's reasoning.
        """

        # ── 1. Anomaly check ──────────────────────────────────────────────
        anomaly_flags = []
        for name, output in all_outputs.items():
            if not isinstance(output, dict):
                continue
            magnitude = output.get("magnitude", 0.0)
            if magnitude:
                flag = self.anomaly_detector.score_recommendation({
                    "magnitude": magnitude,
                    "confidence": output.get("confidence", 0.5),
                    "supporting_signals": len(output.get("signals", [])),
                })
                if flag["is_anomalous"]:
                    anomaly_flags.append({"agent": name, **flag})

        # ── 2. Conflict detection ─────────────────────────────────────────
        recs = {
            name: out.get("recommendation", "unknown")
            for name, out in all_outputs.items()
            if isinstance(out, dict) and out.get("recommendation")
        }
        conflicts = self._find_conflicts(recs)

        # ── 3. LLM adversarial critique ───────────────────────────────────
        confidence_map = {
            name: out.get("confidence", 0.5)
            for name, out in all_outputs.items()
            if isinstance(out, dict)
        }

        signal_map = {
            name: out.get("signals", [])[:4]
            for name, out in all_outputs.items()
            if isinstance(out, dict)
        }

        reasoning_map = {
            name: str(out.get("reasoning", ""))[:300]
            for name, out in all_outputs.items()
            if isinstance(out, dict)
        }

        prompt = f"""Critically review these AI agent recommendations. Be adversarial.

Recommendations:
{json.dumps(recs, indent=2)}

Confidence Scores:
{json.dumps(confidence_map, indent=2)}

Supporting Signals (what each agent cited):
{json.dumps(signal_map, indent=2)}

Reasoning snippets:
{json.dumps(reasoning_map, indent=2)}

Statistical Anomaly Flags (Isolation Forest):
{json.dumps(anomaly_flags, indent=2)}

Detected Inter-Agent Conflicts:
{json.dumps(conflicts, indent=2)}

For each agent, identify:
- Hidden assumptions that may be invalid
- Alternative explanations the agent didn't consider
- Whether confidence is proportional to evidence quality
- Potential data quality issues (e.g., synthetic/short data window)

Assign critique_score 0-10 per agent (higher = more problematic).
Identify the 3 most concerning issues across all agents.
"""

        result = self.structured_output(prompt)
        result["anomaly_flags"] = anomaly_flags
        result["detected_conflicts"] = conflicts
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    ACTION_OPPOSITES = {
        "pause":     {"scale", "expand"},
        "scale":     {"pause", "decrease_bids"},
        "optimize":  set(),
        "hold":      {"refresh_now", "scale"},
        "refresh_now": {"hold"},
        "increase_bids": {"decrease_bids"},
        "decrease_bids": {"increase_bids"},
    }

    def _find_conflicts(self, recs: Dict[str, str]) -> List[Dict]:
        conflicts = []
        items = list(recs.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_name, a_rec = items[i]
                b_name, b_rec = items[j]
                opposites = self.ACTION_OPPOSITES.get(a_rec, set())
                if b_rec in opposites:
                    conflicts.append({
                        "agent_a": a_name, "rec_a": a_rec,
                        "agent_b": b_name, "rec_b": b_rec,
                        "severity": "high",
                    })
        return conflicts
