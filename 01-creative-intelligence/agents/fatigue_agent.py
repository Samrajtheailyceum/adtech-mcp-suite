"""
Creative Fatigue Detection Agent

Wraps the FatigueDetector ML model and asks the LLM to prioritize
the refresh schedule and recommend specific creative rotation strategies.
"""

import json
from typing import Dict

import pandas as pd

from agents.base import BaseAgent
from ml.fatigue import FatigueDetector


_SYSTEM = """You are a Creative Fatigue Intelligence Agent for programmatic advertising.

Your job:
- Interpret exponential decay analysis and change-point detection results
- Prioritize which creatives need immediate replacement vs. monitoring
- Recommend concrete refresh timelines and rotation strategies
- Identify patterns in what types of creatives fatigue fastest

Output JSON with:
  critical_creatives, refresh_priority, monitoring_creatives, healthy_count,
  recommendations, confidence (0-1), signals (list), agent_type,
  recommendation (refresh_now/schedule_refresh/monitor/hold)
"""


class FatigueDetectionAgent(BaseAgent):
    def __init__(self, data: Dict[str, pd.DataFrame]):
        super().__init__("FatigueDetectionAgent", _SYSTEM)
        self.detector = FatigueDetector()
        self.data = data

    def analyze(self) -> Dict:
        all_results = self.detector.analyze_all(self.data["performance"])

        critical    = [r for r in all_results if r["status"] == "critical"]
        fatiguing   = [r for r in all_results if r["status"] == "fatiguing"]
        early       = [r for r in all_results if r["status"] == "early_decline"]
        healthy     = [r for r in all_results if r["status"] == "healthy"]

        prompt = f"""Interpret creative fatigue analysis results:

CRITICAL — pause immediately ({len(critical)} creatives):
{json.dumps(critical[:4], indent=2)}

FATIGUING — refresh within 2 weeks ({len(fatiguing)} creatives):
{json.dumps(fatiguing[:4], indent=2)}

EARLY DECLINE — monitor closely ({len(early)} creatives):
{json.dumps(early[:3], indent=2)}

HEALTHY: {len(healthy)} creatives performing normally.

Provide a refresh schedule and rotation strategy.
Which formats fatigue fastest? What's the typical half-life?
"""

        result = self.structured_output(prompt)
        result["agent_type"] = "fatigue"
        result["raw_counts"] = {
            "critical": len(critical),
            "fatiguing": len(fatiguing),
            "early_decline": len(early),
            "healthy": len(healthy),
        }
        result["critical_ids"] = [r["creative_id"] for r in critical[:5]]
        result.setdefault("confidence", 0.82)
        result.setdefault("recommendation", "refresh_now" if critical else "schedule_refresh")
        return result
