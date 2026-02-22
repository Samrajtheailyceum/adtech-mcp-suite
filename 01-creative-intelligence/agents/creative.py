"""
Creative Performance Agent

Uses the Random Forest classifier to tier creatives and surfaces
the feature importance as an explainability layer for creative teams.
"""

import json
from typing import Dict, Optional

import pandas as pd

from agents.base import BaseAgent
from ml.classifier import CreativeClassifier


_SYSTEM = """You are a Creative Performance Intelligence Agent for programmatic advertising.

Your job:
- Interpret Random Forest classification outputs (Strong / Moderate / Weak tiers)
- Identify what separates top performers from the rest
- Surface actionable creative intelligence grounded in data
- Flag underperformers with likely root causes

Structure your analysis as JSON with these keys:
  summary, top_performers, underperformers, recommendations,
  confidence (0-1), signals (list of key findings), agent_type,
  recommendation (single action word: optimize/pause/scale/test)
"""


class CreativePerformanceAgent(BaseAgent):
    def __init__(self, classifier: CreativeClassifier, data: Dict[str, pd.DataFrame]):
        super().__init__("CreativePerformanceAgent", _SYSTEM)
        self.classifier = classifier
        self.data = data

    def analyze(self, campaign_id: Optional[str] = None) -> Dict:
        perf = self.data["performance"]
        if campaign_id:
            ids = self.data["creatives"][
                self.data["creatives"]["campaign_id"] == campaign_id
            ]["creative_id"].tolist()
            perf = perf[perf["creative_id"].isin(ids)]

        predictions = self.classifier.predict(perf)
        features = self.classifier.top_features()

        by_tier = {}
        for tier in ["Strong", "Moderate", "Weak"]:
            bucket = [p for p in predictions if p["tier"] == tier]
            by_tier[tier] = {
                "count": len(bucket),
                "avg_confidence": round(
                    sum(p["confidence"] for p in bucket) / max(len(bucket), 1), 3
                ),
                "examples": [p["creative_id"] for p in bucket[:3]],
            }

        prompt = f"""Interpret this Random Forest creative classification output:

Tier Distribution:
{json.dumps(by_tier, indent=2)}

Top 5 Strong Creatives:
{json.dumps([p for p in predictions if p["tier"] == "Strong"][:5], indent=2)}

Bottom 5 Weak Creatives:
{json.dumps([p for p in predictions if p["tier"] == "Weak"][:5], indent=2)}

Key Performance Drivers (RF Feature Importance):
{json.dumps(features, indent=2)}

Provide creative performance intelligence with actionable recommendations.
Focus on what separates Strong from Weak, and what the team should do next.
"""

        result = self.structured_output(prompt)
        result["agent_type"] = "creative_performance"
        result["ml_tier_summary"] = by_tier
        result["top_features"] = features
        result.setdefault("confidence", 0.75)
        return result
