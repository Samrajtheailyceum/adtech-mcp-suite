"""
Creative Brief Generation Agent

Synthesizes all specialist agent outputs into a structured creative brief
that a design/copy team can act on directly.
"""

import json
from typing import Dict

from agents.base import BaseAgent


_SYSTEM = """You are a Creative Strategy Agent for programmatic advertising.

Your job:
- Translate performance data and agent insights into an actionable creative brief
- Connect data signals directly to creative decisions
- Give specific, concrete direction (not vague suggestions)
- Ground every recommendation in the evidence provided

Output JSON with:
  brief_title, campaign_objective, primary_audience, secondary_audience,
  creative_direction, copy_guidance, format_recommendations, do_list, dont_list,
  kpis_to_beat, confidence (0-1), signals (list), agent_type,
  recommendation (create_new/iterate/scale_existing)
"""


class BriefGenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("BriefGenerationAgent", _SYSTEM)

    def generate(self, agent_insights: Dict[str, Dict]) -> Dict:
        """Generate brief from the combined outputs of all specialist agents."""

        creative_recs = agent_insights.get("creative", {}).get("recommendations", [])
        top_segments  = agent_insights.get("audience", {}).get("top_segments", [])
        fatigue_recs  = agent_insights.get("fatigue", {}).get("recommendations", [])
        bid_context   = agent_insights.get("bidding", {}).get("bid_recommendations", [])
        features      = agent_insights.get("creative", {}).get("top_features", {})

        prompt = f"""Generate a creative brief from these performance intelligence inputs:

CREATIVE PERFORMANCE INSIGHTS (what's working):
{json.dumps(creative_recs if isinstance(creative_recs, list) else [str(creative_recs)], indent=2)}

TOP AUDIENCE SEGMENTS to target:
{json.dumps(top_segments[:3] if isinstance(top_segments, list) else [], indent=2)}

FATIGUE SIGNALS (what to avoid repeating):
{json.dumps(fatigue_recs if isinstance(fatigue_recs, list) else [str(fatigue_recs)], indent=2)}

BIDDING CONTEXT (when/where ads will run):
{json.dumps(bid_context if isinstance(bid_context, list) else [str(bid_context)], indent=2)}

TOP PERFORMANCE DRIVERS (RF feature importance):
{json.dumps(features, indent=2)}

Write a brief that a creative director can hand directly to designers and copywriters.
Be specific. Connect every instruction to the data.
"""

        result = self.structured_output(prompt)
        result["agent_type"] = "brief"
        result.setdefault("confidence", 0.78)
        result.setdefault("recommendation", "create_new")
        return result
