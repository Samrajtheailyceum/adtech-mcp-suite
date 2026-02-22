"""
Audience Intelligence Agent

Computes segment-level performance (CTR, CPA, reach × efficiency)
and uses LLM reasoning to identify targeting opportunities.
"""

import json
from typing import Dict

import numpy as np
import pandas as pd

from agents.base import BaseAgent


_SYSTEM = """You are an Audience Intelligence Agent for programmatic advertising.

Your job:
- Identify high-value audience segments from performance data
- Surface audience-creative affinity patterns
- Flag segments with poor efficiency (high reach, low CTR = wasted budget)
- Recommend targeting adjustments with rationale

Output JSON with:
  top_segments, underperforming_segments, recommendations,
  confidence (0-1), signals (list), agent_type,
  recommendation (optimize/exclude/expand/test)
"""


class AudienceIntelligenceAgent(BaseAgent):
    def __init__(self, data: Dict[str, pd.DataFrame]):
        super().__init__("AudienceIntelligenceAgent", _SYSTEM)
        self.data = data

    def analyze(self) -> Dict:
        segs = self.data["segments"].copy()

        # Efficiency = CTR × log(Reach) — rewards high-CTR AND large-reach segments
        segs["efficiency"] = segs["ctr"] * np.log1p(segs["reach"])

        top = segs.nlargest(5, "efficiency")[
            ["age_group", "gender", "index", "reach", "ctr", "cpa"]
        ].round(4).to_dict("records")

        bottom = segs.nsmallest(5, "ctr")[
            ["age_group", "gender", "index", "reach", "ctr", "cpa"]
        ].round(4).to_dict("records")

        # Best single age×gender combo
        affinity = segs.groupby(["age_group", "gender"])["ctr"].mean()
        best_combo = affinity.idxmax()  # (age_group, gender)

        # High reach but low CTR = budget sink
        waste = segs[(segs["reach"] > segs["reach"].quantile(0.7)) &
                     (segs["ctr"] < segs["ctr"].quantile(0.3))]
        waste_summary = waste[["age_group", "gender", "reach", "ctr"]].head(3).to_dict("records")

        prompt = f"""Analyze audience segment performance data:

Top 5 Segments by Efficiency (CTR × log(Reach)):
{json.dumps(top, indent=2)}

Lowest CTR Segments:
{json.dumps(bottom, indent=2)}

Best Age × Gender Combo: {best_combo[1]} / {best_combo[0]}

High-Reach Low-CTR Segments (budget waste candidates):
{json.dumps(waste_summary, indent=2)}

Provide audience targeting recommendations. Flag segments to exclude, expand, or test.
"""

        result = self.structured_output(prompt)
        result["agent_type"] = "audience"
        result.setdefault("confidence", 0.70)
        return result
