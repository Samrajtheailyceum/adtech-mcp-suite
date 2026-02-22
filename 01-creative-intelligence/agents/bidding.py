"""
Bidding Dynamics Agent

Analyzes the intraday bid landscape to find optimal windows for
spend efficiency: peak win rate hours, low-competition windows,
and floor-price spread opportunities.
"""

import json
from typing import Dict

import numpy as np
import pandas as pd

from agents.base import BaseAgent


_SYSTEM = """You are a Bidding Dynamics Agent for programmatic advertising.

Your job:
- Identify optimal bid timing windows (highest win rate, lowest competition)
- Analyse floor price dynamics and bid spread opportunities
- Recommend bid multipliers by time-of-day
- Quantify expected improvement in win rate

Output JSON with:
  hourly_opportunities, bid_recommendations, floor_analysis,
  confidence (0-1), signals (list), agent_type,
  recommendation (increase_bids/decrease_bids/shift_budget/hold),
  magnitude (decimal, e.g. 0.15 = 15% expected win-rate improvement)
"""


class BiddingDynamicsAgent(BaseAgent):
    def __init__(self, data: Dict[str, pd.DataFrame]):
        super().__init__("BiddingDynamicsAgent", _SYSTEM)
        self.data = data

    def analyze(self) -> Dict:
        bids = self.data["bid_landscape"].copy()

        hourly = bids.groupby("hour").agg(
            win_rate=("win_rate", "mean"),
            avg_bid=("avg_bid", "mean"),
            competition=("competition_index", "mean"),
            floor=("floor_price", "mean"),
        ).round(3)

        # Bid value score = win_rate / (competition + 0.1) — finds sweet spots
        hourly["value_score"] = (hourly["win_rate"] / (hourly["competition"] + 0.1)).round(3)

        peak_win_hour = int(hourly["win_rate"].idxmax())
        best_value_hour = int(hourly["value_score"].idxmax())
        low_comp_hour = int(hourly["competition"].idxmin())

        top_hours = hourly.nlargest(3, "value_score").to_dict("index")
        worst_hours = hourly.nsmallest(3, "value_score").to_dict("index")

        # Floor spread analysis
        bids["floor_spread"] = bids["avg_bid"] - bids["floor_price"]
        avg_spread = round(float(bids["floor_spread"].mean()), 2)
        tight_spread_pct = round(
            float((bids["floor_spread"] < 0.20).mean()) * 100, 1
        )

        prompt = f"""Analyze this programmatic bid landscape:

Peak Win-Rate Hour: {peak_win_hour}:00
Best Value Hour (win_rate / competition): {best_value_hour}:00
Lowest Competition Hour: {low_comp_hour}:00

Top 3 Hours by Value Score:
{json.dumps(top_hours, indent=2)}

Worst 3 Hours (avoid or reduce bids):
{json.dumps(worst_hours, indent=2)}

Floor Price Analysis:
  Average bid-floor spread: ${avg_spread}
  Hours with tight spread (<$0.20): {tight_spread_pct}% — risk of floor-price loss

Recommend bid multipliers by hour and quantify expected win-rate improvement.
Include 'magnitude' as a decimal (e.g. 0.18 = 18% improvement expected).
"""

        result = self.structured_output(prompt)
        result["agent_type"] = "bidding"
        result.setdefault("confidence", 0.72)
        result.setdefault("magnitude", 0.12)
        return result
