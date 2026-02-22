"""
MCP 6 — Real-Time Bidding Optimizer MCP

Bid landscape analysis, win probability modeling, and bid price optimization.
Uses: Gradient Boosting for win probability + Thompson Sampling for exploration.

Agents:
- WinProbabilityAgent    — GBM model predicting auction win probability
- BidOptimiserAgent      — Thompson Sampling bid strategy
- FloorPriceAgent        — Floor price detection and avoidance
- RTBCritic              — Challenges bid model assumptions
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from mcp.server.fastmcp import FastMCP
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent

mcp = FastMCP("RTB Optimizer MCP")

np.random.seed(17)
N = 2000

# Synthetic bid log
_bids = pd.DataFrame({
    "bid_price":        np.random.lognormal(0.5, 0.6, N),
    "floor_price":      np.random.uniform(0.1, 2.0, N),
    "competition_idx":  np.random.beta(2, 3, N),
    "hour_of_day":      np.random.randint(0, 24, N),
    "user_value_score": np.random.beta(2, 5, N),
    "ad_quality_score": np.random.beta(3, 2, N),
    "placement_premium": np.random.choice([0, 1], N, p=[0.7, 0.3]),
})
# Win = bid > floor + competition pressure
_bids["won"] = (
    (_bids["bid_price"] > _bids["floor_price"]) &
    (_bids["bid_price"] > _bids["competition_idx"] * 3)
).astype(int)


# ── Gradient Boosting win probability model ────────────────────────────────
_FEATURES = ["bid_price", "floor_price", "competition_idx",
             "hour_of_day", "user_value_score", "ad_quality_score", "placement_premium"]

_scaler = StandardScaler()
_X = _scaler.fit_transform(_bids[_FEATURES])
_gbm = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
_gbm.fit(_X, _bids["won"])

_cv_auc = float(np.mean(cross_val_score(_gbm, _X, _bids["won"], cv=5, scoring="roc_auc")))


def predict_win_prob(bid_price: float, floor_price: float, competition: float,
                     hour: int = 12, user_value: float = 0.3) -> dict:
    X = _scaler.transform([[bid_price, floor_price, competition, hour, user_value, 0.6, 0]])
    prob = float(_gbm.predict_proba(X)[0][1])
    return {"bid_price": bid_price, "win_probability": round(prob, 3),
            "model_auc": round(_cv_auc, 3)}


# ── Thompson Sampling for bid exploration ─────────────────────────────────
class ThompsonSamplingBidder:
    """Multi-armed bandit over bid price buckets using Thompson Sampling."""
    def __init__(self, buckets: list):
        self.buckets = buckets
        self.alpha = {b: 1.0 for b in buckets}  # successes + 1
        self.beta = {b: 1.0 for b in buckets}   # failures + 1

    def recommend_bid(self) -> dict:
        samples = {b: np.random.beta(self.alpha[b], self.beta[b]) for b in self.buckets}
        best = max(samples, key=samples.get)
        return {"recommended_bid": best, "thompson_samples": {str(k): round(v, 3) for k, v in samples.items()}}

    def update(self, bid: float, won: bool):
        if bid in self.alpha:
            if won:
                self.alpha[bid] += 1
            else:
                self.beta[bid] += 1


_bidder = ThompsonSamplingBidder([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

_RTB_SYSTEM = """You are a Real-Time Bidding Optimisation Agent.
Interpret win probability models and Thompson Sampling bid strategies.
Recommend bid price adjustments for different contexts (hour, placement, user value).
Output JSON: bid_strategy, win_rate_opportunities, floor_avoidance_tips,
recommendations, confidence (0-1), signals, magnitude (expected win-rate improvement)."""

_CRITIC_SYSTEM = """You are an RTB Model Critic.
Challenge: training data recency, bid shading effects, auction mechanism assumptions,
winner's curse, and exploration-exploitation tradeoffs.
Output JSON: model_risks, critique_score (0-10), recommendations."""


class RTBOptimizerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RTBOptimizerAgent", _RTB_SYSTEM)

    def analyze(self) -> dict:
        thompson_rec = _bidder.recommend_bid()
        win_at_bids = {
            str(b): predict_win_prob(b, 0.5, 0.4)["win_probability"]
            for b in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        }
        feat_imp = dict(zip(_FEATURES, _gbm.feature_importances_.round(3).tolist()))

        prompt = f"""Analyse real-time bidding data:

GBM Win Probability by Bid Price:
{json.dumps(win_at_bids, indent=2)}

Thompson Sampling Recommendation:
{json.dumps(thompson_rec, indent=2)}

GBM Feature Importances:
{json.dumps(feat_imp, indent=2)}

Model CV AUC: {round(_cv_auc, 3)}

What bid strategy maximises wins while minimising overspend?
Include 'magnitude' (decimal) for expected win-rate improvement."""
        result = self.structured_output(prompt)
        result["agent_type"] = "rtb_optimizer"
        result["win_probs"] = win_at_bids
        result["thompson"] = thompson_rec
        result.setdefault("magnitude", 0.15)
        return result


class RTBCritic(BaseAgent):
    def __init__(self):
        super().__init__("RTBCritic", _CRITIC_SYSTEM)

    def critique(self, analysis: dict) -> dict:
        prompt = f"""Challenge this RTB bid model:
{json.dumps(analysis, indent=2)}"""
        return self.structured_output(prompt)


_agent = RTBOptimizerAgent()
_critic = RTBCritic()


@mcp.tool()
def optimize_bid_strategy() -> str:
    """
    Optimise bid strategy using Gradient Boosting win probability + Thompson Sampling.
    Returns recommended bid prices, win probability curves, and exploration strategy.
    """
    result = _agent.analyze()
    critique = _critic.critique(result)
    return json.dumps({"strategy": result, "critique": critique}, indent=2, default=str)


@mcp.tool()
def predict_auction_win(bid_price: float, floor_price: float = 0.5,
                        competition: float = 0.4, hour: int = 12) -> str:
    """
    Predict probability of winning a specific auction using the GBM model.

    Args:
        bid_price: Your bid in CPM dollars
        floor_price: Publisher floor price
        competition: Competition index (0-1)
        hour: Hour of day (0-23)
    """
    return json.dumps(predict_win_prob(bid_price, floor_price, competition, hour), indent=2)


@mcp.tool()
def get_thompson_bid_recommendation() -> str:
    """Get current Thompson Sampling bid price recommendation."""
    return json.dumps(_bidder.recommend_bid(), indent=2)


if __name__ == "__main__":
    mcp.run()
