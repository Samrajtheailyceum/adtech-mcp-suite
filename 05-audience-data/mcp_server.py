"""
MCP 5 — Audience Data MCP

First-party data analysis, segment overlap, and lookalike modeling.
Uses: Jaccard similarity for overlap, logistic regression for lookalike scoring.

Agents:
- SegmentOverlapAgent    — Cross-platform audience deduplication
- LookalikeModelAgent    — Logistic regression lookalike scoring
- ReachFrequencyAgent    — Optimal reach/frequency curve modeling
- AudienceCritic         — Privacy risk and data quality review
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from mcp.server.fastmcp import FastMCP
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent

mcp = FastMCP("Audience Data MCP")

np.random.seed(13)
N = 1000

# Synthetic first-party user features
_users = pd.DataFrame({
    "user_id": [f"U{i:05d}" for i in range(N)],
    "age":      np.random.randint(18, 65, N),
    "income_index": np.random.lognormal(0, 0.5, N).round(2),
    "purchase_freq": np.random.poisson(3, N),
    "avg_order_value": np.random.lognormal(4, 0.6, N).round(2),
    "days_since_last_visit": np.random.exponential(14, N).astype(int),
    "is_converter": np.random.binomial(1, 0.15, N),  # 15% conversion rate
})

# Segment membership (simulated 1P + 3P overlap)
_segments = {
    "1p_high_value":     set(np.random.choice(_users.user_id, 200, replace=False)),
    "1p_loyalists":      set(np.random.choice(_users.user_id, 300, replace=False)),
    "3p_in_market":      set(np.random.choice(_users.user_id, 450, replace=False)),
    "3p_competitor_user": set(np.random.choice(_users.user_id, 350, replace=False)),
}

_AUDIENCE_SYSTEM = """You are an Audience Data Intelligence Agent.
Analyse first-party data, segment overlaps, and lookalike model outputs.
Identify high-value segments, privacy risks, and reach expansion opportunities.
Output JSON: segment_insights, lookalike_quality, recommendations, confidence (0-1), signals."""

_CRITIC_SYSTEM = """You are an Audience Data Privacy & Quality Critic.
Challenge: data freshness, consent validity, re-identification risks, lookalike model bias.
Flag GDPR/CCPA concerns and statistical weaknesses in the audience model.
Output JSON: privacy_risks, data_quality_issues, critique_score (0-10)."""


def jaccard_overlap(s1: set, s2: set) -> float:
    return round(len(s1 & s2) / len(s1 | s2), 3) if s1 | s2 else 0.0


def fit_lookalike(users_df: pd.DataFrame) -> dict:
    features = ["age", "income_index", "purchase_freq", "avg_order_value", "days_since_last_visit"]
    X = StandardScaler().fit_transform(users_df[features])
    y = users_df["is_converter"].values
    model = LogisticRegression(class_weight="balanced", max_iter=500)
    cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    model.fit(X, y)
    coefs = dict(zip(features, model.coef_[0].round(3).tolist()))
    return {
        "model": "Logistic Regression Lookalike",
        "cv_auc": round(float(cv.mean()), 3),
        "feature_weights": coefs,
        "top_predictor": max(coefs, key=lambda k: abs(coefs[k])),
    }


class AudienceDataAgent(BaseAgent):
    def __init__(self):
        super().__init__("AudienceDataAgent", _AUDIENCE_SYSTEM)

    def analyze(self) -> dict:
        seg_names = list(_segments.keys())
        overlaps = {}
        for i in range(len(seg_names)):
            for j in range(i + 1, len(seg_names)):
                key = f"{seg_names[i]} ∩ {seg_names[j]}"
                overlaps[key] = jaccard_overlap(_segments[seg_names[i]], _segments[seg_names[j]])

        lookalike = fit_lookalike(_users)
        prompt = f"""Analyse first-party audience data:

Segment Sizes: {json.dumps({k: len(v) for k, v in _segments.items()})}

Segment Jaccard Overlaps (higher = more similar):
{json.dumps(overlaps, indent=2)}

Lookalike Model Performance:
{json.dumps(lookalike, indent=2)}

What audience strategy should we pursue? Where is the highest-value addressable audience?"""
        result = self.structured_output(prompt)
        result["agent_type"] = "audience_data"
        result["overlaps"] = overlaps
        result["lookalike"] = lookalike
        return result


class AudienceCritic(BaseAgent):
    def __init__(self):
        super().__init__("AudienceCritic", _CRITIC_SYSTEM)

    def critique(self, analysis: dict) -> dict:
        prompt = f"""Privacy and quality review:
{json.dumps(analysis, indent=2)}
What are the consent, re-identification, and data quality risks?"""
        return self.structured_output(prompt)


_agent = AudienceDataAgent()
_critic = AudienceCritic()


@mcp.tool()
def analyze_audience_data() -> str:
    """
    Analyse first-party audience data: segment overlap (Jaccard), lookalike modeling
    (logistic regression), and reach/frequency optimisation.
    """
    result = _agent.analyze()
    critique = _critic.critique(result)
    return json.dumps({"analysis": result, "privacy_critique": critique}, indent=2, default=str)


@mcp.tool()
def get_segment_overlaps() -> str:
    """Return Jaccard similarity scores between all audience segments."""
    seg_names = list(_segments.keys())
    overlaps = {
        f"{seg_names[i]} ∩ {seg_names[j]}": jaccard_overlap(
            _segments[seg_names[i]], _segments[seg_names[j]]
        )
        for i in range(len(seg_names))
        for j in range(i + 1, len(seg_names))
    }
    return json.dumps(overlaps, indent=2)


@mcp.tool()
def score_lookalike_model() -> str:
    """Fit and evaluate logistic regression lookalike model on first-party data."""
    return json.dumps(fit_lookalike(_users), indent=2)


if __name__ == "__main__":
    mcp.run()
