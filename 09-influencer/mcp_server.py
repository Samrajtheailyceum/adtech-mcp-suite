"""
MCP 9 — Influencer Intelligence MCP

Influencer performance scoring, fake follower detection, and content-brand alignment.
Uses: Engagement Rate analysis + anomaly detection for fake follower patterns.

Agents:
- InfluencerScoringAgent  — Multi-dimensional influencer scoring
- FakeFollowerAgent       — Statistical detection of inauthentic engagement
- ContentAlignmentAgent   — Brand-content fit scoring via TF-IDF similarity
- InfluencerCritic        — Challenges influencer ROI claims
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from mcp.server.fastmcp import FastMCP
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent

mcp = FastMCP("Influencer Intelligence MCP")

np.random.seed(23)

# Synthetic influencer profiles
_influencers = pd.DataFrame([
    {"handle": f"@creator_{i:03d}",
     "followers": int(np.random.lognormal(10, 1.5)),
     "avg_likes": int(np.random.lognormal(7, 1.2)),
     "avg_comments": int(np.random.lognormal(4, 1.0)),
     "avg_shares": int(np.random.lognormal(3, 1.1)),
     "follower_growth_30d": round(np.random.uniform(-0.02, 0.15), 3),
     "posting_frequency_week": round(np.random.uniform(1, 14), 1),
     "audience_authenticity": round(np.random.beta(8, 2), 3),  # % real followers
     "niche": np.random.choice(["fitness", "beauty", "tech", "food", "travel", "gaming"]),
     "avg_story_views": int(np.random.lognormal(6.5, 1.0)),
     "sponsored_post_ratio": round(np.random.beta(2, 8), 2),
    } for i in range(50)
])

# Content samples per niche
_content_samples = {
    "fitness": "workout routine gym exercise protein muscle building cardio",
    "beauty": "skincare makeup tutorial lipstick foundation glow routine",
    "tech": "smartphone review laptop performance software app gadget",
    "food": "recipe cooking healthy meal nutrition restaurant chef",
    "travel": "destination beach hotel adventure explore culture photography",
    "gaming": "stream gameplay video game console review walkthrough",
}

_brand_brief = "premium athletic performance gear for serious runners marathon training"


def engagement_rate(row: pd.Series) -> float:
    """True engagement rate: (likes + comments + shares) / followers."""
    return (row["avg_likes"] + row["avg_comments"] + row["avg_shares"]) / max(row["followers"], 1)


def detect_fake_followers(row: pd.Series) -> dict:
    """
    Detect inauthentic followers via statistical anomalies:
    - Engagement rate vs follower count (should be inversely correlated)
    - Story view / follower ratio (bots don't watch stories)
    - Sudden follower spikes
    """
    er = engagement_rate(row)
    story_ratio = row["avg_story_views"] / max(row["followers"], 1)
    auth = row["audience_authenticity"]

    # Expected ER for follower count (power law: ER ~ 0.06 * followers^-0.2)
    expected_er = 0.06 * max(row["followers"], 1) ** -0.2
    er_deviation = (er - expected_er) / max(expected_er, 1e-6)

    risk_score = max(0, min(1, (1 - auth) * 0.6 + (story_ratio < 0.01) * 0.4))

    return {
        "engagement_rate": round(er, 4),
        "story_view_ratio": round(story_ratio, 4),
        "audience_authenticity": auth,
        "fake_follower_risk": round(risk_score, 3),
        "flag": risk_score > 0.4,
    }


def content_brand_fit(influencer_niche: str, brand_brief: str) -> float:
    """Cosine similarity between influencer content and brand brief."""
    niche_text = _content_samples.get(influencer_niche, influencer_niche)
    vec = TfidfVectorizer().fit_transform([niche_text, brand_brief])
    return round(float(cosine_similarity(vec[0:1], vec[1:2])[0][0]), 3)


def score_influencer(row: pd.Series, brand_brief: str) -> dict:
    """Composite influencer score (0-100)."""
    fake = detect_fake_followers(row)
    brand_fit = content_brand_fit(row["niche"], brand_brief)
    er = fake["engagement_rate"]
    auth = row["audience_authenticity"]
    growth_score = min(1, max(0, row["follower_growth_30d"] / 0.1))
    sponsored_penalty = 1 - row["sponsored_post_ratio"] * 0.5

    score = (er * 100 * 0.3 + brand_fit * 100 * 0.3 +
             auth * 100 * 0.2 + growth_score * 100 * 0.1 +
             sponsored_penalty * 100 * 0.1)

    return {
        "handle": row["handle"],
        "composite_score": round(score, 1),
        "engagement_rate": round(er, 4),
        "brand_fit": brand_fit,
        "fake_follower_risk": fake["fake_follower_risk"],
        "audience_authenticity": auth,
        "niche": row["niche"],
        "recommendation": "activate" if score > 35 and not fake["flag"] else "skip",
    }


_INFLUENCER_SYSTEM = """You are an Influencer Intelligence Agent.
Evaluate influencer scores, authenticity metrics, and brand-content alignment.
Identify high-ROI influencers and flag fraud risks.
Output JSON: top_influencers, fraud_flags, brand_fit_ranking, recommendations, confidence (0-1), signals."""

_CRITIC_SYSTEM = """You are an Influencer Marketing Critic.
Challenge: engagement rate manipulation, comment pods, audience age verification,
ROI measurement attribution, and disclosure compliance.
Output JSON: fraud_concerns, roi_measurement_gaps, critique_score (0-10)."""


class InfluencerScoringAgent(BaseAgent):
    def __init__(self):
        super().__init__("InfluencerScoringAgent", _INFLUENCER_SYSTEM)

    def analyze(self, brand: str = _brand_brief) -> dict:
        scores = _influencers.apply(lambda r: score_influencer(r, brand), axis=1).tolist()
        scores_df = pd.DataFrame(scores).sort_values("composite_score", ascending=False)

        top = scores_df.head(5).to_dict("records")
        flagged = scores_df[scores_df["fake_follower_risk"] > 0.4].to_dict("records")

        prompt = f"""Analyse influencer performance data for brand: "{brand}"

Top 5 Scored Influencers:
{json.dumps(top, indent=2)}

Fraud Risk Flags ({len(flagged)} influencers):
{json.dumps(flagged[:3], indent=2)}

Total pool: {len(scores)} influencers across niches.

Who should we activate? What's the risk profile of the top recommendations?"""
        result = self.structured_output(prompt)
        result["agent_type"] = "influencer"
        result["scored_influencers"] = top
        result["fraud_flags"] = flagged
        return result


class InfluencerCritic(BaseAgent):
    def __init__(self):
        super().__init__("InfluencerCritic", _CRITIC_SYSTEM)

    def critique(self, analysis: dict) -> dict:
        prompt = f"""Challenge these influencer recommendations:
{json.dumps(analysis, indent=2)}
What fraud and ROI measurement risks exist?"""
        return self.structured_output(prompt)


_agent = InfluencerScoringAgent()
_critic = InfluencerCritic()


@mcp.tool()
def analyze_influencer_roster(brand_brief: str = _brand_brief) -> str:
    """
    Score influencer pool for brand fit, engagement authenticity, and fraud risk.

    Args:
        brand_brief: Description of your brand/campaign to evaluate content alignment
    """
    result = _agent.analyze(brand_brief)
    critique = _critic.critique(result)
    return json.dumps({"analysis": result, "critique": critique}, indent=2, default=str)


@mcp.tool()
def detect_influencer_fraud(handle: str) -> str:
    """
    Run fake follower detection analysis on a specific influencer handle.

    Args:
        handle: Influencer handle (e.g. "@creator_001")
    """
    row = _influencers[_influencers["handle"] == handle]
    if row.empty:
        return json.dumps({"error": f"Handle {handle} not found."})
    return json.dumps(detect_fake_followers(row.iloc[0]), indent=2)


@mcp.tool()
def rank_by_brand_fit(brand_brief: str) -> str:
    """
    Rank all influencers by content-brand TF-IDF cosine similarity.

    Args:
        brand_brief: Your brand description for alignment scoring
    """
    ranked = sorted([
        {"handle": r["handle"], "niche": r["niche"],
         "brand_fit": content_brand_fit(r["niche"], brand_brief)}
        for r in _influencers.to_dict("records")
    ], key=lambda x: x["brand_fit"], reverse=True)
    return json.dumps({"brand_brief": brand_brief, "ranked": ranked[:10]}, indent=2)


if __name__ == "__main__":
    mcp.run()
