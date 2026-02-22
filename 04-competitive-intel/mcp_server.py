"""
MCP 4 — Competitive Intelligence MCP

Tracks competitor ad activity, creative themes, and spend share.
Uses: TF-IDF creative clustering + competitive positioning matrix.

Agents:
- SpendShareAgent       — Estimates SOV (share of voice) by brand
- CreativeThemeAgent    — Clusters competitor creative messaging with KMeans
- PositioningAgent      — Maps competitive whitespace for differentiation
- CompetitiveCritic     — Questions intelligence freshness and reliability
"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mcp.server.fastmcp import FastMCP
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent

mcp = FastMCP("Competitive Intelligence MCP")


# ── Synthetic competitive data ─────────────────────────────────────────────
np.random.seed(11)
BRANDS = ["OurBrand", "Competitor_A", "Competitor_B", "Competitor_C", "Competitor_D"]
CHANNELS = ["search", "display", "social", "video", "ctv"]

_spend_df = pd.DataFrame([
    {"brand": b, "channel": c,
     "est_weekly_spend": round(np.random.lognormal(9, 0.7)),
     "est_impressions": round(np.random.lognormal(15, 0.8)),
     "avg_cpm": round(np.random.uniform(2, 25), 2)}
    for b in BRANDS for c in CHANNELS
])

_creatives = [
    {"brand": "OurBrand",      "headline": "Performance you can measure — real ROAS guaranteed"},
    {"brand": "OurBrand",      "headline": "The smartest way to grow your brand online"},
    {"brand": "Competitor_A",  "headline": "Reach millions with precision targeting"},
    {"brand": "Competitor_A",  "headline": "Data-driven campaigns at scale"},
    {"brand": "Competitor_B",  "headline": "Affordable advertising for every business size"},
    {"brand": "Competitor_B",  "headline": "Start your campaign for free today"},
    {"brand": "Competitor_C",  "headline": "AI-powered ad optimization 24/7"},
    {"brand": "Competitor_C",  "headline": "Beat your competitors with intelligent bidding"},
    {"brand": "Competitor_D",  "headline": "Premium inventory premium results"},
    {"brand": "Competitor_D",  "headline": "Brand-safe environments only"},
]
_creative_df = pd.DataFrame(_creatives)

_INTEL_SYSTEM = """You are a Competitive Intelligence Agent for advertising.
Analyse competitor spend share, creative messaging, and positioning gaps.
Identify whitespace opportunities our brand can exploit.
Output JSON: sov_analysis, competitor_themes, whitespace_opportunities,
recommendations, confidence (0-1), signals."""

_CRITIC_SYSTEM = """You are a Competitive Intelligence Critic.
Challenge the reliability of estimated spend data. Flag: sampling bias,
recency issues, self-reported vs. actual spend gaps, and correlation != causation.
Output JSON: data_reliability_concerns, critique_score (0-10), recommendations."""


# ── KMeans creative clustering ─────────────────────────────────────────────
def cluster_creatives(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["headline"])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels

    clusters = {}
    for c in range(n_clusters):
        cluster_brands = df[df["cluster"] == c]["brand"].tolist()
        cluster_headlines = df[df["cluster"] == c]["headline"].tolist()
        clusters[f"cluster_{c}"] = {
            "brands": cluster_brands,
            "headlines": cluster_headlines,
            "theme": f"Cluster {c} messaging theme",
        }
    return clusters


def sov_by_channel(spend_df: pd.DataFrame) -> dict:
    sov = spend_df.groupby(["channel", "brand"])["est_weekly_spend"].sum().reset_index()
    total = sov.groupby("channel")["est_weekly_spend"].transform("sum")
    sov["sov"] = (sov["est_weekly_spend"] / total * 100).round(1)
    return sov.pivot(index="channel", columns="brand", values="sov").fillna(0).round(1).to_dict()


class CompetitiveIntelAgent(BaseAgent):
    def __init__(self):
        super().__init__("CompetitiveIntelAgent", _INTEL_SYSTEM)

    def analyze(self) -> dict:
        sov = sov_by_channel(_spend_df)
        clusters = cluster_creatives(_creative_df)
        prompt = f"""Analyse competitive intelligence data:

Share of Voice by Channel:
{json.dumps(sov, indent=2)}

Creative Theme Clusters (KMeans on TF-IDF):
{json.dumps(clusters, indent=2)}

Identify: where is OurBrand under-indexed? What messaging angles are unclaimed?
What should we do differently from competitors?"""
        result = self.structured_output(prompt)
        result["agent_type"] = "competitive_intel"
        result["sov"] = sov
        result["creative_clusters"] = clusters
        return result


class CompetitiveCritic(BaseAgent):
    def __init__(self):
        super().__init__("CompetitiveCritic", _CRITIC_SYSTEM)

    def critique(self, analysis: dict) -> dict:
        prompt = f"""Critique this competitive intelligence analysis:
{json.dumps(analysis, indent=2)}
How reliable is the spend estimation? What's missing?"""
        return self.structured_output(prompt)


_agent = CompetitiveIntelAgent()
_critic = CompetitiveCritic()


@mcp.tool()
def analyze_competitive_landscape() -> str:
    """
    Analyse competitor spend share, creative messaging clusters, and positioning gaps.
    Uses KMeans clustering on TF-IDF creative vectors to identify messaging themes.
    """
    result = _agent.analyze()
    critique = _critic.critique(result)
    return json.dumps({"analysis": result, "critique": critique}, indent=2, default=str)


@mcp.tool()
def get_share_of_voice() -> str:
    """Return estimated share of voice by channel and brand."""
    return json.dumps(sov_by_channel(_spend_df), indent=2)


@mcp.tool()
def cluster_competitor_creatives(n_clusters: int = 3) -> str:
    """
    Cluster competitor creative headlines using KMeans + TF-IDF to identify messaging themes.

    Args:
        n_clusters: Number of theme clusters (default 3)
    """
    return json.dumps(cluster_creatives(_creative_df, n_clusters), indent=2)


if __name__ == "__main__":
    mcp.run()
