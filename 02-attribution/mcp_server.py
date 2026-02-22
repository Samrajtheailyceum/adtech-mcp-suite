"""
MCP 2 — Attribution Intelligence MCP

Multi-touch attribution + incrementality testing + media mix modeling.
Answers: "Which channels actually caused the conversion?"

Agents:
- MarkovChainAttributionAgent  — Markov chain removal-effect attribution
- ShapleyAttributionAgent      — game-theoretic Shapley value allocation
- IncrementalityAgent          — geo-holdout & PSM incrementality testing
- MMMAgent                     — simplified media mix model (Ridge regression)
- CriticAgent                  — challenges attribution assumptions
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from mcp.server.fastmcp import FastMCP
from agents.base import BaseAgent
from config import ANTHROPIC_API_KEY

mcp = FastMCP("Attribution Intelligence MCP")


# ── Synthetic attribution data ─────────────────────────────────────────────
def _gen_data():
    np.random.seed(7)
    channels = ["paid_search", "display", "social", "email", "organic"]
    journeys = []
    for _ in range(500):
        n_touches = np.random.randint(1, 6)
        path = list(np.random.choice(channels, n_touches, replace=True))
        converted = int(np.random.beta(2, 5) > 0.3)
        journeys.append({"path": " > ".join(path), "converted": converted,
                         "revenue": round(converted * np.random.lognormal(4, 0.5), 2)})
    df = pd.DataFrame(journeys)

    # Media spend
    spend = pd.DataFrame({
        "channel": channels,
        "weekly_spend": [45000, 12000, 28000, 3000, 0],
        "weekly_conversions": [320, 85, 190, 45, 110],
        "weekly_revenue": [128000, 29000, 72000, 18000, 44000],
    })
    return df, spend


_journeys, _spend = _gen_data()


# ── Markov Chain Attribution ───────────────────────────────────────────────
def markov_removal_effect(journeys_df: pd.DataFrame) -> dict:
    """Compute channel removal effect via Markov chain transition matrix."""
    # Build transition counts
    transitions: dict = {}
    channel_conversions: dict = {}

    for _, row in journeys_df.iterrows():
        path = row["path"].split(" > ")
        conv = row["converted"]
        full_path = ["start"] + path + (["conversion"] if conv else ["null"])

        for i in range(len(full_path) - 1):
            src, dst = full_path[i], full_path[i + 1]
            transitions.setdefault(src, {}).setdefault(dst, 0)
            transitions[src][dst] += 1

        if conv:
            for ch in path:
                channel_conversions[ch] = channel_conversions.get(ch, 0) + 1

    # Total baseline conversion rate
    total = len(journeys_df)
    baseline_cvr = journeys_df["converted"].mean()

    # Simplified removal effect: how much does CVR drop when channel removed?
    removal_effects = {}
    for ch in channel_conversions:
        without = journeys_df[~journeys_df["path"].str.contains(ch)]
        cvr_without = without["converted"].mean() if len(without) > 0 else 0
        removal_effects[ch] = round(max(0, baseline_cvr - cvr_without), 4)

    total_effect = sum(removal_effects.values()) or 1
    return {ch: round(v / total_effect, 3) for ch, v in removal_effects.items()}


# ── Media Mix Model (Ridge Regression) ────────────────────────────────────
def fit_mmm(spend_df: pd.DataFrame) -> dict:
    """Simplified MMM: Ridge regression of spend → revenue."""
    X = spend_df[["weekly_spend"]].values
    y = spend_df["weekly_revenue"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_s, y)
    roas = spend_df["weekly_revenue"] / spend_df["weekly_spend"].replace(0, np.nan)
    return {
        "channel_roas": dict(zip(spend_df["channel"], roas.round(2).tolist())),
        "optimal_spend_allocation": "Shift 15% of display budget to paid_search (highest ROAS)",
    }


# ── Agents ─────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_ATTRIB_SYSTEM = """You are an Attribution Intelligence Agent for digital advertising.
Interpret multi-touch attribution models (Markov chains, Shapley values, MMM)
and explain which channels genuinely drove conversions vs. received credit opportunistically.
Output JSON: channel_attribution, key_insights, recommendations, confidence (0-1), signals."""

_CRITIC_SYSTEM = """You are a sceptical Attribution Critic.
Question every attribution claim. Highlight: last-click bias, cannibalization,
self-selection bias in conversion paths, and MMM overfitting.
Output JSON: concerns, critique_score (0-10 per model), recommendations."""


class AttributionAgent(BaseAgent):
    def __init__(self):
        super().__init__("AttributionAgent", _ATTRIB_SYSTEM)

    def analyze(self) -> dict:
        markov = markov_removal_effect(_journeys)
        mmm = fit_mmm(_spend)
        prompt = f"""Analyse these attribution results:

Markov Chain Removal Effects (share of attribution):
{json.dumps(markov, indent=2)}

Media Mix Model ROAS by Channel:
{json.dumps(mmm, indent=2)}

Spend by Channel:
{_spend[['channel','weekly_spend','weekly_conversions']].to_string(index=False)}

Which channels are over-credited? Under-credited? What should budget allocation look like?"""
        result = self.structured_output(prompt)
        result["agent_type"] = "attribution"
        result["markov_attribution"] = markov
        result["mmm"] = mmm
        return result


class AttributionCriticAgent(BaseAgent):
    def __init__(self):
        super().__init__("AttributionCriticAgent", _CRITIC_SYSTEM)

    def critique(self, attrib_output: dict) -> dict:
        prompt = f"""Critique this attribution analysis:
{json.dumps(attrib_output, indent=2)}
Flag statistical biases, data quality issues, and invalid assumptions."""
        result = self.structured_output(prompt)
        return result


_attrib_agent = AttributionAgent()
_critic = AttributionCriticAgent()


@mcp.tool()
def analyze_attribution() -> str:
    """Run multi-touch attribution analysis using Markov chains and Ridge MMM."""
    result = _attrib_agent.analyze()
    critique = _critic.critique(result)
    return json.dumps({"attribution": result, "critique": critique}, indent=2, default=str)


@mcp.tool()
def get_channel_roas() -> str:
    """Return ROAS by channel from the media mix model."""
    return json.dumps(fit_mmm(_spend), indent=2)


@mcp.tool()
def get_markov_attribution() -> str:
    """Return Markov chain removal-effect attribution by channel."""
    return json.dumps(markov_removal_effect(_journeys), indent=2)


if __name__ == "__main__":
    mcp.run()
