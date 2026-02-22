"""
MCP 10 — Creative Production Intelligence MCP

Automates the brief → creative concept → A/B test design pipeline.
Uses: Monte Carlo simulation for test sizing, decision tree for asset prioritisation.

Agents:
- ConceptGeneratorAgent   — Generates creative concepts from brief
- ABTestDesignAgent       — Statistical test design with Monte Carlo power analysis
- AssetPrioritiserAgent   — Decision tree for asset production order
- ProductionCritic        — Challenges creative and statistical assumptions
"""

import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from mcp.server.fastmcp import FastMCP
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent

mcp = FastMCP("Creative Production Intelligence MCP")

np.random.seed(29)

# Historical asset performance data for decision tree training
_asset_history = {
    "features": [
        # [format_video, duration_long, has_offer, target_age_young, high_spend_budget, high_priority]
        [1, 0, 1, 1, 1],  # video_short + offer + young = high priority
        [1, 1, 0, 0, 0],  # video_long + no offer + old = low
        [0, 0, 1, 1, 1],  # image + offer + young = high
        [0, 0, 0, 1, 0],  # image + no offer + young = medium
        [1, 0, 0, 0, 1],  # video_short + no offer + old = high (brand)
        [1, 1, 1, 0, 1],  # video_long + offer + old = high
        [0, 0, 0, 0, 0],  # image + no offer + old = low
        [1, 0, 1, 0, 1],  # video_short + offer + old = high
    ],
    "labels": ["high", "low", "high", "medium", "high", "high", "low", "high"]
}

_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
_tree.fit(_asset_history["features"], _asset_history["labels"])


def monte_carlo_power(baseline_cvr: float, min_detectable_effect: float,
                      alpha: float = 0.05, target_power: float = 0.80,
                      simulations: int = 5000) -> dict:
    """
    Monte Carlo simulation for A/B test sample size estimation.
    More robust than analytical formulas for non-normal distributions.
    """
    variant_cvr = baseline_cvr * (1 + min_detectable_effect)
    sample_sizes = range(100, 20001, 200)

    for n in sample_sizes:
        detections = 0
        for _ in range(simulations):
            ctrl = np.random.binomial(n, baseline_cvr) / n
            var = np.random.binomial(n, variant_cvr) / n
            # Simplified z-test
            pooled_p = (ctrl * n + var * n) / (2 * n)
            se = np.sqrt(2 * pooled_p * (1 - pooled_p) / n)
            z = abs(var - ctrl) / (se + 1e-8)
            if z > 1.96:  # alpha = 0.05
                detections += 1

        power = detections / simulations
        if power >= target_power:
            return {
                "required_sample_per_variant": n,
                "total_sample": n * 2,
                "simulated_power": round(power, 3),
                "baseline_cvr": baseline_cvr,
                "variant_cvr": round(variant_cvr, 4),
                "min_detectable_effect_pct": round(min_detectable_effect * 100, 1),
                "simulations_run": simulations,
            }

    return {"required_sample_per_variant": ">20000", "note": "Effect too small to detect efficiently"}


def prioritise_assets(assets: list) -> list:
    """Use decision tree to prioritise creative asset production order."""
    scored = []
    for asset in assets:
        features = [
            int(asset.get("format") == "video"),
            int(asset.get("duration", "") == "30s"),
            int(asset.get("has_offer", False)),
            int(asset.get("target_age_group", "") in ["18-24", "25-34"]),
            int(asset.get("budget", 0) > 10000),
        ]
        priority = _tree.predict([features])[0]
        prob = float(max(_tree.predict_proba([features])[0]))
        scored.append({**asset, "priority": priority, "confidence": round(prob, 3)})
    return sorted(scored, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])


_CONCEPT_SYSTEM = """You are a Creative Production Agent for digital advertising.
Generate creative concepts from briefs and design A/B testing structures.
Be specific: name the concept, describe the visual treatment, write the headline and CTA.
Output JSON: concepts (list of 3), recommended_test_structure, production_sequence,
confidence (0-1), signals, recommendation (produce_all/produce_top2/iterate_brief)."""

_CRITIC_SYSTEM = """You are a Creative Production Critic.
Challenge: creative concept differentiation, A/B test contamination, production timeline,
asset reuse opportunities, and whether concepts actually test different hypotheses.
Output JSON: concept_weaknesses, test_design_issues, critique_score (0-10)."""


class CreativeProductionAgent(BaseAgent):
    def __init__(self):
        super().__init__("CreativeProductionAgent", _CONCEPT_SYSTEM)

    def generate(self, brief: str, baseline_cvr: float = 0.10) -> dict:
        test_design = monte_carlo_power(baseline_cvr, 0.10)  # 10% lift

        sample_assets = [
            {"name": "Hero Video", "format": "video", "duration": "15s",
             "has_offer": True, "target_age_group": "25-34", "budget": 15000},
            {"name": "Static Image", "format": "image", "has_offer": True,
             "target_age_group": "35-44", "budget": 8000},
            {"name": "Carousel", "format": "carousel", "has_offer": False,
             "target_age_group": "18-24", "budget": 6000},
        ]
        prioritised = prioritise_assets(sample_assets)

        prompt = f"""Generate 3 creative concepts for this brief:

Brief: {brief}

A/B Test Design (Monte Carlo power analysis):
{json.dumps(test_design, indent=2)}

Prioritised Asset Production Order (Decision Tree):
{json.dumps(prioritised, indent=2)}

Create 3 distinct creative concepts that test different hypotheses.
Each should differ in: value proposition, visual treatment, or audience angle.
Specify headline, visual direction, and CTA for each."""
        result = self.structured_output(prompt)
        result["agent_type"] = "creative_production"
        result["test_design"] = test_design
        result["asset_priority"] = prioritised
        return result


class ProductionCritic(BaseAgent):
    def __init__(self):
        super().__init__("ProductionCritic", _CRITIC_SYSTEM)

    def critique(self, concepts: dict) -> dict:
        prompt = f"""Challenge these creative concepts and test design:
{json.dumps(concepts, indent=2)}
Are the concepts truly differentiated? Does the test design have statistical issues?"""
        return self.structured_output(prompt)


_agent = CreativeProductionAgent()
_critic = ProductionCritic()


@mcp.tool()
def generate_creative_concepts(brief: str, baseline_cvr: float = 0.10) -> str:
    """
    Generate 3 creative concepts from a brief, with A/B test design and asset prioritisation.
    Uses Monte Carlo simulation for sample size and Decision Tree for production ordering.

    Args:
        brief: Creative brief text
        baseline_cvr: Current conversion rate (e.g. 0.10 = 10%)
    """
    result = _agent.generate(brief, baseline_cvr)
    critique = _critic.critique(result)
    return json.dumps({"production_plan": result, "critique": critique}, indent=2, default=str)


@mcp.tool()
def design_ab_test(
    baseline_cvr: float, min_detectable_effect_pct: float = 10.0,
    target_power: float = 0.80,
) -> str:
    """
    Design an A/B test using Monte Carlo power analysis.

    Args:
        baseline_cvr: Current conversion rate (e.g. 0.10 = 10%)
        min_detectable_effect_pct: Minimum % lift to detect (e.g. 10.0 = 10%)
        target_power: Statistical power target (default 0.80 = 80%)
    """
    return json.dumps(
        monte_carlo_power(baseline_cvr, min_detectable_effect_pct / 100, target_power=target_power),
        indent=2
    )


@mcp.tool()
def prioritise_creative_assets(assets: list[dict]) -> str:
    """
    Prioritise creative asset production order using a Decision Tree classifier.

    Args:
        assets: List of asset dicts with keys: name, format, has_offer, target_age_group, budget
    """
    return json.dumps({"prioritised_assets": prioritise_assets(assets)}, indent=2)


if __name__ == "__main__":
    mcp.run()
