"""
Ad Tech Intelligence MCP Server

Exposes the full multi-agent system as MCP tools.
Run with: python mcp_server.py
"""

import json
from mcp.server.fastmcp import FastMCP

from data.synthetic import generate_campaign_data
from ml.classifier import CreativeClassifier
from ml.anomaly import AnomalyDetector
from agents.orchestrator import OrchestratorAgent

mcp = FastMCP("AdTech Intelligence MCP")

# Lazy-initialised singleton
_orchestrator: OrchestratorAgent = None


def _init() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    print("[AdTech MCP] Initialising system...")
    data = generate_campaign_data(n_creatives=50, n_days=30)

    classifier = CreativeClassifier()
    train_result = classifier.train(data["performance"])
    print(f"  ✓ Random Forest trained — CV accuracy: {train_result['cv_accuracy']:.1%}")

    anomaly = AnomalyDetector()
    anomaly.fit_performance_detector(data["performance"])
    anomaly.fit_bid_detector(data["bid_landscape"])
    print("  ✓ Isolation Forest fitted")

    _orchestrator = OrchestratorAgent(data, {"classifier": classifier, "anomaly_detector": anomaly})
    print("  ✓ Agent hierarchy ready\n")
    return _orchestrator


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def run_full_analysis(campaign_id: str = None) -> str:
    """
    Run the complete multi-agent ad tech analysis pipeline.

    Executes all 6 specialist agents (Creative, Audience, Bidding, Fatigue, Brief, Critic)
    coordinated by the Orchestrator. Uses Random Forest classification, Isolation Forest
    anomaly detection, and weighted ensemble voting to produce a final recommendation.

    Args:
        campaign_id: Optional filter (e.g. "CAMP_001"). Analyses all campaigns if omitted.
    """
    orch = _init()
    print(f"\n[AdTech MCP] run_full_analysis(campaign_id={campaign_id})")
    result = orch.run_full_analysis(campaign_id)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def analyze_creative_performance(campaign_id: str = None) -> str:
    """
    Classify creatives into Strong / Moderate / Weak tiers using Random Forest.
    Returns feature importance, confidence scores, and actionable recommendations.
    """
    orch = _init()
    result = orch.agents["creative"].analyze(campaign_id)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def analyze_audience_segments() -> str:
    """
    Identify high-value audience segments using efficiency scoring (CTR × log(Reach)).
    Flags budget-wasting segments and recommends targeting adjustments.
    """
    orch = _init()
    result = orch.agents["audience"].analyze()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def analyze_bid_landscape() -> str:
    """
    Analyse the programmatic bid landscape for win-rate optimisation.
    Finds optimal bidding hours, competition pressure, and floor-price dynamics.
    """
    orch = _init()
    result = orch.agents["bidding"].analyze()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def detect_creative_fatigue() -> str:
    """
    Detect creative fatigue using exponential decay fitting + CUSUM change-point detection.
    Returns critical/fatiguing/healthy classification with days-until-fatigue predictions.
    """
    orch = _init()
    result = orch.agents["fatigue"].analyze()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def generate_creative_brief() -> str:
    """
    Generate a data-driven creative brief by synthesising all agent insights.
    Translates performance data into specific direction for designers and copywriters.
    """
    orch = _init()
    insights = {
        "creative": orch.agents["creative"].analyze(),
        "audience": orch.agents["audience"].analyze(),
        "fatigue":  orch.agents["fatigue"].analyze(),
        "bidding":  orch.agents["bidding"].analyze(),
    }
    result = orch.agents["brief"].generate(insights)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def run_adversarial_critique() -> str:
    """
    Run the Critic Agent against all specialist agent outputs.
    Uses Isolation Forest anomaly detection + adversarial LLM prompting to flag
    weak assumptions, conflicts, and low-confidence claims.
    """
    orch = _init()
    outputs = {
        "creative": orch.agents["creative"].analyze(),
        "audience": orch.agents["audience"].analyze(),
        "bidding":  orch.agents["bidding"].analyze(),
        "fatigue":  orch.agents["fatigue"].analyze(),
    }
    result = orch.agents["critic"].critique(outputs)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_system_status() -> str:
    """Return the initialisation status of all agents and ML models."""
    global _orchestrator
    if _orchestrator is None:
        return json.dumps({"status": "not_initialised",
                           "message": "Call any analysis tool to initialise."})
    return json.dumps({
        "status": "ready",
        "algorithms": [
            "Random Forest — creative tier classification (200 trees, 5-fold CV)",
            "Isolation Forest — performance + bid anomaly detection",
            "Exponential Decay Fitting — creative fatigue curves",
            "CUSUM Change-Point Detection — fatigue onset timing",
            "Weighted Ensemble Voting — multi-agent aggregation",
        ],
        "agents": list(_orchestrator.agents.keys()) + ["orchestrator"],
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
