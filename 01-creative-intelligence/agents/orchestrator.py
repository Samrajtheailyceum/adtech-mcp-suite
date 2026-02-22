"""
Orchestrator Agent — Master Coordinator

Controls the full analysis pipeline:
  1. Spawn all specialist agents
  2. Route their outputs to the Brief Agent for synthesis
  3. Route everything to the Critic Agent for adversarial review
  4. Feed all outputs into the Ensemble Engine for weighted aggregation
  5. Produce a final executive recommendation

This mirrors a management hierarchy: specialists → synthesis → critique → decision.
"""

import json
import time
from typing import Dict, Optional

import pandas as pd

from agents.base import BaseAgent
from agents.creative import CreativePerformanceAgent
from agents.audience import AudienceIntelligenceAgent
from agents.bidding import BiddingDynamicsAgent
from agents.fatigue_agent import FatigueDetectionAgent
from agents.brief import BriefGenerationAgent
from agents.critic import CriticAgent
from ml.classifier import CreativeClassifier
from ml.anomaly import AnomalyDetector
from ml.ensemble import EnsembleEngine


_SYSTEM = """You are the Master Orchestrator of an Ad Tech Intelligence System.

You receive outputs from 6 specialist agents, a critique from the Adversarial Critic,
and a weighted ensemble decision. Your job is to synthesize these into a coherent,
prioritised executive action plan.

Rules:
- Resolve conflicts using the ensemble confidence and critic scores
- Discount any agent recommendation with a critique_score > 7
- Always flag items with HIGH_UNCERTAINTY for human review
- Prioritise actions by expected business impact (ROAS > CPA > win rate)

Output JSON with:
  executive_summary, priority_actions (ranked list), resolved_conflicts,
  confidence_assessment, human_review_items, next_steps (with owner: creative/media/data)
"""


class OrchestratorAgent(BaseAgent):
    def __init__(self, data: Dict[str, pd.DataFrame], ml_models: Dict):
        super().__init__("OrchestratorAgent", _SYSTEM)

        classifier: CreativeClassifier = ml_models["classifier"]
        anomaly_detector: AnomalyDetector = ml_models["anomaly_detector"]

        self.ensemble = EnsembleEngine()
        self.agents = {
            "creative":  CreativePerformanceAgent(classifier, data),
            "audience":  AudienceIntelligenceAgent(data),
            "bidding":   BiddingDynamicsAgent(data),
            "fatigue":   FatigueDetectionAgent(data),
            "brief":     BriefGenerationAgent(),
            "critic":    CriticAgent(anomaly_detector),
        }
        self.data = data

    # ── Main pipeline ──────────────────────────────────────────────────────

    def run_full_analysis(self, campaign_id: Optional[str] = None) -> Dict:
        """
        Execute the full hierarchical multi-agent pipeline.

        Pipeline order matters:
          specialist agents (parallel-conceptually) →
          brief agent (needs specialist outputs) →
          critic agent (needs all outputs) →
          ensemble engine (needs all outputs + critique) →
          orchestrator synthesis (needs everything)
        """
        t0 = time.time()
        specialist_outputs: Dict[str, Dict] = {}

        # ── Stage 1: Specialist agents ─────────────────────────────────────
        print("  [1/5] Specialist agents running...")
        specialist_outputs["creative"] = self.agents["creative"].analyze(campaign_id)
        specialist_outputs["audience"] = self.agents["audience"].analyze()
        specialist_outputs["bidding"]  = self.agents["bidding"].analyze()
        specialist_outputs["fatigue"]  = self.agents["fatigue"].analyze()

        # ── Stage 2: Brief synthesis ───────────────────────────────────────
        print("  [2/5] Brief agent synthesising insights...")
        specialist_outputs["brief"] = self.agents["brief"].generate(specialist_outputs)

        # ── Stage 3: Adversarial critique ─────────────────────────────────
        print("  [3/5] Critic agent reviewing all outputs...")
        critique = self.agents["critic"].critique(specialist_outputs)

        # ── Stage 4: Ensemble aggregation ─────────────────────────────────
        print("  [4/5] Ensemble engine aggregating votes...")
        ensemble_inputs = [
            {
                "agent_type":  name,
                "recommendation": out.get("recommendation", "hold"),
                "confidence":  out.get("confidence", 0.5),
                "reasoning":   str(out.get("reasoning", "")),
                "signals":     out.get("signals", []),
                "magnitude":   out.get("magnitude", 0.0),
            }
            for name, out in specialist_outputs.items()
            if isinstance(out, dict)
        ]
        ensemble = self.ensemble.aggregate(ensemble_inputs)

        # ── Stage 5: Orchestrator synthesis ───────────────────────────────
        print("  [5/5] Orchestrator synthesising final recommendation...")
        final = self._synthesise(specialist_outputs, critique, ensemble, campaign_id)

        elapsed = round(time.time() - t0, 2)
        final.update({
            "execution_time_seconds": elapsed,
            "pipeline_stages": [
                "specialist_agents", "brief_synthesis",
                "adversarial_critique", "ensemble_voting",
                "orchestrator_decision",
            ],
            "agents_run": list(specialist_outputs.keys()) + ["critic", "orchestrator"],
        })

        return final

    # ── Private ────────────────────────────────────────────────────────────

    def _synthesise(
        self,
        outputs: Dict,
        critique: Dict,
        ensemble: Dict,
        campaign_id: Optional[str],
    ) -> Dict:
        agent_summaries = {
            name: {
                "recommendation": out.get("recommendation"),
                "confidence": out.get("confidence"),
            }
            for name, out in outputs.items()
            if isinstance(out, dict)
        }

        critique_scores = {}
        for item in critique.get("agent_critiques", []):
            if isinstance(item, dict):
                critique_scores[item.get("agent_name", "")] = item.get("critique_score", 0)

        prompt = f"""Synthesise these multi-agent outputs into a final executive recommendation:

ENSEMBLE DECISION:
{json.dumps(ensemble, indent=2)}

AGENT RECOMMENDATIONS & CONFIDENCE:
{json.dumps(agent_summaries, indent=2)}

CRITIC SCORES PER AGENT (0-10, higher = more problematic):
{json.dumps(critique_scores, indent=2)}

MAJOR CONCERNS FROM CRITIC:
{json.dumps(critique.get("major_concerns", []), indent=2)}

INTER-AGENT CONFLICTS:
{json.dumps(critique.get("detected_conflicts", []), indent=2)}

Campaign scope: {campaign_id or "all campaigns"}

Produce a prioritised executive action plan. Discard recommendations with
critique_score > 7. Flag HIGH_UNCERTAINTY items for human review.
Assign each next_step an owner: creative / media / data.
"""

        result = self.structured_output(prompt)
        result["ensemble_result"] = ensemble
        result["critique_summary"] = {
            "major_concerns": critique.get("major_concerns", []),
            "overall_quality_score": critique.get("overall_quality_score"),
            "anomaly_flags": critique.get("anomaly_flags", []),
        }
        result["agent_snapshots"] = agent_summaries
        return result
