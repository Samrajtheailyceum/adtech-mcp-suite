"""
Weighted Ensemble Decision Engine

Aggregates recommendations from all specialist agents using confidence-weighted
soft voting — the same principle as Random Forest's ensemble but applied to
agent outputs instead of decision trees.

Key properties:
- Each agent has a prior weight (tunable from historical accuracy)
- Each vote is scaled by the agent's stated confidence
- Agreement score quantifies how aligned agents are
- High uncertainty triggers a human-review flag
"""

from collections import Counter
from typing import Any, Dict, List


# Prior weights per agent type — reflects relative domain precision
# (fatigue signals are high-precision; brief agent is more subjective)
AGENT_WEIGHTS: Dict[str, float] = {
    "creative_performance": 1.2,
    "audience":             1.0,
    "bidding":              1.1,
    "fatigue":              1.3,
    "brief":                0.7,
}


class EnsembleEngine:
    """Confidence-weighted soft voting across agent recommendations."""

    def aggregate(self, agent_outputs: List[Dict[str, Any]]) -> Dict:
        """
        Aggregate agent recommendations into a single weighted decision.

        Each output must contain:
          - agent_type: str
          - recommendation: str
          - confidence: float [0,1]
          - signals: List[str]
          - reasoning: str
          - magnitude: float  (expected impact, e.g. 0.15 = 15% change)
        """
        if not agent_outputs:
            return {"error": "No agent outputs provided."}

        votes: Dict[str, float] = Counter()
        total_weight = 0.0
        all_signals: List[str] = []
        reasoning_map: Dict[str, List[str]] = {}

        for out in agent_outputs:
            agent_type   = out.get("agent_type", "unknown")
            rec          = out.get("recommendation", "hold")
            confidence   = float(out.get("confidence", 0.5))
            prior_weight = AGENT_WEIGHTS.get(agent_type, 1.0)

            weighted_vote = confidence * prior_weight
            votes[rec] += weighted_vote
            total_weight += prior_weight

            all_signals.extend(out.get("signals", []))
            reasoning_map.setdefault(rec, []).append(str(out.get("reasoning", "")))

        # Normalise
        norm_votes = {
            action: round(v / total_weight, 3) for action, v in votes.items()
        }

        winner = max(norm_votes, key=norm_votes.get)
        winner_score = norm_votes[winner]

        # Agreement: fraction of agents that voted with the winner
        n = len(agent_outputs)
        n_agree = sum(1 for o in agent_outputs if o.get("recommendation") == winner)
        agreement = round(n_agree / n, 3)

        # Uncertainty: 1 - margin between 1st and 2nd place
        sorted_scores = sorted(norm_votes.values(), reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0)
        uncertainty = round(1.0 - min(1.0, margin * 2), 3)

        return {
            "final_recommendation": winner,
            "confidence":           round(winner_score, 3),
            "agreement_score":      agreement,
            "uncertainty":          uncertainty,
            "vote_breakdown":       norm_votes,
            "n_agents":             n,
            "n_agreeing":           n_agree,
            "key_signals":          list(dict.fromkeys(all_signals))[:10],
            "top_reasoning":        reasoning_map.get(winner, [])[:2],
            "quality_flag":         self._quality(winner_score, agreement, uncertainty),
        }

    @staticmethod
    def _quality(confidence: float, agreement: float, uncertainty: float) -> str:
        if confidence > 0.70 and agreement > 0.70 and uncertainty < 0.30:
            return "HIGH_CONFIDENCE — proceed autonomously"
        elif confidence > 0.50 and agreement > 0.50:
            return "MODERATE_CONFIDENCE — proceed with monitoring"
        elif uncertainty > 0.60:
            return "HIGH_UNCERTAINTY — human review recommended"
        return "LOW_CONFIDENCE — gather more data before acting"
