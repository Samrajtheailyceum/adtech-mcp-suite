"""
MCP 3 — Brand Safety MCP

Classifies content environments for brand suitability.
Uses keyword vectorization + Naive Bayes classification + LLM semantic review.

Agents:
- ContentClassifierAgent    — ML-based content category classification
- SentimentSafetyAgent      — Toxicity + sentiment scoring
- ContextualTargetingAgent  — Finds brand-safe contextual segments
- SafetyCriticAgent         — Adversarially tests safety boundaries
"""

import json
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from mcp.server.fastmcp import FastMCP
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent

mcp = FastMCP("Brand Safety MCP")


# ── Training data (simplified) ─────────────────────────────────────────────
_TRAIN = [
    ("luxury cars performance review", "safe"),
    ("cooking recipe healthy meals", "safe"),
    ("sports highlights championship", "safe"),
    ("finance investment portfolio growth", "safe"),
    ("travel destination beach vacation", "safe"),
    ("violence shooting conflict war", "unsafe"),
    ("drugs illegal substance abuse", "unsafe"),
    ("hate speech discrimination offensive", "unsafe"),
    ("adult explicit sexual content", "unsafe"),
    ("fake news misinformation conspiracy", "unsafe"),
    ("political controversy divisive debate", "sensitive"),
    ("alcohol drinking beer wine", "sensitive"),
    ("gambling casino betting odds", "sensitive"),
    ("death obituary funeral services", "sensitive"),
    ("medical health disease symptoms", "sensitive"),
]

_texts, _labels = zip(*_TRAIN)
_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", MultinomialNB(alpha=0.5)),
])
_pipeline.fit(_texts, _labels)

# GARM brand safety tiers (Global Alliance for Responsible Media standard)
GARM_TIERS = {
    "floor": ["child exploitation", "terrorism", "hate speech", "illegal drugs"],
    "brand_suitability": ["adult content", "gambling", "alcohol", "politics"],
    "safe": ["news", "entertainment", "sports", "lifestyle", "finance"],
}

_SAFETY_SYSTEM = """You are a Brand Safety Agent for programmatic advertising.
Evaluate content environments for brand suitability using GARM standards.
Consider: content category, sentiment, contextual risk, brand alignment.
Output JSON: safety_tier (safe/sensitive/unsafe), garm_category, risk_score (0-1),
suitability_explanation, contextual_keywords, recommendation (run/exclude/review)."""

_CRITIC_SYSTEM = """You are a Brand Safety Critic. Challenge safety classifications.
Find edge cases where 'safe' content could be contextually risky for specific brands.
Flag over-blocking (missing valid inventory) and under-blocking (missing risks).
Output JSON: missed_risks, over_blocks, edge_cases, overall_critique."""


class ContentClassifierAgent(BaseAgent):
    def __init__(self):
        super().__init__("ContentClassifierAgent", _SAFETY_SYSTEM)

    def classify(self, content: str, brand: str = "generic") -> dict:
        ml_pred = _pipeline.predict([content])[0]
        ml_prob = _pipeline.predict_proba([content])[0]
        classes = _pipeline.classes_
        prob_map = dict(zip(classes, ml_prob.round(3).tolist()))

        prompt = f"""Classify this content for brand safety:

Content: "{content}"
Brand: {brand}
ML Classification: {ml_pred} (probabilities: {prob_map})

Evaluate GARM suitability. Is this safe for a {brand} brand to advertise next to?"""
        result = self.structured_output(prompt)
        result["ml_classification"] = ml_pred
        result["ml_probabilities"] = prob_map
        return result


class SafetyCriticAgent(BaseAgent):
    def __init__(self):
        super().__init__("SafetyCriticAgent", _CRITIC_SYSTEM)

    def critique(self, classification: dict, content: str) -> dict:
        prompt = f"""Adversarially review this brand safety classification:
Content: "{content}"
Classification: {json.dumps(classification, indent=2)}
What risks were missed? What was over-blocked?"""
        return self.structured_output(prompt)


_classifier = ContentClassifierAgent()
_critic = SafetyCriticAgent()


@mcp.tool()
def classify_content_safety(content: str, brand: str = "generic") -> str:
    """
    Classify content for brand safety using TF-IDF + Naive Bayes + LLM semantic review.

    Args:
        content: The page content or URL description to classify
        brand: Brand name for contextual suitability (e.g. "Nike", "Pfizer")
    """
    result = _classifier.classify(content, brand)
    critique = _critic.critique(result, content)
    return json.dumps({"classification": result, "safety_critique": critique},
                      indent=2, default=str)


@mcp.tool()
def get_garm_tiers() -> str:
    """Return GARM brand safety tier definitions."""
    return json.dumps(GARM_TIERS, indent=2)


@mcp.tool()
def bulk_classify_urls(urls: list[str], brand: str = "generic") -> str:
    """
    Classify multiple content URLs/descriptions for brand safety in batch.

    Args:
        urls: List of content descriptions or URLs
        brand: Brand name for contextual evaluation
    """
    results = []
    for url in urls[:10]:  # cap at 10
        pred = _pipeline.predict([url])[0]
        prob = float(max(_pipeline.predict_proba([url])[0]))
        results.append({"content": url, "tier": pred, "confidence": round(prob, 3)})
    return json.dumps({"batch_results": results, "brand": brand}, indent=2)


if __name__ == "__main__":
    mcp.run()
