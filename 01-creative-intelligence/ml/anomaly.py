"""
Isolation Forest Anomaly Detector

Two distinct detectors:
1. Performance anomalies — unusual CTR/CVR/ROAS combinations in campaign data
2. Agent recommendation anomalies — flags outlier suggestions from sub-agents

Isolation Forest works by random recursive partitioning. Anomalies get isolated
with fewer splits (shorter average path length) because they sit in sparse regions
of feature space. This makes it O(n log n) and highly effective for high-dim data.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

warnings.filterwarnings("ignore")


class AnomalyDetector:
    """Isolation Forest for performance and recommendation anomaly detection."""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self._models: Dict[str, IsolationForest] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._fitted: Dict[str, bool] = {}

    # ── Fitting ────────────────────────────────────────────────────────────

    def fit_performance_detector(self, perf_df: pd.DataFrame) -> None:
        cols = ["ctr", "cvr", "cpa", "roas", "frequency"]
        X = perf_df[cols].fillna(0).values
        scaler = StandardScaler()
        model = IsolationForest(
            n_estimators=200, contamination=self.contamination, random_state=42, n_jobs=-1
        )
        model.fit(scaler.fit_transform(X))
        self._models["perf"] = model
        self._scalers["perf"] = scaler
        self._fitted["perf"] = True

    def fit_bid_detector(self, bid_df: pd.DataFrame) -> None:
        cols = ["avg_bid", "win_rate", "competition_index", "floor_price"]
        X = bid_df[cols].fillna(0).values
        scaler = StandardScaler()
        model = IsolationForest(
            n_estimators=150, contamination=self.contamination, random_state=42
        )
        model.fit(scaler.fit_transform(X))
        self._models["bid"] = model
        self._scalers["bid"] = scaler
        self._fitted["bid"] = True

    # ── Detection ──────────────────────────────────────────────────────────

    def detect_performance_anomalies(self, perf_df: pd.DataFrame) -> pd.DataFrame:
        """Return rows with anomalous performance patterns, sorted by severity."""
        if not self._fitted.get("perf"):
            raise RuntimeError("Fit performance detector first.")

        cols = ["ctr", "cvr", "cpa", "roas", "frequency"]
        X = self._scalers["perf"].transform(perf_df[cols].fillna(0).values)

        preds = self._models["perf"].predict(X)
        scores = self._models["perf"].score_samples(X)
        score_p10 = float(np.percentile(scores, 10))
        score_p25 = float(np.percentile(scores, 25))

        result = perf_df.copy()
        result["is_anomaly"] = preds == -1
        result["anomaly_score"] = scores
        result["severity"] = result["anomaly_score"].apply(
            lambda s: "high" if s < score_p10 else "medium" if s < score_p25 else "low"
        )

        return (
            result[result["is_anomaly"]]
            .sort_values("anomaly_score")
            .reset_index(drop=True)
        )

    def score_recommendation(self, rec: Dict[str, Any]) -> Dict:
        """
        Score how anomalous an agent recommendation is.
        Used by the Critic Agent to flag outlier suggestions.

        A simple Z-score on the magnitude of the recommended change.
        """
        magnitude = abs(float(rec.get("magnitude", 0.0)))
        confidence = float(rec.get("confidence", 0.5))
        n_signals = int(rec.get("supporting_signals", 1))

        # High-magnitude + low-confidence = suspicious
        suspicion = magnitude / max(confidence, 0.1) / max(n_signals, 1)
        is_anomalous = magnitude > 0.35 or (magnitude > 0.20 and confidence < 0.4)

        return {
            "is_anomalous": is_anomalous,
            "suspicion_score": round(suspicion, 3),
            "critique": (
                f"Outlier recommendation: {magnitude:.0%} change with only "
                f"{confidence:.0%} confidence. Verify with additional signals."
            ) if is_anomalous else "Recommendation within expected bounds.",
        }
