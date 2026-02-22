"""
Random Forest Creative Performance Classifier

Classifies creatives into Strong / Moderate / Weak tiers based on aggregated
performance metrics. Provides feature importance for explainability.

Why Random Forest here:
- Handles correlated features (CTR, CVR, ROAS are all related) gracefully via bagging
- Feature importance is free — directly interpretable for creative teams
- No scaling required — works on raw metric magnitudes
- Robust to outliers from attribution noise in real campaigns
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List

warnings.filterwarnings("ignore")


FEATURES = [
    "ctr_mean", "ctr_std", "cvr_mean",
    "cpa_mean", "roas_mean", "frequency_mean",
    "days_active", "impression_volume", "conversion_volume",
]


class CreativeClassifier:
    """Random Forest classifier for creative performance tiers."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=3,
            class_weight="balanced",   # handles class imbalance (most creatives are Moderate)
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False
        self.feature_importances_: Dict[str, float] = {}
        self.classes_: List[str] = []

    # ── Feature engineering ────────────────────────────────────────────────

    def _aggregate(self, perf_df: pd.DataFrame) -> pd.DataFrame:
        """Collapse daily time-series into per-creative feature vectors."""
        agg = perf_df.groupby("creative_id").agg(
            ctr_mean=("ctr", "mean"),
            ctr_std=("ctr", "std"),
            cvr_mean=("cvr", "mean"),
            cpa_mean=("cpa", "mean"),
            roas_mean=("roas", "mean"),
            frequency_mean=("frequency", "mean"),
            days_active=("date", "count"),
            impression_volume=("impressions", "sum"),
            conversion_volume=("conversions", "sum"),
        ).reset_index().fillna(0)
        return agg

    def _label(self, agg: pd.DataFrame) -> pd.Series:
        """Label creatives by ROAS/CPA percentile thresholds."""
        roas_p33 = agg["roas_mean"].quantile(0.33)
        roas_p66 = agg["roas_mean"].quantile(0.66)
        cpa_med = agg["cpa_mean"].median()

        def classify(row):
            if row["roas_mean"] >= roas_p66 and row["cpa_mean"] < cpa_med:
                return "Strong"
            elif row["roas_mean"] <= roas_p33:
                return "Weak"
            return "Moderate"

        return agg.apply(classify, axis=1)

    # ── Training ───────────────────────────────────────────────────────────

    def train(self, perf_df: pd.DataFrame) -> Dict:
        """Train on historical performance data. Returns diagnostics."""
        agg = self._aggregate(perf_df)
        labels = self._label(agg)

        X = agg[FEATURES].values
        y = labels.values

        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")
        self.model.fit(X, y)
        self.is_trained = True
        self.classes_ = list(self.model.classes_)

        self.feature_importances_ = dict(zip(FEATURES, self.model.feature_importances_))

        return {
            "cv_accuracy": round(float(cv_scores.mean()), 3),
            "cv_std": round(float(cv_scores.std()), 3),
            "class_distribution": dict(pd.Series(y).value_counts()),
            "feature_importances": {
                k: round(float(v), 3)
                for k, v in sorted(
                    self.feature_importances_.items(), key=lambda x: x[1], reverse=True
                )
            },
        }

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, perf_df: pd.DataFrame, creative_id: str = None) -> List[Dict]:
        """Predict performance tier for all (or one) creative."""
        if not self.is_trained:
            raise RuntimeError("Train the classifier first.")

        agg = self._aggregate(perf_df)
        if creative_id:
            agg = agg[agg["creative_id"] == creative_id]

        X = agg[FEATURES].values
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        results = []
        for i, (_, row) in enumerate(agg.iterrows()):
            prob_map = dict(zip(self.classes_, probs[i].tolist()))
            results.append({
                "creative_id": row["creative_id"],
                "tier": preds[i],
                "confidence": round(float(max(probs[i])), 3),
                "probabilities": {k: round(float(v), 3) for k, v in prob_map.items()},
                "metrics": {
                    "ctr_mean": round(float(row["ctr_mean"]), 4),
                    "roas_mean": round(float(row["roas_mean"]), 2),
                    "cpa_mean": round(float(row["cpa_mean"]), 2),
                    "days_active": int(row["days_active"]),
                },
            })

        return sorted(results, key=lambda x: x["confidence"], reverse=True)

    def top_features(self, n: int = 5) -> Dict[str, float]:
        return dict(
            sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:n]
        )
