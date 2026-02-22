"""
Creative Fatigue Detector

Combines three techniques:
1. Exponential decay curve fitting  — models the natural lifecycle of a creative
2. Linear regression on recent window — measures current rate of change
3. CUSUM change point detection — finds when decay accelerated

The combination gives both a "current state" diagnosis and a forward prediction
of days until the creative drops below the fatigue threshold.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# CTR must drop this much from peak before we call it "fatigued"
FATIGUE_THRESHOLD_PCT = 0.35


def _exp_decay(t, A, k, c):
    """A * exp(-k*t) + c — the canonical creative fatigue curve."""
    return A * np.exp(-k * t) + c


class FatigueDetector:
    """Time-series fatigue analysis for individual creatives."""

    def analyze(self, perf_df: pd.DataFrame, creative_id: str) -> Dict:
        """Full fatigue diagnosis for one creative."""
        data = (
            perf_df[perf_df["creative_id"] == creative_id]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(data) < 4:
            return {"creative_id": creative_id, "status": "insufficient_data",
                    "days_until_fatigue": None, "decay_rate": None}

        t = np.arange(len(data))
        ctr = data["ctr"].values

        # ── 1. Fit exponential decay ───────────────────────────────────────
        decay = self._fit_decay(t, ctr)

        # ── 2. Recent trend (last 7 days) ─────────────────────────────────
        window = ctr[-min(7, len(ctr)):]
        t_window = np.arange(len(window))
        slope, _, r_val, p_val, _ = linregress(t_window, window)

        # ── 3. CUSUM change point ─────────────────────────────────────────
        change_point = self._cusum_change_point(ctr)

        # ── 4. Fatigue prediction ─────────────────────────────────────────
        peak_ctr = float(ctr.max())
        current_ctr = float(ctr[-1])
        fatigue_ctr = peak_ctr * (1 - FATIGUE_THRESHOLD_PCT)
        ctr_drop_pct = (peak_ctr - current_ctr) / max(peak_ctr, 1e-8)

        days_until = self._predict_days(current_ctr, slope, fatigue_ctr)
        status = self._classify(ctr_drop_pct, slope, r_val ** 2)

        return {
            "creative_id": creative_id,
            "status": status,
            "current_ctr": round(current_ctr, 4),
            "peak_ctr": round(peak_ctr, 4),
            "ctr_drop_pct": round(ctr_drop_pct * 100, 1),
            "recent_slope": round(float(slope), 6),
            "trend_r2": round(float(r_val ** 2), 3),
            "change_point_day": change_point,
            "decay_half_life_days": decay.get("half_life"),
            "days_until_fatigue": days_until,
            "recommendation": self._rec(status, days_until),
        }

    def analyze_all(self, perf_df: pd.DataFrame) -> List[Dict]:
        """Analyze all creatives, sorted by urgency (soonest fatigue first)."""
        results = [
            self.analyze(perf_df, cid)
            for cid in perf_df["creative_id"].unique()
        ]
        return sorted(results, key=lambda x: x.get("days_until_fatigue") or 999)

    # ── Private helpers ────────────────────────────────────────────────────

    def _fit_decay(self, t: np.ndarray, ctr: np.ndarray) -> Dict:
        try:
            popt, _ = curve_fit(
                _exp_decay, t, ctr,
                p0=[ctr[0], 0.05, max(ctr.min(), 0.001)],
                bounds=([0, 1e-4, 0], [1, 2, 1]),
                maxfev=5000,
            )
            A, k, c = popt
            return {"A": float(A), "k": float(k), "c": float(c),
                    "half_life": round(np.log(2) / max(k, 1e-6), 1)}
        except Exception:
            return {}

    def _cusum_change_point(self, ctr: np.ndarray) -> Optional[int]:
        """CUSUM: find the day where cumulative deviation from mean is largest."""
        if len(ctr) < 5:
            return None
        cusum = np.cumsum(ctr - ctr.mean())
        cp = int(np.argmax(np.abs(cusum)))
        return cp if cp > 0 else None

    def _predict_days(self, current: float, slope: float, threshold: float) -> Optional[int]:
        if slope >= 0:
            return None  # not declining
        if current <= threshold:
            return 0    # already fatigued
        days = int((threshold - current) / slope)
        return max(0, min(days, 90))

    def _classify(self, drop_pct: float, slope: float, r2: float) -> str:
        if drop_pct > 0.5 or (slope < -0.001 and r2 > 0.65):
            return "critical"
        elif drop_pct > 0.3 or (slope < -0.0005 and r2 > 0.40):
            return "fatiguing"
        elif slope < 0 and r2 > 0.25:
            return "early_decline"
        return "healthy"

    def _rec(self, status: str, days: Optional[int]) -> str:
        return {
            "critical": "URGENT: Pause creative and rotate replacement immediately.",
            "fatiguing": f"Schedule creative refresh within {days or 7} days.",
            "early_decline": "Prepare refresh variant. Monitor daily.",
            "healthy": "Creative healthy. Continue standard optimization.",
        }.get(status, "No action required.")
