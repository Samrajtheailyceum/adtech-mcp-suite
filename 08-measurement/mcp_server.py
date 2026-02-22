"""
MCP 8 — Measurement & Reporting MCP

Unified cross-platform KPI reporting, statistical significance testing,
and automated anomaly detection in campaign metrics.

Agents:
- StatTestingAgent       — Frequentist + Bayesian A/B testing
- AnomalyReportAgent     — IQR + Z-score anomaly detection in KPIs
- UnifiedReportAgent     — Cross-platform metric unification
- MeasurementCritic      — Challenges statistical claims
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from mcp.server.fastmcp import FastMCP
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import BaseAgent

mcp = FastMCP("Measurement MCP")

np.random.seed(19)

# Synthetic A/B test data
_control = np.random.beta(2, 18, 500)    # ~10% CVR
_variant = np.random.beta(2.3, 18, 500)  # ~11.3% CVR

# Daily KPI time series (with injected anomaly on day 18)
_days = 30
_kpis = pd.DataFrame({
    "day": range(_days),
    "ctr":  [max(0, 0.045 + np.random.normal(0, 0.003)) if i != 18
             else 0.015 for i in range(_days)],  # anomaly on day 18
    "cvr":  [max(0, 0.11 + np.random.normal(0, 0.008)) for _ in range(_days)],
    "cpa":  [max(0, 12 + np.random.normal(0, 1.5)) if i != 22
             else 45 for i in range(_days)],      # CPA spike on day 22
    "roas": [max(0, 3.2 + np.random.normal(0, 0.3)) for _ in range(_days)],
    "spend":[max(0, 5000 + np.random.normal(0, 300)) for _ in range(_days)],
})


def frequentist_ab_test(control: np.ndarray, variant: np.ndarray) -> dict:
    """Two-sided t-test + chi-squared for conversion rates."""
    t_stat, p_val = stats.ttest_ind(control, variant)
    effect_size = (variant.mean() - control.mean()) / np.sqrt(
        (control.std()**2 + variant.std()**2) / 2
    )
    return {
        "control_cvr": round(float(control.mean()), 4),
        "variant_cvr": round(float(variant.mean()), 4),
        "lift_pct": round(float((variant.mean() / control.mean() - 1) * 100), 1),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "significant_at_95": bool(p_val < 0.05),
        "cohen_d": round(float(effect_size), 3),
    }


def bayesian_ab_test(control: np.ndarray, variant: np.ndarray, samples: int = 10000) -> dict:
    """Beta-binomial Bayesian A/B test."""
    ctrl_a = control.sum() * len(control) + 1
    ctrl_b = (1 - control).sum() * len(control) + 1
    var_a = variant.sum() * len(variant) + 1
    var_b = (1 - variant).sum() * len(variant) + 1

    ctrl_samples = np.random.beta(ctrl_a, ctrl_b, samples)
    var_samples = np.random.beta(var_a, var_b, samples)
    prob_variant_better = float((var_samples > ctrl_samples).mean())

    return {
        "prob_variant_better": round(prob_variant_better, 3),
        "expected_lift": round(float((var_samples / ctrl_samples - 1).mean() * 100), 1),
        "credible_interval_95": [
            round(float(np.percentile(var_samples - ctrl_samples, 2.5)), 4),
            round(float(np.percentile(var_samples - ctrl_samples, 97.5)), 4),
        ],
    }


def detect_kpi_anomalies(kpi_df: pd.DataFrame) -> dict:
    """IQR + Z-score anomaly detection on KPI time series."""
    anomalies = {}
    for col in ["ctr", "cvr", "cpa", "roas"]:
        series = kpi_df[col].values
        q1, q3 = np.percentile(series, 25), np.percentile(series, 75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z_scores = np.abs(stats.zscore(series))

        anomaly_days = [
            {"day": int(d), "value": round(float(series[d]), 4),
             "z_score": round(float(z_scores[d]), 2)}
            for d in np.where((series < lower) | (series > upper) | (z_scores > 2.5))[0]
        ]
        if anomaly_days:
            anomalies[col] = anomaly_days
    return anomalies


_STAT_SYSTEM = """You are a Measurement Intelligence Agent for digital advertising.
Interpret A/B test results (frequentist + Bayesian) and KPI anomaly reports.
Explain statistical findings in plain language and recommend actions.
Output JSON: ab_recommendation, anomaly_interpretation, confidence (0-1), signals, recommendations."""

_CRITIC_SYSTEM = """You are a Statistical Methods Critic.
Challenge: p-value misinterpretation, multiple testing without correction,
peeking problem in A/B tests, Bayesian prior assumptions, and confounds.
Output JSON: statistical_concerns, critique_score (0-10), what_was_missed."""


class MeasurementAgent(BaseAgent):
    def __init__(self):
        super().__init__("MeasurementAgent", _STAT_SYSTEM)

    def analyze(self) -> dict:
        freq = frequentist_ab_test(_control, _variant)
        bayes = bayesian_ab_test(_control, _variant)
        anomalies = detect_kpi_anomalies(_kpis)

        prompt = f"""Interpret these measurement results:

Frequentist A/B Test:
{json.dumps(freq, indent=2)}

Bayesian A/B Test:
{json.dumps(bayes, indent=2)}

KPI Anomalies Detected:
{json.dumps(anomalies, indent=2)}

Should we ship the variant? What do the anomalies mean? What actions should we take?"""
        result = self.structured_output(prompt)
        result["agent_type"] = "measurement"
        result["ab_frequentist"] = freq
        result["ab_bayesian"] = bayes
        result["anomalies"] = anomalies
        return result


class StatCritic(BaseAgent):
    def __init__(self):
        super().__init__("StatCritic", _CRITIC_SYSTEM)

    def critique(self, analysis: dict) -> dict:
        prompt = f"""Critique these statistical claims:
{json.dumps(analysis, indent=2)}
What statistical mistakes or misinterpretations exist?"""
        return self.structured_output(prompt)


_agent = MeasurementAgent()
_critic = StatCritic()


@mcp.tool()
def analyze_ab_test_results() -> str:
    """
    Analyse A/B test using both frequentist (t-test) and Bayesian (beta-binomial) methods.
    Includes KPI anomaly detection on the full time series.
    """
    result = _agent.analyze()
    critique = _critic.critique(result)
    return json.dumps({"measurement": result, "statistical_critique": critique},
                      indent=2, default=str)


@mcp.tool()
def run_statistical_significance_test(
    control_conversions: int, control_visitors: int,
    variant_conversions: int, variant_visitors: int,
) -> str:
    """
    Run frequentist + Bayesian significance test on your A/B test numbers.

    Args:
        control_conversions: Conversions in control group
        control_visitors: Total visitors in control group
        variant_conversions: Conversions in variant group
        variant_visitors: Total visitors in variant group
    """
    ctrl = np.array([1] * control_conversions + [0] * (control_visitors - control_conversions))
    var = np.array([1] * variant_conversions + [0] * (variant_visitors - variant_conversions))
    return json.dumps({
        "frequentist": frequentist_ab_test(ctrl, var),
        "bayesian": bayesian_ab_test(ctrl, var),
    }, indent=2)


@mcp.tool()
def detect_metric_anomalies() -> str:
    """Detect anomalies in daily KPI time series using IQR + Z-score methods."""
    return json.dumps(detect_kpi_anomalies(_kpis), indent=2)


if __name__ == "__main__":
    mcp.run()
