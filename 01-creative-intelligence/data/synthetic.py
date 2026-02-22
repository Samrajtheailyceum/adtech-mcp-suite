"""
Synthetic Ad Campaign Data Generator

Generates realistic campaign data with:
- Exponential CTR decay (simulates real creative fatigue)
- Log-normal impression distributions (real traffic follows power laws)
- Correlated audience segments (age/gender performance matrices)
- Intraday bid landscape with competition dynamics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_campaign_data(n_creatives: int = 50, n_days: int = 30) -> dict:
    """Generate synthetic ad campaign data across 4 DataFrames."""
    np.random.seed(42)
    random.seed(42)

    creative_ids = [f"CRE_{i:04d}" for i in range(n_creatives)]
    formats = ["video_15s", "video_30s", "image_static", "carousel", "story"]
    campaigns = ["CAMP_001", "CAMP_002", "CAMP_003"]

    # ── Creative metadata ──────────────────────────────────────────────────
    creatives = []
    for cid in creative_ids:
        fmt = random.choice(formats)
        # Beta distribution gives realistic CTR spread (most creatives are mediocre)
        base_ctr = np.random.beta(2, 20)
        base_cvr = np.random.beta(1.5, 15)
        creatives.append({
            "creative_id": cid,
            "format": fmt,
            "campaign_id": random.choice(campaigns),
            "launch_date": datetime.now() - timedelta(days=random.randint(5, n_days)),
            "base_ctr": base_ctr,
            "base_cvr": base_cvr,
        })

    # ── Daily performance time series ──────────────────────────────────────
    # Each creative follows an exponential decay curve + noise
    performance_records = []
    for creative in creatives:
        days_active = (datetime.now() - creative["launch_date"]).days

        for day in range(min(days_active, n_days)):
            date = creative["launch_date"] + timedelta(days=day)

            # Fatigue: exponential decay A*exp(-k*t) + noise
            k = np.random.uniform(0.02, 0.08)  # decay rate varies per creative
            fatigue_factor = np.exp(-k * day) + np.random.normal(0, 0.02)
            fatigue_factor = max(0.05, fatigue_factor)

            # Log-normal impressions (power law traffic distribution)
            impressions = int(np.random.lognormal(10, 0.5))
            ctr = max(0, creative["base_ctr"] * fatigue_factor + np.random.normal(0, 0.002))
            cvr = max(0, creative["base_cvr"] * fatigue_factor + np.random.normal(0, 0.001))

            clicks = int(impressions * ctr)
            conversions = int(clicks * cvr)
            spend = impressions * np.random.uniform(0.005, 0.02)

            performance_records.append({
                "creative_id": creative["creative_id"],
                "campaign_id": creative["campaign_id"],
                "format": creative["format"],
                "date": date.strftime("%Y-%m-%d"),
                "day_of_campaign": day,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": round(spend, 2),
                "ctr": round(ctr, 4),
                "cvr": round(cvr, 4),
                "cpa": round(spend / max(conversions, 1), 2),
                "roas": round((conversions * 45) / max(spend, 0.01), 2),
                "frequency": round(np.random.uniform(1.0, 8.0), 1),
            })

    # ── Audience segment performance ────────────────────────────────────────
    segments = []
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    genders = ["M", "F", "Unknown"]

    for creative in creatives[:15]:
        for age in age_groups:
            for gender in genders:
                # Different age/gender combos have different affinities
                age_multiplier = {"18-24": 1.2, "25-34": 1.4, "35-44": 1.1,
                                   "45-54": 0.9, "55+": 0.7}[age]
                gender_multiplier = {"M": 1.0, "F": 1.15, "Unknown": 0.85}[gender]
                affinity = age_multiplier * gender_multiplier

                segments.append({
                    "creative_id": creative["creative_id"],
                    "age_group": age,
                    "gender": gender,
                    "index": round(affinity * np.random.lognormal(0, 0.3), 2),
                    "reach": int(np.random.lognormal(8, 1)),
                    "ctr": round(max(0, creative["base_ctr"] * affinity *
                                    np.random.uniform(0.6, 1.8)), 4),
                    "cpa": round(np.random.uniform(5, 80), 2),
                })

    # ── Intraday bid landscape ─────────────────────────────────────────────
    # Competition peaks during business hours, floor prices vary by campaign
    bid_data = []
    for hour in range(24):
        for campaign in campaigns:
            # Business hours have higher competition
            hour_competition = 0.5 + 0.5 * np.sin((hour - 6) * np.pi / 12) \
                               if 6 <= hour <= 22 else 0.3

            bid_data.append({
                "hour": hour,
                "campaign_id": campaign,
                "avg_bid": round(np.random.uniform(0.5, 5.0) * (1 + hour_competition), 2),
                "win_rate": round(np.random.beta(3, 7) * (1 - hour_competition * 0.3), 3),
                "competition_index": round(hour_competition + np.random.uniform(-0.1, 0.1), 3),
                "floor_price": round(np.random.uniform(0.1, 1.0), 2),
            })

    return {
        "creatives": pd.DataFrame(creatives),
        "performance": pd.DataFrame(performance_records),
        "segments": pd.DataFrame(segments),
        "bid_landscape": pd.DataFrame(bid_data),
    }
