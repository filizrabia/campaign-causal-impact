"""Synthetic daily sales for a regional campaign test with a known true lift.

One test region gets a campaign; several control regions do not. All regions
share a common demand signal (trend, weekly and yearly seasonality, shocks),
so the controls carry information about what the test region would have done
without the campaign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CampaignData:
    df: pd.DataFrame  # date, test, control_1..k
    campaign_start: pd.Timestamp
    campaign_end: pd.Timestamp
    true_effect: pd.Series  # daily incremental sales in the test region

    @property
    def controls(self) -> list[str]:
        return [c for c in self.df.columns if c.startswith("control_")]


def make_campaign_data(
    n_days: int = 365,
    n_controls: int = 6,
    campaign_day: int = 280,
    campaign_days: int = 42,
    lift: float = 0.08,
    seed: int = 11,
) -> CampaignData:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    common = (
        1.0
        + 0.0006 * t
        + 0.12 * np.sin(2 * np.pi * t / 365.25)
        + 0.08 * np.where(dates.dayofweek >= 5, 1, 0)
    )
    # Shared shocks (e.g. weather, holidays) as a slowly varying random walk.
    shocks = np.cumsum(rng.normal(0, 0.006, n_days))
    common = common + shocks - shocks.mean()

    df = pd.DataFrame({"date": dates})
    weights = rng.uniform(0.6, 1.4, n_controls)
    for i, w in enumerate(weights, start=1):
        own = rng.normal(0, 0.025, n_days)
        df[f"control_{i}"] = (8_000 * w * (common + own)).round(1)

    base_test = 10_000 * (common + rng.normal(0, 0.025, n_days))
    # The campaign ramps up over the first week and keeps a small tail after it ends.
    effect = np.zeros(n_days)
    end = campaign_day + campaign_days
    ramp = np.minimum(1.0, (t[campaign_day:end] - campaign_day + 1) / 7)
    effect[campaign_day:end] = lift * 10_000 * common[campaign_day:end] * ramp
    df["test"] = (base_test + effect).round(1)

    df = df[["date", "test"] + [f"control_{i}" for i in range(1, n_controls + 1)]]
    return CampaignData(
        df=df,
        campaign_start=dates[campaign_day],
        campaign_end=dates[end - 1],
        true_effect=pd.Series(effect, index=dates, name="true_effect"),
    )
