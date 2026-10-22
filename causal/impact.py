"""Estimate the incremental effect of a campaign with a synthetic control.

1. **Counterfactual.** On the pre-campaign period, regress the test region on
   the control regions (intercept plus non-negative weights). Applied to the
   campaign period, this predicts what the test region would have sold
   without the campaign.
2. **Effect.** Actual minus counterfactual, per day and cumulated.
3. **Uncertainty.** Pre-period residuals are resampled in blocks to build
   intervals around the counterfactual.
4. **Placebo tests.** The same analysis is run on fake campaign dates inside
   the pre-period, where the true effect is zero. If the real estimate is
   larger than almost all placebo estimates, the effect is unlikely to be noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


@dataclass
class ImpactResult:
    daily: pd.DataFrame  # actual, counterfactual, effect, lower, upper
    weights: pd.Series
    pre_period_mape: float
    cumulative_effect: float
    cumulative_interval: tuple[float, float]
    relative_lift: float
    relative_interval: tuple[float, float]
    placebo_effects: np.ndarray | None = None

    @property
    def placebo_p_value(self) -> float | None:
        if self.placebo_effects is None or len(self.placebo_effects) == 0:
            return None
        # Share of placebo runs with a relative effect at least as large as the real one.
        hits = np.sum(np.abs(self.placebo_effects) >= abs(self.relative_lift))
        return float((hits + 1) / (len(self.placebo_effects) + 1))

    def summary(self) -> str:
        lo, hi = self.cumulative_interval
        rlo, rhi = self.relative_interval
        lines = [
            f"Pre-period fit (MAPE): {self.pre_period_mape:.2%}",
            f"Cumulative incremental sales: {self.cumulative_effect:,.0f} (90% interval {lo:,.0f} to {hi:,.0f})",
            f"Relative lift: {self.relative_lift:+.2%} (90% interval {rlo:+.2%} to {rhi:+.2%})",
        ]
        if self.placebo_p_value is not None:
            lines.append(
                f"Placebo test: {len(self.placebo_effects)} fake start dates, p = {self.placebo_p_value:.3f}"
            )
        return "\n".join(lines)


def _fit_weights(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Intercept (free) plus non-negative control weights."""
    A = np.column_stack([np.ones(len(y)), X])
    lower = np.r_[-np.inf, np.zeros(X.shape[1])]
    upper = np.full(X.shape[1] + 1, np.inf)
    return lsq_linear(A, y, bounds=(lower, upper)).x


def _predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + X @ coef[1:]


def estimate_impact(
    df: pd.DataFrame,
    test: str,
    controls: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_boot: int = 500,
    block: int = 7,
    seed: int = 0,
) -> ImpactResult:
    df = df.set_index("date") if "date" in df.columns else df
    pre = df.loc[: start - pd.Timedelta(days=1)]
    post = df.loc[start:end]

    coef = _fit_weights(pre[test].to_numpy(), pre[controls].to_numpy())
    pre_fit = _predict(coef, pre[controls].to_numpy())
    resid = pre[test].to_numpy() - pre_fit
    mape = float(np.mean(np.abs(resid) / pre[test].to_numpy()))

    counterfactual = _predict(coef, post[controls].to_numpy())
    actual = post[test].to_numpy()

    # Block bootstrap of pre-period residuals for counterfactual uncertainty.
    rng = np.random.default_rng(seed)
    n_post = len(post)
    boot_cum, boot_daily = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, len(resid) - block, size=n_post // block + 1)
        noise = np.concatenate([resid[s : s + block] for s in starts])[:n_post]
        cf = counterfactual + noise
        boot_daily.append(actual - cf)
        boot_cum.append(np.sum(actual - cf) / np.sum(cf))
    boot_daily = np.array(boot_daily)
    boot_abs = boot_daily.sum(axis=1)

    effect = actual - counterfactual
    daily = pd.DataFrame(
        {
            "actual": actual,
            "counterfactual": counterfactual,
            "effect": effect,
            "lower": np.percentile(boot_daily, 5, axis=0),
            "upper": np.percentile(boot_daily, 95, axis=0),
        },
        index=post.index,
    )
    return ImpactResult(
        daily=daily,
        weights=pd.Series(coef[1:], index=controls, name="weight"),
        pre_period_mape=mape,
        cumulative_effect=float(effect.sum()),
        cumulative_interval=(float(np.percentile(boot_abs, 5)), float(np.percentile(boot_abs, 95))),
        relative_lift=float(effect.sum() / counterfactual.sum()),
        relative_interval=(float(np.percentile(boot_cum, 5)), float(np.percentile(boot_cum, 95))),
    )


def control_sensitivity(
    df: pd.DataFrame,
    test: str,
    controls: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Re-run the analysis leaving out one control region at a time.

    If the estimate depends heavily on a single control, the conclusion is
    fragile: that region may have had its own shock, or be affected by the
    campaign itself.
    """
    rows = [("all controls", estimate_impact(df, test, controls, start, end, n_boot=200))]
    for c in controls:
        rest = [x for x in controls if x != c]
        rows.append((f"without {c}", estimate_impact(df, test, rest, start, end, n_boot=200)))
    return pd.DataFrame(
        {
            "relative_lift": [r.relative_lift for _, r in rows],
            "lower": [r.relative_interval[0] for _, r in rows],
            "upper": [r.relative_interval[1] for _, r in rows],
            "pre_period_mape": [r.pre_period_mape for _, r in rows],
        },
        index=[name for name, _ in rows],
    )


def placebo_test(
    df: pd.DataFrame,
    test: str,
    controls: list[str],
    start: pd.Timestamp,
    length_days: int,
    min_train_days: int = 90,
    step_days: int = 7,
) -> np.ndarray:
    """Relative 'effects' estimated at fake campaign dates before the real one."""
    df = df.set_index("date") if "date" in df.columns else df
    pre = df.loc[: start - pd.Timedelta(days=1)]
    effects = []
    first = pre.index[0] + pd.Timedelta(days=min_train_days)
    last = pre.index[-1] - pd.Timedelta(days=length_days - 1)
    for fake_start in pd.date_range(first, last, freq=f"{step_days}D"):
        fake_end = fake_start + pd.Timedelta(days=length_days - 1)
        res = estimate_impact(pre.reset_index(), test, controls, fake_start, fake_end, n_boot=1)
        effects.append(res.relative_lift)
    return np.array(effects)
