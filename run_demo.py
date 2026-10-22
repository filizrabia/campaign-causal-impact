"""Measure the lift of a regional campaign against control regions.

    python run_demo.py

Writes charts to ``images/``, the data to ``data/``, and a summary to ``results.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from causal.data import make_campaign_data
from causal.impact import control_sensitivity, estimate_impact, placebo_test

ROOT = Path(__file__).parent
(ROOT / "images").mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)

BLUE, ORANGE, GREY = "#3B5BDB", "#E8590C", "#868E96"
plt.rcParams.update(
    {"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10}
)


def main():
    data = make_campaign_data()
    data.df.to_csv(ROOT / "data" / "regional_daily_sales.csv", index=False)
    start, end = data.campaign_start, data.campaign_end
    length = (end - start).days + 1

    result = estimate_impact(data.df, "test", data.controls, start, end)
    result.placebo_effects = placebo_test(data.df, "test", data.controls, start, length)

    # Full-period counterfactual for plotting (fit on pre-period only).
    df = data.df.set_index("date")
    window = df.loc[start - np.timedelta64(70, "D") : end]
    full = estimate_impact(data.df, "test", data.controls, window.index[0], end, n_boot=200)
    plot = full.daily

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    ax = axes[0]
    ax.plot(plot.index, plot["actual"], color=GREY, lw=1.2, label="Actual (test region)")
    ax.plot(plot.index, plot["counterfactual"], color=BLUE, lw=1.5, ls="--", label="Counterfactual")
    ax.set_title("Test region: actual vs. what would have happened without the campaign", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    eff = result.daily
    ax.fill_between(eff.index, eff["lower"], eff["upper"], color=BLUE, alpha=0.18, lw=0)
    ax.plot(eff.index, eff["effect"], color=BLUE, lw=1.5, label="Estimated daily effect")
    ax.plot(eff.index, data.true_effect[start:end], color=ORANGE, lw=1.2, label="True effect")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Daily incremental sales (90% interval)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[2]
    ax.plot(eff.index, eff["effect"].cumsum(), color=BLUE, lw=1.8, label="Estimated")
    ax.plot(eff.index, data.true_effect[start:end].cumsum(), color=ORANGE, lw=1.2, label="True")
    ax.set_title("Cumulative incremental sales", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for a in axes:
        a.axvspan(start, end, color=ORANGE, alpha=0.05)
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "impact.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.hist(result.placebo_effects * 100, bins=12, color=GREY, alpha=0.8, label="Placebo start dates")
    ax.axvline(result.relative_lift * 100, color=BLUE, lw=2, label="Real campaign")
    ax.set_xlabel("Estimated lift (%)")
    ax.set_title("Placebo test: real lift vs. fake campaign dates", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "placebo.png")
    plt.close(fig)

    spill = make_campaign_data(spillover={4: 0.6})
    sens = control_sensitivity(spill.df, "test", spill.controls, spill.campaign_start, spill.campaign_end)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ypos = np.arange(len(sens))
    ax.errorbar(
        sens["relative_lift"] * 100, ypos,
        xerr=[(sens["relative_lift"] - sens["lower"]) * 100, (sens["upper"] - sens["relative_lift"]) * 100],
        fmt="o", color=BLUE, capsize=3,
    )
    true_rel_spill = float(spill.true_effect[start:end].sum()) / (
        spill.df.set_index("date").loc[start:end, "test"].sum() - float(spill.true_effect[start:end].sum())
    )
    ax.axvline(true_rel_spill * 100, color=ORANGE, lw=1.5, label="True lift")
    ax.set_yticks(ypos, sens.index)
    ax.invert_yaxis()
    ax.set_xlabel("Estimated lift (%)")
    ax.set_title("Leave-one-out check with spillover into control 4", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "spillover.png")
    plt.close(fig)

    true_cum = float(data.true_effect[start:end].sum())
    true_rel = true_cum / (df.loc[start:end, "test"].sum() - true_cum)
    text = [
        "## Results",
        "",
        f"Campaign: {start.date()} to {end.date()} ({length} days), test region vs {len(data.controls)} controls.",
        "",
        "```",
        result.summary(),
        "```",
        "",
        f"Ground truth: {true_cum:,.0f} incremental sales, {true_rel:+.2%} relative lift.",
        "",
        "### Control weights",
        "",
        result.weights.round(3).to_frame().to_markdown(),
    ]
    (ROOT / "results.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
