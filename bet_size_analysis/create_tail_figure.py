"""Create a tail-focused figure for the largest unusual bets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "bet_size_analysis_sample.csv"
PNG_FILE = HERE / "bet_size_tail_figure.png"
PDF_FILE = HERE / "bet_size_tail_figure.pdf"

INK = "#17212B"
MUTED = "#66717C"
GRID = "#D8DEE3"
ACCENT = "#176B87"
EXPECTED = "#D65A3A"
BACKGROUND = "#F7F3EA"


def wilson_interval(wins, total, z=1.96):
    rate = wins / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return center - margin, center + margin


def main():
    df = pd.read_csv(INPUT_FILE)
    thresholds = [
        ("All bets", 0.00),
        ("Largest 25%", 0.75),
        ("Largest 10%", 0.90),
        ("Largest 5%", 0.95),
        ("Largest 2.5%", 0.975),
        ("Largest 1%", 0.99),
    ]

    rows = []
    for label, quantile in thresholds:
        cutoff = df["trade_value_usd"].quantile(quantile)
        sample = df[df["trade_value_usd"] >= cutoff]
        wins = int(sample["bet_won_num"].sum())
        low, high = wilson_interval(wins, len(sample))
        rows.append(
            {
                "label": label,
                "cutoff": cutoff,
                "trades": len(sample),
                "wins": wins,
                "win_rate": sample["bet_won_num"].mean(),
                "expected_rate": sample["price"].mean(),
                "ci_low": low,
                "ci_high": high,
            }
        )
    summary = pd.DataFrame(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )

    figure, axis = plt.subplots(figsize=(11.5, 7.5), facecolor=BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    x = np.arange(len(summary))
    lower_error = summary["win_rate"] - summary["ci_low"]
    upper_error = summary["ci_high"] - summary["win_rate"]

    axis.errorbar(
        x,
        summary["win_rate"],
        yerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        color=ACCENT,
        markerfacecolor=BACKGROUND,
        markeredgewidth=2.5,
        markersize=10,
        capsize=6,
        linewidth=2.2,
        label="Observed win rate · 95% Wilson CI",
        zorder=3,
    )
    axis.plot(
        x,
        summary["expected_rate"],
        color=EXPECTED,
        marker="s",
        markersize=6,
        linewidth=2,
        label="Average market-implied probability",
        zorder=2,
    )

    axis.axvspan(3.5, 5.5, color=EXPECTED, alpha=0.07, zorder=0)
    axis.text(
        4.5,
        axis.get_ylim()[1] if axis.get_ylim()[1] else 0.1,
        "Extreme tail",
        ha="center",
        va="top",
        color=EXPECTED,
        fontsize=9,
        fontweight="bold",
    )

    labels = []
    for row in summary.itertuples():
        cutoff = f"${row.cutoff:,.0f}+" if row.label != "All bets" else "$1,000+"
        labels.append(f"{row.label}\n{cutoff}")
    axis.set_xticks(x, labels)
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.set_ylabel("Win rate", labelpad=12)
    axis.set_xlabel("Nested bet-size threshold", labelpad=14)
    axis.set_ylim(0, max(summary["ci_high"].max(), summary["expected_rate"].max()) * 1.28)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")

    for index, row in summary.iterrows():
        axis.annotate(
            f"{int(row.wins)}/{int(row.trades)} wins",
            (index, row.ci_high),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=MUTED,
        )

    figure.subplots_adjust(left=0.09, right=0.97, top=0.75, bottom=0.20)
    figure.text(
        0.09,
        0.92,
        "The most extreme bets did not outperform",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.09,
        0.86,
        "Observed win rates briefly rise in the largest 5%, then fall to zero in the extreme upper tail.",
        fontsize=11.5,
        color=MUTED,
    )
    figure.text(
        0.09,
        0.055,
        "Notes: Threshold groups are nested, not independent. $1,000+ bets priced from 1% to below 10%. "
        "Wide confidence intervals reflect sparse extreme-tail samples.",
        fontsize=8.5,
        color=MUTED,
    )

    figure.savefig(PNG_FILE, dpi=300, facecolor=BACKGROUND)
    figure.savefig(PDF_FILE, facecolor=BACKGROUND)
    summary.to_csv(HERE / "bet_size_tail_summary.csv", index=False)
    print(f"Saved {PNG_FILE}")
    print(f"Saved {PDF_FILE}")


if __name__ == "__main__":
    main()
