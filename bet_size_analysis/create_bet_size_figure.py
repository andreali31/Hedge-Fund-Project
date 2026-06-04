"""Create a publication-ready figure for bet size and performance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter


HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "bet_size_analysis_sample.csv"
PNG_FILE = HERE / "bet_size_performance_figure.png"
PDF_FILE = HERE / "bet_size_performance_figure.pdf"

INK = "#17212B"
MUTED = "#66717C"
GRID = "#D8DEE3"
LOSS = "#A7B0B8"
WIN = "#D65A3A"
ACCENT = "#176B87"
BACKGROUND = "#F7F3EA"


def dollars(value, _):
    if value >= 1_000:
        return f"${value / 1_000:g}k"
    return f"${value:g}"


def wilson_interval(wins, total, z=1.96):
    rate = wins / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return center - margin, center + margin


def main():
    df = pd.read_csv(INPUT_FILE)
    df["bet_won"] = df["bet_won"].astype(bool)
    df["size_quintile"] = pd.qcut(
        df["trade_value_usd"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )

    summary = (
        df.groupby("size_quintile", observed=True)
        .agg(
            trades=("bet_won", "size"),
            wins=("bet_won", "sum"),
            median_usd=("trade_value_usd", "median"),
            win_rate=("bet_won", "mean"),
        )
        .reset_index()
    )
    intervals = summary.apply(
        lambda row: wilson_interval(row["wins"], row["trades"]), axis=1
    )
    summary["ci_low"] = [interval[0] for interval in intervals]
    summary["ci_high"] = [interval[1] for interval in intervals]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )

    figure = plt.figure(figsize=(14, 7.8), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.08, 0.92], left=0.07, right=0.97, top=0.78, bottom=0.18, wspace=0.24
    )
    distribution_ax = figure.add_subplot(grid[0, 0], facecolor=BACKGROUND)
    quintile_ax = figure.add_subplot(grid[0, 1], facecolor=BACKGROUND)

    bins = np.geomspace(df["trade_value_usd"].min(), df["trade_value_usd"].max(), 24)
    distribution_ax.hist(
        df.loc[~df["bet_won"], "trade_value_usd"],
        bins=bins,
        color=LOSS,
        alpha=0.8,
        label=f"Lost ({(~df['bet_won']).sum():,})",
    )
    distribution_ax.hist(
        df.loc[df["bet_won"], "trade_value_usd"],
        bins=bins,
        color=WIN,
        alpha=0.9,
        label=f"Won ({df['bet_won'].sum():,})",
    )
    distribution_ax.set_xscale("log")
    distribution_ax.xaxis.set_major_formatter(FuncFormatter(dollars))
    distribution_ax.set_xlabel("Bet value, USD · logarithmic scale", labelpad=10)
    distribution_ax.set_ylabel("Number of bets", labelpad=10)
    distribution_ax.set_title("A · Bet-size distribution", loc="left", pad=14, fontsize=13)
    distribution_ax.legend(frameon=False, loc="upper right")
    distribution_ax.grid(axis="y", color=GRID, linewidth=0.8)
    distribution_ax.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(summary))
    lower_error = summary["win_rate"] - summary["ci_low"]
    upper_error = summary["ci_high"] - summary["win_rate"]
    quintile_ax.errorbar(
        x,
        summary["win_rate"],
        yerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        color=ACCENT,
        markerfacecolor=BACKGROUND,
        markeredgewidth=2.4,
        markersize=9,
        capsize=5,
        linewidth=2,
    )
    quintile_ax.plot(x, summary["win_rate"], color=ACCENT, alpha=0.35, linewidth=1.5)
    quintile_ax.set_xticks(
        x,
        [
            f"{row.size_quintile}\n${row.median_usd:,.0f}"
            for row in summary.itertuples()
        ],
    )
    quintile_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    quintile_ax.set_xlabel("Bet-size quintile · median bet", labelpad=10)
    quintile_ax.set_ylabel("Win rate with 95% Wilson CI", labelpad=10)
    quintile_ax.set_title("B · Win rate by bet-size quintile", loc="left", pad=14, fontsize=13)
    quintile_ax.grid(axis="y", color=GRID, linewidth=0.8)
    quintile_ax.spines[["top", "right"]].set_visible(False)
    quintile_ax.set_ylim(0, max(summary["ci_high"]) * 1.22)

    for index, row in summary.iterrows():
        quintile_ax.annotate(
            f"{int(row.wins)}/{int(row.trades)}",
            (index, row.ci_high),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=MUTED,
        )

    figure.text(
        0.07,
        0.93,
        "Larger bets do not show reliably better outcomes",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.07,
        0.875,
        "Among 1,988 unusual bets priced from 1% to below 10%, win rates remain low across the full size distribution.",
        fontsize=11.5,
        color=MUTED,
    )
    figure.text(
        0.07,
        0.035,
        "Notes: $1,000+ bets only. Confidence intervals are Wilson 95% intervals. "
        "Primary clustered regression p = 0.730; statistical association does not establish private information.",
        fontsize=8.5,
        color=MUTED,
    )

    figure.savefig(PNG_FILE, dpi=300, facecolor=BACKGROUND)
    figure.savefig(PDF_FILE, facecolor=BACKGROUND)
    print(f"Saved {PNG_FILE}")
    print(f"Saved {PDF_FILE}")


if __name__ == "__main__":
    main()
