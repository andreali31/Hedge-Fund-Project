"""
Test whether larger unusual bets have better subsequent performance.

The primary model is a logistic regression with standard errors clustered by
market. A within-market fixed-effects model and a within-market permutation
test provide stricter robustness checks.
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BETS_FILE = os.path.join(PROJECT_DIR, "unusual_bets.json")
EVENTS_FILE = os.path.join(PROJECT_DIR, "politics_events_0-10000.json")
OUTPUT_DIR = SCRIPT_DIR

MIN_PRICE = 0.01
MAX_PRICE = 0.10
MIN_DOLLAR = 1_000
TAKER_FEE = 0.02
N_PERMUTATIONS = 5_000
RANDOM_SEED = 20260604


def parse_json_list(value):
    if isinstance(value, list):
        return value
    try:
        return json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []


def build_market_lookup(events):
    lookup = {}
    for event in events:
        for market in event.get("markets", []):
            market_id = str(market.get("id", ""))
            outcomes = parse_json_list(market.get("outcomes"))
            try:
                outcome_prices = [float(x) for x in parse_json_list(market.get("outcomePrices"))]
            except (TypeError, ValueError):
                outcome_prices = []

            resolved = bool(outcome_prices) and max(outcome_prices) == 1.0
            winning_outcome = outcomes[outcome_prices.index(1.0)] if resolved else None
            lookup[market_id] = {
                "winning_outcome": winning_outcome,
                "market_volume": float(market.get("volumeNum") or market.get("volume") or 0),
                "resolution_time": market.get("closedTime") or market.get("endDate"),
            }
    return lookup


def prepare_data(bets, markets):
    df = pd.DataFrame(bets).copy()
    df["market_id"] = df["market_id"].astype(str)
    df["trade_value_usd"] = pd.to_numeric(df["dollar_value"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["trade_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    df["winning_outcome"] = df["market_id"].map(
        lambda x: markets.get(x, {}).get("winning_outcome")
    )
    df["market_volume"] = df["market_id"].map(
        lambda x: markets.get(x, {}).get("market_volume", 0)
    )
    df["resolution_time"] = pd.to_datetime(
        df["market_id"].map(lambda x: markets.get(x, {}).get("resolution_time")),
        utc=True,
        errors="coerce",
    )

    df = df[
        df["winning_outcome"].notna()
        & df["price"].between(MIN_PRICE, MAX_PRICE, inclusive="left")
        & (df["trade_value_usd"] >= MIN_DOLLAR)
    ].copy()

    df["bet_won"] = (
        df["outcome"].astype(str).str.strip().str.lower()
        == df["winning_outcome"].astype(str).str.strip().str.lower()
    )
    df["bet_won_num"] = df["bet_won"].astype(int)
    df["log2_size"] = np.log2(df["trade_value_usd"])
    df["price_sq"] = df["price"] ** 2
    df["log_volume"] = np.log1p(df["market_volume"])
    df["days_to_resolution"] = (
        (df["resolution_time"] - df["trade_time"]).dt.total_seconds() / 86_400
    ).clip(lower=0)
    df["days_missing"] = df["days_to_resolution"].isna().astype(int)
    df["days_to_resolution"] = df["days_to_resolution"].fillna(
        df["days_to_resolution"].median()
    )
    df["net_roi"] = (
        df["bet_won_num"] / df["price"] - 1 - TAKER_FEE
    ) / (1 + TAKER_FEE)
    return df


def size_quintiles(df):
    work = df.copy()
    work["size_quintile"] = pd.qcut(
        work["trade_value_usd"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    return work.groupby("size_quintile", observed=True).agg(
        trades=("bet_won", "size"),
        median_bet_usd=("trade_value_usd", "median"),
        mean_price=("price", "mean"),
        win_rate=("bet_won", "mean"),
        mean_net_roi=("net_roi", "mean"),
    )


def within_market_permutation_test(df):
    usable = df.groupby("market_id").filter(
        lambda group: group["log2_size"].nunique() > 1 and group["bet_won_num"].nunique() > 1
    ).copy().reset_index(drop=True)
    usable["x_demeaned"] = usable["log2_size"] - usable.groupby("market_id")["log2_size"].transform("mean")
    usable["y_demeaned"] = usable["bet_won_num"] - usable.groupby("market_id")["bet_won_num"].transform("mean")

    denominator = np.dot(usable["x_demeaned"], usable["x_demeaned"])
    observed_slope = np.dot(usable["x_demeaned"], usable["y_demeaned"]) / denominator

    rng = np.random.default_rng(RANDOM_SEED)
    groups = [group.index.to_numpy() for _, group in usable.groupby("market_id")]
    original = usable["log2_size"].to_numpy()
    y_demeaned = usable["y_demeaned"].to_numpy()
    permuted_slopes = np.empty(N_PERMUTATIONS)

    for i in range(N_PERMUTATIONS):
        permuted = original.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        permuted_demeaned = permuted - pd.Series(permuted).groupby(usable["market_id"].to_numpy()).transform("mean").to_numpy()
        permuted_slopes[i] = np.dot(permuted_demeaned, y_demeaned) / np.dot(
            permuted_demeaned, permuted_demeaned
        )

    p_value = (np.sum(np.abs(permuted_slopes) >= abs(observed_slope)) + 1) / (
        N_PERMUTATIONS + 1
    )
    return usable, observed_slope, p_value


def coefficient_table(result):
    table = pd.DataFrame(
        {
            "coefficient": result.params,
            "std_error": result.bse,
            "p_value": result.pvalues,
            "ci_low": result.conf_int()[0],
            "ci_high": result.conf_int()[1],
        }
    )
    return table


def main():
    with open(BETS_FILE) as file:
        bets = json.load(file)
    with open(EVENTS_FILE) as file:
        events = json.load(file)

    df = prepare_data(bets, build_market_lookup(events))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    quintiles = size_quintiles(df)
    spearman = stats.spearmanr(df["log2_size"], df["net_roi"])

    controls = "log2_size + price + price_sq + log_volume + days_to_resolution + days_missing"
    logit_market = smf.glm(
        f"bet_won_num ~ {controls}", data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["market_id"]})
    logit_wallet = smf.glm(
        f"bet_won_num ~ {controls}", data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["wallet"]})

    within_market = smf.ols(
        f"bet_won_num ~ {controls} + C(market_id)", data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["market_id"]})

    permutation_sample, permutation_slope, permutation_p = within_market_permutation_test(df)

    market_table = coefficient_table(logit_market)
    wallet_table = coefficient_table(logit_wallet)
    within_table = coefficient_table(within_market)
    odds_ratio_per_doubling = np.exp(logit_market.params["log2_size"])
    odds_ci = np.exp(logit_market.conf_int().loc["log2_size"])

    df.to_csv(f"{OUTPUT_DIR}/bet_size_analysis_sample.csv", index=False)
    quintiles.to_csv(f"{OUTPUT_DIR}/bet_size_quintiles.csv")
    market_table.to_csv(f"{OUTPUT_DIR}/bet_size_logit_market_clustered.csv")
    wallet_table.to_csv(f"{OUTPUT_DIR}/bet_size_logit_wallet_clustered.csv")
    within_table.to_csv(f"{OUTPUT_DIR}/bet_size_within_market_model.csv")

    report = f"""BET SIZE AND PERFORMANCE STATISTICAL ANALYSIS

Sample
------
Trades: {len(df):,}
Markets: {df['market_id'].nunique():,}
Wallets: {df['wallet'].nunique():,}
Wins: {df['bet_won_num'].sum():,}
Win rate: {df['bet_won_num'].mean():.3%}
Filter: ${MIN_DOLLAR:,}+ bets with prices from {MIN_PRICE:.0%} to below {MAX_PRICE:.0%}

Data limitation
---------------
Liquidity is missing for nearly all trades, so log market volume is used as the
available market-scale control. Statistical association is not proof of insider
information.

Unadjusted return relationship
------------------------------
Spearman correlation, log2 bet size versus net ROI: {spearman.statistic:.6f}
Naive Spearman p-value: {spearman.pvalue:.6f}

Primary logistic regression, standard errors clustered by market
----------------------------------------------------------------
log2 bet-size coefficient: {logit_market.params['log2_size']:.6f}
Clustered standard error: {logit_market.bse['log2_size']:.6f}
p-value: {logit_market.pvalues['log2_size']:.6f}
Odds ratio for each doubling of bet size: {odds_ratio_per_doubling:.6f}
95% odds-ratio CI: [{odds_ci.iloc[0]:.6f}, {odds_ci.iloc[1]:.6f}]

Wallet-clustered robustness check
---------------------------------
log2 bet-size coefficient: {logit_wallet.params['log2_size']:.6f}
p-value: {logit_wallet.pvalues['log2_size']:.6f}

Within-market fixed-effects linear probability model
----------------------------------------------------
Change in win probability per doubling of bet size: {within_market.params['log2_size']:.6f}
Market-clustered standard error: {within_market.bse['log2_size']:.6f}
p-value: {within_market.pvalues['log2_size']:.6f}

Within-market permutation test
------------------------------
Usable trades: {len(permutation_sample):,}
Usable markets: {permutation_sample['market_id'].nunique():,}
Observed within-market slope: {permutation_slope:.6f}
Two-sided permutation p-value ({N_PERMUTATIONS:,} permutations): {permutation_p:.6f}

Interpretation rule
-------------------
A positive log2 bet-size coefficient with a small p-value would support the
claim that larger bets perform better after controls. It would not establish
that private or insider information caused the relationship.

Size quintiles
--------------
{quintiles.to_string()}
"""
    with open(f"{OUTPUT_DIR}/bet_size_statistics_report.txt", "w") as file:
        file.write(report)
    print(report)
    print(f"\nSaved detailed outputs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
