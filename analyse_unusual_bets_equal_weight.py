"""
analyse_unusual_bets.py
───────────────────────
Self-contained signal analysis using unusual_bets.json + politics_events.json.
No API calls needed — works entirely from the files you already have.

Run:
    python3 analyse_unusual_bets.py
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

# ─── Config ──────────────────────────────────────────────────────────────────

UNUSUAL_BETS_FILE  = "unusual_bets.json"
EVENTS_FILE        = "politics_events_0-10000.json"

# Signal thresholds
MIN_DOLLAR        = 1_000    # absolute floor for a "large" bet
MIN_PCT_LIQUIDITY = 0.02     # must be ≥ 2% of market liquidity
DEEP_OTM_MIN      = 0.01     # ignore near-zero noise
DEEP_OTM_MAX      = 0.20     # deep OTM ceiling

# Price buckets (lo_inclusive, hi_exclusive, label)
PRICE_BUCKETS = [
    (0.01, 0.03, "1–3%"),
    (0.03, 0.05, "3–5%"),
    (0.05, 0.10, "5–10%"),
    (0.10, 0.15, "10–15%"),
    (0.15, 0.20, "15–20%"),
]

TAKER_FEE = 0.02   # 2% taker fee
EQUAL_STAKE_USD = 100  # equal simulated spend per qualifying trade
MIN_TRADES = 15    # minimum per bucket for stats

# ─── Step 1 · Load & join ────────────────────────────────────────────────────

print("Loading data …")
with open(UNUSUAL_BETS_FILE) as f:
    bets = json.load(f)

with open(EVENTS_FILE) as f:
    events = json.load(f)

# Build market-level lookup so each bet is matched to its actual market and outcome.
market_lookup = {}
for ev in events:
    for mkt in ev.get("markets", []):
        mid = str(mkt.get("id", ""))
        if not mid:
            continue
        try:
            outcomes = json.loads(mkt.get("outcomes", "[]"))
            outcome_prices = [float(x) for x in json.loads(mkt.get("outcomePrices", "[]"))]
        except (TypeError, ValueError, json.JSONDecodeError):
            outcomes, outcome_prices = [], []

        resolved = bool(outcome_prices) and max(outcome_prices) == 1.0
        winning_outcome = outcomes[outcome_prices.index(1.0)] if resolved else None
        market_lookup[mid] = {
            "resolved": resolved,
            "winning_outcome": winning_outcome,
            "liquidity": float(mkt.get("liquidityNum") or mkt.get("liquidity") or 0),
        }

# Build DataFrame from bets
df = pd.DataFrame(bets)

# Normalise field names
df = df.rename(columns={
    "dollar_value" : "trade_value_usd",
    "price"        : "price",
    "outcome"      : "bet_side",       # Yes or No
    "event_id"     : "event_id",
})

df["price"]           = pd.to_numeric(df["price"], errors="coerce")
df["trade_value_usd"] = pd.to_numeric(df["trade_value_usd"], errors="coerce")
df["timestamp"]       = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")

# Join resolution data
df["resolved"]        = df["market_id"].astype(str).map(lambda x: market_lookup.get(x, {}).get("resolved", False))
df["winning_outcome"] = df["market_id"].astype(str).map(lambda x: market_lookup.get(x, {}).get("winning_outcome"))
df["liquidity"]       = df["market_id"].astype(str).map(lambda x: market_lookup.get(x, {}).get("liquidity", 0.0))

# ─── Step 2 · Compute whether THIS bet won ───────────────────────────────────
# Match the selected outcome to the winning outcome for this specific market.

def bet_won(row):
    if pd.isna(row["winning_outcome"]):
        return None
    return str(row["bet_side"]).strip().lower() == str(row["winning_outcome"]).strip().lower()

df["bet_won"] = df.apply(bet_won, axis=1)

# ─── Step 3 · Compute pct_of_liquidity ───────────────────────────────────────

df["pct_of_liquidity"] = (
    df["trade_value_usd"] / df["liquidity"].replace(0, float("nan"))
).round(4)

# ─── Step 4 · Overview ───────────────────────────────────────────────────────

total        = len(df)
matched      = df["resolved"].sum()
with_outcome = df["bet_won"].notna().sum()

print(f"\n── Dataset Overview ─────────────────────────────────────────")
print(f"  Total unusual bets:          {total:,}")
print(f"  Matched to resolved events:  {matched:,}")
print(f"  Bets with known outcome:     {with_outcome:,}")
print(f"  Price range:                 {df['price'].min():.3f} – {df['price'].max():.3f}")
print(f"  Dollar value range:          ${df['trade_value_usd'].min():,.0f} – ${df['trade_value_usd'].max():,.0f}")
print(f"  Bet sides:                   {df['bet_side'].value_counts().to_dict()}")

# ─── Step 5 · Filter to signal trades ────────────────────────────────────────

settled = df[df["bet_won"].notna()].copy()
settled["bet_won"] = settled["bet_won"].astype(bool)

signal = settled[
    (settled["price"] >= DEEP_OTM_MIN) &
    (settled["price"] <  DEEP_OTM_MAX) &
    (settled["trade_value_usd"] >= MIN_DOLLAR)
].copy()

print(f"\n── After Signal Filter (price<{DEEP_OTM_MAX:.0%}, usd≥${MIN_DOLLAR:,}) ──")
print(f"  Qualifying trades: {len(signal):,}")

# ─── Step 6 · Bucket analysis ────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)

rows = []
for lo, hi, label in PRICE_BUCKETS:
    bucket = signal[(signal["price"] >= lo) & (signal["price"] < hi)]
    n = len(bucket)
    if n < MIN_TRADES:
        continue
    wins         = bucket["bet_won"].sum()
    realised     = wins / n
    mean_price   = bucket["price"].mean()
    ci_lo, ci_hi = wilson_ci(wins, n)
    edge         = realised - mean_price
    se           = np.sqrt(mean_price * (1 - mean_price) / n)
    t_stat       = edge / se if se > 0 else np.nan
    p_val        = 1 - stats.norm.cdf(t_stat) if not np.isnan(t_stat) else np.nan
    rows.append({
        "bucket"        : label,
        "n"             : n,
        "mean_price"    : round(mean_price, 4),
        "realised_rate" : round(realised, 4),
        "ci_lo"         : round(ci_lo, 4),
        "ci_hi"         : round(ci_hi, 4),
        "edge"          : round(edge, 4),
        "t_stat"        : round(t_stat, 3) if not np.isnan(t_stat) else np.nan,
        "p_value"       : round(p_val, 4)  if not np.isnan(p_val)  else np.nan,
        "significant"   : bool(p_val < 0.05) if not np.isnan(p_val) else False,
    })

bucket_df = pd.DataFrame(rows)
print(f"\n── Bucket Statistics ────────────────────────────────────────")
print(bucket_df.to_string(index=False))

# ─── Step 7 · Equal-weight profitability ─────────────────────────────────────
# Equal-weight simulation:
# - trade_value_usd is still used only to decide whether a historical bet qualifies.
# - Every qualifying trade gets the same simulated spend: EQUAL_STAKE_USD.
# - shares = EQUAL_STAKE_USD / price
# - gross_payoff = shares × $1 if bet won
# - net_pnl = gross_payoff - simulated_spend - fee

signal = signal.copy()
signal["simulated_spend"] = EQUAL_STAKE_USD
signal["shares"]          = signal["simulated_spend"] / signal["price"]
signal["fee"]             = signal["simulated_spend"] * TAKER_FEE
signal["gross_payoff"]    = signal["bet_won"].astype(float) * signal["shares"]
signal["net_pnl"]         = signal["gross_payoff"] - signal["simulated_spend"] - signal["fee"]
signal["roi"]             = signal["net_pnl"] / (signal["simulated_spend"] + signal["fee"])

# Sanity check
wins  = signal["bet_won"].sum()
total = len(signal)
print(f"\n── Equal-Weight Profitability Sanity Check ─────────────────")
print(f"  Equal stake per trade: ${EQUAL_STAKE_USD:,.2f}")
print(f"  Winning trades: {wins} / {total} ({wins/total:.1%})")
print(f"  Sample rows:")
print(signal[["price","trade_value_usd","simulated_spend","shares","bet_won","gross_payoff","net_pnl","roi"]].head(5).to_string())

total_invested = (signal["simulated_spend"] + signal["fee"]).sum()
total_pnl      = signal["net_pnl"].sum()
win_rate       = signal["bet_won"].mean()
avg_roi        = signal["roi"].mean()
sharpe         = avg_roi / signal["roi"].std() if signal["roi"].std() > 0 else np.nan

print(f"\n── Equal-Weight Profitability Summary ──────────────────────")
print(f"  Trades simulated:    {len(signal):,}")
print(f"  Stake per trade:     ${EQUAL_STAKE_USD:,.2f}")
print(f"  Total invested:      ${total_invested:,.2f}")
print(f"  Total P&L:           ${total_pnl:,.2f}")
print(f"  Overall ROI:         {total_pnl/max(total_invested,1):.1%}")
print(f"  Win rate:            {win_rate:.1%}")
print(f"  Avg trade ROI:       {avg_roi:.1%}")
print(f"  Sharpe ratio:        {sharpe:.2f}" if not np.isnan(sharpe) else "  Sharpe ratio:        N/A")

# ─── Step 8 · Threshold scan ─────────────────────────────────────────────────

print(f"\n── Threshold Scan (edge = realised − implied) ───────────────")
scan_rows = []
for min_usd in [500, 1_000, 2_500, 5_000, 10_000]:
    for max_prob in [0.05, 0.10, 0.15, 0.20]:
        sub = settled[
            (settled["price"] >= DEEP_OTM_MIN) &
            (settled["price"] <  max_prob) &
            (settled["trade_value_usd"] >= min_usd)
        ]
        n = len(sub)
        if n < MIN_TRADES:
            scan_rows.append({"min_usd": min_usd, "max_prob": f"<{max_prob:.0%}", "n": n, "edge": np.nan, "p_val": np.nan})
            continue
        mp = sub["price"].mean()
        rr = sub["bet_won"].mean()
        se = np.sqrt(mp * (1-mp) / n)
        t  = (rr - mp) / se if se > 0 else np.nan
        p  = 1 - stats.norm.cdf(t) if not np.isnan(t) else np.nan
        scan_rows.append({"min_usd": min_usd, "max_prob": f"<{max_prob:.0%}", "n": n,
                          "edge": round(rr - mp, 4), "p_val": round(p, 4) if not np.isnan(p) else np.nan})

scan_df = pd.DataFrame(scan_rows)
pivot = scan_df.pivot_table(index="min_usd", columns="max_prob", values="edge", aggfunc="first")
print(pivot.to_string())

# ─── Save ─────────────────────────────────────────────────────────────────────


# ─── Step 9 · Focused 1–10% Profitability ────────────────────────────────────

signal_focused = settled[
    (settled["price"] >= 0.01) &
    (settled["price"] <  0.10) &
    (settled["trade_value_usd"] >= MIN_DOLLAR)
].copy()

signal_focused["simulated_spend"] = EQUAL_STAKE_USD
signal_focused["shares"]          = signal_focused["simulated_spend"] / signal_focused["price"]
signal_focused["fee"]             = signal_focused["simulated_spend"] * TAKER_FEE
signal_focused["gross_payoff"]    = signal_focused["bet_won"].astype(float) * signal_focused["shares"]
signal_focused["net_pnl"]         = signal_focused["gross_payoff"] - signal_focused["simulated_spend"] - signal_focused["fee"]
signal_focused["roi"]             = signal_focused["net_pnl"] / (signal_focused["simulated_spend"] + signal_focused["fee"])

wins   = signal_focused["bet_won"].sum()
total  = len(signal_focused)
ti     = (signal_focused["simulated_spend"] + signal_focused["fee"]).sum()
tp     = signal_focused["net_pnl"].sum()
sharpe = signal_focused["roi"].mean() / signal_focused["roi"].std() if signal_focused["roi"].std() > 0 else np.nan

print(f"\n── Focused 1–10% Profitability ──────────────────────────────")
print(f"  Trades:              {total:,}  ({wins} wins, {total-wins} losses)")
print(f"  Win rate:            {wins/total:.1%}")
print(f"  Stake per trade:     ${EQUAL_STAKE_USD:,.2f}")
print(f"  Total invested:      ${ti:,.2f}")
print(f"  Total P&L:           ${tp:,.2f}")
print(f"  Overall ROI:         {tp/max(ti,1):.1%}")
print(f"  Avg trade ROI:       {signal_focused['roi'].mean():.1%}")
print(f"  Median trade ROI:    {signal_focused['roi'].median():.1%}")
print(f"  Sharpe ratio:        {sharpe:.2f}" if not np.isnan(sharpe) else "  Sharpe ratio:        N/A")

print(f"\n  Per-bucket breakdown:")
for lo, hi, label in [(0.01, 0.03, "1–3%"), (0.03, 0.05, "3–5%"), (0.05, 0.10, "5–10%")]:
    b = signal_focused[(signal_focused["price"] >= lo) & (signal_focused["price"] < hi)]
    if len(b) == 0:
        continue
    bwins = b["bet_won"].sum()
    bti   = (b["simulated_spend"] + b["fee"]).sum()
    btp   = b["net_pnl"].sum()
    print(f"    {label:6s}  n={len(b):3d}  win={bwins/len(b):.1%}  "
          f"invested=${bti:,.0f}  P&L=${btp:,.0f}  ROI={btp/max(bti,1):.1%}")

import os
os.makedirs("data", exist_ok=True)
signal_focused.to_csv("data/signal_trades_focused_equal_weight.csv", index=False)
print(f"\nSaved signal_trades_focused_equal_weight.csv to data/")
