# BEM Final

Analysis of unusual Polymarket bets, including equal-weighting, bet-size robustness, and tail-risk figures.

## Files

- `analyse_unusual_bets_equal_weight.py`: corrected equal-weight profitability analysis.
- `unusual_bets.json`: unusual bet dataset.
- `politics_events_0-10000.json`: politics event and market metadata.
- `bet_size_analysis/`: bet-size statistical analysis, generated tables, reports, and figures.
- `data/`: generated equal-weight signal trade output.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python analyse_unusual_bets_equal_weight.py
cd bet_size_analysis
../.venv/bin/python analyse_bet_size_statistics.py
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib ../.venv/bin/python create_bet_size_figure.py
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib ../.venv/bin/python create_tail_figure.py
```
