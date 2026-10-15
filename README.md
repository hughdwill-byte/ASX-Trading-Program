# ASX Trading Program

A research and backtesting tool for ASX equities. It ingests historical price data, engineers features (including ASIC short-position data and news sentiment), trains machine-learning classifiers, and produces daily buy/sell signal recommendations — all from a single CLI.

> **Disclaimer:** This is a personal research project, not financial advice. Signals are experimental; trade at your own risk.

## Features

- **Data ingestion** — downloads and caches historical OHLCV data for ASX symbols (via yfinance / EOD sources).
- **Short-position tracking** — `update_short_positions.py` pulls ASIC short-interest data into `short_positions.json` for use as a model feature.
- **ML signal models** — Random Forest and Histogram Gradient Boosting classifiers with time-series cross-validation and probability calibration (`ml/train_model.py`), plus an LSTM sequence model (`ml/train_sequence_model.py`) for long-range dependencies.
- **Backtesting** — walk-forward backtests with reports written to `reports/` (see `Testsreports.txt` and `reports/backtest*.json`).
- **Recommendations** — scans the market and ranks top candidates with capital-aware position sizing.
- **Sentiment** — optional VADER news-sentiment features.

## Live dashboard

**https://hughdwill-byte.github.io/ASX-Trading-Program/** shows today's TimesFM picks, forecast charts,
and a cost-inclusive six-month backtest against the All Ordinaries. The `Daily picks website` GitHub
Action rebuilds it every weekday at 5:17pm AEST (`.github/workflows/pages.yml`); run it on demand from
the Actions tab. One-time setup: **Settings → Pages → Source: GitHub Actions**.

## Strategy research (monthly)

The dashboard's recommendation comes from `strat-backtest`: 22 cost-aware monthly strategy variants
(risk-adjusted momentum, low volatility, a blend, two TimesFM signals; 5 or 8 positions; optional
All Ords trend filter) tested from 2016 against VAS.AX (total return), with walk-forward selection,
an untouched 2024-2026 hold-out, deflated Sharpe ratio and probability of backtest overfitting.
A strategy is recommended only if it passes all nine acceptance tests; otherwise the page says to hold
the index ETF. The `Strategy research` workflow runs it monthly (or from the Actions tab) and commits
`results/`. Design notes and deviations: `NOTES_strategy_v2.md`.

```bash
python stocktrade.py strat-backtest      # grid + validation -> results/
python stocktrade.py strat-recommend     # today's tier and next-open orders
python stocktrade.py strat-publish       # merge into docs/data/latest.json
```

## Quick start

```bash
pip install -r requirements.txt
python stocktrade.py ingest                          # download prices (uses config.example.yaml if no config.yaml)
python stocktrade.py tfm-recommend --capital 5000    # tomorrow's picks
python stocktrade.py tfm-publish && python -m http.server -d docs   # dashboard at http://localhost:8000
```

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml    # optional: edit capital, brokerage, filters
```

Without a `config.yaml` the program uses `config.example.yaml`. Optional extras for the legacy
commands: `pip install vaderSentiment colorama`.

## Usage

All commands run through `stocktrade.py`:

```bash
python stocktrade.py ingest                       # download & cache historical data
python stocktrade.py backtest --symbol BHP        # backtest one symbol
python stocktrade.py live --symbol BHP            # evaluate latest signal, simulate order
python stocktrade.py recommend --top 5            # scan symbols, rank best candidates
python stocktrade.py yesterday                    # evaluate yesterday's signals
```

Global options: `--config path/to/config.yaml`, `--log-level DEBUG`.

## TimesFM fast path

`tfm-backtest` / `tfm-recommend` use Google's [TimesFM 2.5](https://github.com/google-research/timesfm)
(zero-shot, Apache-2.0 weights) instead of training a classifier per symbol per day. Each rebalance
date runs one batched forecast over every symbol passing the liquidity filters, ranks by forecast
return, and holds the top N for `--hold` sessions (signal at close t, entry at open t+1, costs and
slippage from `config.yaml`). Forecasts are cached in `data/timesfm_cache/`, so re-runs are seconds.

```bash
pip install "timesfm[torch]" scikit-learn pyyaml yfinance   # first run downloads ~800MB of weights
python stocktrade.py ingest
python stocktrade.py tfm-backtest --start-date 2025-04-01 --end-date 2025-09-30 --top 5 --hold 5
python stocktrade.py tfm-backtest --start-date 2025-04-01 --end-date 2025-09-30 --signal momentum  # baseline
python stocktrade.py tfm-recommend --top 5 --capital 5000
python ml/test_timesfm_strategy.py                          # self-check, no weights needed
```

The report compares against the equal-weighted eligible universe and `--benchmark` (default `^AORD`).
Speed: ~20 forecasts/s on a 4-core CPU (a 6-month weekly backtest over ~400 stocks ≈ 8 min cold);
a CUDA GPU is used automatically. `--checkpoint` accepts a local folder with `model.safetensors`.

## Repo layout

| Path | Purpose |
|---|---|
| `stocktrade.py` | Main CLI: ingest, backtest, live, recommend, yesterday |
| `ml/` | Dataset builder, RF/HGB training, LSTM sequence model, saved `model.pkl` |
| `update_short_positions.py` | Refresh ASIC short-position data (JSON) |
| `update_short_positions_csv.py` | CSV variant of the short-position updater |
| `available_symbols.json` | Cached ASX symbol universe |
| `reports/` | Backtest output |
| `MLM` | Notes on the sequence-model integration |
