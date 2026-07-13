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

## Setup

```bash
pip install numpy pandas requests scikit-learn yfinance pyyaml vaderSentiment colorama
python stocktrade.py generate-config   # writes a sample config.yaml
```

Edit `config.yaml` to set capital, risk parameters, and data sources.

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
