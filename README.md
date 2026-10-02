# ASX Trading Program

A research and backtesting toolkit for ASX equities. It:

- ingests historical price data
- engineers features, including ASIC short-position data and news sentiment
- trains machine-learning classifiers
- produces ranked daily buy/sell recommendations

All of it runs from a single CLI.

> **Disclaimer:** This is a personal research project, not financial advice. Signals are experimental, so trade at your own risk.

## Features

- **Data ingestion.** Downloads and caches daily OHLCV data from yfinance, EODHD or WebLink.
- **Short-position tracking.** `update_short_positions.py` pulls ASIC short-interest data into `short_positions.json`, which the models use as a feature.
- **ML signal models.** Random Forest and Histogram Gradient Boosting classifiers with time-series cross-validation and probability calibration (`ml/train_model.py`). There is also a PyTorch LSTM sequence model (`ml/train_sequence_model.py`) for long-range dependencies.
- **Backtesting.** Walk-forward backtests, with reports written to `reports/`.
- **Recommendations.** Scans the market and ranks the top candidates, sizing positions to the available capital.
- **Sentiment.** Optional VADER news-sentiment features.

## Setup

Requires Python 3.10+.

```bash
pip install numpy pandas requests scikit-learn joblib yfinance pyyaml
pip install vaderSentiment colorama   # optional: sentiment features, coloured output
pip install torch                     # optional: LSTM sequence model
python stocktrade.py generate-config  # writes a sample config.yaml
```

Edit `config.yaml` to set capital, risk parameters, symbols and data sources (plus an EODHD API key if you use that source). `config.yaml` is git-ignored, so keys stay local.

## Usage

```bash
python stocktrade.py ingest [--exchange AU --limit 200]   # download & cache historical data
python stocktrade.py backtest --symbol BHP                # backtest one symbol
python stocktrade.py live --symbol BHP                    # evaluate latest signal, simulate order
python stocktrade.py recommend --top 5 [--capital 10000]  # scan symbols, rank best candidates
python stocktrade.py yesterday [--symbol BHP]             # compare yesterday's predictions to outcomes
```

Global options are `--config path/to/config.yaml` and `--log-level DEBUG`. Run any command with `-h` to see all of its flags.

### Training the models

```bash
python ml/train_model.py --config config.yaml            # tabular RF/HGB → ml/model.pkl
python ml/train_sequence_model.py --config config.yaml \
  --sequence-length 60 --epochs 30 --folds 5             # LSTM sequence model
python update_short_positions.py                         # refresh ASIC short data
```

See `MLM` for notes on the sequence-model design and the optional MLflow tracking.

## Project layout

| Path | Purpose |
|---|---|
| `stocktrade.py` | Main CLI: ingest, backtest, live, recommend, yesterday, generate-config |
| `ml/dataset_builder.py` | Feature and label construction for tabular and sequence datasets |
| `ml/train_model.py` | RF/HGB training with time-series CV, saved to `ml/model.pkl` |
| `ml/train_sequence_model.py`, `ml/sequence_model.py` | LSTM sequence model and its training loop |
| `ml/predict.py` | Loads trained models for inference |
| `update_short_positions.py` / `_csv.py` | Refresh ASIC short-position data (JSON / CSV) |
| `available_symbols.json` | Cached ASX symbol universe |
| `reports/`, `Testsreports.txt` | Backtest output |
| `MLM` | Sequence-model integration notes |
