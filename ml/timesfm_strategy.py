"""TimesFM cross-sectional strategy: fast backtest + live recommendations.

Instead of training a classifier per symbol per day (the legacy ``yesterday`` path),
this runs ONE batched TimesFM forecast per rebalance date across every symbol that
passes the liquidity filters, ranks by forecast return, and holds the top N for
``hold`` sessions. Signals use data up to the close of day t; trades enter at the
open of t+1, so there is no look-ahead.

Forecasts are cached in ``data/timesfm_cache`` (one file per date+context), so
re-running with a different --top / --hold / --rank is near-instant.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stocktrade import (
    FilterConfig,
    _is_cache_valid,
    ensure_directory,
    project_path,
    sanitize_for_json,
    yf,
)

logger = logging.getLogger(__name__)

# ponytail: one forecast per (date, context) covers any --hold up to 32 sessions.
HORIZON = 32
DEFAULT_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"  # Apache-2.0 weights (3.0 is non-commercial)
MOMENTUM_LOOKBACK = 20
MIN_CONTEXT = 64

# contexts -> (median, q10, q90), each shaped [n_series, HORIZON]
Forecaster = Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def load_timesfm(
    checkpoint: str = DEFAULT_CHECKPOINT, context: int = 512, batch_size: int = 64
) -> Forecaster:
    """Load TimesFM 2.5 from a Hugging Face repo id or a local directory holding model.safetensors."""
    import timesfm
    import torch

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        checkpoint, torch_compile=torch.cuda.is_available()
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=context,
            max_horizon=HORIZON,
            normalize_inputs=True,
            per_core_batch_size=batch_size,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    def forecast(contexts: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # quantile channel 0 is the mean, 1..9 are the 10th..90th percentiles.
        _, quantiles = model.forecast(horizon=HORIZON, inputs=list(contexts))
        return quantiles[:, :, 5], quantiles[:, :, 1], quantiles[:, :, 9]

    return forecast


# -------- Data -------------------------------------------------------------------------


def load_panel(ingestor, symbols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Wide (date x symbol) frames from the cached per-symbol OHLCV CSVs."""
    frames: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = ingestor.fetch_ohlcv(sym)
        except Exception as err:
            logger.debug("Skipping %s: %s", sym, err)
            continue
        frames[sym] = df[~df.index.duplicated(keep="last")].sort_index()
    if not frames:
        raise ValueError("No OHLCV data available. Run `stocktrade.py ingest` first.")
    panel = {
        col: pd.DataFrame({sym: df[col] for sym, df in frames.items()}).sort_index()
        for col in ("open", "close", "adj_close", "volume")
    }
    # Adjust the open the same way as the close so open->close returns include dividends/splits.
    panel["adj_open"] = panel["open"] * (panel["adj_close"] / panel["close"])
    return panel


def eligibility(panel: Dict[str, pd.DataFrame], filters: FilterConfig) -> pd.DataFrame:
    """Boolean (date x symbol): passes the same prefilters as the legacy path, using only data <= date."""
    traded = panel["adj_close"].notna()
    history = traded.cumsum()
    mask = traded & (history >= MIN_CONTEXT)
    if not filters.enabled:
        return mask
    window = max(1, filters.avg_volume_window)
    close = panel["close"]
    dollar_volume = (
        panel["volume"].rolling(window, min_periods=1).mean()
        * close.rolling(window, min_periods=1).mean()
    )
    return (
        mask
        & (history >= filters.min_trading_days)
        & (close >= filters.min_price)
        & (dollar_volume >= filters.min_avg_dollar_volume)
    )


def load_benchmark(symbol: str, base_path: pathlib.Path, start: str) -> Optional[pd.Series]:
    path = project_path(str(base_path), f"benchmark_{symbol.lstrip('^')}.csv")
    if not _is_cache_valid(path, 24 * 60 * 60) and yf is not None:
        try:
            df = yf.download(symbol, start=start, auto_adjust=True, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                ensure_directory(path.parent)
                df[["Close"]].to_csv(path)
        except Exception as err:
            logger.warning("Benchmark %s download failed: %s", symbol, err)
    if not path.exists():
        return None
    series = pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.sort_index()


# -------- Signals ----------------------------------------------------------------------


def forecast_on(
    date: pd.Timestamp,
    symbols: Sequence[str],
    adj: pd.DataFrame,
    forecaster: Forecaster,
    context: int,
    cache_dir: pathlib.Path,
) -> Dict[str, np.ndarray]:
    """TimesFM forecasts for `symbols` using closes up to and including `date`, cached per date."""
    path = cache_dir / f"ctx{context}_{date:%Y%m%d}.npz"
    store: Dict[str, Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
    if path.exists():
        z = np.load(path)
        for i, sym in enumerate(z["symbols"]):
            store[str(sym)] = (z["last"][i], z["med"][i], z["q10"][i], z["q90"][i])

    missing = [s for s in symbols if s not in store]
    if missing:
        history = adj.loc[:date, missing]
        contexts = [history[s].dropna().to_numpy()[-context:] for s in missing]
        med, q10, q90 = forecaster(contexts)
        for j, sym in enumerate(missing):
            store[sym] = (contexts[j][-1], med[j], q10[j], q90[j])
        ensure_directory(cache_dir)
        keys = list(store)
        np.savez(
            path,
            symbols=np.array(keys),
            last=np.array([store[k][0] for k in keys]),
            med=np.stack([store[k][1] for k in keys]),
            q10=np.stack([store[k][2] for k in keys]),
            q90=np.stack([store[k][3] for k in keys]),
        )

    return {
        "last": np.array([store[s][0] for s in symbols], dtype=float),
        "med": np.stack([store[s][1] for s in symbols]).astype(float),
        "q10": np.stack([store[s][2] for s in symbols]).astype(float),
        "q90": np.stack([store[s][3] for s in symbols]).astype(float),
    }


def pick(
    i: int,
    panel: Dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    args,
    forecaster: Optional[Forecaster],
    cache_dir: pathlib.Path,
    min_edge: float,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Rank eligible symbols on row i; return (top picks, full eligible universe)."""
    adj = panel["adj_close"]
    universe = list(eligible.columns[eligible.iloc[i].to_numpy()])
    if not universe:
        return [], universe
    if args.signal == "momentum":
        if i < MOMENTUM_LOOKBACK:
            return [], universe
        expected = (adj.iloc[i][universe] / adj.iloc[i - MOMENTUM_LOOKBACK][universe] - 1).to_numpy()
        score = expected
        low = high = np.full(len(universe), np.nan)
    else:
        fc = forecast_on(adj.index[i], universe, adj, forecaster, args.context, cache_dir)
        h = args.hold - 1
        expected = fc["med"][:, h] / fc["last"] - 1
        low = fc["q10"][:, h] / fc["last"] - 1
        high = fc["q90"][:, h] / fc["last"] - 1
        # sharpe: reward per unit of forecast uncertainty, so volatile penny stocks don't dominate.
        score = expected / np.maximum(high - low, 1e-6) if args.rank == "sharpe" else expected

    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))
    picks = [
        {"symbol": universe[j], "expected": float(expected[j]), "score": float(score[j]),
         "q10": float(low[j]), "q90": float(high[j])}
        for j in order
        if np.isfinite(score[j]) and expected[j] > min_edge
    ][: args.top]
    return picks, universe


# -------- Backtest ---------------------------------------------------------------------


def run_backtest(
    panel: Dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    args,
    forecaster: Optional[Forecaster],
    cache_dir: pathlib.Path,
    initial_capital: float,
    trade_cost: float,
    slippage_bps: float,
    benchmark: Optional[pd.Series] = None,
) -> Dict[str, object]:
    idx = panel["adj_close"].index
    start, end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date or idx[-1])
    positions = np.flatnonzero((idx >= start) & (idx <= end))
    slip = slippage_bps / 10_000
    hold = args.hold

    equity = bench_equity = universe_equity = float(initial_capital)
    periods: List[Dict[str, object]] = []
    trades: List[Dict[str, object]] = []
    started = time.monotonic()

    rebalances = [i for i in positions[::hold] if i + hold < len(idx)]
    for n, i in enumerate(rebalances, start=1):
        if equity <= 0:
            logger.warning("Equity exhausted; stopping backtest.")
            break
        signal_date, exit_date = idx[i], idx[i + hold]  # entry is the open of idx[i + 1]
        picks, universe = pick(i, panel, eligible, args, forecaster, cache_dir, 2 * slip)
        entry_px = panel["adj_open"].iloc[i + 1]
        exit_px = panel["adj_close"].iloc[i + 1 : i + hold + 1].ffill().iloc[-1]

        alloc = equity / args.top
        pnl = 0.0
        for p in picks:
            r = exit_px[p["symbol"]] / entry_px[p["symbol"]] - 1
            if not np.isfinite(r):  # no open price on entry day: couldn't buy, stays cash
                continue
            net = (1 + r) * (1 - slip) / (1 + slip) - 1
            trade_pnl = alloc * net - 2 * trade_cost
            pnl += trade_pnl
            trades.append({"signal_date": signal_date, "exit_date": exit_date, **p,
                           "return": float(r), "pnl": float(trade_pnl)})
        period_return = pnl / equity if equity > 0 else 0.0
        equity += pnl

        universe_ret = float(np.nanmean((exit_px[universe] / entry_px[universe] - 1).to_numpy())) if universe else 0.0
        universe_equity *= 1 + (universe_ret if np.isfinite(universe_ret) else 0.0)
        bench_ret = float("nan")
        if benchmark is not None:
            b = benchmark.reindex([signal_date, exit_date], method="ffill")
            bench_ret = float(b.iloc[1] / b.iloc[0] - 1)
            bench_equity *= 1 + (bench_ret if np.isfinite(bench_ret) else 0.0)

        periods.append({"signal_date": signal_date, "exit_date": exit_date, "picks": len(picks),
                        "universe": len(universe), "return": period_return,
                        "universe_return": universe_ret, "benchmark_return": bench_ret,
                        "equity": equity})
        logger.info("Rebalance %d/%d %s | picks=%d universe=%d ret=%+.2f%% | %.1fs elapsed",
                    n, len(rebalances), signal_date.date(), len(picks), len(universe),
                    100 * period_return, time.monotonic() - started)

    returns = pd.Series([p["return"] for p in periods], dtype=float)
    return {
        "settings": {k: getattr(args, k) for k in ("start_date", "end_date", "top", "hold", "signal", "rank", "context")},
        "strategy": _summarise(returns, equity, initial_capital, hold),
        "universe_equal_weight": _summarise(
            pd.Series([p["universe_return"] for p in periods], dtype=float), universe_equity, initial_capital, hold),
        "benchmark": (_summarise(pd.Series([p["benchmark_return"] for p in periods], dtype=float),
                                 bench_equity, initial_capital, hold) if benchmark is not None else None),
        "trades_count": len(trades),
        "hit_rate": float(np.mean([t["return"] > 0 for t in trades])) if trades else float("nan"),
        "avg_trade_return": float(np.mean([t["return"] for t in trades])) if trades else float("nan"),
        "elapsed_seconds": time.monotonic() - started,
        "periods": periods,
        "trades": trades,
    }


def _summarise(returns: pd.Series, final_equity: float, initial: float, hold: int) -> Dict[str, float]:
    returns = returns.fillna(0.0)
    per_year = 252 / hold
    curve = initial * (1 + returns).cumprod()
    std = returns.std()
    return {
        "final_equity": float(final_equity),
        "total_return": float(final_equity / initial - 1),
        "cagr": float((final_equity / initial) ** (per_year / max(1, len(returns))) - 1) if final_equity > 0 else -1.0,
        "sharpe": float(returns.mean() / std * math.sqrt(per_year)) if std and std > 0 else float("nan"),
        "max_drawdown": float((curve / curve.cummax() - 1).min()) if len(curve) else 0.0,
    }


# -------- CLI --------------------------------------------------------------------------


def add_cli_args(parser) -> None:
    parser.add_argument("--top", type=int, default=5, help="Positions held per rebalance (default 5).")
    parser.add_argument("--hold", type=int, default=5, help=f"Sessions per holding period, 1-{HORIZON} (default 5).")
    parser.add_argument("--signal", choices=["timesfm", "momentum"], default="timesfm",
                        help="Ranking signal; momentum is a cheap baseline to check TimesFM adds value.")
    parser.add_argument("--rank", choices=["sharpe", "return"], default="sharpe",
                        help="Rank by forecast return / forecast spread (sharpe) or raw forecast return.")
    parser.add_argument("--context", type=int, default=512, help="TimesFM context length in sessions (default 512).")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Hugging Face repo id or local folder containing model.safetensors.")
    parser.add_argument("--batch-size", type=int, default=64, help="TimesFM inference batch size.")
    parser.add_argument("--limit", type=int, help="Limit number of symbols loaded.")
    parser.add_argument("--benchmark", default="^AORD", help="Benchmark index ticker (default ^AORD, All Ordinaries).")


def run_cli(args, orchestrator) -> int:
    from ml.dataset_builder import _resolve_symbol_list

    if not 1 <= args.hold <= HORIZON:
        raise SystemExit(f"--hold must be between 1 and {HORIZON}.")
    config = orchestrator.config
    started = time.monotonic()
    symbols = _resolve_symbol_list(None, config, orchestrator.ingestor, args.limit)
    panel = load_panel(orchestrator.ingestor, symbols)
    eligible = eligibility(panel, config.filters)
    logger.info("Loaded %d symbols in %.1fs.", panel["adj_close"].shape[1], time.monotonic() - started)

    forecaster = load_timesfm(args.checkpoint, args.context, args.batch_size) if args.signal == "timesfm" else None
    cache_dir = project_path(str(config.data.base_path), "timesfm_cache")

    if args.command == "tfm-recommend":
        return _print_recommendations(panel, eligible, args, forecaster, cache_dir, config)

    benchmark = load_benchmark(args.benchmark, config.data.base_path, config.data.start_date)
    report = run_backtest(panel, eligible, args, forecaster, cache_dir, config.backtest.initial_capital,
                          config.backtest.trade_cost, config.backtest.slippage_bps, benchmark)
    _print_backtest(report, args)
    path = project_path("reports", f"tfm_backtest_{args.signal}_{args.start_date}_{args.end_date or 'latest'}.json")
    ensure_directory(path.parent)
    path.write_text(json.dumps(sanitize_for_json(report), indent=2), encoding="utf-8")
    print(f"\nReport written to {path}")
    return 0


def _print_backtest(report: Dict[str, object], args) -> None:
    print(f"\n{'Signal':<10} {'Exit':<12} {'Picks':>5} {'Univ':>5} {'Strat%':>8} {'UnivEW%':>8} {'Bench%':>8} {'Equity':>12}")
    print("-" * 76)
    pct = lambda x: f"{100 * x:>+8.2f}" if np.isfinite(x) else f"{'n/a':>8}"
    for p in report["periods"]:
        print(f"{p['signal_date']:%Y-%m-%d} {p['exit_date']:%Y-%m-%d} {p['picks']:>6} {p['universe']:>5} "
              f"{pct(p['return'])} {pct(p['universe_return'])} {pct(p['benchmark_return'])} {p['equity']:>12.2f}")
    print(f"\n{'':<24} {'Total%':>8} {'CAGR%':>8} {'Sharpe':>7} {'MaxDD%':>8}")
    for label, key in (("Strategy", "strategy"), ("Universe equal-weight", "universe_equal_weight"),
                       (f"Benchmark {args.benchmark}", "benchmark")):
        s = report.get(key)
        if s:
            print(f"{label:<24} {100 * s['total_return']:>+8.2f} {100 * s['cagr']:>+8.2f} "
                  f"{s['sharpe']:>7.2f} {100 * s['max_drawdown']:>8.2f}")
    print(f"\nTrades: {report['trades_count']} | Hit rate: {100 * report['hit_rate']:.1f}% | "
          f"Avg trade return: {100 * report['avg_trade_return']:+.2f}% | Ran in {report['elapsed_seconds']:.1f}s")


def _print_recommendations(panel, eligible, args, forecaster, cache_dir, config) -> int:
    i = len(panel["adj_close"].index) - 1
    picks, universe = pick(i, panel, eligible, args, forecaster, cache_dir, 2 * config.backtest.slippage_bps / 10_000)
    capital = args.capital or config.backtest.initial_capital
    budget = capital * (1 - config.risk.min_cash_buffer) / args.top
    print(f"\nTimesFM picks for next open after {panel['adj_close'].index[i]:%Y-%m-%d} "
          f"(hold {args.hold} sessions, {len(universe)} eligible)")
    print(f"{'Symbol':<8} {'Price':>8} {'Fcst%':>7} {'Q10%':>7} {'Q90%':>7} {'Score':>7} {'Qty':>6} {'Spend':>9}")
    print("-" * 66)
    for p in picks:
        price = float(panel["close"].iloc[i][p["symbol"]])
        qty = math.floor(budget / price) if price > 0 else 0
        print(f"{p['symbol']:<8} {price:>8.3f} {100 * p['expected']:>+7.2f} {100 * p['q10']:>+7.2f} "
              f"{100 * p['q90']:>+7.2f} {p['score']:>7.3f} {qty:>6} {qty * price:>9.2f}")
    if not picks:
        print("No symbol has a positive forecast edge after costs; stay in cash.")
    return 0
