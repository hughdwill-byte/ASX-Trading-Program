"""Self-check for the TimesFM strategy backtest (no model weights or network needed).

Run: python ml/test_timesfm_strategy.py
"""

import pathlib
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.timesfm_strategy import HORIZON, eligibility, run_backtest  # noqa: E402
from stocktrade import FilterConfig  # noqa: E402


def make_panel(days=120):
    idx = pd.bdate_range("2024-01-01", periods=days)
    growth = {"UP": 1.01, "FLAT": 1.0, "DOWN": 0.99}
    close = pd.DataFrame({s: 10 * g ** np.arange(days) for s, g in growth.items()}, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])  # no overnight gaps
    return {"open": open_, "close": close, "adj_close": close, "adj_open": open_,
            "volume": close * 0 + 1e6}


def test_backtest():
    panel = make_panel()
    seen_last = []

    def extrapolate(contexts):
        seen_last.extend(c[-1] for c in contexts)
        rate = np.array([c[-1] / c[-2] for c in contexts])[:, None]
        med = np.array([c[-1] for c in contexts])[:, None] * rate ** np.arange(1, HORIZON + 1)
        return med, med * 0.98, med * 1.02

    args = SimpleNamespace(start_date="2024-04-01", end_date=None, top=1, hold=2,
                           signal="timesfm", rank="sharpe", context=512)
    eligible = eligibility(panel, FilterConfig(enabled=False))
    with tempfile.TemporaryDirectory() as tmp:
        cache = pathlib.Path(tmp)
        report = run_backtest(panel, eligible, args, extrapolate, cache, 1000.0, 0.0, 0.0)
        periods = report["periods"]
        assert periods and all(p["picks"] == 1 for p in periods)
        assert all(t["symbol"] == "UP" for t in report["trades"])
        # entry at next open (= signal close), exit at close 2 sessions later.
        expected = 1000.0 * 1.01 ** (2 * len(periods))
        assert abs(report["strategy"]["final_equity"] - expected) < 1e-6, report["strategy"]

        # No look-ahead: every context ends exactly at its signal-date close.
        signal_closes = {round(panel["adj_close"].loc[p["signal_date"], s], 9)
                         for p in periods for s in panel["adj_close"].columns}
        assert {round(v, 9) for v in seen_last} <= signal_closes

        # Costs: two legs of $1 per trade.
        costly = run_backtest(panel, eligible, args, extrapolate, cache, 1000.0, 1.0, 0.0)
        assert costly["strategy"]["final_equity"] < report["strategy"]["final_equity"]

        # Cache: re-run must not call the model again.
        def boom(_):
            raise AssertionError("forecaster called despite cache")

        cached = run_backtest(panel, eligible, args, boom, cache, 1000.0, 0.0, 0.0)
        assert cached["strategy"]["final_equity"] == report["strategy"]["final_equity"]


if __name__ == "__main__":
    test_backtest()
    print("ok")
