"""Point-in-time universe and signals for the monthly strategy engine (strategy v2).

Everything is computed from data stamped at or before the close of day t. Frames are
precomputed once for every date so each backtest variant only does lookups.
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIGNALS = ("MOM_12_1", "RAMOM", "RESMOM", "LOWVOL", "COMBO", "TFM_REL", "TFM_RAW", "VOL_63")
TRADEABLE_SIGNALS = ("RAMOM", "RESMOM", "COMBO", "LOWVOL", "TFM_REL", "TFM_RAW")


class Context:
    """Precomputed calendar, prices, eligibility inputs and signal frames."""

    def __init__(self, panel: Dict[str, pd.DataFrame], aord: Optional[pd.Series], cfg: dict,
                 tfm_dir: Optional[pathlib.Path] = None, asic: Optional[pd.DataFrame] = None) -> None:
        adj = panel["adj_close"]
        counts = adj.notna().sum(axis=1)
        # Calendar: days on which a reasonable share of the market traded (drops stray Yahoo rows).
        keep = counts >= max(5, 0.3 * float(counts.median()))
        self.dates = adj.index[keep]
        cut = lambda f: f.loc[self.dates]
        self.A = cut(adj)
        self.C = cut(panel["close"])
        self.V = cut(panel["volume"]).fillna(0.0)
        self.AO = cut(panel["adj_open"])
        self.A_last = self.A.ffill()  # valuation: last known adjusted close
        self.cfg = cfg
        self.aord = aord.reindex(self.dates, method="ffill") if aord is not None else None
        self.tfm_dir = tfm_dir
        self.asic = asic.reindex(self.dates).ffill() if asic is not None else None  # usable SI per date
        self._tfm_cache: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
        self._build()

    # ---- precompute -------------------------------------------------------------------
    def _build(self) -> None:
        A, C, V = self.A, self.C, self.V
        ff = A.ffill(limit=5)
        ln = np.log(ff)
        r = np.log(A).diff()
        self.VOL_252 = r.rolling(252, min_periods=200).std() * math.sqrt(252)
        self.VOL_63 = r.rolling(63, min_periods=50).std() * math.sqrt(252)
        self.MOM_12_1 = ff.shift(21) / ff.shift(252) - 1
        self.RAMOM = (ln.shift(21) - ln.shift(252)) / (self.VOL_252 * math.sqrt(231 / 252))
        self.LOWVOL = -self.VOL_252
        # Residual momentum: strip the VAS-beta part of each stock's daily return. Beta uses only returns up to
        # the previous day (shift 1), and the 12-1 window ends 21 days back, so nothing after row i is used.
        m = r[self.cfg["core_ticker"].replace(".AX", "")]
        beta = r.rolling(756, min_periods=500).cov(m).div(m.rolling(756, min_periods=500).var(), axis=0).shift(1)
        e = (r - beta.mul(m, axis=0)).shift(21)
        self.RESMOM = e.rolling(231, min_periods=200).sum() / (e.rolling(231, min_periods=200).std() * math.sqrt(231))

        e = self.cfg
        dv = C * V
        self.MDV60 = dv.rolling(60, min_periods=60).median()
        present = A.notna()
        base = (
            present
            & (present.cumsum() >= e["min_history_days"])
            & (present.rolling(252, min_periods=1).sum() >= e["min_presence"] * 252)
            & (C >= e["min_price"])
            & (self.MDV60 >= e["min_mdv60"])
            & ((V == 0).rolling(20, min_periods=1).sum() <= e["max_zero_volume_days"])
            & ((A / A.shift(1) - 1).abs().fillna(0).rolling(252, min_periods=1).max() <= e["max_daily_move"])
        )
        excluded = set(e.get("exclude_tickers", [])) | {e["core_ticker"].replace(".AX", "")}
        base.loc[:, [c for c in base.columns if c in excluded]] = False
        self.base = base

    # ---- lookups ----------------------------------------------------------------------
    def eligible(self, i: int) -> List[str]:
        """eligibility_v2: base filters, then top `universe_size` by MDV60 at row i."""
        row = self.base.iloc[i]
        names = row.index[row.to_numpy()]
        if len(names) == 0:
            return []
        mdv = self.MDV60.iloc[i][names].sort_values(ascending=False, kind="mergesort")
        return list(mdv.index[: self.cfg["universe_size"]])

    def scores(self, signal: str, i: int, names: Sequence[str]) -> pd.Series:
        """Higher = better. NaN scores are dropped."""
        names = list(names)
        if not names:
            return pd.Series(dtype=float)
        if signal in ("RAMOM", "RESMOM", "LOWVOL", "MOM_12_1", "VOL_63"):
            s = getattr(self, signal).iloc[i][names]
        elif signal == "COMBO":
            s = 0.5 * zscore(self.RAMOM.iloc[i][names]) + 0.5 * zscore(self.LOWVOL.iloc[i][names])
        elif signal in ("TFM_REL", "TFM_RAW"):
            s = self.tfm_scores(signal, self.dates[i]).reindex(names)
        else:
            raise ValueError(f"Unknown signal {signal}")
        return s.astype(float).dropna()

    def ranks(self, signal: str, i: int) -> pd.Series:
        """Rank 1 = best among the eligible set. Ties: higher MDV60, then ticker."""
        names = self.eligible(i)
        s = self.scores(signal, i, names)
        if s.empty:
            return pd.Series(dtype=int)
        frame = pd.DataFrame({"s": s, "m": self.MDV60.iloc[i][s.index], "t": s.index})
        frame = frame.sort_values(["s", "m", "t"], ascending=[False, False, True])
        return pd.Series(np.arange(1, len(frame) + 1), index=frame.index)

    def si(self, i: int, ticker: str) -> float:
        if self.asic is None or ticker not in self.asic.columns:
            return float("nan")
        return float(self.asic.iloc[i][ticker])

    def tfm_scores(self, signal: str, date: pd.Timestamp) -> pd.Series:
        transform = "rel" if signal == "TFM_REL" else "raw"
        cache = self._tfm_cache.setdefault(transform, {})
        if date not in cache:
            path = (self.tfm_dir or pathlib.Path("data/tfm_cache")) / f"{transform}_{date:%Y-%m-%d}.csv"
            if not path.exists():
                cache[date] = pd.Series(dtype=float)
            else:
                df = pd.read_csv(path, index_col="ticker")
                cache[date] = df["score"]
        return cache[date]

    def decision_rows(self, start: Optional[str] = None, end: Optional[str] = None) -> List[int]:
        """Rows that are the last trading day of their month and have a next day to execute on."""
        d = self.dates
        month_end = np.r_[d[1:].month != d[:-1].month, False]
        rows = np.flatnonzero(month_end)
        if start:
            rows = rows[d[rows] >= pd.Timestamp(start)]
        if end:
            rows = rows[d[rows] <= pd.Timestamp(end)]
        return [int(i) for i in rows]


def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score after winsorising at the 2.5th / 97.5th percentiles."""
    s = s.astype(float)
    valid = s.dropna()
    if len(valid) < 3:
        return s * np.nan
    lo, hi = np.percentile(valid, [2.5, 97.5])
    w = s.clip(lo, hi)
    sd = w.std()
    return (w - w.mean()) / sd if sd and sd > 0 else w * 0.0


def tfm_cache_scores(ctx: Context, transform: str, date_row: int, names: Sequence[str], forecaster,
                     bench: pd.Series, context_len: int = 512, horizon: int = 21) -> pd.DataFrame:
    """TimesFM forecasts for one decision date. rel: ln A - ln B (B = core ETF); raw: price level."""
    hist = ctx.A.iloc[: date_row + 1][list(names)]
    series, keep = [], []
    lnb = np.log(bench.iloc[: date_row + 1])
    for t in names:
        a = hist[t]
        if transform == "rel":
            x = (np.log(a) - lnb).dropna().to_numpy()[-context_len:]
        else:
            x = a.dropna().to_numpy()[-context_len:]
        if len(x) >= 64:
            series.append(x)
            keep.append(t)
    if not series:
        return pd.DataFrame(columns=["ticker", "median", "q10", "q90", "score"])
    med, q10, q90 = forecaster(series)
    h = horizon - 1
    last = np.array([s[-1] for s in series])
    if transform == "rel":
        m = med[:, h] - last
        w = q90[:, h] - q10[:, h]
    else:
        m = med[:, h] / last - 1
        w = (q90[:, h] - q10[:, h]) / last
    score = m / np.maximum(w, 1e-9)
    return pd.DataFrame({"ticker": keep, "median": med[:, h], "q10": q10[:, h], "q90": q90[:, h], "score": score})
