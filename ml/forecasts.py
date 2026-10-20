"""Causal calibration of signal ranks to horizon arithmetic returns (research optimiser input, F14/F27).

A rank is not a return. For decision row i the calibration uses only earlier decisions r whose traded label
(entry at the open of r+1, exit at the close of r+horizon) is complete by the close of i. Two pooled,
shrunk estimates:
  - market level m: mean label of the core ETF over the training decisions, shrunk to 0 by T/(T+T0);
  - cross-sectional slope b: per-date OLS slope of demeaned labels on centred percentile rank, averaged
    over dates and shrunk to 0 by T/(T+Tb). Its standard error comes from the date-to-date dispersion
    (each date one observation, so same-date cross-sectional correlation is not double counted).
mu_i = m + b (p_i - 1/2). mean-estimation uncertainty mu_se_i = sqrt(se_m^2 + (p_i - 1/2)^2 se_b^2), kept
separate from predictive dispersion (not estimated here).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CALIBRATION_ID = "pooled_rank_slope_shrunk_v1"


@dataclasses.dataclass
class ForecastBundle:
    as_of: pd.Timestamp
    available_at: pd.Timestamp
    trained_until: Optional[pd.Timestamp]   # last decision used for training
    label_cutoff: Optional[pd.Timestamp]    # latest label end used (<= as_of)
    mu: pd.Series                           # expected arithmetic return over the horizon, fraction
    mu_se: pd.Series                        # standard error of the estimated mean
    horizon: int
    target: str = "arithmetic_return_next_open_to_close_h"
    units: str = "fraction_per_horizon"
    model_id: str = ""
    calibration_id: str = CALIBRATION_ID
    n_train_dates: int = 0
    status: str = "ok"                      # ok | insufficient_history | missing_signal

    def identity(self) -> str:
        blob = json.dumps({"as_of": str(self.as_of), "trained_until": str(self.trained_until), "h": self.horizon,
                           "model": self.model_id, "cal": self.calibration_id, "target": self.target,
                           "mu": {k: round(float(v), 10) for k, v in self.mu.sort_index().items()}}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _percentile(ranks: pd.Series) -> pd.Series:
    n = len(ranks)
    return 1 - (ranks - 1) / max(n - 1, 1)


def traded_label(ctx, r: int, horizon: int, names) -> pd.Series:
    """Entry at the open of r+1, exit at the close of r+horizon (both adjusted: a research label)."""
    names = list(names)
    return (ctx.A_last.iloc[r + horizon][names] / ctx.AO.iloc[r + 1][names] - 1).replace([np.inf, -np.inf], np.nan)


def calibrate(ctx, signal: str, i: int, horizon: int = 21, train_decisions: int = 60, t0_market: float = 60.0,
              t0_slope: float = 24.0, rank_cache: Optional[Dict] = None, min_dates: int = 12) -> ForecastBundle:
    core = ctx.cfg["core_ticker"].replace(".AX", "")

    def ranks_at(r: int) -> pd.Series:
        rk = rank_cache.get((signal, r)) if rank_cache is not None else None
        return ctx.ranks(signal, r) if rk is None else rk

    all_rows = [r for r in ctx.decision_rows() if r + horizon <= i]  # labels complete by the close of i
    train = all_rows[-train_decisions:]
    slopes: List[float] = []
    market: List[float] = []
    for r in train:
        rk = ranks_at(r)
        if len(rk) >= 10:
            y = traded_label(ctx, r, horizon, rk.index)
            p = _percentile(rk)
            ok = y.notna()
            if ok.sum() >= 10:
                pc = p[ok] - p[ok].mean()
                yc = y[ok] - y[ok].mean()
                denom = float((pc ** 2).sum())
                if denom > 0:
                    slopes.append(float((pc * yc).sum() / denom))
        m = traded_label(ctx, r, horizon, [core]).iloc[0] if core in ctx.A.columns else np.nan
        if np.isfinite(m):
            market.append(float(m))
    now = ranks_at(i)
    as_of = ctx.dates[i]
    status = "ok"
    if len(slopes) < min_dates:
        status = "insufficient_history"
    T, Tm = len(slopes), len(market)
    b = float(np.mean(slopes)) * T / (T + t0_slope) if T else 0.0
    se_b = float(np.std(slopes, ddof=1) / np.sqrt(T)) if T > 1 else float("nan")
    m = float(np.mean(market)) * Tm / (Tm + t0_market) if Tm else 0.0
    se_m = float(np.std(market, ddof=1) / np.sqrt(Tm)) if Tm > 1 else float("nan")
    p = _percentile(now) if len(now) else pd.Series(dtype=float)
    mu = m + b * (p - 0.5)
    mu_se = np.sqrt(se_m ** 2 + (p - 0.5) ** 2 * se_b ** 2)
    mu[core] = m
    mu_se[core] = se_m
    return ForecastBundle(as_of=as_of, available_at=as_of, trained_until=ctx.dates[train[-1]] if train else None,
                          label_cutoff=ctx.dates[train[-1] + horizon] if train else None, mu=mu.astype(float),
                          mu_se=mu_se.astype(float), horizon=horizon, model_id=f"rank:{signal}",
                          n_train_dates=T, status=status if len(now) else "missing_signal")
