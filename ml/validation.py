"""Validation for strategy v2: metrics, bootstrap CIs, PSR/DSR, PBO (CSCV), rank IC, walk-forward."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import pathlib
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

EULER = 0.5772156649
PERIODS = {
    "design": ("2016-03-01", "2019-06-30"),
    "walk_forward": ("2019-07-01", "2024-06-30"),
    "hold_out": ("2024-07-01", "2099-12-31"),
    "full": ("2016-01-01", "2099-12-31"),
}
WF_YEARS = (2019, 2020, 2021, 2022, 2023)


def window(s: pd.Series, period: str) -> pd.Series:
    a, b = PERIODS[period]
    return s.loc[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]


# ---- core statistics ---------------------------------------------------------------------

def cagr(monthly: pd.Series) -> float:
    if len(monthly) == 0:
        return float("nan")
    growth = float((1 + monthly).prod())
    return growth ** (12 / len(monthly)) - 1 if growth > 0 else -1.0


def sharpe(monthly: pd.Series) -> float:
    sd = monthly.std()
    return float(monthly.mean() / sd * math.sqrt(12)) if len(monthly) > 2 and sd > 0 else float("nan")


def max_drawdown(monthly: pd.Series) -> float:
    curve = (1 + monthly).cumprod()
    return float((curve / curve.cummax().clip(lower=1.0) - 1).min()) if len(curve) else float("nan")


def stationary_bootstrap_ci(x: Sequence[float], mean_block: float = 3.0, n_boot: int = 5000,
                            seed: int = 42, level: float = 0.90) -> tuple:
    """Politis-Romano stationary bootstrap CI of the mean, annualised x12."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 6:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    new_block = rng.random((n_boot, n)) < 1.0 / mean_block
    new_block[:, 0] = True
    starts = rng.integers(n, size=(n_boot, n))
    k = np.arange(n)
    last = np.maximum.accumulate(np.where(new_block, k, 0), axis=1)  # position where current block began
    idx = (np.take_along_axis(starts, last, axis=1) + (k - last)) % n
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo * 12), float(hi * 12)


def psr(returns: Sequence[float], sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012), non-annualised per-period SR."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurt()) + 3.0  # non-excess kurtosis
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / denom))


def dsr(returns: Sequence[float], trial_srs: Sequence[float], n_trials: int) -> Dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014). trial_srs are per-period (monthly) SRs."""
    v = float(np.var(trial_srs, ddof=1)) if len(trial_srs) > 1 else 0.0
    n = max(2, int(n_trials))
    sr0 = math.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e)))
    return {"dsr": psr(returns, sr0), "psr": psr(returns, 0.0), "sr0_monthly": sr0, "trials": n,
            "trial_sr_var": v}


def pbo_cscv(M: np.ndarray, S: int = 10) -> Dict[str, object]:
    """Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2017).

    M: T x N matrix of per-period returns for N trials. Metric: Sharpe.
    """
    T, N = M.shape
    if N < 2 or T < S * 2:
        return {"pbo": float("nan"), "logits": [], "combinations": 0}
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for combo in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[c] for c in combo])
        oos_idx = np.concatenate([blocks[c] for c in range(S) if c not in combo])
        sr = lambda X: X.mean(0) / np.where(X.std(0, ddof=1) > 0, X.std(0, ddof=1), np.nan)
        is_sr, oos_sr = sr(M[is_idx]), sr(M[oos_idx])
        best = int(np.nanargmax(is_sr))
        rank = (pd.Series(oos_sr).rank(method="average").iloc[best])  # 1 = worst, N = best
        w = rank / (N + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "logits": logits.round(4).tolist(), "combinations": len(logits)}


def newey_west_t(x: Sequence[float], lags: int = 3) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags + 3:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for L in range(1, lags + 1):
        var += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(var / n)) if var > 0 else float("nan")


def rank_ic(ctx, signal: str, rows: List[int], bench: pd.Series = None, horizon: int = 21) -> Dict[str, float]:
    """Monthly Spearman IC between the score at the close of i and the TRADED label: entry at the open of i+1,
    exit at the close of i+horizon (both known only after the decision). Subtracting a common benchmark return
    would not change cross-sectional ranks (nor neutralise beta), so none is subtracted; `bench` is kept for
    call compatibility. Monthly decisions with a 21-session horizon do not overlap; NW(3) t-stat allows for
    residual serial dependence."""
    ics, dates = [], []
    for i in rows:
        if i + horizon >= len(ctx.dates):
            break
        names = ctx.eligible(i)
        s = ctx.scores(signal, i, names)
        if len(s) < 10:
            continue
        fwd = (ctx.A_last.iloc[i + horizon][s.index] / ctx.AO.iloc[i + 1][s.index] - 1).replace([np.inf, -np.inf], np.nan)
        ok = fwd.notna()
        if ok.sum() < 10:
            continue
        ics.append(spearmanr(s[ok], fwd[ok]).statistic)
        dates.append(ctx.dates[i])
    ics = np.array(ics, dtype=float)
    return {"mean_ic": float(np.nanmean(ics)) if len(ics) else float("nan"),
            "nw_t": newey_west_t(ics, 3), "pct_positive": float(np.mean(ics > 0)) if len(ics) else float("nan"),
            "months": int(len(ics)), "label": f"next open -> close +{horizon} sessions (adjusted)"}


# ---- metrics table -------------------------------------------------------------------------

def daily_drawdown(equity: pd.Series) -> float:
    """Daily marked-to-market max drawdown (the start NAV counts as a peak)."""
    eq = equity.dropna()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")


def metrics(result: Dict, bench_m: pd.Series, ew_m: pd.Series, period: str, cash_rate: float = 0.0) -> Dict[str, float]:
    """Monthly metrics over a period. `sharpe` = zero-rate Sharpe (x sqrt 12); `sharpe_cash_excess` subtracts
    `cash_rate` (annual); `ir` = information ratio vs the benchmark; `mdd` = month-end drawdown (gate e);
    `mdd_daily` = daily marked-to-market drawdown (diagnostic); `sortino` uses downside deviation over ALL
    months versus a 0 target."""
    m = window(result["monthly"], period)
    b = bench_m.reindex(m.index)
    e = ew_m.reindex(m.index)
    out = {"months": int(len(m))}
    if len(m) < 3:
        return out
    years = len(m) / 12
    ex = (m - b).dropna()
    te = float(ex.std() * math.sqrt(12)) if len(ex) > 2 else float("nan")
    eq = result["equity"]
    a, z = PERIODS[period]
    eqw = eq.loc[(eq.index >= m.index[0] - pd.offsets.MonthEnd(1)) & (eq.index <= m.index[-1])]
    avg_eq = float(eqw.mean()) if len(eqw) else float("nan")
    orders = result["orders"]
    if len(orders):
        o = orders[(orders["date"] > m.index[0] - pd.offsets.MonthEnd(1)) & (orders["date"] <= m.index[-1])]
    else:
        o = orders
    sells = float(o.loc[o["side"] == "SELL", "value"].sum()) if len(o) else 0.0
    brok = float(o["brokerage"].sum()) if len(o) else 0.0
    slip_cost = float(o["slippage"].sum()) if len(o) else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(m.to_numpy(), 0.0) ** 2)))
    holds = [h for h in result["holdings"] if h["exit"] is not None
             and pd.Timestamp(h["exit"]) >= m.index[0] - pd.offsets.MonthEnd(1) and pd.Timestamp(h["exit"]) <= m.index[-1]]
    days = [(pd.Timestamp(h["exit"]) - pd.Timestamp(h["entry"])).days for h in holds]
    cov = np.cov(m.loc[b.dropna().index], b.dropna()) if b.notna().sum() > 2 else None
    out.update({
        "cagr": cagr(m), "vol": float(m.std() * math.sqrt(12)), "sharpe": sharpe(m),
        "sortino": float(m.mean() / downside * math.sqrt(12)) if downside and downside > 0 else float("nan"),
        "mdd": max_drawdown(m),
        "mdd_daily": daily_drawdown(eq.loc[(eq.index >= m.index[0] - pd.offsets.MonthEnd(1)) & (eq.index <= m.index[-1])]),
        "sharpe_cash_excess": sharpe(m - cash_rate / 12),
        "nav_invalid": bool((eq <= 0).any()),
        "hit_rate": float((ex > 0).mean()) if len(ex) else float("nan"),
        "turnover": sells / avg_eq / years if avg_eq else float("nan"),
        "orders_per_year": len(o) / years,
        "brokerage_drag": brok / avg_eq / years if avg_eq else float("nan"),
        "slippage_drag": slip_cost / avg_eq / years if avg_eq else float("nan"),
        "excess_vs_bench": float(ex.mean() * 12) if len(ex) else float("nan"),
        "tracking_error": te,
        "ir": float(ex.mean() * 12 / te) if te and te > 0 else float("nan"),
        "beta": float(cov[0, 1] / cov[1, 1]) if cov is not None and cov[1, 1] > 0 else float("nan"),
        "excess_vs_ew": float((m - e).dropna().mean() * 12) if e.notna().any() else float("nan"),
        "avg_positions": float(np.mean(result["positions"])) if result["positions"] else 0.0,
        "avg_holding_days": float(np.mean(days)) if days else float("nan"),
        "pct_disposals_lt_365": float(np.mean([d < 365 for d in days])) if days else float("nan"),
    })
    out["calmar"] = out["cagr"] / abs(out["mdd"]) if out["mdd"] and out["mdd"] < 0 else float("nan")
    lo, hi = stationary_bootstrap_ci(ex.to_numpy())
    out["excess_ci_lo"], out["excess_ci_hi"] = lo, hi
    out["cost_drag"] = out["brokerage_drag"] + out["slippage_drag"]
    return out


def cost_drag(result: Dict) -> Dict[str, float]:
    eq = result["equity"]
    years = max(1e-9, len(result["monthly"]) / 12)
    avg = float(eq.mean())
    c = result["costs"]
    return {"brokerage": c["brokerage"] / avg / years, "slippage": c["slippage"] / avg / years,
            "total": (c["brokerage"] + c["slippage"]) / avg / years}


# ---- selection -----------------------------------------------------------------------------

def turnover_before(result: Dict, cut: pd.Timestamp) -> float:
    """Annual sell turnover / average equity using only orders and equity before `cut` (tie-break input)."""
    eq = result["equity"]
    eq = eq.loc[eq.index < cut]
    o = result["orders"]
    if not len(eq) or float(eq.mean()) <= 0:
        return 0.0
    sells = float(o.loc[(o["side"] == "SELL") & (o["date"] < cut), "value"].sum()) if len(o) else 0.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 12)
    return sells / float(eq.mean()) / years


def _turnover_at(turnover, vid: str, cut: pd.Timestamp) -> float:
    return float(turnover(vid, cut)) if callable(turnover) else float(turnover.get(vid, 0.0))


def walk_forward_select(monthly: Dict[str, pd.Series], turnover) -> Dict[str, object]:
    """On each 1 July, pick the variant with the best net Sharpe using only months before that date (ties: lower
    turnover before that date; `turnover` is a function (variant, cut) or a dict for tests). `monthly` here is
    the spliced series of each pick's own account - a DIAGNOSTIC only; the investable walk-forward is the
    continuous account built by `portfolio.run_backtest_v2(schedule=picks)`."""
    pieces, picks = [], []
    for y in WF_YEARS:
        cut = pd.Timestamp(f"{y}-07-01")
        scored = []
        for vid, s in monthly.items():
            past = s.loc[s.index < cut]
            if len(past) >= 12:
                scored.append((sharpe(past), -_turnover_at(turnover, vid, cut), vid))
        if not scored:
            continue
        scored.sort(reverse=True)
        vid = scored[0][2]
        picks.append({"year": y, "variant": vid, "from": cut.date().isoformat()})
        nxt = monthly[vid]
        pieces.append(nxt.loc[(nxt.index >= cut) & (nxt.index < pd.Timestamp(f"{y + 1}-07-01"))])
    series = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return {"monthly": series, "picks": picks}


HOLD_OUT_NOTE = ("July 2024 onwards was already examined before these repairs: for any new change it is historical "
                 "diagnostic evidence, not an untouched hold-out. Only data after an experiment's freeze time is "
                 "prospective.")


def _registry(path: pathlib.Path) -> Dict[str, object]:
    raw = json.loads(path.read_text()) if path.exists() else {}
    if raw.get("schema") == "frozen_registry_v1":
        return raw
    legacy = [{"setting": k, **v, "experiment_id": f"legacy-{v.get('config_hash')}", "legacy": True}
              for k, v in raw.items() if isinstance(v, dict)]
    return {"schema": "frozen_registry_v1", "experiments": legacy}


def experiment_id(hashes: Dict[str, str], cut: str, setting: str, engine: str = "") -> str:
    blob = json.dumps({"hashes": sorted(hashes.values()), "cut": cut, "setting": setting, "engine": engine})
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def freeze_choice(monthly: Dict[str, pd.Series], turnover, hashes: Dict[str, str],
                  path: pathlib.Path, setting: str, engine: str = "") -> Dict[str, object]:
    """Append-only experiment registry. The same candidate set (variant hashes + selection cut + engine id) reuses
    its original frozen choice; a changed set registers a NEW experiment. Entries are never deleted or rewritten."""
    cut = pd.Timestamp("2024-07-01")
    reg = _registry(path)
    eid = experiment_id(hashes, "2024-07-01", setting, engine)
    for e in reg["experiments"]:
        if e.get("experiment_id") == eid and e.get("setting") == setting:
            return {**e, "reused": True}
    scored = [(sharpe(s.loc[s.index < cut]), -_turnover_at(turnover, v, cut), v) for v, s in monthly.items()
              if len(s.loc[s.index < cut]) >= 12]
    scored.sort(reverse=True)
    vid = scored[0][2]
    entry = {"setting": setting, "experiment_id": eid, "variant": vid, "config_hash": hashes[vid],
             "timestamp": time.time(), "selected_using_data_before": "2024-07-01", "candidates": len(hashes),
             "prospective_from": pd.Timestamp.now().date().isoformat(), "hold_out_note": HOLD_OUT_NOTE}
    reg["experiments"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    tmp.replace(path)
    return {**entry, "reused": False}


def hold_out_gate(path: pathlib.Path, setting: str, evaluation_start: float, eid: Optional[str] = None) -> str:
    """Raise unless a frozen choice for this setting (and experiment, when given) exists and was registered
    before this hold-out evaluation began."""
    if not path.exists():
        raise RuntimeError("Hold-out evaluation refused: results/frozen_choice.json is missing.")
    found = [e for e in _registry(path)["experiments"] if e.get("setting") == setting
             and (eid is None or e.get("experiment_id") == eid)]
    if not found:
        raise RuntimeError(f"Hold-out evaluation refused: no frozen choice for {setting}.")
    choice = found[-1]
    if choice["timestamp"] > evaluation_start:
        raise RuntimeError("Hold-out evaluation refused: frozen choice is newer than the evaluation start.")
    return choice["variant"]


def acceptance(frozen: Dict[str, float], frozen_full: Dict[str, float], wf: Dict[str, float],
               bench_wf: Dict[str, float], bench_ho: Dict[str, float], bench_full: Dict[str, float],
               dsr_val: float, pbo_val: float, drag: float, neighbours_pass: int, n_eff: int) -> Dict[str, object]:
    beats = lambda s, b: (s.get("cagr", -1) > b.get("cagr", 0)) and (s.get("sharpe", -9) >= b.get("sharpe", 9))
    c = {
        "a": bool(beats(wf, bench_wf) and beats(frozen, bench_ho)),
        "b": bool(frozen_full.get("excess_vs_bench", -1) - 0.015 > 0),
        "c": bool(dsr_val >= 0.95),
        "d": bool(pbo_val <= 0.25),
        "e": bool(frozen_full.get("mdd", -1) >= bench_full.get("mdd", 0) - 0.05),
        "f": bool(drag <= 0.01),
        "g": bool(neighbours_pass >= 2),
        "h": bool(frozen_full.get("excess_vs_bench", -1) > 0 and frozen_full.get("excess_ci_lo", -1) > -0.01),
        "i": bool(n_eff >= 3),
    }
    labels = {
        "a": "investable walk-forward selection AND the frozen choice's later window must both beat VAS "
             "(CAGR strictly greater, zero-rate Sharpe at least VAS's)",
        "b": "full-period annual arithmetic excess minus a fixed 1.5-point hurdle must be positive "
             "(a hurdle/sensitivity, not a survivorship correction)",
        "c": "deflated Sharpe ratio (full frozen series; trial variance from the selection window) must be at least 0.95",
        "d": "probability of backtest overfitting (CSCV on the selection window) must be at most 0.25",
        "e": "month-end max drawdown must be within 5 points of VAS's",
        "f": "annual cost drag (brokerage + slippage) must be at most 1.0%",
        "g": "at least 2 of 3 registered neighbours must pass the full criterion (a)",
        "h": "annual excess must be positive with a 90% block-bootstrap CI lower bound above -1%",
        "i": "minimum fee-aware funded satellite capacity must be at least 3 positions at this capital",
    }
    if all(c.values()):
        tier = "LIVE"
    elif c["a"] and c["d"] and c["e"] and c["f"] and c["i"] and dsr_val >= 0.80:
        tier = "PAPER"
    else:
        tier = "FALLBACK"
    return {"tier": tier, "passes": c, "reasons": [f"({k}) {labels[k]}" for k, ok in c.items() if not ok]}
