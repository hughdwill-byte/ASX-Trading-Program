"""Monthly research backtest for OptPolicy (uncapped optimiser and its registered controls). RESEARCH/PAPER ONLY.

Lifecycle per decision row i (last session of the month): causal forecast calibration and covariance from data
up to the close of i -> optimiser proposal in integer shares at the close of i -> execution at the open of
i+1 through `optimizer.revalidate_at_execution` (sells first, buys resized down, missing open = rejected) ->
fills recorded once in an `accounting.Ledger` (slippage inside the fill price) -> daily NAV from holdings.
Prices are split/dividend-adjusted, matching the legacy engine, so quantities are synthetic research
quantities (`executable_quantities=False`).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml import optimizer as opt
from ml.accounting import Ledger
from ml.forecasts import calibrate
from ml.policy_spec import OptPolicy
from ml.portfolio import brokerage_fee, funded_capacity, slippage
from ml.risk_model import covariance
from ml.signals import Context


def _fixed_targets(p: opt.Problem, x: np.ndarray, status: str, info: dict) -> Dict[str, object]:
    """Integer book for prescribed weights (controls): floor to shares, drop sub-floor opens / small changes,
    then trim the largest buy one share at a time until cash, reserve and limits hold."""
    q = opt._round_target(p, np.minimum(x, p.name_limit))
    value = np.abs(q - p.shares) * p.price
    q = np.where((p.shares == 0) & (q > 0) & (value < p.opening_floor), 0, q)
    small = (p.shares > 0) & (q > 0) & (q != p.shares) & (value < p.order_floor)
    q = np.where(small, p.shares, q)
    e = opt.evaluate(p, q)
    while not e["feasible"] and np.any(q > p.shares):
        buys = np.where(q > p.shares, (q - p.shares) * p.price, -1)
        k = int(np.argmax(buys))
        q[k] -= 1
        if q[k] > p.shares[k] and (q[k] - p.shares[k]) * p.price[k] < (p.opening_floor if p.shares[k] == 0 else p.order_floor):
            q[k] = p.shares[k]
        e = opt.evaluate(p, q)
    return opt._decision(p, q, status if e["feasible"] else "no_feasible_proposal", info, time.monotonic())


def decide(ctx: Context, pol: OptPolicy, cfg: dict, i: int, shares: Dict[str, int], cash: float,
           rank_cache: Optional[Dict] = None, returns: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """One decision at the close of row i for the given book. Returns the optimiser decision plus inputs'
    identities. Never reads rows after i."""
    core = cfg["core_ticker"].replace(".AX", "")
    ranks = rank_cache.get((pol.signal, i)) if rank_cache is not None else None
    ranks = ctx.ranks(pol.signal, i) if ranks is None else ranks
    fb = calibrate(ctx, pol.signal, i, pol.horizon, pol.train_decisions, rank_cache=rank_cache)
    held = sorted(t for t, q in shares.items() if q)
    meta = {"forecast_status": fb.status, "forecast_id": fb.identity(), "trained_until": fb.trained_until,
            "candidates": int(len(ranks))}
    if fb.status != "ok":
        return {"status": "no_trade_forecast_unavailable", "orders": [], "count": len(held), **meta}
    names = sorted(set(ranks.index) | set(held) | {core})
    px = ctx.A_last.iloc[i].reindex(names).to_numpy(dtype=float)
    market = float(fb.mu.get(core, 0.0))
    mu = fb.mu.reindex(names).fillna(market).to_numpy()   # held but unscored: neutral market forecast, no buys
    R = returns if returns is not None else ctx.A.pct_change(fill_method=None)
    risk = covariance(R.iloc[max(0, i - 251): i + 1], names, horizon=pol.horizon)
    stale = ~(ctx.age.iloc[i].reindex(names).to_numpy(dtype=float) < cfg["stale_days"])
    mdv = ctx.MDV60.iloc[i].reindex(names).to_numpy(dtype=float)
    spread = np.array([slippage(cfg, 0, etf=True) if t == core else slippage(cfg, float(np.nan_to_num(m)))
                       for t, m in zip(names, mdv)]) * pol.slip_mult
    p = opt.Problem(
        names=names, price=np.where(np.isfinite(px), px, 1.0), shares=np.array([shares.get(t, 0) for t in names]),
        cash=cash, mu=mu, cov=risk["cov"].to_numpy(), gamma=pol.gamma,
        fee=lambda side, value: brokerage_fee(cfg, side, value) * pol.slip_mult,
        r_cash=cfg.get("cash_rate_annual", 0.0) * pol.horizon / 252, cash_reserve=pol.cash_reserve,
        name_limit=np.array([1.0 if t == core else pol.name_limit for t in names]),
        risk_budget=pol.risk_budget, turnover_limit=pol.turnover_limit,
        participation=np.where(np.isfinite(mdv), pol.participation * mdv, 0.0),
        can_buy=np.array([(t in ranks.index or t == core) for t in names]) & np.isfinite(px),
        tradable=~stale & np.isfinite(px), spread=spread,
        impact=np.full(len(names), pol.impact), opening_floor=max(cfg["min_parcel"], cfg["min_position"]),
        order_floor=cfg["min_order"], max_evals=20_000, time_limit=20.0)
    meta.update({"risk_model": risk["model_id"], "thin_history": risk["thin"], "shrinkage": risk["shrinkage"]})
    if pol.selection == "rank" or pol.holding_cap is not None or pol.sizing != "covariance_cost_aware":
        d = _control(p, pol, ranks, core, names, risk["cov"])
    else:
        d = opt.solve(p)
    d.update(meta)
    stocks = {t: v["pre_nav_weight"] for t, v in d.get("targets", {}).items() if t != core and v["shares"]}
    tot = sum(stocks.values())
    d["effective_count"] = 1 / sum((w / tot) ** 2 for w in stocks.values()) if tot > 0 else 0.0
    d["stock_count"] = len(stocks)
    return d


def _control(p: opt.Problem, pol: OptPolicy, ranks: pd.Series, core: str, names: List[str], cov: pd.DataFrame):
    idx = {t: k for k, t in enumerate(names)}
    if pol.selection == "rank":
        chosen = [t for t in ranks.index[: pol.holding_cap]] if pol.holding_cap else \
            [t for t in ranks.index if p.mu[idx[t]] > p.mu[idx[core]]]
    else:
        cont = opt.solve_continuous(p)
        if cont["status"] != "optimal":
            return opt._decision(p, p.shares.copy(), "solver_failed", {"continuous": cont}, time.monotonic())
        x = pd.Series(cont["x"], index=names)
        chosen = [t for t in x.drop(core).sort_values(ascending=False, kind="stable").index if x[t] > 1e-4]
        if pol.holding_cap:
            chosen = chosen[: pol.holding_cap]
        if pol.sizing == "covariance_cost_aware":   # capped variable weight: optimise inside the capped set
            keep = np.array([t in chosen or t == core for t in names])
            q = p.shares.copy()
            for k in np.flatnonzero(~keep & (p.shares > 0)):   # the cap forces exits of names outside the set
                q[k] = 0
            pre = opt.evaluate(p, q)
            sub = opt.Problem(**{**p.__dict__, "shares": np.where(keep, p.shares, 0), "cash": pre["cash_post"],
                                 "can_buy": p.can_buy & keep})
            d = opt.solve(sub)
            final = np.array([d["targets"].get(t, {"shares": 0})["shares"] for t in names])
            return opt._decision(p, final, d["status"], d["solver"], time.monotonic())
    x = np.zeros(len(names))
    invest = max(0.0, 1 - p.cash_reserve - 0.01)   # leave ~1% for estimated costs; trimmed exactly below
    if chosen:
        if pol.sizing == "inverse_vol":
            iv = np.array([1 / np.sqrt(cov.loc[t, t]) for t in chosen])
            w = iv / iv.sum()
        else:
            w = np.full(len(chosen), 1 / len(chosen))
        for t, wt in zip(chosen, w):
            x[idx[t]] = invest * wt
    return _fixed_targets(p, x, "control_targets", {"chosen": chosen})


def run_backtest_opt(ctx: Context, pol: OptPolicy, cfg: dict, rank_cache: Optional[Dict] = None,
                     end: Optional[str] = None) -> Dict[str, object]:
    core = cfg["core_ticker"].replace(".AX", "")
    rows = ctx.decision_rows(cfg.get("start"), end)
    end_ts = pd.Timestamp(end) if end else ctx.dates[-1]
    L = Ledger(pol.capital, settlement_days=0, account_id=pol.id)   # declared: same-day sale proceeds fund buys
    returns = ctx.A.pct_change(fill_method=None)
    orders, rejected, decisions, snaps, holds = [], [], [], [], []
    entry: Dict[str, pd.Timestamp] = {}
    costs = {"brokerage": 0.0, "slippage": 0.0}
    started = time.monotonic()
    for i in rows:
        j = i + 1
        if j >= len(ctx.dates) or ctx.dates[j] > end_ts:
            break
        quarter_skip = pol.alpha_cadence == "Q" and ctx.dates[i].month not in (3, 6, 9, 12)
        d = {"status": "no_trade_off_cadence", "orders": []} if quarter_skip else \
            decide(ctx, pol, cfg, i, dict(L.shares), L.available, rank_cache, returns)
        decisions.append({"date": ctx.dates[i], **{k: d.get(k) for k in (
            "status", "stock_count", "effective_count", "objective", "residual_cash", "violations", "forecast_status",
            "seconds", "candidates")}, "orders": len(d["orders"])})
        if d["orders"]:
            p = opt.Problem(names=[o["ticker"] for o in d["orders"]], price=np.array([o["est_price"] for o in d["orders"]]),
                            shares=np.array([L.shares.get(o["ticker"], 0) for o in d["orders"]]), cash=L.available,
                            mu=np.zeros(len(d["orders"])), cov=np.eye(len(d["orders"])), gamma=0.0,
                            fee=lambda side, value: brokerage_fee(cfg, side, value) * pol.slip_mult,
                            cash_reserve=0.0,
                            spread=np.array([o["est_adverse"] / o["est_value"] if o["est_value"] else 0 for o in d["orders"]]),
                            opening_floor=max(cfg["min_parcel"], cfg["min_position"]), order_floor=cfg["min_order"])
            nav = L.available + sum(q * float(ctx.A_last.iloc[i][t]) for t, q in L.shares.items() if q)
            p.cash_reserve = pol.cash_reserve * nav / p.nav   # the policy reserve, on full pre-trade NAV
            opens = {o["ticker"]: float(ctx.AO.iloc[j][o["ticker"]]) for o in d["orders"]}
            for o in opt.revalidate_at_execution(p, d["orders"], opens):
                if o["status"] == "rejected":
                    rejected.append({"decision_date": ctx.dates[i], "date": ctx.dates[j], **o})
                    continue
                k = p.names.index(o["ticker"])
                raw = o["exec_price"]
                fill = raw * (1 + p.spread[k]) if o["side"] == "BUY" else raw * (1 - p.spread[k])
                fee = p.fee(o["side"], o["shares"] * raw)
                est = o["shares"] * fill + fee if o["side"] == "BUY" else 0.0
                if o["side"] == "BUY" and est > L.available:
                    rejected.append({"decision_date": ctx.dates[i], "date": ctx.dates[j], **o, "why": "cash at fill"})
                    continue
                before = L.shares.get(o["ticker"], 0)
                oid = L.request(ctx.dates[i], o["ticker"], o["side"], o["shares"], est_debit=est)
                L.fill(oid, ctx.dates[j], o["shares"], fill, fee)
                costs["brokerage"] += fee
                costs["slippage"] += o["shares"] * raw * p.spread[k]
                orders.append({"decision_date": ctx.dates[i], "date": ctx.dates[j], "ticker": o["ticker"],
                               "side": o["side"], "shares": o["shares"], "price": fill, "value": o["shares"] * fill,
                               "brokerage": fee, "slippage": o["shares"] * raw * p.spread[k], "reason": o["kind"],
                               "status": "filled" if o["status"] == "fillable" else "partial"})
                after = L.shares.get(o["ticker"], 0)
                if before == 0 and after > 0:
                    entry[o["ticker"]] = ctx.dates[j]
                elif before > 0 and after == 0:
                    holds.append({"ticker": o["ticker"], "entry": entry.pop(o["ticker"], None), "exit": ctx.dates[j]})
        snaps.append((j, {t: q for t, q in L.shares.items() if q}, L.available))
    if not snaps:
        raise ValueError("No executable decision dates in range.")
    idx = ctx.dates
    tickers = sorted({t for _, b, _ in snaps for t in b} | {core})
    exec_rows = [j for j, _, _ in snaps]
    held = pd.DataFrame(np.nan, index=idx, columns=tickers)
    held.iloc[exec_rows] = [[float(b.get(t, 0)) for t in tickers] for _, b, _ in snaps]
    held = held.ffill().fillna(0.0)
    cash_s = pd.Series(np.nan, index=idx)
    cash_s.iloc[exec_rows] = [c for _, _, c in snaps]
    cash_s = cash_s.ffill()
    haircut = (1 - cfg["stale_haircut"] * (~(ctx.age[tickers] < cfg["stale_days"]))).where(held > 0, 1.0)
    equity = (held * ctx.A_last[tickers].fillna(0.0) * haircut).sum(axis=1) + cash_s
    equity.iloc[: exec_rows[0]] = pol.capital
    equity = equity.loc[: end_ts].ffill()
    dec_dates = ctx.dates[rows[: len(snaps)]]
    month_ends = ctx.dates[ctx.decision_rows(cfg.get("start"))]
    month_ends = month_ends[(month_ends >= dec_dates[0]) & (month_ends <= equity.index[-1])]
    for t in L.shares:
        if L.shares[t]:
            holds.append({"ticker": t, "entry": entry.get(t), "exit": None})
    dec = pd.DataFrame(decisions)
    books = pd.Series([" ".join(sorted(b)) for _, b, _ in snaps], index=dec_dates, dtype=object)
    return {
        "variant": pol, "policy": pol.to_json(), "equity": equity,
        "monthly": equity.reindex(month_ends).pct_change().dropna(), "orders": pd.DataFrame(orders),
        "rejected": pd.DataFrame(rejected), "decisions": dec, "books": books, "holdings": holds,
        "positions": [len([t for t in b if t != core]) for _, b, _ in snaps], "costs": costs,
        "n_eff": funded_capacity(cfg, pol.capital, pol.slip_mult), "capacity_rule": "funded_capacity(capital) >= 3",
        "skipped_reason": None, "final_book": dict(L.shares), "cash": L.available, "ledger_hash": L.state_hash(),
        "valuation_date": equity.index[-1], "price_basis": "adjusted_total_return_synthetic",
        "executable_quantities": False, "calendar_id": getattr(ctx, "calendar_id", None),
        "runtime_seconds": time.monotonic() - started, "research_only": True,
    }
