"""Cost-aware monthly portfolio engine (strategy v2): run_backtest_v2.

Decisions at the close of the last trading day of each month (row i); orders fill at the
adjusted open of row i+1 with tiered slippage, integer shares and fixed brokerage per order.
Dividends enter only through adjusted prices; franking credits are ignored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml.signals import Context

DEFAULTS = {
    "capital": 10000.0, "core_weight": 0.0, "core_ticker": "VAS.AX", "signal": "RAMOM",
    "n_positions": 5, "buffer_mult": 3, "min_position": 500.0, "min_order": 500.0,
    "broker": "stake", "min_parcel": 500.0,
    "regime_filter": True, "regime_band": 0.02, "regime_sma": 200, "si_screen": False, "si_max": 0.05,
    "brokerage": 6.50, "slippage_tiers": {"high": 0.0010, "mid": 0.0025, "low": 0.0050, "etf": 0.0005},
    "slippage_mdv_high": 10e6, "slippage_mdv_mid": 2e6, "cost_stress_mult": 2.0,
    "stale_days": 10, "stale_haircut": 0.30, "cash_rate_annual": 0.0, "core_drift_band": 0.10,
    "asic_lag_days": 4, "universe_size": 250, "min_history_days": 300, "min_presence": 0.95,
    "min_price": 1.00, "min_mdv60": 2_000_000.0, "max_zero_volume_days": 2, "max_daily_move": 2.0,
    "exclude_tickers": ["STW", "SFY", "A200", "VAS"], "start": "2016-03-31",
    "tax_defer_days": 35, "tax_defer_exit_pct": 0.8, "tax_defer_max_dd": 0.15,
    "tax_rate": 0.32,  # marginal rate for the after-tax estimate (30% bracket + 2% Medicare levy)
}


@dataclasses.dataclass(frozen=True)
class Variant:
    signal: str = "RAMOM"
    n: int = 5
    regime: bool = True
    buffer_mult: int = 3
    core_weight: float = 0.0
    capital: float = 10000.0
    si_screen: bool = False
    slip_mult: float = 1.0
    rebalance: str = "M"       # "Q": trade only on March/June/September/December decisions
    tax_aware: bool = False    # defer rank sells of winners held 330-364 days (50% CGT discount)

    @property
    def id(self) -> str:
        tag = f"{self.signal}_N{self.n}_{'reg' if self.regime else 'noreg'}_B{self.buffer_mult}"
        if self.si_screen:
            tag += "_SI"
        if self.rebalance == "Q":
            tag += "_Q"
        if self.tax_aware:
            tag += "_TAX"
        if self.slip_mult != 1.0:
            tag += f"_slipx{self.slip_mult:g}"
        return tag

    @property
    def setting(self) -> str:
        return f"{int(self.capital)}_core{self.core_weight:g}"

    def config_hash(self) -> str:
        return hashlib.sha1(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def slippage(cfg: dict, mdv: float, etf: bool = False) -> float:
    t = cfg["slippage_tiers"]
    if etf:
        return t["etf"]
    if mdv >= cfg["slippage_mdv_high"]:
        return t["high"]
    if mdv >= cfg["slippage_mdv_mid"]:
        return t["mid"]
    return t["low"]


def update_regime(state: Optional[bool], price: float, sma: float, band: float) -> Optional[bool]:
    """Hysteresis: ON above SMA*(1+band), OFF below SMA*(1-band), otherwise unchanged."""
    if not np.isfinite(sma) or not np.isfinite(price):
        return state
    if state is None:
        return bool(price > sma)
    if price > sma * (1 + band):
        return True
    if price < sma * (1 - band):
        return False
    return state


def brokerage_fee(cfg: dict, side: str, value: float) -> float:
    """Brokerage for one order. Fees come from the research report and are UNVERIFIED: owner should check
    hellostake.com/au/pricing and cmcmarkets.com/en-au/stockbroking/pricing.
    The engine places at most one buy per ticker per day, so CMC's "first buy of the day" always holds."""
    broker = cfg.get("broker", "flat")
    if broker == "flat":
        return cfg["brokerage"]
    if broker == "stake":
        return 3.0 if value <= 30000 else 0.0001 * value
    if broker == "cmc":
        return 0.0 if side == "BUY" and value <= 1000 else max(11.0, 0.001 * value)
    raise ValueError(f"Unknown broker model: {broker}")


def run_backtest_v2(ctx: Context, v: Variant, cfg: dict, rank_cache: Optional[Dict] = None,
                    end: Optional[str] = None) -> Dict[str, object]:
    core = cfg["core_ticker"].replace(".AX", "")
    if core not in ctx.A.columns:
        raise ValueError(f"Core ETF {core} missing from the panel; run ingest.")
    rows = ctx.decision_rows(cfg.get("start"), end)
    if not rows:
        raise ValueError("No decision dates in range.")
    A_last, AO = ctx.A_last, ctx.AO
    min_order = max(cfg["min_order"], cfg["min_parcel"])
    fee = lambda side, value: brokerage_fee(cfg, side, value) * v.slip_mult  # cost stress scales fees too

    sat0 = (1 - v.core_weight) * v.capital
    n_eff = min(v.n, int(sat0 // max(cfg["min_position"], cfg["min_parcel"])))
    skipped_reason = None
    core_target = v.core_weight
    if n_eff < 3:
        skipped_reason = f"satellite disabled: A${sat0:,.0f} / A${cfg['min_position']:,.0f} floor < 3 positions"
        core_target, n_eff = 1.0, 0
    K = v.buffer_mult * max(n_eff, 1)

    cash = {"core": 0.0, "sat": 0.0}
    shares: Dict[str, int] = {}          # satellite stocks
    regime_etf = 0                       # satellite slice parked in the core ETF while regime is OFF
    core_sh = 0
    entry: Dict[str, pd.Timestamp] = {}
    entry_px: Dict[str, float] = {}
    inelig: Dict[str, int] = {}
    orders: List[dict] = []
    holds: List[dict] = []
    snaps: List[tuple] = []              # (exec row, {ticker: shares}, total cash)
    positions_count: List[int] = []
    regime_on: Optional[bool] = None
    costs = {"brokerage": 0.0, "slippage": 0.0}

    def trade(j: int, ticker: str, side: str, qty: int, sleeve: str, slip: float, px: Optional[float] = None,
              reason: str = "") -> float:
        """Fill qty at adjusted open of row j (or px). Returns net cash change."""
        if qty <= 0:
            return 0.0
        raw = float(AO.iloc[j][ticker]) if px is None else px
        if not np.isfinite(raw) or raw <= 0:
            raw = float(A_last.iloc[j - 1][ticker])
        fill = raw * (1 + slip) if side == "BUY" else raw * (1 - slip)
        value = qty * fill
        brokerage = fee(side, value)
        costs["brokerage"] += brokerage
        costs["slippage"] += qty * raw * slip
        orders.append({"date": ctx.dates[j], "ticker": ticker, "side": side, "shares": qty, "price": fill,
                       "value": value, "brokerage": brokerage, "slippage": qty * raw * slip,
                       "sleeve": sleeve, "reason": reason})
        return -(value + brokerage) if side == "BUY" else value - brokerage

    def buy_amount(j: int, ticker: str, amount: float, sleeve: str, slip: float, reason: str = "") -> int:
        px = float(AO.iloc[j][ticker])
        if not np.isfinite(px) or px <= 0:
            return 0
        qty = int((amount - fee("BUY", amount)) // (px * (1 + slip)))
        qty = int((amount - fee("BUY", qty * px * (1 + slip))) // (px * (1 + slip)))  # refine with the fee at the real size
        if 0 < qty and qty * px < cfg["min_parcel"]:  # round up to the minimum parcel if the sleeve's cash covers it
            up = int(-(-cfg["min_parcel"] // px))
            if up * px * (1 + slip) + fee("BUY", up * px * (1 + slip)) <= cash[sleeve]:
                qty = up
        if qty <= 0 or qty * px < min_order:
            return 0
        cash[sleeve] += trade(j, ticker, "BUY", qty, sleeve, slip, reason=reason)
        return qty

    etf_slip = slippage(cfg, 0, etf=True) * v.slip_mult

    for k, i in enumerate(rows):
        j = i + 1
        if j >= len(ctx.dates):
            break
        px_i = A_last.iloc[i]
        if k == 0:  # inception
            cash["core"] = core_target * v.capital
            cash["sat"] = v.capital - cash["core"]
            if cash["core"] >= min_order:
                core_sh += buy_amount(j, core, cash["core"], "core", etf_slip, "inception")

        # Regime state on ^AORD vs its SMA, evaluated only at decision dates (hysteresis band).
        if v.regime and ctx.aord is not None and n_eff:
            p = ctx.aord.iloc[: i + 1]
            sma = p.iloc[-cfg["regime_sma"]:].mean() if len(p.dropna()) >= cfg["regime_sma"] else np.nan
            regime_on = update_regime(regime_on, float(p.iloc[-1]), float(sma), cfg["regime_band"])
        on = True if not v.regime or regime_on is None else regime_on

        trade_month = v.rebalance != "Q" or ctx.dates[i].month in (3, 6, 9, 12)
        if n_eff:
            ranks = rank_cache[(v.signal, i)] if rank_cache is not None else ctx.ranks(v.signal, i)
            elig = set(ranks.index)
            # 1. Sells: buffer, persistent ineligibility, stale price, or regime OFF.
            for t in list(shares):
                last_valid = ctx.A[t].iloc[: i + 1].last_valid_index()
                stale = last_valid is None or (i - ctx.dates.get_loc(last_valid)) >= cfg["stale_days"]
                inelig[t] = 0 if t in elig else inelig.get(t, 0) + 1
                rank = ranks.get(t, np.inf)
                reason = ("stale" if stale else "regime_off" if not on else "rank" if rank > K
                          else "ineligible" if inelig[t] >= 2 else "")
                if not trade_month and not stale:
                    continue
                if not reason:
                    continue
                if (v.tax_aware and reason == "rank" and 365 - cfg["tax_defer_days"] <= (ctx.dates[j] - entry[t]).days < 365
                        and px_i[t] > entry_px[t] and rank <= cfg["tax_defer_exit_pct"] * len(ranks)
                        and px_i[t] >= entry_px[t] * (1 - cfg["tax_defer_max_dd"])):
                    continue  # defer: a winner a few weeks short of the 12-month CGT discount
                slip = slippage(cfg, float(ctx.MDV60.iloc[i].get(t, 0) or 0)) * v.slip_mult
                if stale:
                    px = float(px_i[t]) * (1 - cfg["stale_haircut"])
                    cash["sat"] += trade(j, t, "SELL", shares[t], "sat", 0.0, px=px, reason="stale")
                else:
                    cash["sat"] += trade(j, t, "SELL", shares[t], "sat", slip, reason=reason)
                holds.append({"ticker": t, "entry": entry.pop(t), "exit": ctx.dates[j]})
                entry_px.pop(t, None)
                del shares[t]
                inelig.pop(t, None)

            sat_value = cash["sat"] + sum(shares[t] * px_i[t] for t in shares) + regime_etf * px_i[core]
            target = sat_value / n_eff

            if not trade_month:
                pass
            elif not on:
                if cash["sat"] >= min_order:
                    regime_etf += buy_amount(j, core, cash["sat"], "sat", etf_slip, "regime_off")
            else:
                if regime_etf:
                    cash["sat"] += trade(j, core, "SELL", regime_etf, "sat", etf_slip, reason="regime_on")
                    regime_etf = 0
                # 5. Trim only a holding that has grown past 2x target.
                for t in list(shares):
                    if shares[t] * px_i[t] > 2 * target:
                        qty = int(shares[t] - target // px_i[t])
                        if qty * px_i[t] >= min_order:
                            slip = slippage(cfg, float(ctx.MDV60.iloc[i].get(t, 0) or 0)) * v.slip_mult
                            cash["sat"] += trade(j, t, "SELL", qty, "sat", slip, reason="trim")
                            shares[t] -= qty
                # 4. Fill empty slots with the best-ranked names inside the top N.
                for t in ranks.index:
                    if len(shares) >= n_eff:
                        break
                    if ranks[t] > n_eff or t in shares or not np.isfinite(ctx.A.iloc[i][t]):
                        continue  # never buy a name with no price at the decision close
                    if v.si_screen and ctx.si(i, t) > cfg["si_max"]:
                        continue
                    slip = slippage(cfg, float(ctx.MDV60.iloc[i][t])) * v.slip_mult
                    qty = buy_amount(j, t, min(target, cash["sat"]), "sat", slip, f"rank {int(ranks[t])}")
                    if qty:
                        shares[t] = qty
                        entry[t] = ctx.dates[j]
                        entry_px[t] = orders[-1]["price"]
                        inelig[t] = 0

        # Core/satellite rebalance: June decision executes on the first trading day of July.
        if v.core_weight > 0 and n_eff and ctx.dates[j].month == 7 and ctx.dates[i].month == 6:
            total = cash["core"] + cash["sat"] + core_sh * px_i[core] + regime_etf * px_i[core] + \
                sum(shares[t] * px_i[t] for t in shares)
            w = (core_sh * px_i[core] + cash["core"]) / total
            if abs(w - v.core_weight) > cfg["core_drift_band"]:
                diff = (v.core_weight - w) * total
                if diff > 0 and cash["sat"] >= diff:
                    cash["sat"] -= diff
                    cash["core"] += diff
                    core_sh += buy_amount(j, core, cash["core"], "core", etf_slip, "rebalance")
                elif diff < 0:
                    qty = int(-diff // px_i[core])
                    if qty * px_i[core] >= min_order:
                        cash["core"] += trade(j, core, "SELL", qty, "core", etf_slip, reason="rebalance")
                        core_sh -= qty
                        cash["sat"] += cash["core"]
                        cash["core"] = 0.0

        positions_count.append(len(shares))
        book = dict(shares)
        book[core] = core_sh + regime_etf
        snaps.append((j, book, cash["core"] + cash["sat"]))

    # Daily equity from holdings snapshots (shares change only on execution days).
    idx = ctx.dates
    first = snaps[0][0]
    tickers = sorted({t for _, b, _ in snaps for t in b})
    exec_rows = [j for j, _, _ in snaps]
    held = pd.DataFrame(np.nan, index=idx, columns=tickers)
    held.iloc[exec_rows] = [[float(b.get(t, 0)) for t in tickers] for _, b, _ in snaps]
    held = held.ffill().fillna(0.0)
    cash_s = pd.Series(np.nan, index=idx)
    cash_s.iloc[exec_rows] = [c for _, _, c in snaps]
    cash_s = cash_s.ffill()
    equity = (held * A_last[tickers].fillna(0.0)).sum(axis=1) + cash_s
    equity.iloc[: first] = v.capital
    equity = equity.loc[: ctx.dates[min(rows[-1] + 1, len(idx) - 1)]].ffill()

    dec_dates = ctx.dates[rows]
    monthly = equity.reindex(dec_dates).pct_change().dropna()

    for t in shares:
        holds.append({"ticker": t, "entry": entry[t], "exit": None})
    # What was held after each decision (index = decision date), for the dashboard's history view.
    books = pd.Series([" ".join(sorted(t for t, q in b.items() if q)) for _, b, _ in snaps],
                      index=dec_dates[: len(snaps)], dtype=object)
    return {
        "books": books,
        "variant": v, "equity": equity, "monthly": monthly, "orders": pd.DataFrame(orders),
        "holdings": holds, "positions": positions_count, "costs": costs, "n_eff": n_eff,
        "skipped_reason": skipped_reason, "final_book": {**shares, core: core_sh + regime_etf},
        "regime_on": regime_on,
    }


def after_tax_monthly(res: Dict[str, object], rate: float) -> pd.Series:
    """Monthly returns after an estimate of Australian capital gains tax on realised gains.

    FIFO lots per ticker; capital losses (current and carried forward) offset non-discount gains first;
    the 50% discount applies to what remains of gains on lots held over 12 months. Each financial
    year's tax is paid at the first month-end on or after 31 October following it (or the last month
    if the data ends first), as a fraction of equity so later returns compound on the smaller balance.
    ponytail: ignores dividend tax and franking (adjusted prices fold dividends into gains), brokerage
    in the cost base and personal circumstances; a rough guide, not tax advice.
    """
    monthly: pd.Series = res["monthly"]
    orders: pd.DataFrame = res["orders"]
    if not rate or orders.empty or monthly.empty:
        return monthly.copy()
    lots: Dict[str, deque] = defaultdict(deque)
    fy_parts: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # short gain, long gain, loss
    for o in orders.sort_values("date", kind="stable").itertuples(index=False):
        if o.side == "BUY":
            lots[o.ticker].append([int(o.shares), float(o.price), o.date])
            continue
        qty, fy = int(o.shares), o.date.year + (o.date.month >= 7)
        while qty > 0 and lots[o.ticker]:
            lot = lots[o.ticker][0]
            q = min(qty, lot[0])
            gain = q * (float(o.price) - lot[1])
            if gain < 0:
                fy_parts[fy][2] -= gain
            else:
                fy_parts[fy][1 if (o.date - lot[2]).days > 365 else 0] += gain
            lot[0] -= q
            qty -= q
            if not lot[0]:
                lots[o.ticker].popleft()
    frac = pd.Series(0.0, index=monthly.index)
    carry = 0.0
    for fy in sorted(fy_parts):
        short, long_, loss = fy_parts[fy]
        loss += carry
        used = min(loss, short)
        short, loss = short - used, loss - used
        used = min(loss, long_)
        long_, carry = long_ - used, loss - used
        tax = rate * (short + long_ / 2)
        if tax <= 0:
            continue
        due = monthly.index[monthly.index >= pd.Timestamp(fy, 10, 31)]
        when = due[0] if len(due) else monthly.index[-1]
        frac[when] = 1 - (1 - frac[when]) * (1 - tax / float(res["equity"].loc[when]))
    return (1 + monthly) * (1 - frac) - 1


def ew_universe(ctx: Context, cfg: dict, end: Optional[str] = None) -> pd.Series:
    """Equal-weight eligible universe, monthly, gross (no costs): entry at next open, valued at next decision close."""
    rows = ctx.decision_rows(cfg.get("start"), end)
    out = {}
    for a, b in zip(rows[:-1], rows[1:]):
        names = ctx.eligible(a)
        if not names:
            continue
        r = ctx.A_last.iloc[b][names] / ctx.AO.iloc[a + 1][names] - 1
        out[ctx.dates[b]] = float(r.replace([np.inf, -np.inf], np.nan).mean())
    return pd.Series(out, dtype=float)


def build_rank_cache(ctx: Context, signals, end: Optional[str] = None, start: Optional[str] = None) -> Dict:
    rows = ctx.decision_rows(start, end)
    return {(s, i): ctx.ranks(s, i) for s in signals for i in rows}
