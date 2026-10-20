"""Raw-share account ledger (AccountState): physical integer shares, AUD cash balances, FIFO tax lots,
order requests vs fills, settlement, corporate actions and external cashflows.

Rules (declared, checked after every event by `check()`):
- shares are non-negative integers (a split's fractional entitlement is paid as cash in lieu);
- available cash never goes negative (no margin); reserved cash/shares cover open requests;
- NAV = available + reserved + unsettled + dividends receivable + sum(shares * raw mark);
- a request is not a fill; every fill has a unique id, references an open order, never exceeds the
  remaining requested quantity; a fill price already includes slippage, so slippage is never charged again;
- sale proceeds settle after `settlement_days` sessions (0 = immediately available);
- a dividend entitlement is fixed at the ex-date (shares held at the start of that day) and paid once.
Brokerage is part of the cost base of a buy and reduces the proceeds of a sale (lot records keep both).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import deque
from typing import Dict, List, Optional

import pandas as pd

EPS = 1e-6


class Ledger:
    def __init__(self, cash: float = 0.0, settlement_days: int = 2, account_id: str = "paper",
                 calendar: Optional[pd.DatetimeIndex] = None) -> None:
        self.account_id = account_id
        self.settlement_days = settlement_days
        self.calendar = calendar
        self.available = 0.0
        self.reserved = 0.0
        self.unsettled: List[list] = []          # [settle_date, amount]
        self.receivable: Dict[tuple, list] = {}  # (ticker, ex_date) -> [pay_date, amount, paid]
        self.shares: Dict[str, int] = {}
        self.reserved_shares: Dict[str, int] = {}
        self.lots: Dict[str, deque] = {}         # ticker -> deque of [qty, cost_per_share incl. brokerage, date]
        self.orders: Dict[str, dict] = {}
        self.fills: List[dict] = []
        self.cashflows: List[dict] = []
        self.realised: List[dict] = []
        self.actions_applied: set = set()
        self._ids = itertools.count(1)
        if cash:
            self.deposit(pd.Timestamp("1970-01-01"), cash, "opening balance")

    # ---- cash ------------------------------------------------------------------------------
    def deposit(self, date, amount: float, note: str = "") -> None:
        if amount < 0:
            raise ValueError("use withdraw()")
        self.available += amount
        self.cashflows.append({"date": pd.Timestamp(date), "amount": amount, "note": note})
        self.check()

    def withdraw(self, date, amount: float, note: str = "") -> None:
        if amount > self.available + EPS:
            raise ValueError("withdrawal exceeds available cash")
        self.available -= amount
        self.cashflows.append({"date": pd.Timestamp(date), "amount": -amount, "note": note})
        self.check()

    def _settle_date(self, date) -> pd.Timestamp:
        date = pd.Timestamp(date)
        if self.calendar is not None:
            later = self.calendar[self.calendar > date]
            return later[self.settlement_days - 1] if self.settlement_days and len(later) >= self.settlement_days else date
        return date + pd.offsets.BDay(self.settlement_days)

    def process(self, date) -> None:
        """Settle sale proceeds and pay dividends due on or before `date`."""
        date = pd.Timestamp(date)
        keep = []
        for settle, amount in self.unsettled:
            if settle <= date:
                self.available += amount
            else:
                keep.append([settle, amount])
        self.unsettled = keep
        for key, rec in self.receivable.items():
            if not rec[2] and rec[0] <= date:
                self.available += rec[1]
                rec[2] = True
                self.cashflows.append({"date": rec[0], "amount": rec[1], "note": f"dividend {key[0]}", "internal": True})
        self.check()

    # ---- orders ------------------------------------------------------------------------------
    def request(self, date, ticker: str, side: str, qty: int, est_debit: float = 0.0) -> str:
        """Record an order request. Buys reserve `est_debit` cash; sells reserve the shares."""
        if qty <= 0 or int(qty) != qty:
            raise ValueError("quantity must be a positive integer")
        if side == "BUY":
            if est_debit > self.available + EPS:
                raise ValueError("insufficient available cash for the request")
            self.available -= est_debit
            self.reserved += est_debit
        else:
            free = self.shares.get(ticker, 0) - self.reserved_shares.get(ticker, 0)
            if qty > free:
                raise ValueError("sell request exceeds unreserved shares")
            self.reserved_shares[ticker] = self.reserved_shares.get(ticker, 0) + qty
        oid = f"{self.account_id}-O{next(self._ids)}"
        self.orders[oid] = {"id": oid, "date": pd.Timestamp(date), "ticker": ticker, "side": side, "qty": int(qty),
                            "filled": 0, "reserved_cash": est_debit if side == "BUY" else 0.0, "status": "requested"}
        self.check()
        return oid

    def fill(self, oid: str, date, qty: int, price: float, fee: float) -> str:
        """Record an actual (possibly partial) fill. `price` already includes slippage."""
        o = self.orders.get(oid)
        if o is None or o["status"] not in ("requested", "partial"):
            raise ValueError(f"fill references no open order: {oid}")
        if qty <= 0 or int(qty) != qty or o["filled"] + qty > o["qty"]:
            raise ValueError("fill exceeds the remaining requested quantity")
        date = pd.Timestamp(date)
        if date < o["date"]:
            raise ValueError("fill dated before its request")
        value = qty * price
        t = o["ticker"]
        if o["side"] == "BUY":
            debit = value + fee
            use = min(o["reserved_cash"], debit)
            self.reserved -= use
            o["reserved_cash"] -= use
            if debit - use > self.available + EPS:
                raise ValueError("fill debit exceeds reserved plus available cash")
            self.available -= debit - use
            self.shares[t] = self.shares.get(t, 0) + qty
            self.lots.setdefault(t, deque()).append([qty, debit / qty, date])
        else:
            self.reserved_shares[t] -= qty
            self.shares[t] -= qty
            proceeds = value - fee
            remaining = qty
            while remaining:
                lot = self.lots[t][0]
                take = min(remaining, lot[0])
                self.realised.append({"date": date, "ticker": t, "qty": take, "proceeds": proceeds * take / qty,
                                      "cost_base": lot[1] * take, "acquired": lot[2]})
                lot[0] -= take
                remaining -= take
                if not lot[0]:
                    self.lots[t].popleft()
            settle = self._settle_date(date)
            if settle <= date:
                self.available += proceeds
            else:
                self.unsettled.append([settle, proceeds])
        o["filled"] += qty
        o["status"] = "filled" if o["filled"] == o["qty"] else "partial"
        if o["status"] == "filled" and o["side"] == "BUY" and o["reserved_cash"]:
            self.available += o["reserved_cash"]
            self.reserved -= o["reserved_cash"]
            o["reserved_cash"] = 0.0
        fid = f"{oid}-F{len(self.fills) + 1}"
        self.fills.append({"id": fid, "order": oid, "date": date, "ticker": t, "side": o["side"], "qty": int(qty),
                           "price": price, "fee": fee})
        self.check()
        return fid

    def cancel(self, oid: str, why: str = "cancelled") -> None:
        o = self.orders[oid]
        if o["status"] not in ("requested", "partial"):
            return
        if o["side"] == "BUY":
            self.available += o["reserved_cash"]
            self.reserved -= o["reserved_cash"]
            o["reserved_cash"] = 0.0
        else:
            self.reserved_shares[o["ticker"]] -= o["qty"] - o["filled"]
        o["status"] = "cancelled" if not o["filled"] else "partial_cancelled"
        o["why"] = why
        self.check()

    # ---- corporate actions -------------------------------------------------------------------
    def split(self, date, ticker: str, ratio: float, cash_in_lieu_price: float = 0.0) -> None:
        """`ratio` new shares per old share (2.0 = 2-for-1, 0.1 = 1-for-10 consolidation)."""
        key = ("split", ticker, pd.Timestamp(date))
        if key in self.actions_applied:
            return
        self.actions_applied.add(key)
        old = self.shares.get(ticker, 0)
        if not old:
            return
        exact = old * ratio
        new = int(exact + 1e-9)
        if self.reserved_shares.get(ticker):
            raise ValueError("split with open sell requests: cancel and re-request")
        lots = deque()
        for q, cost, d in self.lots[ticker]:
            lots.append([q * ratio, cost / ratio, d])   # fractional until the cash-in-lieu below
        frac = exact - new
        if frac > 1e-9:
            self.available += frac * cash_in_lieu_price
            take = frac
            while take > 1e-9:                          # FIFO: cash-in-lieu disposes of the oldest fraction
                lot = lots[0]
                part = min(take, lot[0])
                self.realised.append({"date": pd.Timestamp(date), "ticker": ticker, "qty": part,
                                      "proceeds": part * cash_in_lieu_price, "cost_base": part * lot[1],
                                      "acquired": lot[2]})
                lot[0] -= part
                take -= part
                if lot[0] <= 1e-9:
                    lots.popleft()
        for lot in lots:
            lot[0] = round(lot[0], 6)
        self.lots[ticker] = deque(lot for lot in lots if lot[0] > 1e-9)
        self.shares[ticker] = new
        self.check()

    def dividend(self, ex_date, pay_date, ticker: str, per_share: float) -> None:
        """Entitlement fixed by shares held at the ex-date; paid (once) by `process` on the pay date."""
        key = (ticker, pd.Timestamp(ex_date))
        if key in self.receivable:
            return
        held = self.shares.get(ticker, 0)
        if held:
            self.receivable[key] = [pd.Timestamp(pay_date), held * per_share, False]
        self.check()

    # ---- valuation / checks ------------------------------------------------------------------
    def cash_balances(self) -> float:
        return (self.available + self.reserved + sum(a for _, a in self.unsettled)
                + sum(r[1] for r in self.receivable.values() if not r[2]))

    def nav(self, marks: Dict[str, float]) -> Dict[str, object]:
        missing = [t for t, q in self.shares.items() if q and not (t in marks and marks[t] == marks[t])]
        value = sum(q * marks[t] for t, q in self.shares.items() if q and t not in missing)
        return {"nav": self.cash_balances() + value, "unmarked": missing, "complete": not missing}

    def check(self) -> None:
        assert self.available >= -EPS, f"negative available cash {self.available}"
        assert self.reserved >= -EPS
        for t, q in self.shares.items():
            assert isinstance(q, int) and q >= 0, (t, q)
            assert 0 <= self.reserved_shares.get(t, 0) <= q, f"reservation conflict {t}"
            lot_q = sum(lot[0] for lot in self.lots.get(t, ()))
            assert abs(lot_q - q) < 1e-6, f"lots {lot_q} != shares {q} for {t}"
        open_cash = sum(o["reserved_cash"] for o in self.orders.values())
        assert abs(open_cash - self.reserved) < 1e-6, "reserved cash does not match open orders"
        ids = [f["id"] for f in self.fills]
        assert len(ids) == len(set(ids)), "duplicate fill ids"

    def state_hash(self) -> str:
        blob = json.dumps({"shares": {k: v for k, v in sorted(self.shares.items()) if v},
                           "cash": round(self.cash_balances(), 6), "orders": sorted(
                               (o["id"], o["status"], o["filled"]) for o in self.orders.values())}, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def discount_eligible(acquired, disposed) -> bool:
    """CGT discount needs the asset owned for at least 12 months, excluding the acquisition and disposal
    days (ATO). E.g. bought 1 Jan 2020 -> earliest eligible disposal 2 Jan 2021. Verify current ATO rules
    and the account's eligibility (individual/trust vs company, residency) before relying on it."""
    return pd.Timestamp(disposed) >= pd.Timestamp(acquired) + pd.DateOffset(years=1) + pd.Timedelta(days=1)


def import_holdings(path) -> Dict[str, object]:
    """Load actual holdings/cash from a local CSV (columns: ticker, shares[, cost_base, acquired]) with an
    optional `CASH` row, or JSON {"cash": x, "holdings": {ticker: shares}, "as_of": date}. Returns the
    parsed state and its hash; raises on fractional/negative shares."""
    import pathlib

    path = pathlib.Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        cash = float(raw.get("cash", 0.0))
        holdings = {str(k).replace(".AX", "").upper(): v for k, v in raw.get("holdings", {}).items()}
        as_of = raw.get("as_of")
    else:
        df = pd.read_csv(path)
        df["ticker"] = df["ticker"].astype(str).str.replace(".AX", "", regex=False).str.upper()
        cash = float(df.loc[df["ticker"] == "CASH", "shares"].sum())
        holdings = dict(zip(df.loc[df["ticker"] != "CASH", "ticker"], df.loc[df["ticker"] != "CASH", "shares"]))
        as_of = None
    for t, q in holdings.items():
        if float(q) != int(float(q)) or float(q) < 0:
            raise ValueError(f"holding {t}: shares must be a non-negative integer, got {q}")
    holdings = {t: int(float(q)) for t, q in holdings.items() if int(float(q))}
    if cash < 0:
        raise ValueError("cash must be non-negative")
    blob = json.dumps({"cash": round(cash, 2), "holdings": dict(sorted(holdings.items()))})
    return {"cash": cash, "holdings": holdings, "as_of": as_of, "source": str(path),
            "account_hash": hashlib.sha1(blob.encode()).hexdigest()[:12]}
