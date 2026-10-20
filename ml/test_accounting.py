"""Ledger reconciliation checks (splits, dividends, fees, cashflows, settlement, partial fills).

Run: python ml/test_accounting.py
"""

import json
import pathlib
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.accounting import Ledger, discount_eligible, import_holdings  # noqa: E402

D = pd.Timestamp


def raises(fn, *a):
    try:
        fn(*a)
    except (ValueError, AssertionError):
        return True
    raise AssertionError(f"{fn.__name__} should have raised")


def test_partial_fills_fees_and_reservations():
    L = Ledger(10_000.0)
    oid = L.request(D("2024-01-02"), "BHP", "BUY", 100, est_debit=1_010.0)
    assert L.available == 8_990.0 and L.reserved == 1_010.0
    raises(L.fill, oid, D("2024-01-01"), 10, 10.0, 0.0)          # no fill before its request
    L.fill(oid, D("2024-01-03"), 60, 10.02, 3.0)                  # slippage is in the price; fee once
    L.fill(oid, D("2024-01-03"), 40, 10.03, 0.0)
    raises(L.fill, oid, D("2024-01-03"), 1, 10.0, 0.0)            # nothing left to fill
    raises(L.fill, "paper-O99", D("2024-01-03"), 1, 10.0, 0.0)
    cost = 60 * 10.02 + 3 + 40 * 10.03
    assert L.shares["BHP"] == 100 and L.reserved == 0 and abs(L.available - (10_000 - cost)) < 1e-9
    assert abs(sum(q * c for q, c, _ in L.lots["BHP"]) - cost) < 1e-9     # brokerage in the cost base
    assert abs(L.nav({"BHP": 10.5})["nav"] - (10_000 - cost + 1_050)) < 1e-9
    o2 = L.request(D("2024-01-04"), "CBA", "BUY", 10, est_debit=1_000.0)
    L.fill(o2, D("2024-01-05"), 4, 100.0, 3.0)
    L.cancel(o2, "unfilled at close")
    assert L.orders[o2]["status"] == "partial_cancelled" and L.reserved == 0 and L.shares["CBA"] == 4
    assert len({f["id"] for f in L.fills}) == len(L.fills) == 3


def test_sells_settlement_and_oversell():
    L = Ledger(1_000.0, settlement_days=2)
    o = L.request(D("2024-03-01"), "XYZ", "BUY", 50, est_debit=503.0)
    L.fill(o, D("2024-03-01"), 50, 10.0, 3.0)
    raises(L.request, D("2024-03-04"), "XYZ", "SELL", 51)
    s = L.request(D("2024-03-04"), "XYZ", "SELL", 30)
    raises(L.request, D("2024-03-04"), "XYZ", "SELL", 21)         # only 20 unreserved
    L.fill(s, D("2024-03-04"), 30, 12.0, 3.0)
    before = L.available
    L.process(D("2024-03-05"))
    assert L.available == before and L.unsettled                  # T+1: still unsettled
    nav = L.nav({"XYZ": 12.0})["nav"]
    L.process(D("2024-03-06"))
    assert abs(L.available - (before + 357.0)) < 1e-9 and not L.unsettled
    assert abs(L.nav({"XYZ": 12.0})["nav"] - nav) < 1e-9           # settlement moves cash, not NAV
    r = L.realised[0]
    assert r["qty"] == 30 and abs(r["proceeds"] - 357.0) < 1e-9 and abs(r["cost_base"] - 30 * 503 / 50) < 1e-9
    raises(L.withdraw, D("2024-03-07"), 1e9)
    L.withdraw(D("2024-03-07"), 100.0, "owner withdrawal")
    assert [c["amount"] for c in L.cashflows] == [1_000.0, -100.0]  # external cashflows kept separately


def test_split_dividend_no_double_counting():
    L = Ledger(5_000.0, settlement_days=0)
    o = L.request(D("2024-05-01"), "ABC", "BUY", 101, est_debit=2_100.0)
    L.fill(o, D("2024-05-01"), 101, 20.0, 0.0)
    nav0 = L.nav({"ABC": 20.0})["nav"]
    L.split(D("2024-06-03"), "ABC", 1.5, cash_in_lieu_price=20 / 1.5)
    L.split(D("2024-06-03"), "ABC", 1.5, cash_in_lieu_price=20 / 1.5)   # re-applied: ignored
    assert L.shares["ABC"] == 151
    assert abs(L.nav({"ABC": 20 / 1.5})["nav"] - nav0) < 1e-9          # value-neutral incl. cash in lieu
    L.dividend(D("2024-07-01"), D("2024-07-20"), "ABC", 0.5)
    L.dividend(D("2024-07-01"), D("2024-07-20"), "ABC", 0.5)            # duplicate announcement: ignored
    o = L.request(D("2024-07-02"), "ABC", "BUY", 10, est_debit=140.0)   # bought after ex: no entitlement
    L.fill(o, D("2024-07-02"), 10, 13.0, 0.0)
    cash = L.available
    L.process(D("2024-07-19"))
    assert L.available == cash
    L.process(D("2024-07-20"))
    L.process(D("2024-07-25"))
    assert abs(L.available - (cash + 151 * 0.5)) < 1e-9


def test_discount_rule_and_import():
    assert not discount_eligible("2020-01-01", "2021-01-01") and discount_eligible("2020-01-01", "2021-01-02")
    with tempfile.TemporaryDirectory() as tmp:
        csv = pathlib.Path(tmp) / "h.csv"
        csv.write_text("ticker,shares\nBHP.AX,10\nCASH,1234.5\n")
        a = import_holdings(csv)
        assert a["holdings"] == {"BHP": 10} and a["cash"] == 1234.5
        js = pathlib.Path(tmp) / "h.json"
        js.write_text(json.dumps({"cash": 1234.5, "holdings": {"BHP": 10}}))
        assert import_holdings(js)["account_hash"] == a["account_hash"]
        csv.write_text("ticker,shares\nBHP,10.5\n")
        raises(import_holdings, csv)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all accounting checks passed")
