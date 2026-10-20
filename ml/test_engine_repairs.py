"""Regression fixtures for the engine repairs F01-F05 (synthetic data only; no network).

Run: python ml/test_engine_repairs.py
"""

import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.forecasts import calibrate  # noqa: E402
from ml.portfolio import (DEFAULTS, Variant, affordable_qty, after_tax_monthly, brokerage_fee,  # noqa: E402
                          funded_capacity, run_backtest_v2)
from ml.signals import Context  # noqa: E402

CFG = dict(DEFAULTS, min_mdv60=0.0, exclude_tickers=[], min_history_days=1, min_presence=0.0)


def panel_from(adj, volume=5e5, opens=None):
    open_ = adj.shift(1).fillna(adj.iloc[0]) if opens is None else opens
    return {"adj_close": adj, "close": adj.copy(), "open": open_, "volume": adj * 0 + volume, "adj_open": open_}


def flat(days=320, n=10, px=10.0):
    idx = pd.bdate_range("2016-01-01", periods=days)
    return pd.DataFrame({f"T{i:02d}": px for i in range(n)} | {"VAS": 50.0}, index=idx)


def ranks_series(order):
    return pd.Series(np.arange(1, len(order) + 1), index=order)


def run(adj, v, cfg, order_fn, opens=None):
    cfg = dict(cfg, start=str(adj.index[70].date()))
    ctx = Context(panel_from(adj, opens=opens), None, cfg)
    rows = ctx.decision_rows(cfg["start"])
    rc = {("RAMOM", i): order_fn(k, i) for k, i in enumerate(rows)}
    return ctx, rows, run_backtest_v2(ctx, v, cfg, rc)


def debit(o):
    return float((o["value"] + o["brokerage"]).sum())


def test_f01_cmc_three_positions_no_overdraft():
    cfg = dict(CFG, broker="cmc", min_position=500.0, min_order=500.0, min_parcel=500.0,
               slippage_tiers={"high": 0.0025, "mid": 0.0025, "low": 0.0025, "etf": 0.0025})
    fee = lambda value: brokerage_fee(cfg, "BUY", value)
    # old two-pass sizing: 100 shares x 10.025 = 1,002.50 + A$11 = 1,013.50 per name -> A$3,040.50 from A$3,030
    assert affordable_qty(10.0, 0.0025, 1010.0, fee) == 99            # 992.48, free CMC buy under A$1,000
    names = [f"T{i:02d}" for i in range(10)]
    _, _, r = run(flat(), Variant(n=3, regime=False, buffer_mult=1, capital=3030), cfg, lambda k, i: ranks_series(names))
    buys = r["orders"][r["orders"]["side"] == "BUY"]
    first = buys[buys["date"] == buys["date"].min()]
    assert len(first) == 3 and debit(first) <= 3030 + 1e-9, debit(first)
    assert r["cash"] >= 0 and (r["equity"] > 0).all()


def test_f01_fee_boundaries_and_expensive_shares():
    for broker in ("flat", "stake", "cmc"):
        cfg = dict(CFG, broker=broker)
        fee = lambda value: brokerage_fee(cfg, "BUY", value)
        for budget in (499.0, 500.0, 999.0, 1000.0, 1001.0, 1011.0, 1012.0, 29_999.0, 30_003.0, 30_010.0, 40_004.0, 40_010.0):
            for px in (0.37, 1.0, 9.99, 100.0, 333.33, 600.0):
                for slip in (0.0, 0.0025):
                    q = affordable_qty(px, slip, budget, fee)
                    fill = px * (1 + slip)
                    assert q * fill + fee(q * fill) <= budget + 1e-9, (broker, budget, px, q)
                    assert (q + 1) * fill + fee((q + 1) * fill) > budget - 1e-9, (broker, budget, px, q)  # maximal
    stake = lambda value: brokerage_fee(dict(CFG, broker="stake"), "BUY", value)
    assert affordable_qty(100.0, 0.0, 30_003.0, stake) == 300             # 30,000 + A$3
    assert affordable_qty(100.0, 0.0, 30_002.9, stake) == 299
    assert affordable_qty(100.0, 0.0, 40_004.0, stake) == 400             # 40,000 + 0.01% = A$4
    cmc = lambda value: brokerage_fee(dict(CFG, broker="cmc"), "BUY", value)
    assert affordable_qty(10.0, 0.0, 1010.0, cmc) == 100 and affordable_qty(10.0, 0.0, 1010.99, cmc) == 100
    assert affordable_qty(10.0, 0.0, 1011.0, cmc) == 100                  # 1,010 would cost A$1,021
    # expensive share: one A$400 share fits A$700 but is below the A$500 parcel -> no buy, never rounded up
    cfg = dict(CFG, broker="stake")
    adj = flat(px=400.0)
    names = [f"T{i:02d}" for i in range(10)]
    _, _, r = run(adj, Variant(n=3, regime=False, capital=2100), cfg, lambda k, i: ranks_series(names))
    assert r["n_eff"] == 3
    stock_buys = r["orders"][(r["orders"]["side"] == "BUY") & (r["orders"]["ticker"] != "VAS")] if len(r["orders"]) else []
    assert len(stock_buys) == 0 or (stock_buys["shares"] * 400 >= 500).all()


def test_capacity_is_fee_aware():
    cfg = dict(CFG, broker="stake")
    assert funded_capacity(cfg, 1500.0) == 2      # nominal 1500 // 500 = 3, but fees and slippage overdraw
    assert funded_capacity(cfg, 4000.0) == 7      # nominal 8
    names = [f"T{i:02d}" for i in range(10)]
    _, _, r = run(flat(), Variant(n=3, regime=False, capital=1500), cfg, lambda k, i: ranks_series(names))
    assert r["n_nominal"] == 3 and r["n_eff"] == 0 and "disabled" in r["skipped_reason"]
    _, _, r = run(flat(), Variant(n=8, regime=False, capital=4000), cfg, lambda k, i: ranks_series(names))
    assert r["n_nominal"] == 8 and r["n_eff"] == 7
    buys = r["orders"][r["orders"]["side"] == "BUY"]
    assert debit(buys[buys["date"] == buys["date"].min()]) <= 4000 and (buys["value"] >= 500).all()


def test_f02_missing_open_no_fill():
    adj = flat()
    opens = adj.shift(1).fillna(adj.iloc[0])
    cfg = dict(CFG, broker="flat", start=str(adj.index[70].date()))
    ctx = Context(panel_from(adj, opens=opens), None, cfg)
    j = ctx.decision_rows(cfg["start"])[0] + 1
    opens.iloc[j, opens.columns.get_loc("T00")] = np.nan  # T00 has no open on the first execution day
    names = [f"T{i:02d}" for i in range(10)]
    _, _, r = run(adj, Variant(n=3, regime=False, capital=9000), cfg, lambda k, i: ranks_series(names), opens=opens)
    first = r["orders"][r["orders"]["date"] == adj.index[j]]
    assert "T00" not in set(first["ticker"])
    rej = r["rejected"]
    assert len(rej) == 1 and rej.iloc[0]["ticker"] == "T00" and rej.iloc[0]["status"] == "rejected"
    assert abs(r["cash"] + debit(r["orders"]) - 9000) < 1e-6   # cash moved only by real fills
    # a sell with no open stays on the book
    adj2 = flat()
    opens2 = adj2.shift(1).fillna(adj2.iloc[0])
    ctx2 = Context(panel_from(adj2, opens=opens2), None, cfg)
    rows = ctx2.decision_rows(cfg["start"])
    opens2.iloc[rows[1] + 1, opens2.columns.get_loc("T00")] = np.nan
    swap = ["T05", "T06", "T07", "T08", "T09", "T01", "T02", "T03", "T04", "T00"]
    _, _, r2 = run(adj2, Variant(n=3, regime=False, buffer_mult=1, capital=9000), cfg,
                   lambda k, i: ranks_series(names if k == 0 else swap), opens=opens2)
    o = r2["orders"]
    t00 = o[(o["ticker"] == "T00") & (o["side"] == "SELL")]
    assert len(t00) == 1 and t00.iloc[0]["date"] == adj2.index[rows[2] + 1]  # sold at the next review instead


def test_f03_causal_calendar():
    idx = pd.bdate_range("2016-01-01", periods=400)
    early = pd.DataFrame({f"E{i}": 10.0 for i in range(6)}, index=idx)
    late = pd.DataFrame({f"L{i:02d}": 10.0 for i in range(30)}, index=idx)
    late.iloc[:100] = np.nan                     # thirty tickers list on day 100
    full = pd.concat([early, late], axis=1).assign(VAS=50.0)
    ctx_full = Context(panel_from(full), None, CFG)
    ctx_prefix = Context(panel_from(full.iloc[:90]), None, CFG)
    assert len(ctx_full.dates) == 400                       # old full-panel median rule erased the first 100
    assert ctx_full.dates[:90].equals(ctx_prefix.dates)
    assert ctx_full.calendar_id == "causal_count_v1"
    stray = full.copy()
    stray.iloc[200, 1:] = np.nan                            # a stray row with 1 priced ticker is not a session
    assert len(Context(panel_from(stray), None, CFG).dates) == 399


def test_f04_risk_exit_cadence_and_missing_rank():
    names = [f"T{i:02d}" for i in range(10)]
    without_t00 = [n for n in names if n != "T00"]
    cfg = dict(CFG, broker="flat", min_position=2500.0, min_order=1000.0)
    # quarterly alpha cadence: T00 drops out of the eligible/scored set in non-quarter months -> sold after
    # the 2nd consecutive monthly miss (monthly protective cadence), not delayed to the next quarter
    adj = flat(days=420)
    ctx = Context(panel_from(adj), None, dict(cfg, start=str(adj.index[70].date())))
    rows = ctx.decision_rows(str(adj.index[70].date()))
    months = [ctx.dates[i].month for i in rows]
    k0 = next(k for k, m in enumerate(months) if m == 6)     # bought at a June (quarter) decision
    order = lambda k, i: ranks_series(names if k <= k0 else without_t00)
    _, _, r = run(adj, Variant(n=3, regime=False, capital=9000, rebalance="Q"), cfg,
                  lambda k, i: order(k, i) if k >= k0 else ranks_series(names))
    o = r["orders"]
    sell = o[(o["ticker"] == "T00") & (o["side"] == "SELL")]
    assert len(sell) == 1 and sell.iloc[0]["reason"] == "ineligible"
    assert sell.iloc[0]["decision_date"].month == 8           # Jul miss 1, Aug miss 2: risk exit off-quarter
    # monthly: a single month with a missing rank does not trigger an immediate rank sale
    one_gap = lambda k, i: ranks_series(without_t00 if k == 1 else names)
    _, _, r = run(flat(), Variant(n=3, regime=False, capital=9000), cfg, one_gap)
    o = r["orders"]
    assert not ((o["ticker"] == "T00") & (o["side"] == "SELL")).any()


def test_f05_valuation_through_latest_session():
    idx = pd.bdate_range("2016-01-01", "2017-03-23")
    adj = pd.DataFrame({f"T{i:02d}": np.linspace(10, 12, len(idx)) for i in range(10)} | {"VAS": 50.0}, index=idx)
    cfg = dict(CFG, broker="flat", min_position=2500.0, min_order=1000.0)
    names = [f"T{i:02d}" for i in range(10)]
    _, _, r = run(adj, Variant(n=3, regime=False, capital=9000), cfg, lambda k, i: ranks_series(names))
    assert r["valuation_date"] == pd.Timestamp("2017-03-23") and r["equity"].index[-1] == pd.Timestamp("2017-03-23")
    assert r["last_execution_date"] == pd.Timestamp("2017-03-01")
    assert r["monthly"].index[-1] == pd.Timestamp("2017-02-28")   # completed months only
    held = {t: q for t, q in r["final_book"].items() if q}
    marks = adj.iloc[-1]
    assert abs(r["equity"].iloc[-1] - (r["cash"] + sum(q * marks[t] for t, q in held.items()))) < 1e-6
    # an explicit end date is respected: nothing executes after it
    cfg_end = dict(cfg, start=str(adj.index[70].date()))
    ctx = Context(panel_from(adj), None, cfg_end)
    rc = {("RAMOM", i): ranks_series(names) for i in ctx.decision_rows(cfg_end["start"])}
    r2 = run_backtest_v2(ctx, Variant(n=3, regime=False, capital=9000), cfg_end, rc, end="2016-12-31")
    assert r2["orders"]["date"].max() <= pd.Timestamp("2016-12-31") and r2["equity"].index[-1] <= pd.Timestamp("2016-12-31")


def test_f25_tax_not_pulled_forward():
    months = pd.date_range("2021-01-31", "2022-03-31", freq="ME")
    d = pd.Timestamp
    orders = pd.DataFrame([{"date": d("2021-02-01"), "ticker": "X", "side": "BUY", "shares": 100, "price": 10.0, "brokerage": 3.0},
                           {"date": d("2021-12-01"), "ticker": "X", "side": "SELL", "shares": 100, "price": 15.0, "brokerage": 3.0}])
    res = {"monthly": pd.Series(0.01, index=months), "orders": orders, "equity": pd.Series(1e5, index=months)}
    out = after_tax_monthly(res, 0.32)
    assert np.allclose(out.values, 0.01)                         # FY22 tax is due 31 Oct 2022, after the data
    assert abs(out.attrs["unpaid_tax"]["FY2022"] - 0.32 * (500 - 3 - 3)) < 1e-9   # brokerage in the cost base


def test_forecast_calibration_is_causal():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2014-01-01", periods=900)
    adj = pd.DataFrame({f"T{i:02d}": 10 * np.exp(np.cumsum(rng.normal(0.0002 * i, 0.01, 900))) for i in range(15)}, index=idx)
    adj["VAS"] = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, 900)))
    cfg = dict(CFG, min_history_days=260)
    ctx = Context(panel_from(adj), None, cfg)
    i = ctx.decision_rows()[-6]
    fb = calibrate(ctx, "RAMOM", i, min_dates=3)
    assert fb.label_cutoff <= ctx.dates[i] and fb.trained_until < ctx.dates[i]
    later = adj.copy()
    later.loc[later.index > ctx.dates[i]] *= rng.uniform(0.5, 1.5, later.loc[later.index > ctx.dates[i]].shape)
    fb2 = calibrate(Context(panel_from(later), None, cfg), "RAMOM", i, min_dates=3)
    assert fb.mu.round(12).equals(fb2.mu.round(12)) and fb.identity() == fb2.identity()
    assert "VAS" in fb.mu.index and fb.status == "ok" and np.isfinite(fb.mu_se.drop("VAS")).all()


def test_research_pipeline_offline_and_immutable_freeze():
    """strat-backtest end to end on synthetic data in a temp dir (never results/): continuous walk-forward
    account, append-only frozen registry (a rerun reuses, never re-freezes), complete neighbour criterion."""
    import json
    import tempfile
    from types import SimpleNamespace

    from ml import strategy_cli as sc

    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2015-06-01", "2025-03-31")
    mkt = rng.normal(0.0003, 0.008, len(idx))
    cols = {f"T{i:02d}": 20 * np.exp(np.cumsum(mkt + rng.normal(0.00005 * (i % 5), 0.013, len(idx)))) for i in range(14)}
    cols["VAS"] = 50 * np.exp(np.cumsum(mkt))
    adj = pd.DataFrame(cols, index=idx)
    cfg = dict(CFG, broker="stake", min_history_days=260)
    ctx = Context(panel_from(adj, volume=2e6), None, cfg)
    saved = sc.RESULTS, sc.SETTINGS
    with tempfile.TemporaryDirectory() as tmp:
        try:
            sc.RESULTS, sc.SETTINGS = pathlib.Path(tmp), [(25000.0, 0.0)]
            ctx.tfm_dir = pathlib.Path(tmp)
            assert sc.strat_backtest(ctx, cfg, SimpleNamespace(end=None)) == 0
            v1 = json.loads((pathlib.Path(tmp) / "validation.json").read_text())["settings"]["25000_core0"]
            reg1 = json.loads((pathlib.Path(tmp) / "frozen_choice.json").read_text())
            assert sc.strat_backtest(ctx, cfg, SimpleNamespace(end=None)) == 0
            v2 = json.loads((pathlib.Path(tmp) / "validation.json").read_text())["settings"]["25000_core0"]
            reg2 = json.loads((pathlib.Path(tmp) / "frozen_choice.json").read_text())
            hold = pathlib.Path(tmp) / "holdings.csv"
            hold.write_text("ticker,shares\nT01,10\nCASH,500\n")
            today = ctx.dates[-1]
            sc.strat_recommend(ctx, dict(cfg, capital=25000.0, core_weight=0.0),
                               SimpleNamespace(capital=None, holdings=str(hold)), today=today)
            rec = json.loads((pathlib.Path(tmp) / "recommendation.json").read_text())
            sc.strat_recommend(ctx, dict(cfg, capital=26000.0, core_weight=0.0), SimpleNamespace(capital=None), today=today)
            rec_other = json.loads((pathlib.Path(tmp) / "recommendation.json").read_text())
            sc.strat_recommend(ctx, dict(cfg, capital=25000.0, core_weight=0.0), SimpleNamespace(capital=None),
                               today=today + pd.offsets.BDay(10))
            rec_stale = json.loads((pathlib.Path(tmp) / "recommendation.json").read_text())
            vpath = pathlib.Path(tmp) / "validation.json"   # force PAPER to exercise the proposal path
            vj = json.loads(vpath.read_text())
            vj["settings"]["25000_core0"]["tier"] = "PAPER"
            vpath.write_text(json.dumps(vj))
            last_month_end = ctx.dates[ctx.decision_rows()[-1]]
            sc.strat_recommend(ctx, dict(cfg, capital=25000.0, core_weight=0.0), SimpleNamespace(capital=None),
                               today=last_month_end)
            rec_paper = json.loads((pathlib.Path(tmp) / "recommendation.json").read_text())
            paper_rows = (pathlib.Path(tmp) / "paper").glob("*.csv")
            paper_text = "".join(f.read_text() for f in paper_rows)
        finally:
            sc.RESULTS, sc.SETTINGS = saved
    assert len(reg1["experiments"]) == len(reg2["experiments"]) == 1 and reg2["experiments"] == reg1["experiments"]
    assert v2["frozen"]["reused"] and not v1["frozen"]["reused"] and v1["frozen"]["variant"] == v2["frozen"]["variant"]
    assert all("passes_a_full" in n for n in v1["neighbours"]) and "hold_out_note" in v1
    assert v1["wf_select"]["months"] > 0 and v1["capacity"]["funded_n_eff"] >= 3
    assert rec["execution_orders"] == [] and rec["account"]["cash"] == 500 and rec["setting"] == "25000_core0"
    assert any("synthetic" in b for b in rec["execution_blockers"]) and rec["data_state"] == "current"
    assert all(o["status"] == "estimate" for o in rec["paper_proposals"])
    assert rec_other["setting"] is None and rec_other["tier"] == "FALLBACK" and "not inherited" in rec_other["reasons"][0]
    assert rec_stale["data_state"] == "stale" and rec_stale["paper_proposals"] == []
    assert rec_paper["paper_book"]["date"] == str(ctx.dates[-1].date()) and rec_paper["execution_orders"] == []
    assert str(ctx.dates[-1].date()) in paper_text   # paper NAV dated by its real valuation session


def test_f20_asic_retry_and_as_of_join():
    import tempfile
    from types import SimpleNamespace

    from ml import data_asic

    class Fake:
        def get(self, url, timeout):
            return SimpleNamespace(status_code=404, ok=False, content=b"")

    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        data_asic.download(base, "2026-09-01", "2026-10-16", pause=0, session=Fake(), today=pd.Timestamp("2026-10-20"))
        missing = set((base / "missing.txt").read_text().split())
    assert "20260901" in missing and "20261016" not in missing          # recent 404 retried next run
    si = pd.DataFrame({"BHP": [0.02, 0.03]}, index=pd.to_datetime(["2026-01-03", "2026-01-20"]))  # 3 Jan = Saturday
    sessions = pd.bdate_range("2026-01-02", "2026-01-30")
    out = data_asic.as_of(si, sessions, max_age_days=10)
    assert np.isnan(out.loc["2026-01-02", "BHP"]) and out.loc["2026-01-05", "BHP"] == 0.02   # weekend update kept
    assert np.isnan(out.loc["2026-01-14", "BHP"]) and out.loc["2026-01-20", "BHP"] == 0.03   # stale -> missing


if __name__ == "__main__":
    started = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"all engine repair checks passed in {time.time() - started:.1f}s")
