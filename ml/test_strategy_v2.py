"""Self-check for strategy v2 (synthetic data only; no network, no TimesFM weights).

Run: python ml/test_strategy_v2.py
"""

import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml import validation as val  # noqa: E402
from ml.portfolio import brokerage_fee, DEFAULTS, Variant, after_tax_monthly, run_backtest_v2, update_regime  # noqa: E402
from ml.signals import Context, zscore  # noqa: E402

CFG = dict(DEFAULTS, broker="flat", min_position=2500.0, min_order=1000.0, min_mdv60=0.0, exclude_tickers=[], min_history_days=260, start="2016-01-29")


def panel_from(adj: pd.DataFrame, volume: float = 1e6, opens=None):
    close = adj.copy()
    open_ = adj.shift(1).fillna(adj.iloc[0]) if opens is None else opens
    vol = pd.DataFrame(volume, index=adj.index, columns=adj.columns)
    return {"adj_close": adj, "close": close, "open": open_, "volume": vol, "adj_open": open_}


def random_panel(n=12, days=700, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=days)
    cols = {f"T{i:02d}": 10 * np.exp(np.cumsum(rng.normal(0.0003 * (i - n / 2), 0.015, days))) for i in range(n)}
    cols["VAS"] = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, days)))
    adj = pd.DataFrame(cols, index=idx)
    aord = pd.Series(5000 * adj["VAS"] / adj["VAS"].iloc[0], index=idx)
    return panel_from(adj), aord


def test_signals():
    panel, aord = random_panel()
    ctx = Context(panel, aord, CFG)
    i = len(ctx.dates) - 1
    a = panel["adj_close"]["T03"].to_numpy()
    r = np.diff(np.log(a))[-252:]
    vol = r.std(ddof=1) * math.sqrt(252)
    assert abs(ctx.MOM_12_1.iloc[i]["T03"] - (a[-22] / a[-253] - 1)) < 1e-12
    assert abs(ctx.VOL_252.iloc[i]["T03"] - vol) < 1e-12
    ramom = (math.log(a[-22]) - math.log(a[-253])) / (vol * math.sqrt(231 / 252))
    assert abs(ctx.RAMOM.iloc[i]["T03"] - ramom) < 1e-9
    names = ctx.eligible(i)
    combo = ctx.scores("COMBO", i, names)
    exp = 0.5 * zscore(ctx.RAMOM.iloc[i][names]) + 0.5 * zscore(-ctx.VOL_252.iloc[i][names])
    assert np.allclose(combo.sort_index(), exp.dropna().sort_index())


def test_no_lookahead():
    panel, aord = random_panel()
    ctx = Context(panel, aord, CFG)
    rows = ctx.decision_rows(CFG["start"])
    i = rows[len(rows) // 2]
    base = run_backtest_v2(ctx, Variant(n=4, regime=False), CFG)
    t = ctx.dates[i]
    rng = np.random.default_rng(1)
    altered = {k: v.copy() for k, v in panel.items()}
    for k in ("adj_close", "close", "open", "adj_open"):
        altered[k].loc[altered[k].index > t] *= rng.uniform(0.5, 1.5, altered[k].loc[altered[k].index > t].shape)
    ctx2 = Context(altered, aord.where(aord.index <= t, aord * 3), CFG)
    assert ctx.ranks("RAMOM", i).equals(ctx2.ranks("RAMOM", i))
    assert ctx.RESMOM.iloc[i].equals(ctx2.RESMOM.iloc[i])
    alt = run_backtest_v2(ctx2, Variant(n=4, regime=False), CFG)
    key = lambda o: o.loc[o["date"] <= ctx.dates[i + 1], ["date", "ticker", "side"]].reset_index(drop=True)
    assert key(base["orders"]).equals(key(alt["orders"]))


def test_resmom_market_neutral():
    rng = np.random.default_rng(3)
    days = 3000
    idx = pd.bdate_range("2005-01-01", periods=days)
    mkt = rng.normal(0.0003, 0.01, days)
    cols = {f"T{i:02d}": 10 * np.exp(np.cumsum(1.5 * mkt + rng.normal(0, 0.012, days))) for i in range(6)}
    cols["VAS"] = 50 * np.exp(np.cumsum(mkt))
    ctx = Context(panel_from(pd.DataFrame(cols, index=idx)), None, dict(CFG, min_history_days=1, min_presence=0.0))
    lm = np.log(ctx.A["VAS"])
    trailing = (lm.shift(21) - lm.shift(252))  # market's own 12-1 return over the same window
    # windows overlap heavily, so single-stock correlations are noisy; the average over stocks is not
    res, raw = [], []
    for t in (f"T{i:02d}" for i in range(6)):
        both = pd.concat([ctx.RESMOM[t], ctx.RAMOM[t], trailing], axis=1).dropna()
        res.append(both.iloc[:, 0].corr(both.iloc[:, 2]))
        raw.append(both.iloc[:, 1].corr(both.iloc[:, 2]))
    assert abs(np.mean(res)) < 0.1 and np.mean(raw) > 0.5


def flat_ctx(days=320, n=10, mdv_high=None):
    idx = pd.bdate_range("2016-01-01", periods=days)
    adj = pd.DataFrame({f"T{i:02d}": 10.0 for i in range(n)} | {"VAS": 50.0}, index=idx)
    p = panel_from(adj, volume=5e5)  # $5m/day -> mid slippage tier
    if mdv_high:
        p["volume"][mdv_high] = 2e6  # 10 x 2e6 = 20m dollar turnover -> high-liquidity tier
    cfg = dict(CFG, min_history_days=1, min_presence=0.0, start=str(idx[70].date()))  # MDV60 needs 60 days
    return Context(p, None, cfg), cfg


def ranks_series(order):
    return pd.Series(np.arange(1, len(order) + 1), index=order)


def test_buffer_and_costs():
    ctx, cfg = flat_ctx(mdv_high="T00")
    rows = ctx.decision_rows(cfg["start"])
    names = [f"T{i:02d}" for i in range(10)]
    v = Variant(n=3, regime=False, buffer_mult=2, capital=9000)  # N=3, K=6
    rc = {("RAMOM", rows[0]): ranks_series(names)}
    # month 2: T00 at rank 4 (kept, <=K), T01 at rank 6 (kept), T02 at rank 7 (sold); fill from rank<=3
    rc[("RAMOM", rows[1])] = ranks_series(["T05", "T06", "T07", "T00", "T08", "T01", "T02", "T09", "T03", "T04"])
    for i in rows[2:]:
        rc[("RAMOM", i)] = rc[("RAMOM", rows[1])]
    r = run_backtest_v2(ctx, v, cfg, rc)
    o = r["orders"]
    first = o[o["date"] == ctx.dates[rows[0] + 1]]
    assert set(first["ticker"]) == {"T00", "T01", "T02"} and (first["side"] == "BUY").all()
    second = o[o["date"] == ctx.dates[rows[1] + 1]]
    assert list(second.loc[second["side"] == "SELL", "ticker"]) == ["T02"]
    assert list(second.loc[second["side"] == "BUY", "ticker"]) == ["T05"]
    # costs: brokerage per order, tiered slippage, integer shares
    assert abs(r["costs"]["brokerage"] - 6.5 * len(o)) < 1e-9
    t00 = first[first["ticker"] == "T00"].iloc[0]
    t01 = first[first["ticker"] == "T01"].iloc[0]
    assert abs(t00["price"] - 10 * 1.001) < 1e-9 and abs(t01["price"] - 10 * 1.0025) < 1e-9
    assert all(float(x).is_integer() for x in o["shares"])
    # min_order: no stock is bought when every order would be below it (A$9,000 funds one A$5,000 floor ->
    # satellite disabled, account holds the core ETF)
    r2 = run_backtest_v2(ctx, v, dict(cfg, min_order=5000), rc)
    assert r2["n_eff"] == 0 and set(r2["orders"]["ticker"]) == {"VAS"}


def test_broker_fees():
    for v, flat, stake, cmc_buy, cmc_sell in [(500, 6.5, 3, 0, 11), (1000, 6.5, 3, 0, 11), (1001, 6.5, 3, 11, 11),
                                              (30000, 6.5, 3, 30, 30), (40000, 6.5, 4, 40, 40)]:
        for broker, buy, sell in (("flat", flat, flat), ("stake", stake, stake), ("cmc", cmc_buy, cmc_sell)):
            c = {"broker": broker, "brokerage": 6.5}
            assert abs(brokerage_fee(c, "BUY", v) - buy) < 1e-9 and abs(brokerage_fee(c, "SELL", v) - sell) < 1e-9, (broker, v)


def test_min_parcel():
    ctx, cfg = flat_ctx()
    cfg = dict(cfg, broker="stake", min_position=500.0, min_order=500.0, min_parcel=500.0)
    rows = ctx.decision_rows(cfg["start"])
    rc = {("RAMOM", i): ranks_series([f"T{k:02d}" for k in range(10)]) for i in rows}
    r = run_backtest_v2(ctx, Variant(n=5, regime=False, capital=2000), cfg, rc)
    # was n_eff == 4 (nominal 2000 // 500): four A$500 parcels plus fees and slippage overdraw A$2,000
    assert r["n_nominal"] == 4 and r["n_eff"] == 3
    buys = r["orders"][r["orders"]["side"] == "BUY"]
    assert len(buys) and (buys["value"] >= 500).all()
    assert abs(r["costs"]["brokerage"] - 3.0 * len(r["orders"])) < 1e-9


def _tax_ctx(final_px):
    idx = pd.bdate_range("2016-01-01", periods=560)
    adj = pd.DataFrame({f"T{i:02d}": 10.0 for i in range(10)} | {"VAS": 50.0}, index=idx)
    adj.iloc[200:, 0] = final_px  # T00 moves after entry (~row 71)
    cfg = dict(CFG, min_history_days=1, min_presence=0.0, start=str(idx[70].date()))
    return Context(panel_from(adj, volume=5e5), None, cfg), cfg


def _t00_sale(final_px, tax_aware):
    ctx, cfg = _tax_ctx(final_px)
    rows = ctx.decision_rows(cfg["start"])
    names = [f"T{i:02d}" for i in range(10)]
    bad = ["T05", "T06", "T07", "T08", "T00", "T01", "T02", "T09", "T03", "T04"]  # T00 rank 5, outside K=3
    entry = ctx.dates[rows[0] + 1]
    rc = {("RAMOM", i): ranks_series(bad if (ctx.dates[i + 1] - entry).days >= 330 else names) for i in rows}
    v = Variant(n=3, regime=False, buffer_mult=1, capital=9000, tax_aware=tax_aware)
    o = run_backtest_v2(ctx, v, cfg, rc)["orders"]
    sells = o[(o["ticker"] == "T00") & (o["side"] == "SELL")]
    return None if sells.empty else (sells.iloc[0]["date"] - entry).days


def test_tax_aware_deferral():
    assert Variant(rebalance="Q", tax_aware=True).id == "RAMOM_N5_reg_B3_Q_TAX"
    plain = _t00_sale(12.0, False)
    assert 330 <= plain < 365                      # rank sell as soon as T00 falls outside K
    held = _t00_sale(12.0, True)                   # in profit, 330-364 days: deferred
    assert held is not None and held >= 365
    assert _t00_sale(8.0, True) == plain           # at a loss: sold immediately


def test_quarterly_only_trades_in_quarter_months():
    ctx, cfg = flat_ctx(days=400)
    rows = ctx.decision_rows(cfg["start"])
    names = [f"T{i:02d}" for i in range(10)]
    rc = {("RAMOM", i): ranks_series(names[k % 3:] + names[:k % 3]) for k, i in enumerate(rows)}
    v = Variant(n=3, regime=False, buffer_mult=1, capital=9000, rebalance="Q")
    r = run_backtest_v2(ctx, v, cfg, rc)
    ok = {ctx.dates[i + 1] for i in rows if ctx.dates[i].month in (3, 6, 9, 12)}
    assert len(r["orders"]) and set(r["orders"]["date"]) <= ok
    assert len(r["monthly"]) == len(rows) - 1      # snapshots stay monthly


def test_regime():
    s = None
    s = update_regime(s, 101, 100, 0.02)   # initial: above SMA -> ON
    assert s is True
    assert update_regime(True, 99, 100, 0.02) is True     # inside band: unchanged
    assert update_regime(True, 97.9, 100, 0.02) is False  # below band: OFF
    assert update_regime(False, 101.5, 100, 0.02) is False
    assert update_regime(False, 102.1, 100, 0.02) is True
    # only evaluated at month-end: a mid-month crash that recovers by month-end causes no sells
    ctx, cfg = flat_ctx()
    aord = pd.Series(np.linspace(100, 130, len(ctx.dates)), index=ctx.dates)  # steady uptrend: regime ON
    rows = ctx.decision_rows(cfg["start"])
    mid = rows[-3] - 8
    aord.iloc[mid:mid + 3] = 50.0
    ctx.aord = aord
    rc = {("RAMOM", i): ranks_series([f"T{k:02d}" for k in range(10)]) for i in rows}
    cfg = dict(cfg, regime_sma=20)
    r = run_backtest_v2(ctx, Variant(n=3, regime=True, capital=9000), cfg, rc)
    assert (r["orders"]["reason"] != "regime_off").all()


def test_effective_n_and_stale():
    ctx, cfg = flat_ctx()
    r = run_backtest_v2(ctx, Variant(n=5, regime=False, core_weight=0.6, capital=10000), cfg)
    assert r["n_eff"] == 0 and "disabled" in r["skipped_reason"]
    assert set(r["orders"]["ticker"]) == {"VAS"}
    # stale exit with haircut
    ctx, cfg = flat_ctx()
    rows = ctx.decision_rows(cfg["start"])
    rc = {("RAMOM", i): ranks_series([f"T{k:02d}" for k in range(10)]) for i in rows}
    gap_end = rows[2]
    ctx.A.iloc[gap_end - 12:, ctx.A.columns.get_loc("T00")] = np.nan
    ctx.refresh_age()
    r = run_backtest_v2(ctx, Variant(n=3, regime=False, capital=9000), cfg, rc)
    # was: a "stale" sale at last close x 0.70 (fictitious liquidation, invented cash). Now the holding stays on
    # the book, marked at the haircut, and no order or cash comes from it.
    o = r["orders"]
    assert not ((o["ticker"] == "T00") & (o["side"] == "SELL")).any()
    assert "T00" in r["frozen"] and r["final_book"]["T00"] > 0
    others = sum(q * 10.0 for t, q in r["final_book"].items() if t not in ("T00", "VAS")) + r["final_book"]["VAS"] * 50
    assert abs(r["equity"].iloc[-1] - (r["cash"] + others + r["final_book"]["T00"] * 10 * 0.70)) < 1e-6


def test_selection_and_gate():
    idx = pd.date_range("2016-01-31", "2024-12-31", freq="ME")
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.01, 0.02, len(idx)), index=idx)
    b = pd.Series(rng.normal(0.0, 0.02, len(idx)), index=idx)
    b_future = b.copy()
    b_future[b_future.index >= "2019-07-01"] = 0.05  # only the future favours b
    wf = val.walk_forward_select({"a": a, "b": b_future}, {})
    assert wf["picks"][0]["variant"] == "a"  # 2019 choice uses only data before 2019-07-01
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "frozen_choice.json"
        try:
            val.hold_out_gate(path, "s", time.time())
            raise AssertionError("gate should raise without frozen_choice.json")
        except RuntimeError:
            pass
        val.freeze_choice({"a": a, "b": b}, {}, {"a": "h1", "b": "h2"}, path, "s")
        assert val.hold_out_gate(path, "s", time.time() + 1) == "a"


def test_psr_dsr_pbo_bootstrap():
    r = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.00, 0.01, 0.02, -0.01])
    sr = r.mean() / r.std(ddof=1)
    g3 = pd.Series(r).skew()
    g4 = pd.Series(r).kurt() + 3
    expected = 0.5 * (1 + math.erf((sr * math.sqrt(9) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)) / math.sqrt(2)))
    assert abs(val.psr(r) - expected) < 1e-12
    d = val.dsr(r, [0.1, 0.2, 0.3], 3)
    sr0 = math.sqrt(np.var([0.1, 0.2, 0.3], ddof=1)) * ((1 - val.EULER) * 0.4307272993 + val.EULER * 1.1619569289)
    assert abs(d["sr0_monthly"] - sr0) < 1e-3 and d["dsr"] < d["psr"]
    rng = np.random.default_rng(0)
    T = 120
    dominant = np.column_stack([rng.normal(0.03, 0.02, T)] + [rng.normal(0, 0.02, T) for _ in range(9)])
    assert val.pbo_cscv(dominant)["pbo"] < 0.1
    noise = rng.normal(0, 0.02, (T, 20))
    assert abs(val.pbo_cscv(noise)["pbo"] - 0.5) < 0.15
    x = rng.normal(0.01, 0.03, 60)
    assert val.stationary_bootstrap_ci(x) == val.stationary_bootstrap_ci(x)


def test_publish_and_page():
    from ml import strategy_cli as sc

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        out = tmp / "latest.json"
        out.write_text(json.dumps({"generated_at": "x", "recommendations": []}))
        (tmp / "recommendation.json").write_text(json.dumps({"tier": "FALLBACK", "action_text": "Hold VAS.AX",
                                                              "reasons": ["(c) x"], "orders_next_open": []}))
        (tmp / "strategy_compare.json").write_text(json.dumps([{"id": "RAMOM_N5_reg_B3", "setting": "10000_core0",
                                                                 "period": "full", "cagr": 0.1}]))
        (tmp / "validation.json").write_text(json.dumps({"settings": {}, "ic": {}}))
        sc.strat_publish(dict(DEFAULTS), results_dir=tmp, out=out)
        d = json.loads(out.read_text())
        for key in ("recommendation", "strategy_comparison", "orders_next_open", "benchmark_note", "data_flags"):
            assert key in d, key
        assert d["generated_at"] == "x" and d["recommendations"] == []
    html = pathlib.Path(__file__).resolve().parents[1].joinpath("docs", "index.html").read_text()
    assert 'id="compare-sec"' in html and 'id="rec"' in html
    assert "d.strategy_comparison || []" in html and "const r = d.recommendation;" in html


def test_after_tax_and_fund_filter():
    months = pd.date_range("2020-06-30", "2022-12-31", freq="ME")
    d = pd.Timestamp
    orders = pd.DataFrame([
        {"date": d("2020-07-01"), "ticker": t, "side": "BUY", "shares": 100, "price": 10.0} for t in "XYZ"] + [
        {"date": d("2020-12-01"), "ticker": "X", "side": "SELL", "shares": 100, "price": 15.0},  # +500 short
        {"date": d("2021-01-04"), "ticker": "Y", "side": "SELL", "shares": 100, "price": 6.0},   # -400 loss
        {"date": d("2021-08-02"), "ticker": "Z", "side": "SELL", "shares": 50, "price": 30.0},   # +1000 long
    ])
    res = {"monthly": pd.Series(0.01, index=months), "orders": orders, "equity": pd.Series(1e5, index=months)}
    out = after_tax_monthly(res, 0.32)
    expect = pd.Series(0.01, index=months)
    expect[d("2021-10-31")] = 1.01 * (1 - 0.32 * 100 / 1e5) - 1        # FY21: (500 - 400) taxed in full
    expect[d("2022-10-31")] = 1.01 * (1 - 0.32 * 1000 / 2 / 1e5) - 1    # FY22: discounted long gain
    assert np.allclose(out.values, expect.values), out[out != 0.01]
    assert after_tax_monthly(res, 0.0).equals(res["monthly"])

    from ml.strategy_cli import instrument_types
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "types.json"
        path.write_text(json.dumps({"IVV": "ETF"}))
        asked = []
        fake = lambda t: asked.append(t) or {"BHP": "EQUITY"}.get(t)
        types = instrument_types(["IVV", "BHP", "ZZZ"], path, fetch=fake)
        assert asked == ["BHP", "ZZZ"] and types == {"IVV": "ETF", "BHP": "EQUITY"}
        assert json.loads(path.read_text()) == types  # unknown ZZZ is retried next run, not cached


if __name__ == "__main__":
    started = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"all strategy v2 checks passed in {time.time() - started:.1f}s")
