"""OptPolicy spec round-trips and an end-to-end synthetic research backtest (mechanics only, not performance).

Run: python ml/test_policies.py
"""

import dataclasses
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.policies import run_backtest_opt  # noqa: E402
from ml.policy_spec import OptPolicy, controlled_family, decode, explain_config, neighbours  # noqa: E402
from ml.portfolio import DEFAULTS, Variant, run_backtest_v2  # noqa: E402
from ml.signals import Context  # noqa: E402

CFG = dict(DEFAULTS, min_mdv60=0.0, exclude_tickers=[], min_history_days=260, min_presence=0.0, broker="stake")


def test_null_cap_round_trip_and_hashes():
    p = OptPolicy()
    js = json.dumps(p.to_json(), allow_nan=False)
    assert '"holding_cap": null' in js and '"count_target": null' in js
    back = OptPolicy.from_json(json.loads(js))
    assert back == p and back.id == p.id and back.config_hash() == p.config_hash()
    assert OptPolicy(gamma=8.0).config_hash() != p.config_hash()            # material input changes identity
    assert OptPolicy(capital=50_000).config_hash() == p.config_hash()       # setting is not a trial dimension
    for bad in (dict(holding_cap=0), dict(holding_cap=5), dict(count_target=float("inf"))):
        try:
            OptPolicy(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert OptPolicy(mode="capped_control", holding_cap=5).id.startswith("OPT_RAMOM_cap5_var_")
    assert p.id.startswith("OPT_RAMOM_nocap_var_")
    assert decode("RAMOM_N5_noreg_B1_Q_TAX") == Variant(n=5, regime=False, buffer_mult=1, rebalance="Q", tax_aware=True)
    assert decode(p.to_json()) == p
    assert len(controlled_family()) == 5 and len(neighbours(p)) == 3
    ex = explain_config(DEFAULTS, {"n_positions": 8, "bogus": 1}, {"10000_core0": {"variant": "RAMOM_N5_noreg_B1"}})
    rows = {r["key"]: r for r in ex["rows"]}
    assert rows["n_positions"]["status"] == "shadowed" and rows["n_positions"]["origin"] == "config.yaml"
    assert rows["cash_rate_annual"]["status"] == "inactive" and rows["bogus"]["status"] == "unregistered"
    assert ex["frozen"][0]["frozen_variant"] == "RAMOM_N5_noreg_B1"


def synthetic(n=30, days=900, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2013-01-01", periods=days)
    mkt = rng.normal(0.0003, 0.008, days)
    cols = {f"T{i:02d}": 20 * np.exp(np.cumsum(mkt + rng.normal(0.0001 * (i % 7), 0.012, days))) for i in range(n)}
    cols["VAS"] = 50 * np.exp(np.cumsum(mkt))
    adj = pd.DataFrame(cols, index=idx)
    opens = adj.shift(1).fillna(adj.iloc[0])
    panel = {"adj_close": adj, "close": adj, "open": opens, "adj_open": opens, "volume": adj * 0 + 2e6}
    return Context(panel, None, dict(CFG, start=str(idx[400].date())))


def test_end_to_end_research_backtest():
    ctx = synthetic()
    cfg = dict(CFG, start=str(ctx.dates[400].date()))
    pol = OptPolicy(capital=200_000.0, name_limit=0.15)
    r = run_backtest_opt(ctx, pol, cfg)
    dec = r["decisions"]
    assert len(dec) and set(dec["status"]) <= {"proposal", "no_trade_optimal", "no_trade_forecast_unavailable"}
    assert r["executable_quantities"] is False and r["research_only"]
    marks = ctx.A_last.iloc[-1]
    nav = r["cash"] + sum(q * marks[t] for t, q in r["final_book"].items() if q)
    assert abs(r["equity"].iloc[-1] - nav) < 1e-6 and r["cash"] >= 0
    o = r["orders"]
    assert len(o) and (o["shares"] == o["shares"].astype(int)).all()
    counts = dec["stock_count"].dropna()
    print(f"  synthetic: stock count range {counts.min():.0f}-{counts.max():.0f}, mean effective "
          f"{dec['effective_count'].dropna().mean():.1f}, {len(o)} fills, {r['runtime_seconds']:.1f}s")
    # same inputs -> identical result (deterministic search)
    r2 = run_backtest_opt(ctx, pol, cfg)
    assert r2["ledger_hash"] == r["ledger_hash"]
    # controls run on the same engine and the legacy control on the same panel
    for c in controlled_family():
        rc = run_backtest_opt(ctx, dataclasses.replace(c, capital=200_000.0), cfg)
        if c.holding_cap:
            assert rc["decisions"]["stock_count"].dropna().max() <= c.holding_cap
    legacy = run_backtest_v2(ctx, Variant(n=5, regime=False, capital=200_000.0), cfg)
    assert legacy["monthly"].index.equals(r["monthly"].index)            # identical benchmark intervals


def test_opt_study_entry_point_offline():
    import tempfile
    from types import SimpleNamespace

    from ml import strategy_cli as sc

    ctx = synthetic(n=12, days=720)
    cfg = dict(CFG, start=str(ctx.dates[620].date()))
    with tempfile.TemporaryDirectory() as tmp:   # never results/: synthetic data
        assert sc.strat_opt_study(ctx, cfg, SimpleNamespace(end=None, capitals="20000"), pathlib.Path(tmp)) == 0
        rows = json.loads((pathlib.Path(tmp) / "opt_study" / "compare.json").read_text())
        man = json.loads((pathlib.Path(tmp) / "opt_study" / "manifest.json").read_text())
    ids = {r["id"] for r in rows}
    assert {"BENCH_TR", "RAMOM_N5_noreg_B1", "RAMOM_N8_noreg_B3"} <= ids and sum(i.startswith("OPT_") for i in ids) == 8
    assert man["policies"][3]["holding_cap"] is None and "not for trading" in man["status"]


if __name__ == "__main__":
    started = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"all policy checks passed in {time.time() - started:.1f}s")
