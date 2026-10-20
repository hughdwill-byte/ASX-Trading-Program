"""Dashboard ranking, explanation and publication-contract checks. All fixtures are SYNTHETIC (no network, no models)
and are written to temp dirs, never results/ or docs/data/.

Run: python ml/test_dashboard.py
"""

import json
import math
import pathlib
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml import dashboard_rankings as db  # noqa: E402

DATES = [d.date().isoformat() for d in pd.date_range("2016-04-30", periods=24, freq="ME")]
COHORT = {"engine_id": "E", "calendar_id": "C", "capital": 10000.0, "core_policy": 0.0, "broker": "stake",
          "cost_basis": "x1", "contributions": "none", "price_basis": "adj", "start": DATES[0], "end": DATES[-1]}


def method(mid, monthly, kind="strategy", cohort=None, status="ok", **kw):
    m = {"method_id": mid, "kind": kind, "status": status, "cohort": dict(COHORT, **(cohort or {})),
         "monthly": dict(zip(DATES, monthly)) if monthly is not None else None, "full": kw.pop("full", {})}
    m.update(kw)
    return m


def bench(r=0.005):
    b = method("BENCH_TR", [r] * 24, kind="benchmark", full={"mdd_daily": -0.10})
    b["dates"] = list(DATES)
    return b


def test_historical_rank_by_net_aud_profit_ties_and_losses():
    ms = [method("B", [0.01] * 24), method("A", [0.01] * 24), method("C", [0.04] * 12 + [-0.01] * 12),
          method("D", [0.0] * 24)]
    h = db.historical_ranking(ms, bench())
    order = [(r["rank"], r["method_id"]) for r in h["rows"]]
    assert order[:3] == [(1, "C"), (2, "A"), (2, "B")], order        # tie shares rank, stable id order
    assert [r["rank"] for r in h["rows"]][3] == 4                     # competition ranking 1,2,2,4
    a = next(r for r in h["rows"] if r["method_id"] == "A")
    assert abs(a["final_value"] - 10000 * 1.01 ** 24) < 1e-6 and abs(a["net_profit"] - (a["final_value"] - 10000)) < 1e-9
    assert abs(a["cagr"] - (1.01 ** 12 - 1)) < 1e-12 and abs(a["delta_cagr_vs_vas"] - (1.01 ** 12 - 1.005 ** 12)) < 1e-12
    assert not h["all_lost"] and any(r["method_id"] == "BENCH_TR" for r in h["rows"])
    losers = db.historical_ranking([method("X", [-0.01] * 24), method("Y", [-0.002] * 24)], bench(-0.003))
    assert losers["all_lost"] and losers["rows"][0]["method_id"] == "Y"   # smallest loss, not "profitable winner"


def test_cohort_rejection_and_short_coverage():
    ms = [method("cap", [0.01] * 24, cohort={"capital": 50000.0}), method("brk", [0.01] * 24, cohort={"broker": "cmc"}),
          method("eng", [0.01] * 24, cohort={"engine_id": "OLD"}), method("px", [0.01] * 24, cohort={"price_basis": "raw"}),
          method("ok", [0.01] * 24)]
    short = method("WF", None, kind="walk_forward_account", cohort={"start": DATES[12]})
    short["monthly"] = dict(zip(DATES[12:], [0.02] * 12))
    h = db.historical_ranking(ms + [short], bench())
    assert [r["method_id"] for r in h["rows"]] == ["ok", "BENCH_TR"]   # another capital is never rescaled in
    assert {r["method_id"] for r in h["incompatible"]} == {"cap", "brk", "eng", "px"}
    s = h["short_coverage"][0]
    assert s["method_id"] == "WF" and s["own_interval"]["months"] == 12
    assert abs(s["own_interval"]["vas_total_return"] - (1.005 ** 12 - 1)) < 1e-12   # same-period VAS, not full period
    assert h["common_interval"] == {"start": DATES[0], "end": DATES[-1], "months": 24}  # not shortened


def test_missing_invalid_unfunded_and_after_tax_are_distinct():
    gap = [0.01] * 24
    gap[5] = float("nan")
    ms = [method("nan", gap), method("dead", [0.0] * 10 + [-1.0] + [0.0] * 13), method("none", None),
          method("unf", None, status="unfunded", reason="satellite disabled"), method("cash", [0.0] * 24)]
    h = db.historical_ranking(ms, bench())
    nr = {r["method_id"]: r for r in h["not_ranked"]}
    assert "non-finite" in nr["nan"]["reason"] and "invalid NAV" in nr["dead"]["reason"]
    assert nr["none"]["reason"] == "series missing" and nr["unf"]["status"] == "unfunded"
    cash = next(r for r in h["rows"] if r["method_id"] == "cash")
    assert cash["net_profit"] == 0 and cash["status"] == "ranked"          # funded cash earning zero is a result
    ms = [method("taxed", [0.01] * 24, monthly_after_tax=dict(zip(DATES, [0.009] * 24))), method("untaxed", [0.02] * 24)]
    b = bench()
    assert db.historical_ranking(ms, b, "after_tax_estimate")["state"] == "unavailable"   # no VAS after-tax series
    b["monthly_after_tax"] = dict(zip(DATES, [0.005] * 24))
    t = db.historical_ranking(ms, b, "after_tax_estimate")
    assert [r["method_id"] for r in t["rows"]] == ["taxed", "BENCH_TR"]
    assert t["not_ranked"][0]["method_id"] == "untaxed"                    # never pre-tax substituted
    nob = bench()
    nob["monthly"] = None
    assert db.historical_ranking(ms, nob)["state"] == "unavailable"


def ev(W=0.02, L=0.01, mddW=-0.15, mddL=-0.10, dsr=0.6, passed=2, years=(1, 1, 0, 1), dragW=0.01, dragL=0.01, **kw):
    vas = lambda c, m: {"cagr": c, "mdd_daily": m, "cost_drag": 0.0, "months": 60}
    out = {"W": {"cagr": 0.08 + W, "mdd_daily": mddW, "cost_drag": dragW, "months": 60},
           "L": {"cagr": 0.08 + L, "mdd_daily": mddL, "cost_drag": dragL, "months": 27},
           "VAS_W": vas(0.08, -0.20), "VAS_L": vas(0.08, -0.12), "dsr": dsr,
           "neighbours": {"evaluated": 3, "passed": passed},
           "years": [{"year": str(2019 + i), "method": 0.1 + 0.01 * b, "vas": 0.1} for i, b in enumerate(years)]}
    out.update(kw)
    return out


def test_score_formula_endpoints_units_and_toy_example():
    assert db.g(0) == 50 and db.g(0.1) == 100 and db.g(-0.1) == 0 and db.g(0.2) == 100 and abs(db.g(0.01) - 55) < 1e-12
    for w in db.WEIGHTS.values():
        assert abs(sum(w) - 1) < 1e-12
    assert abs(db.score({"R": 70, "D": 40, "B": 80, "T": 60, "C": 50})["score"] - 61.5) < 1e-9
    toy_b = db.score({"R": 60, "D": 80, "B": 70, "T": 70, "C": 90})
    assert abs(toy_b["score"] - 72.0) < 1e-9 and abs(sum(toy_b["contributions"].values()) - 72.0) < 1e-9
    c = db.components(ev())["components"]
    assert abs(c["R"] - (0.6 * db.g(0.02) + 0.4 * db.g(0.01))) < 1e-12
    assert abs(c["D"] - (0.6 * db.g(0.20 - 0.15) + 0.4 * db.g(0.12 - 0.10))) < 1e-12   # shallower fall = positive gap
    assert abs(c["B"] - 100 * (0.5 * 0.6 + 0.5 * 2 / 3)) < 1e-12 and c["T"] == 75.0
    assert abs(c["C"] - 100 * (1 - 0.01 / 0.03)) < 1e-12
    assert db.components(ev(dragW=0.05, dragL=0.05))["components"]["C"] == 0.0
    month_end_only = ev()
    month_end_only["W"] = {**month_end_only["W"], "mdd_daily": None, "mdd": -0.1}
    x = db.components(month_end_only)
    assert "D" not in x["components"] and db.score(x["components"]) is None   # no month-end substitution, no renorm
    nan_dsr = db.components(ev(dsr=float("nan")))
    assert "B" not in nan_dsr["components"]


def q(group="exploratory", assessment="not_assessed"):
    return {"group": group, "assessment": assessment}


def test_history_leader_can_lose_evidence_ranking_and_groups():
    risky = {"method_id": "RISKY", "status": "ok", "qualification": q(), "evidence": ev(W=0.04, L=0.03, mddW=-0.45,
                                                                                         mddL=-0.30, dsr=0.2, passed=0,
                                                                                         years=(1, 0, 0, 1), dragW=0.025,
                                                                                         dragL=0.025)}
    steady = {"method_id": "STEADY", "status": "ok", "qualification": q(), "evidence": ev(W=0.01, L=0.005)}
    r = db.evidence_ranking([risky, steady])
    rows = r["groups"][2]["rows"]
    assert [x["method_id"] for x in rows] == ["STEADY", "RISKY"]
    h = db.historical_ranking([method("RISKY", [0.03] * 24), method("STEADY", [0.01] * 24)], bench())
    assert h["rows"][0]["method_id"] == "RISKY"                       # history leader != evidence leader
    assert r["no_stock_method_qualifies"] and not r["groups"][0]["rows"] and not r["groups"][1]["rows"]
    paper = dict(steady, method_id="FROZEN", qualification=q("paper", "paper"))
    r2 = db.evidence_ranking([risky, paper, steady])
    assert [x["method_id"] for x in r2["groups"][1]["rows"]] == ["FROZEN"] and r2["groups"][1]["rows"][0]["group_rank"] == 1
    assert [x["group_rank"] for x in r2["groups"][2]["rows"]] == [1, 2]   # group-local ranks
    unscored = {"method_id": "THIN", "status": "ok", "qualification": q(), "evidence": ev(L=0.0)}
    unscored["evidence"]["L"]["months"] = 8
    r3 = db.evidence_ranking([unscored, {"method_id": "NR", "status": "not_run", "reason": "cache missing"}])
    assert not any(g["rows"] for g in r3["groups"]) and {x["method_id"] for x in r3["not_scored"]} == {"THIN", "NR"}
    assert "PBO" not in json.dumps(r3["shared_pbo"]) and "procedure" in r3["shared_pbo"]["scope"]


def test_near_ties_and_sensitivity_within_fixed_group():
    a = {"method_id": "A", "status": "ok", "qualification": q(), "evidence": ev(W=0.010)}
    b = {"method_id": "B", "status": "ok", "qualification": q(), "evidence": ev(W=0.0101)}
    c = {"method_id": "C", "status": "ok", "qualification": q(), "evidence": ev(W=-0.02, mddW=-0.05, mddL=-0.05)}
    r = db.evidence_ranking([a, b, c])
    rows = {x["method_id"]: x for x in r["groups"][2]["rows"]}
    assert rows["A"]["near_tie"] and rows["B"]["near_tie"]
    for x in rows.values():
        lo, hi = x["sensitivity_rank_range"]
        assert 1 <= lo <= x["group_rank"] <= hi <= 3 or lo <= hi <= 3
    assert set(r["groups"][2]["leaders"]) == set(db.WEIGHTS)
    assert r["score_manifest"]["version"] == db.SCORE_VERSION and r["score_manifest"]["weights"]["default"] == [.30, .20, .20, .15, .15]


def test_explanations_reconcile_with_metrics():
    h = db.historical_ranking([method("M", [0.01] * 12 + [0.0] * 12, costs_aud=123.4, full={"mdd_daily": -0.2})], bench())
    rows = {r["method_id"]: r for r in h["rows"]}
    years = db.calendar_years(dict(zip(DATES, [0.01] * 12 + [0.0] * 12)), dict(zip(DATES, [0.005] * 24)), DATES)
    x = db.historical_explanation(rows["M"], rows["BENCH_TR"], years, "cid", h["common_interval"])
    facts = {f["metric"]: f for f in x["facts"]}
    assert facts["net_profit"]["value"] == rows["M"]["net_profit"] and f"{rows['M']['net_profit']:,.0f}" in facts["net_profit"]["text"]
    assert facts["costs_aud"]["value"] == 123.4 and "not deducted again" in facts["costs_aud"]["text"]
    assert facts["attribution"]["value"] is None and facts["attribution"]["source"] == "not_available"
    assert facts["mdd_daily"]["value"] == -0.2
    complete = [y for y in years if y["complete"]]
    assert [y["year"] for y in complete] == ["2017"] and abs(complete[0]["method"] - (1.01 ** 3 - 1)) < 1e-12  # 2016, 2018 partial
    assert not db.historical_explanation(None, rows["BENCH_TR"], [], "cid", h["common_interval"])["available"]
    r = db.evidence_ranking([{"method_id": "M", "status": "ok", "qualification": q("exploratory", "failed"), "evidence": ev()}])
    row = r["groups"][2]["rows"][0]
    e = db.rank_explanation(row, ["(c) deflated Sharpe ..."], [], "cid")
    sf = e["facts"][0]
    assert sf["value"] == row["score"] and f"{row['score']:.1f}" in sf["text"]
    assert any(f["metric"] == "qualification" for f in e["facts"])


def test_method_registry_from_structured_policy():
    from ml.policy_spec import OptPolicy

    d = db.describe_method("X", {"kind": "variant", "signal": "RAMOM", "n": 5, "regime": True, "buffer_mult": 1,
                                 "rebalance": "Q", "tax_aware": True, "si_screen": False})
    assert d["family"] == "RAMOM" and "quarterly reviews, monthly risk exits" in d["chips"]
    assert any("not a full tax model" in c for c in d["chips"]) and "trend filter (checked monthly)" in d["chips"]
    opt = db.describe_method("OPT", {"kind": "opt_policy", **OptPolicy().to_json()})
    assert "no preset holding count" in opt["chips"] and "research only" in opt["chips"]
    assert "optimal" in opt["text"] and "not guaranteed" in opt["text"]
    assert db.describe_method("ZZZ_N5", {"kind": "variant", "signal": "ZZZ"})["text"] is None   # never invented
    assert "TimesFM" in db.describe_method("t", {"kind": "variant", "signal": "TFM_REL", "n": 5, "horizon": 21})["text"]


def test_paper_allocation_unequal_weights_and_cash():
    from ml.strategy_cli import paper_allocation

    when = pd.Timestamp("2026-10-16")
    for n in (0, 1, 5, 24):
        names = [f"S{i:02d}" for i in range(n)]
        marks = pd.DataFrame({t: [10.0 + i] for i, t in enumerate(names)} | {"VAS": [50.0]}, index=[when])
        book = {t: 3 + i for i, t in enumerate(names)} | {"VAS": 0}
        cash = 1234.5
        nav = cash + sum(q * marks.loc[when, t] for t, q in book.items())
        ctx = SimpleNamespace(A_last=marks, cfg={"stale_haircut": 0.3})
        a = paper_allocation(ctx, {"valuation_date": when, "equity": pd.Series([nav], index=[when]),
                                   "final_book": book, "cash": cash, "frozen": []})
        assert a["actual_count"] == n and abs(sum(h["weight"] for h in a["holdings"]) + a["cash_weight"] - 1) < 1e-12
        if n > 1:
            assert len({round(h["weight"], 9) for h in a["holdings"]}) > 1    # actual unequal weights, not 1/n


def _write_results(tmp, engine="E1", dash_engine="E1", tier="FALLBACK"):
    from ml import strategy_cli as sc

    (tmp / "validation.json").write_text(json.dumps({"engine_id": engine, "config_id": "c", "calendar_id": "k",
                                                     "settings": {"10000_core0": {"tier": tier, "reasons": ["(a) x"],
                                                                                  "frozen": {"variant": "RAMOM_N5_noreg_B1"}}}}))
    (tmp / "recommendation.json").write_text(json.dumps({"tier": tier, "setting": "10000_core0", "execution_orders": [],
                                                         "paper_proposals": [], "engine_id": sc.ENGINE_ID,
                                                         "broker_execution_state": "none: this program places no orders",
                                                         "execution_blockers": ["price basis is synthetic"]}))
    (tmp / "dashboard.json").write_text(json.dumps({"schema": db.DASHBOARD_SCHEMA, "engine_id": dash_engine, "config_id": "c",
                                                    "settings": {}, "research_id": "r"}))


def test_publisher_manifest_states_and_scenarios():
    from ml import strategy_cli as sc
    from ml.portfolio import DEFAULTS

    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        out = tmp / "latest.json"
        _write_results(tmp, engine=sc.ENGINE_ID, dash_engine=sc.ENGINE_ID)
        sc.strat_publish(dict(DEFAULTS), results_dir=tmp, out=out)
        d = json.loads(out.read_text())
        assert d["display_schema"] == sc.DISPLAY_SCHEMA and d["publication"]["state"] == "ok" and d["dashboard"]
        assert d["execution_orders"] == [] and "places no orders" in d["recommendation"]["broker_execution_state"]
        sc10 = d["scenarios"]["10000_core0"]
        assert sc10["evaluated"] and sc10["is_published_recommendation"] and sc10["chosen_before_testing"]
        assert d["scenarios"]["25000_core0"] == {**d["scenarios"]["25000_core0"], "evaluated": False, "tier": None}
        _write_results(tmp, engine=sc.ENGINE_ID, dash_engine="OLD")       # mixed versions are rejected
        sc.strat_publish(dict(DEFAULTS), results_dir=tmp, out=out)
        d = json.loads(out.read_text())
        assert d["publication"]["rankings_state"] == "needs_research_refresh" and d["dashboard"] is None
        assert d["publication"]["state"] == "ok"            # research validity itself is still current
        assert any("mixed research versions" in x for x in d["publication"]["rankings_reasons"])
        (tmp / "dashboard.json").unlink()
        sc.strat_publish(dict(DEFAULTS), results_dir=tmp, out=out)
        assert "no ranking artifact" in " ".join(json.loads(out.read_text())["publication"]["rankings_reasons"])
        _write_results(tmp, engine="OLD_ENGINE", dash_engine="OLD_ENGINE")   # validation from another engine
        sc.strat_publish(dict(DEFAULTS), results_dir=tmp, out=out)
        d = json.loads(out.read_text())
        assert d["publication"]["state"] == d["publication"]["rankings_state"] == "needs_research_refresh"


def test_pipeline_diagnostics_are_per_method_and_change_no_decisions():
    """Synthetic end-to-end research run in a temp dir: per-method DSR/neighbours differ from the frozen candidate's,
    the frozen registry and tiers are unaffected by the dashboard producer, and the shared PBO is cohort-level."""
    from ml import strategy_cli as sc
    from ml.portfolio import DEFAULTS
    from ml.signals import Context

    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2015-06-01", "2026-09-30")
    mkt = rng.normal(0.0003, 0.008, len(idx))
    cols = {f"T{i:02d}": 20 * np.exp(np.cumsum(mkt + rng.normal(0.00005 * (i % 5), 0.013, len(idx)))) for i in range(14)}
    cols["VAS"] = 50 * np.exp(np.cumsum(mkt))
    adj = pd.DataFrame(cols, index=idx)
    op = adj.shift(1).fillna(adj.iloc[0])
    cfg = dict(DEFAULTS, min_mdv60=0.0, exclude_tickers=[], min_history_days=260, min_presence=0.0, broker="stake")
    ctx = Context({"adj_close": adj, "close": adj, "open": op, "adj_open": op, "volume": adj * 0 + 2e6}, None, cfg)
    saved = sc.RESULTS, sc.SETTINGS
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        try:
            sc.RESULTS, sc.SETTINGS = tmp, [(25000.0, 0.0)]
            ctx.tfm_dir = tmp
            sc.strat_backtest(ctx, cfg, SimpleNamespace(end=None))
            v = json.loads((tmp / "validation.json").read_text())["settings"]["25000_core0"]
            reg = json.loads((tmp / "frozen_choice.json").read_text())
            dash = json.loads((tmp / "dashboard.json").read_text())
        finally:
            sc.RESULTS, sc.SETTINGS = saved
    s = dash["settings"]["25000_core0"]
    assert dash["schema"] == db.DASHBOARD_SCHEMA and len(reg["experiments"]) == 1
    assert s["frozen"] == v["frozen"]["variant"] == reg["experiments"][0]["variant"] and s["tier"] == v["tier"]
    rows = [r for g in s["evidence"]["groups"] for r in g["rows"]]
    frozen_row = next(r for r in rows if r["method_id"] == s["frozen"])
    assert abs(frozen_row["dsr"]["dsr"] - v["dsr_frozen"]["dsr"]) < 1e-12    # same convention as the engine
    others = {round(r["dsr"]["dsr"], 9) for r in rows if r["method_id"] != s["frozen"]}
    assert len(others) > 1                                                    # method-specific, not copied
    nb_frozen = [n["variant"] for n in frozen_row["neighbours"]["rows"]]
    assert nb_frozen == [n["variant"] for n in v["neighbours"]]
    assert any([n["variant"] for n in r["neighbours"]["rows"]] != nb_frozen for r in rows)
    assert all(r["group"] == "exploratory" for r in rows) and s["evidence"]["no_stock_method_qualifies"]
    assert s["evidence"]["shared_pbo"]["pbo"] == v["pbo"]["pbo"]
    hist = s["historical"]["pre_tax"]
    assert any(r["method_id"] == "BENCH_TR" for r in hist["rows"])
    assert [r["method_id"] for r in hist["short_coverage"]] == ["WF_SELECT"]
    for r in hist["rows"]:
        assert all(math.isfinite(r[k]) for k in ("net_profit", "final_value", "total_return"))
    ref_ids = {r["method_id"] for r in s["references"]}
    assert ref_ids == {"EW_UNIV", "AORD_PRICE"} and not ref_ids & {r["method_id"] for r in hist["rows"]}


if __name__ == "__main__":
    started = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"all dashboard checks passed in {time.time() - started:.1f}s")
