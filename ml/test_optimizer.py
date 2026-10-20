"""Optimiser checks (synthetic inputs; verify mechanics, not ASX returns or investment merit).

Run: python ml/test_optimizer.py
"""

import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.optimizer import Problem, evaluate, exhaustive, revalidate_at_execution, solve, solve_continuous  # noqa: E402
from ml.risk_model import covariance, nearest_psd  # noqa: E402

FREE = lambda side, value: 0.0
FLAT3_BUY = lambda side, value: 3.0 if side == "BUY" else 0.0


def prob(mu, cov, cash=1e6, price=None, shares=None, gamma=4.0, fee=FREE, **kw):
    n = len(mu)
    return Problem(names=[f"A{k:02d}" for k in range(n)], price=np.full(n, 1.0) if price is None else price,
                   shares=np.zeros(n, int) if shares is None else shares, cash=cash, mu=np.asarray(mu, float),
                   cov=np.asarray(cov, float), gamma=gamma, fee=fee, opening_floor=kw.pop("opening_floor", 0.0), **kw)


def test_one_and_two_asset_reference():
    p = prob([0.01], [[0.0025]])
    assert abs(solve_continuous(p)["x"][0] - 0.5) < 1e-5                 # mu / (2 gamma sigma^2)
    d = solve(p)
    assert d["status"] == "proposal" and abs(d["targets"]["A00"]["pre_nav_weight"] - 0.5) < 1e-5
    S = np.array([[0.0025, 0.001], [0.001, 0.0016]])
    mu = np.array([0.01, 0.008])
    ref = np.linalg.solve(S, mu) / (2 * 4.0)
    assert np.allclose(solve_continuous(prob(mu, S))["x"], ref, atol=1e-5), ref


def test_unequal_edges_unequal_weights_and_symmetry():
    S = np.diag([0.0025] * 3)
    x = solve_continuous(prob([0.010, 0.015, 0.020], S, gamma=10))["x"]
    assert x[0] < x[1] < x[2] and x[0] > 0
    sym = solve_continuous(prob([0.015] * 6, 0.0036 * np.eye(6) + 0.0001, gamma=4))["x"]
    assert np.ptp(sym) < 1e-6
    # integer tie handling is deterministic and as even as possible
    pr = lambda: prob([0.015] * 6, 0.0036 * np.eye(6) + 0.0001, cash=10_000.0, price=np.full(6, 100.0), fee=FLAT3_BUY,
                      spread=np.full(6, 0.0025))
    a, b = solve(pr()), solve(pr())
    qa = [a["targets"][k]["shares"] for k in sorted(a["targets"])]
    assert qa == [b["targets"][k]["shares"] for k in sorted(b["targets"])] and max(qa) - min(qa) <= 1


def test_redundant_and_diversifying_assets():
    s2 = 0.0025
    redundant = np.full((2, 2), s2)                      # perfectly correlated: B is A with a worse forecast
    x = solve_continuous(prob([0.02, 0.015], redundant, gamma=10))["x"]
    assert x[1] < 1e-6 and x[0] > 0.3
    S = np.array([[s2, 0.95 * s2, 0], [0.95 * s2, s2, 0], [0, 0, s2]])
    x = solve_continuous(prob([0.02, 0.02, 0.012], S, gamma=10))["x"]
    assert x[2] > x[0] and x[2] > x[1]                   # diversification beats a higher stand-alone forecast


def test_no_purchase_vs_invalid_vs_floors():
    S = 0.0025 * np.eye(3)
    d = solve(prob([-0.001, -0.002, 0.0], S))
    assert d["status"] == "no_trade_optimal" and d["orders"] == [] and d["residual_cash"] == 1e6
    bad = solve(prob([0.01, np.nan, 0.01], S))
    assert bad["status"] == "invalid_input" and bad["orders"] == []
    floors = solve(prob([0.02] * 3, S, cash=1000.0, price=np.full(3, 10.0), opening_floor=5000.0))
    assert floors["status"] == "no_trade_optimal" and floors["orders"] == []


def test_existing_holdings_and_costs():
    S = 0.0025 * np.eye(2)
    p = prob([0.01, 0.01], S, cash=10_000.0, price=np.array([10.0, 10.0]), fee=FLAT3_BUY, spread=np.full(2, 0.0025))
    first = solve(p)
    q = np.array([first["targets"][k]["shares"] for k in ("A00", "A01")])
    cash = first["residual_cash"]
    again = solve(prob([0.01, 0.01], S, cash=cash, price=np.array([10.0, 10.0]), shares=q, fee=FLAT3_BUY,
                       spread=np.full(2, 0.0025)))
    assert again["status"] == "no_trade_optimal" and again["costs"]["fees"] == 0.0   # no fee on unchanged names
    # an old holding below the opening floor is kept, not sold just for being small
    tiny = prob([0.01, 0.01], S, cash=5_000.0, price=np.array([10.0, 10.0]), shares=np.array([5, 0]), fee=FLAT3_BUY,
                opening_floor=500.0, order_floor=500.0)
    d = solve(tiny)
    assert d["targets"]["A00"]["shares"] >= 5


def test_risk_participation_and_frozen():
    S = 0.0036 * np.eye(4) + 0.0001
    p = prob([0.02] * 4, S, risk_budget=0.02, price=np.full(4, 10.0))
    d = solve(p)
    assert d["risk"] <= 0.02 + 1e-9 and d["feasible"]
    p = prob([0.02, 0.01], 0.0025 * np.eye(2), participation=np.array([50_000.0, np.inf]))
    d = solve(p)
    assert d["targets"]["A00"]["target_notional"] <= 50_000 + 1e-6
    # frozen holding alone breaks the risk budget: nothing is erased, no buys, status reports the violation
    p = prob([0.02, 0.02], 0.0025 * np.eye(2), cash=1000.0, price=np.full(2, 10.0), shares=np.array([10_000, 0]),
             tradable=np.array([False, True]), risk_budget=0.01)
    d = solve(p)
    assert d["targets"]["A00"]["shares"] == 10_000 and not any(o["side"] == "BUY" for o in d["orders"])
    assert "risk budget" not in d["violations"] or d["status"] != "proposal"


def test_psd_and_covariance_inputs():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert np.linalg.eigvalsh(nearest_psd(bad)).min() >= 0
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    base = rng.normal(0, 0.01, 300)
    R = pd.DataFrame({"A": base, "B": base + 1e-9 * rng.normal(size=300), "C": 0.0, "D": rng.normal(0, 0.02, 300)},
                     index=idx)
    R.iloc[:250, 3] = np.nan                                  # thin history
    out = covariance(R, ["A", "B", "C", "D", "VAS"], horizon=21)
    S = out["cov"].to_numpy()
    assert np.all(np.isfinite(S)) and np.linalg.eigvalsh(S).min() >= 0
    assert set(out["thin"]) == {"D", "VAS"} and S[2, 2] > 0
    # perturbation: a tiny change in inputs gives a tiny change in the continuous solution
    mu = np.array([0.01, 0.0101, 0.0, 0.012, 0.008])
    x1 = solve_continuous(prob(mu, S, gamma=10))["x"]
    x2 = solve_continuous(prob(mu + 1e-7, S, gamma=10))["x"]
    assert np.abs(x1 - x2).sum() < 0.05


def fixture_24(extra_bad=0):
    n = 24 + extra_bad
    mu = np.r_[np.full(24, 0.015), np.full(extra_bad, -0.01)]
    S = 0.06 ** 2 * np.eye(n) + 0.01 ** 2
    return Problem(names=[f"S{k:02d}" for k in range(n)], price=np.full(n, 100.0), shares=np.zeros(n, int),
                   cash=200_000.0, mu=mu, cov=S, gamma=4.0, fee=FLAT3_BUY, name_limit=np.full(n, 0.35),
                   spread=np.full(n, 0.0025), opening_floor=500.0)


def test_no_cap_funds_more_than_twenty_names():
    p = fixture_24()
    d = solve(p)
    q = sorted(d["targets"][k]["shares"] for k in d["targets"])
    assert d["status"] == "proposal" and d["feasible"] and d["count"] == 24, d["count"]
    # the published feasible example: 84 shares in two names, 83 in 22; A$199,400; fees A$72; adverse A$498.50
    ref = np.array([84, 84] + [83] * 22)
    e_ref = evaluate(p, ref)
    assert e_ref["feasible"] and abs(e_ref["cash_post"] - 29.5) < 1e-6 and e_ref["costs"]["fees"] == 72.0
    assert abs(e_ref["costs"]["adverse"] - 498.5) < 1e-6
    assert d["objective"] >= e_ref["objective"] - 1e-12 and q == sorted(ref.tolist())
    assert d["residual_cash"] < 100 * 1.0025 + 3                  # nothing affordable left over
    # unattractive extra candidates do not force purchases or change the attractive book
    d2 = solve(fixture_24(extra_bad=10))
    assert all(d2["targets"].get(f"S{k:02d}", {"shares": 0})["shares"] == 0 for k in range(24, 34))
    assert d2["count"] == 24


def test_discrete_heuristic_vs_exhaustive_oracle():
    rng = np.random.default_rng(11)
    gaps = []
    for trial in range(12):
        n = 3
        A = rng.normal(0, 0.05, (n, n))
        S = A @ A.T / n + 0.001 * np.eye(n)
        p = Problem(names=["a", "b", "c"], price=rng.uniform(80, 150, n), shares=rng.integers(0, 4, n), cash=600.0,
                    mu=rng.normal(0.01, 0.02, n), cov=S, gamma=3.0, fee=lambda s, v: 3.0, spread=np.full(n, 0.002),
                    opening_floor=100.0, order_floor=100.0)
        d = solve(p)
        q = np.array([d["targets"].get(k, {"shares": 0})["shares"] for k in p.names])
        h = evaluate(p, q)
        o = exhaustive(p, [int(p.nav // px) + 1 for px in p.price])  # covers every affordable book
        assert h["feasible"]
        gaps.append(o["objective"] - h["objective"])
    gaps = np.array(gaps)
    print(f"  heuristic vs exhaustive oracle: max gap {gaps.max():.2e} (fraction of NAV), {np.sum(gaps > 1e-12)}/12 > 0")
    assert gaps.min() >= -1e-12 and gaps.max() < 1e-3


def test_execution_revalidation():
    p = prob([0.01, 0.01], 0.0025 * np.eye(2), cash=1000.0, price=np.full(2, 10.0), fee=FLAT3_BUY, opening_floor=500.0)
    orders = [{"ticker": "A00", "side": "BUY", "shares": 99}, {"ticker": "A01", "side": "BUY", "shares": 50}]
    out = revalidate_at_execution(p, orders, {"A00": 10.5})
    a0 = next(o for o in out if o["ticker"] == "A00")
    assert a0["status"] == "resized" and a0["shares"] * 10.5 + 3 <= 1000      # gap up -> resized down
    assert next(o for o in out if o["ticker"] == "A01")["status"] == "rejected"  # no open, no fill


if __name__ == "__main__":
    started = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"all optimizer checks passed in {time.time() - started:.1f}s")
