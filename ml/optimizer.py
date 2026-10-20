"""Joint stock selection and variable allocation with no preset holding count (F14/F15/F19). RESEARCH ONLY.

Single-period problem in pre-trade-NAV units (N = cash + sum shares * raw price; w = current risky weights;
x = target weights; d = x - w; c = post-cost cash weight; C(d) = estimated trade costs / N):

    max  mu'x + r_cash*c - gamma*x'Sigma*x - lam*C(d) - sum_i rho_i*|x_i - b_i|
    s.t. x >= 0;  c = 1 - sum(x) - C(d) >= cash_reserve
         x_i <= name_limit_i;  sqrt(x'Sigma*x) <= risk_budget;  0.5*sum|d| <= turnover_limit
         N*|d_i| <= participation_i;  d_i <= 0 if not purchase-eligible;  d_i = 0 if frozen

Stage 1 solves the convex continuous relaxation with SciPy SLSQP (d = u - v, u, v >= 0; |x - b| through
auxiliary variables; convex cost C = spread*|d| + impact*|d|^1.5 enters through an inequality, never an
equality). Stage 2 translates targets to integer raw-price shares and runs a deterministic add/drop/resize/
transfer search on the exact discrete objective (per-order fees with their discontinuities, opening parcel
vs economic order floors, full closures, cash/reserve/settlement, limits re-checked after rounding). The
discrete stage is a HEURISTIC: it is compared with no-trade and (in tests) with exhaustive enumeration on
tiny universes, but it is not certified globally optimal. Nothing here places orders.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

OPTIMIZER_ID = "slsqp_relax_plus_discrete_local_search_v1"
TOL = 1e-9
CASH_VIOLATIONS = {"cash reserve", "negative cash", "buys exceed settled cash"}


@dataclasses.dataclass
class Problem:
    names: Sequence[str]
    price: np.ndarray                     # raw tradable price estimate at the decision close (AUD/share)
    shares: np.ndarray                    # current physical shares (int)
    cash: float                           # available cash (AUD)
    mu: np.ndarray                        # expected arithmetic return over the horizon
    cov: np.ndarray                       # matching horizon covariance
    gamma: float
    fee: Callable[[str, float], float]    # exact brokerage for one order (side, value)
    lam: float = 1.0
    r_cash: float = 0.0                   # cash return over the horizon
    cash_reserve: float = 0.0             # fraction of pre-trade NAV
    name_limit: Optional[np.ndarray] = None
    risk_budget: Optional[float] = None   # horizon st.dev of the risky book, fraction of N
    turnover_limit: Optional[float] = None
    participation: Optional[np.ndarray] = None  # max |trade| per name, AUD (e.g. share of lagged turnover)
    can_buy: Optional[np.ndarray] = None
    tradable: Optional[np.ndarray] = None       # False = frozen (suspended/stale): no trade at all
    spread: Optional[np.ndarray] = None         # adverse execution, fraction of traded value
    impact: Optional[np.ndarray] = None         # C_i = impact_i * |d_i|^1.5 (NAV units); stress assumption
    bench: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None
    opening_floor: float = 500.0          # new position minimum value (ASX parcel / policy)
    order_floor: float = 0.0              # top-up / partial-sell minimum value; full closures always allowed
    sale_proceeds_fund_buys: bool = True  # settlement rule: same-day sale proceeds may fund buys
    max_evals: int = 200_000
    time_limit: float = 30.0

    def __post_init__(self) -> None:
        n = len(self.names)
        f = lambda a, d: np.full(n, d, dtype=float) if a is None else np.asarray(a, dtype=float)
        self.price = np.asarray(self.price, dtype=float)
        self.shares = np.asarray(self.shares, dtype=np.int64)
        self.mu = np.asarray(self.mu, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        self.name_limit = f(self.name_limit, 1.0)
        self.participation = f(self.participation, np.inf)
        self.can_buy = np.ones(n, bool) if self.can_buy is None else np.asarray(self.can_buy, bool)
        self.tradable = np.ones(n, bool) if self.tradable is None else np.asarray(self.tradable, bool)
        self.spread = f(self.spread, 0.0)
        self.impact = f(self.impact, 0.0)
        self.bench = f(self.bench, 0.0)
        self.rho = f(self.rho, 0.0)

    @property
    def nav(self) -> float:
        return float(self.cash + self.shares @ self.price)

    def validate(self) -> List[str]:
        bad = []
        n = len(self.names)
        for k, a in (("price", self.price), ("mu", self.mu)):
            if a.shape != (n,) or not np.all(np.isfinite(a)):
                bad.append(f"{k} missing or non-finite")
        if np.any(self.price <= 0):
            bad.append("non-positive price")
        if self.cov.shape != (n, n) or not np.all(np.isfinite(self.cov)):
            bad.append("covariance missing or non-finite")
        elif np.linalg.eigvalsh((self.cov + self.cov.T) / 2).min() < -1e-10:
            bad.append("covariance not PSD")
        if np.any(self.shares < 0) or not np.isfinite(self.cash) or self.cash < 0:
            bad.append("negative holdings or cash")
        if self.nav <= 0:
            bad.append("non-positive NAV")
        return bad


# ---- continuous relaxation ---------------------------------------------------------------------

def _cost(p: Problem, a: np.ndarray) -> float:
    return float(p.spread @ a + p.impact @ np.power(a * a + 1e-16, 0.75))


def _dcost(p: Problem, a: np.ndarray) -> np.ndarray:
    return p.spread + 1.5 * p.impact * np.power(a * a + 1e-16, 0.25)


def solve_continuous(p: Problem, can_buy: Optional[np.ndarray] = None) -> Dict[str, object]:
    from scipy.optimize import minimize

    n, N = len(p.names), p.nav
    w = p.shares * p.price / N
    can_buy = p.can_buy if can_buy is None else can_buy
    part = p.participation / N
    ub_u = np.where(p.tradable & can_buy, np.minimum(part, 1.0), 0.0)
    ub_v = np.where(p.tradable, np.minimum(w, part), 0.0)
    use_t = bool(np.any(p.rho > 0))
    nz = 3 * n if use_t else 2 * n
    S = (p.cov + p.cov.T) / 2

    def split(z):
        u, v = z[:n], z[n:2 * n]
        return u, v, w + u - v, u + v

    def obj(z):
        u, v, x, a = split(z)
        C = _cost(p, a)
        val = p.mu @ x + p.r_cash * (1 - x.sum() - C) - p.gamma * x @ S @ x - p.lam * C
        if use_t:
            val -= p.rho @ z[2 * n:]
        return -val

    def grad(z):
        u, v, x, a = split(z)
        gx = -(p.mu - p.r_cash - 2 * p.gamma * S @ x)
        ga = (p.r_cash + p.lam) * _dcost(p, a)
        g = np.concatenate([gx + ga, -gx + ga])
        return np.concatenate([g, p.rho]) if use_t else g

    cons = []

    def budget(z):
        _, _, x, a = split(z)
        return 1 - x.sum() - _cost(p, a) - p.cash_reserve

    def budget_j(z):
        _, _, _, a = split(z)
        dc = _dcost(p, a)
        j = np.concatenate([-1 - dc, 1 - dc])
        return np.concatenate([j, np.zeros(n)]) if use_t else j

    cons.append({"type": "ineq", "fun": budget, "jac": budget_j})
    lim = np.flatnonzero(p.name_limit < 1.0)
    if len(lim):
        E = np.zeros((len(lim), nz))
        E[np.arange(len(lim)), lim] = -1.0
        E[np.arange(len(lim)), n + lim] = 1.0
        cons.append({"type": "ineq", "fun": lambda z: p.name_limit[lim] - split(z)[2][lim], "jac": lambda z: E})
    if p.risk_budget is not None:
        def risk(z):
            x = split(z)[2]
            return p.risk_budget ** 2 - x @ S @ x

        def risk_j(z):
            g = 2 * S @ split(z)[2]
            j = np.concatenate([-g, g])
            return np.concatenate([j, np.zeros(n)]) if use_t else j
        cons.append({"type": "ineq", "fun": risk, "jac": risk_j})
    if p.turnover_limit is not None:
        J = np.concatenate([-np.ones(2 * n), np.zeros(n)]) if use_t else -np.ones(2 * n)
        cons.append({"type": "ineq", "fun": lambda z: 2 * p.turnover_limit - split(z)[3].sum(), "jac": lambda z: J})
    if use_t:
        I = np.eye(n)
        J1 = np.hstack([-I, I, I])
        J2 = np.hstack([I, -I, I])
        cons.append({"type": "ineq", "fun": lambda z: z[2 * n:] - (split(z)[2] - p.bench), "jac": lambda z: J1})
        cons.append({"type": "ineq", "fun": lambda z: z[2 * n:] + (split(z)[2] - p.bench), "jac": lambda z: J2})
    bounds = list(zip(np.zeros(n), ub_u)) + list(zip(np.zeros(n), ub_v)) + ([(0, None)] * n if use_t else [])
    z0 = np.zeros(nz)
    if use_t:
        z0[2 * n:] = np.abs(w - p.bench)
    started = time.monotonic()
    res = minimize(obj, z0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": 1000, "ftol": 1e-12})
    u, v, x, a = split(res.x)
    x = np.maximum(x, 0.0)
    resid = {"budget": float(budget(res.x)), "x_nonneg": float(x.min()) if n else 0.0}
    if len(lim):
        resid["name_limit"] = float((p.name_limit[lim] - x[lim]).min())
    if p.risk_budget is not None:
        resid["risk"] = float(p.risk_budget - np.sqrt(max(x @ S @ x, 0.0)))
    if p.turnover_limit is not None:
        resid["turnover"] = float(p.turnover_limit - a.sum() / 2)
    ok = res.success and all(r >= -1e-6 for r in resid.values())
    return {"x": x, "d": x - w, "objective": -float(res.fun), "status": "optimal" if ok else "failed",
            "message": str(res.message), "iterations": int(res.get("nit", 0)), "residuals": resid,
            "seconds": time.monotonic() - started, "solver": "scipy.optimize.minimize(SLSQP)", "ftol": 1e-12}


# ---- exact discrete evaluation ---------------------------------------------------------------

def evaluate(p: Problem, q: np.ndarray) -> Dict[str, object]:
    """Exact objective and feasibility of target shares q (one net order per name)."""
    N = p.nav
    dq = q - p.shares
    value = np.abs(dq) * p.price
    traded = dq != 0
    adverse = p.spread * value + p.impact * N * np.power(value / N, 1.5)
    fees = np.zeros(len(q))
    for k in np.flatnonzero(traded):
        fees[k] = p.fee("BUY" if dq[k] > 0 else "SELL", value[k])
    buys = dq > 0
    buy_debit = float((value + adverse + fees)[buys].sum())
    sell_credit = float((value - adverse - fees)[dq < 0].sum())
    cash_post = p.cash - buy_debit + sell_credit
    x = q * p.price / N
    c = cash_post / N
    C = float((fees + adverse).sum()) / N
    S = p.cov
    risk2 = float(x @ S @ x)
    obj = float(p.mu @ x + p.r_cash * c - p.gamma * risk2 - p.lam * C - p.rho @ np.abs(x - p.bench))
    why = []
    if np.any(q < 0):
        why.append("negative shares")
    if np.any(traded & ~p.tradable):
        why.append("trade in a frozen name")
    if np.any(buys & ~p.can_buy):
        why.append("buy of a purchase-ineligible name")
    if np.any(value > p.participation + 1e-6):
        why.append("participation")
    opening = buys & (p.shares == 0)
    if np.any(opening & (value < p.opening_floor - 1e-9)):
        why.append("new position below opening floor")
    small = traded & ~opening & ~((dq < 0) & (q == 0)) & (value < p.order_floor - 1e-9)
    if np.any(small):
        why.append("order below economic floor")
    if buys.any():
        if c < p.cash_reserve - 1e-12:
            why.append("cash reserve")
        if not p.sale_proceeds_fund_buys and buy_debit > p.cash - p.cash_reserve * N + 1e-6:
            why.append("buys exceed settled cash")
        if np.any(buys & (x > p.name_limit + 1e-12)):
            why.append("name limit")
    if cash_post < -1e-6:
        why.append("negative cash")
    w = p.shares * p.price / N
    risk0 = float(w @ S @ w)
    if p.risk_budget is not None and risk2 > p.risk_budget ** 2 + 1e-12 and (buys.any() or risk2 > risk0 + 1e-15):
        why.append("risk budget")
    if p.turnover_limit is not None and value.sum() / N / 2 > p.turnover_limit + 1e-12:
        why.append("turnover")
    return {"objective": obj, "feasible": not why, "violations": why, "cash_post": cash_post, "x": x, "c": c,
            "costs": {"fees": float(fees.sum()), "adverse": float(adverse.sum())}, "fees": fees,
            "adverse": adverse, "risk": float(np.sqrt(max(risk2, 0.0)))}


def _round_target(p: Problem, x: np.ndarray) -> np.ndarray:
    q = np.where(p.tradable, np.floor(x * p.nav / p.price + 1e-9), p.shares).astype(np.int64)
    q = np.where(p.can_buy | (q <= p.shares), q, p.shares)
    return np.maximum(q, 0)


def _one_dim_candidates(p: Problem, q: np.ndarray, k: int, x_cont: Optional[np.ndarray]) -> List[int]:
    """Quantities worth trying for name k with the rest fixed: hold, close, continuous target, the 1-D
    quadratic optimum on each side (fee-free), the cash-limited maximum and +-1 share."""
    if not p.tradable[k]:
        return []
    N, S = p.nav, p.cov
    x = q * p.price / N
    rest = S[k] @ x - S[k, k] * x[k]
    out = {int(p.shares[k]), 0, int(q[k]) + 1, int(q[k]) - 1}
    if x_cont is not None:
        out.add(int(np.floor(x_cont[k] * N / p.price[k] + 1e-9)))
    # cash-limited: everything the current post-trade cash (above the reserve) can buy
    ev = evaluate(p, q)
    spare = ev["cash_post"] - p.cash_reserve * N
    fill = p.price[k] * (1 + p.spread[k])
    cap = int(np.floor(p.name_limit[k] * N / p.price[k] + 1e-9))
    extra = 0
    if spare > 0:
        extra = int(spare // fill)
        while extra > 0 and extra * fill + p.fee("BUY", extra * p.price[k]) > spare + 1e-9:
            extra -= 1
        out.add(int(q[k]) + extra)
    if p.gamma > 0 and S[k, k] > 0:
        for sgn in (1, -1):
            xk = (p.mu[k] - p.r_cash - sgn * (p.lam + p.r_cash) * p.spread[k] - 2 * p.gamma * rest) / (2 * p.gamma * S[k, k])
            s = xk * N / p.price[k]
            out.update({int(np.floor(s)), int(np.ceil(s)), min(int(np.floor(s)), cap, int(q[k]) + extra)})
    if not p.can_buy[k]:
        out = {v for v in out if v <= p.shares[k]}
    return sorted(v for v in out if v >= 0 and v != q[k])


def solve(p: Problem, swap_candidates: int = 25) -> Dict[str, object]:
    """Full proposal: continuous relaxation -> integer shares -> deterministic local search -> compare with
    no-trade. Returns a PolicyDecision-like dict. A failed solve returns the verified no-trade book, never an
    all-cash or sell-everything order."""
    started = time.monotonic()
    bad = p.validate()
    q0 = p.shares.copy()
    if bad:
        return _decision(p, q0, "invalid_input", {"input_errors": bad}, started)
    base = evaluate(p, q0)
    cont = solve_continuous(p)
    info = {"continuous": {k: cont[k] for k in ("status", "objective", "message", "iterations", "residuals",
                                                 "seconds", "solver", "ftol")},
            "no_trade": {"objective": base["objective"], "feasible": base["feasible"], "violations": base["violations"]}}
    if cont["status"] != "optimal" and not base["feasible"]:
        info["mandatory"] = "no-trade violates hard constraints and the relaxation failed"
    # search from the current book and from the rounded relaxation; keep the better
    starts = [q0] + ([_round_target(p, cont["x"])] if cont["status"] == "optimal" else [])
    evals = [0]

    def score(q):
        evals[0] += 1
        e = evaluate(p, q)
        return (e["objective"] if e["feasible"] else -np.inf), e

    def local_search(q, budget):
        best, _ = score(q)
        x_cont = cont["x"] if cont["status"] == "optimal" else None
        stop = evals[0] + budget
        while evals[0] < stop and time.monotonic() - started < p.time_limit:
            move = None
            for k in range(len(q)):                       # deterministic name order
                for v in _one_dim_candidates(p, q, k, x_cont):
                    trial = q.copy()
                    trial[k] = v
                    s, _ = score(trial)
                    if s > best + 1e-12:
                        best, move = s, trial
            if move is None:
                move = _transfer(q, best)
                if move is None:
                    return q, best
                best = score(move)[0]
            q = move
        info.setdefault("budget_exhausted", []).append(int(budget))
        return q, best

    def _transfer(q, best):
        """Move capital between names: cut i's target (or sell it) and spend the freed cash on j.
        Pairs are ordered by conditional marginal contribution given the current book."""
        N = p.nav
        x = q * p.price / N
        g = p.mu - p.r_cash - 2 * p.gamma * p.cov @ x - p.lam * p.spread   # marginal value of one more unit
        held = [k for k in np.argsort(g, kind="stable") if q[k] > 0 and p.tradable[k]][:swap_candidates]
        cand = [k for k in np.argsort(-g, kind="stable") if p.tradable[k] and p.can_buy[k]][:swap_candidates]
        for i in held:
            for step in sorted({1, max(1, int(q[i]) // 10), max(1, int(q[i]) // 2), int(q[i])}):
                cut = q.copy()
                cut[i] -= step
                spare = evaluate(p, cut)["cash_post"] - p.cash_reserve * N
                for j in cand:
                    if j == i or spare <= 0:
                        continue
                    add = int(spare // (p.price[j] * (1 + p.spread[j])))
                    while add > 0:
                        trial = cut.copy()
                        trial[j] += add
                        s, e = score(trial)
                        if not CASH_VIOLATIONS & set(e["violations"]):
                            break
                        add -= 1
                    if add > 0 and s > best + 1e-12:
                        return trial
        return None

    # the current-book start is a cheap safeguard (small universes); most of the budget goes to the relaxation
    share = [p.max_evals // 20, p.max_evals // 2] if len(starts) == 2 else [p.max_evals // 2]
    results = [local_search(q.copy(), b) for q, b in zip(starts, share)]
    # re-solve the relaxation on the chosen active set (plus held names) and polish once more
    q_best, s_best = max(results, key=lambda t: t[1])
    if cont["status"] == "optimal" and np.isfinite(s_best):
        active = (q_best > 0) | (p.shares > 0)
        again = solve_continuous(p, can_buy=p.can_buy & active)
        if again["status"] == "optimal":
            q2, s2 = local_search(_round_target(p, again["x"]), p.max_evals // 2)
            if s2 > s_best + 1e-12:
                q_best, s_best = q2, s2
    info["evaluations"] = evals[0]
    info["discrete_method"] = "heuristic (deterministic add/drop/resize/transfer local search; not certified)"
    if not np.isfinite(s_best) and base["feasible"]:
        return _decision(p, q0, "no_trade_optimal", info, started)
    if not np.isfinite(s_best):
        status = "no_feasible_proposal" if cont["status"] == "optimal" else "solver_failed"
        return _decision(p, q0, status, info, started)
    if base["feasible"] and s_best <= base["objective"] + 1e-12:
        return _decision(p, q0, "no_trade_optimal", info, started)
    return _decision(p, q_best, "proposal", info, started)


def _decision(p: Problem, q: np.ndarray, status: str, info: dict, started: float) -> Dict[str, object]:
    e = evaluate(p, q) if not info.get("input_errors") else None
    N = p.nav
    orders = []
    if e is not None:
        for k in np.flatnonzero(q != p.shares):
            dq = int(q[k] - p.shares[k])
            kind = ("open" if p.shares[k] == 0 else "close" if q[k] == 0 else "top_up" if dq > 0 else "partial_sell")
            orders.append({"ticker": p.names[k], "side": "BUY" if dq > 0 else "SELL", "shares": abs(dq), "kind": kind,
                           "est_price": float(p.price[k]), "est_value": float(abs(dq) * p.price[k]),
                           "est_fee": float(e["fees"][k]), "est_adverse": float(e["adverse"][k]), "status": "proposed"})
    post_nav = e["cash_post"] + float(q @ p.price) if e else N
    held = q > 0
    buy_total = sum(o["est_value"] for o in orders if o["side"] == "BUY")
    targets = {p.names[k]: {"shares": int(q[k]), "target_notional": float(q[k] * p.price[k]),
                            "pre_nav_weight": float(q[k] * p.price[k] / N),
                            "post_nav_weight": float(q[k] * p.price[k] / post_nav) if post_nav > 0 else None}
               for k in np.flatnonzero(held | (p.shares > 0))}
    for o in orders:
        if o["side"] == "BUY" and buy_total > 0:
            o["buy_allocation"] = o["est_value"] / buy_total
    return {"status": status, "optimizer_id": OPTIMIZER_ID, "orders": orders, "targets": targets,
            "count": int(held.sum()), "objective": e["objective"] if e else None,
            "feasible": e["feasible"] if e else False, "violations": e["violations"] if e else [],
            "costs": e["costs"] if e else None, "residual_cash": e["cash_post"] if e else p.cash,
            "risk": e["risk"] if e else None, "pre_trade_nav": N, "post_trade_nav": post_nav,
            "weight_basis": "pre_trade_total_nav", "solver": info, "seconds": time.monotonic() - started,
            "execution_state": "paper_proposal_not_an_order"}


def exhaustive(p: Problem, max_shares: Sequence[int]) -> Dict[str, object]:
    """Oracle for tiny universes: enumerate every integer book 0..max_shares[k] (frozen names fixed)."""
    ranges = [range(0, m + 1) if p.tradable[k] else [int(p.shares[k])] for k, m in enumerate(max_shares)]
    best, arg = -np.inf, None
    for q in itertools.product(*ranges):
        q = np.array(q, dtype=np.int64)
        e = evaluate(p, q)
        if e["feasible"] and e["objective"] > best + 1e-15:
            best, arg = e["objective"], q
    return {"objective": best, "shares": arg}


def revalidate_at_execution(p: Problem, orders: List[dict], open_price: Dict[str, float]) -> List[dict]:
    """Declared execution rule: sells first; each buy is resized DOWN (never up) to fit remaining cash above
    the reserve at the actual open; a missing open rejects the order; sub-floor results are rejected."""
    N = p.nav
    cash = p.cash
    out = []
    for o in sorted(orders, key=lambda o: (o["side"] != "SELL", o["ticker"])):
        k = list(p.names).index(o["ticker"])
        px = open_price.get(o["ticker"])
        r = dict(o)
        if px is None or not np.isfinite(px) or px <= 0:
            out.append({**r, "status": "rejected", "why": "no open price"})
            continue
        if o["side"] == "SELL":
            v = o["shares"] * px
            cash += v - p.spread[k] * v - p.fee("SELL", v)
            out.append({**r, "status": "fillable", "exec_price": px})
            continue
        spare = cash - p.cash_reserve * N
        q = o["shares"]
        fill = px * (1 + p.spread[k])
        while q > 0 and q * fill + p.fee("BUY", q * px) > spare + 1e-9:
            q -= 1
        floor = p.opening_floor if p.shares[k] == 0 else p.order_floor
        if q <= 0 or q * px < floor:
            out.append({**r, "status": "rejected", "why": "unaffordable or below floor at the open"})
            continue
        cash -= q * fill + p.fee("BUY", q * px)
        out.append({**r, "shares": q, "status": "fillable" if q == o["shares"] else "resized", "exec_price": px})
    return out
