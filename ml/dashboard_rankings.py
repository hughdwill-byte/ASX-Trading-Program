"""Dashboard rankings and explanations (display diagnostics only). Pure functions, no market data, no I/O.

Two views per research scenario (capital + core policy + broker + engine + dates + tax basis):

* Past profitability: every compatible, funded, cost-inclusive account (incl. VAS buy-and-hold) ranked by net
  AUD profit over the published common interval. Descriptive, not a forecast.
* Evidence for future use: qualification group first (from the engine's own assessment, never from this score),
  then the declared heuristic "Evidence score /100" (score v1 below). Not a chance of profit, not a forecast.

Nothing here changes trading rules, validation thresholds, the frozen registry, tiers or orders.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional, Sequence

DASHBOARD_SCHEMA = "dashboard_v1"
SCORE_VERSION = "evidence_score_v1"
COMPONENTS = ("R", "D", "B", "T", "C")
WEIGHTS = {"default": (0.30, 0.20, 0.20, 0.15, 0.15),
           "risk_emphasis": (0.20, 0.30, 0.20, 0.15, 0.15),
           "return_emphasis": (0.40, 0.15, 0.20, 0.10, 0.15)}
COVERAGE_RULES = {"W_min_months": 36, "L_min_months": 12, "min_common_years": 3, "neighbours_required": 3}
ANCHORS = {"g": "clip(50 + 500*x): 50 = matches VAS, +/-10 points clips at 100/0",
           "C": "clip(100*(1 - drag/0.03)): 100 at 0% yearly cost drag, 0 at >=3%",
           "B": "100*(0.5*DSR + 0.5*registered neighbour pass fraction)",
           "T": "100*(complete common calendar years beating VAS / complete common calendar years)",
           "window_mix": "0.6*W + 0.4*L"}
NEAR_TIE = 0.5
GROUPS = (("passed", "Passed current research checks"), ("paper", "Paper tracking"),
          ("exploratory", "Exploratory / not qualified"))
COHORT_KEYS = ("engine_id", "calendar_id", "capital", "core_policy", "broker", "cost_basis", "contributions",
               "price_basis", "start", "end")

SIGNAL_TEXT = {
    "RAMOM": "Favours shares with strong gains over roughly the past 12 months excluding the latest month, "
             "relative to how volatile they were.",
    "LOWVOL": "Favours shares whose prices moved least over the past year. Calmer history does not guarantee "
              "smaller future losses.",
    "COMBO": "An equal blend of the standardised momentum (RAMOM) and low-volatility scores. Equal score weights, "
             "not necessarily equal portfolio weights.",
    "RESMOM": "Momentum after removing each share's estimated market-linked movement, scaled by the remaining "
              "volatility. Not a market-neutral portfolio.",
    "TFM_RAW": "Ranks shares by the TimesFM model's forecast price gain relative to its forecast range "
               "(horizon {h} sessions).",
    "TFM_REL": "Ranks shares by the TimesFM model's forecast gain in log price relative to the benchmark, "
               "divided by its forecast range (horizon {h} sessions).",
}


def clip(x: float) -> float:
    return min(100.0, max(0.0, x))


def g(x: float) -> float:
    return clip(50.0 + 500.0 * x)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---- method registry -----------------------------------------------------------------------

def describe_method(method_id: str, policy: Optional[dict]) -> dict:
    """What it does + structured chips, built from the structured policy (never invented for unknown ids)."""
    p = policy or {}
    kind = p.get("kind")
    if method_id == "BENCH_TR":
        return {"family": "VAS", "text": "Buys the VAS ETF once and holds it (dividends reinvested), after the same "
                                         "broker costs. This is the comparison benchmark and the program's fallback.",
                "chips": ["benchmark", "buy and hold"]}
    if method_id == "WF_SELECT":
        return {"family": "WF_SELECT", "text": "One continuous account that switches each 1 July to the method that "
                                               "looked best on data available at that date (costs of switching included).",
                "chips": ["selection procedure", "annual switch"]}
    if method_id == "EW_UNIV":
        return {"family": "reference", "text": "Equal-weight average of all screened shares, before any costs. A gross "
                                               "reference, not an investable account.", "chips": ["reference", "gross"]}
    if method_id == "AORD_PRICE":
        return {"family": "reference", "text": "The All Ordinaries price index without dividends. A price-only "
                                               "reference, not the benchmark.", "chips": ["reference", "price only"]}
    if kind == "opt_policy":
        cap = p.get("holding_cap")
        chips = ["research only", "heuristic search", "no preset holding count" if cap is None else f"at most {cap} holdings",
                 {"covariance_cost_aware": "risk/cost-aware weights", "equal_weight": "equal weights",
                  "inverse_vol": "inverse-volatility weights"}.get(p.get("sizing"), str(p.get("sizing")))]
        return {"family": "OptPolicy", "text": "The model chooses which shares to hold and how much of each, weighing "
                                               "calibrated forecasts, how shares move together, trading costs and limits. "
                                               "Research only; not guaranteed optimal or most profitable.", "chips": chips}
    sig = p.get("signal")
    if kind != "variant" or sig not in SIGNAL_TEXT:
        return {"family": "unknown", "text": None, "chips": [], "unavailable": "no registered description"}
    chips = [f"{p['n']} holdings (fixed count)", "equal-dollar entries",
             "trend filter (checked monthly)" if p.get("regime") else "no trend filter",
             "quarterly reviews, monthly risk exits" if p.get("rebalance") == "Q" else "monthly reviews",
             "no hold buffer" if p.get("buffer_mult") == 1 else f"keeps holdings until rank > {p.get('buffer_mult')}x"]
    if p.get("tax_aware"):
        chips.append("tax-aware sale timing (not a full tax model)")
    if p.get("si_screen"):
        chips.append("short-interest screen")
    return {"family": sig, "text": SIGNAL_TEXT[sig].format(h=p.get("horizon", 21)), "chips": chips}


# ---- shared helpers ------------------------------------------------------------------------

def compound(returns: Sequence[float]) -> float:
    out = 1.0
    for r in returns:
        out *= 1.0 + r
    return out


def cagr_from(total: float, months: int) -> Optional[float]:
    if months <= 0 or total <= 0:
        return None
    return total ** (12.0 / months) - 1.0


def competition_ranks(keys: List[tuple]) -> List[int]:
    """keys are pre-sorted sort keys (value part first); equal value parts share the rank: 1,1,3."""
    ranks, prev = [], None
    for i, k in enumerate(keys):
        ranks.append(ranks[-1] if prev is not None and k[0] == prev else i + 1)
        prev = k[0]
    return ranks


def cohort_mismatch(cohort: dict, reference: dict) -> List[str]:
    return [k for k in COHORT_KEYS if cohort.get(k) != reference.get(k)]


def _series_values(m: dict, dates: List[str], key: str = "monthly"):
    s = m.get(key)
    if not s:
        return None, "series missing"
    vals = [s.get(d) for d in dates]
    if any(v is None for v in vals):
        missing = sum(v is None for v in vals)
        return None, f"{missing} of {len(dates)} months missing in the common interval"
    if not all(_finite(v) for v in vals):
        return None, "non-finite return in the common interval"
    if any(v <= -1 for v in vals):
        return None, "invalid NAV (a month lost 100% or more)"
    return vals, None


# ---- past profitability ----------------------------------------------------------------------

def historical_ranking(methods: List[dict], bench: dict, basis: str = "pre_tax") -> dict:
    """Rank funded, compatible accounts by net AUD profit over the bench's common interval.

    basis: "pre_tax" (default: before tax, after trading costs) or "after_tax_estimate" (uses each method's own
    computed `monthly_after_tax`; a missing series is unranked, never pre-tax substituted)."""
    key = "monthly" if basis == "pre_tax" else "monthly_after_tax"
    dates = list(bench["dates"])
    capital = bench["cohort"]["capital"]
    out = {"basis": basis, "common_interval": {"start": dates[0] if dates else None, "end": dates[-1] if dates else None,
                                               "months": len(dates)},
           "initial_nav": capital, "rows": [], "not_ranked": [], "short_coverage": [], "incompatible": []}
    if bench.get("status", "ok") != "ok":
        out["state"] = "unavailable"
        out["reason"] = f"VAS benchmark account not funded in this scenario: {bench.get('reason')}"
        return out
    bench_vals, why = _series_values(bench, dates, key)
    if bench_vals is None:
        out["state"] = "unavailable"
        out["reason"] = f"VAS benchmark series unavailable for this basis: {why}"
        return out
    bench_total = compound(bench_vals)
    bench_cagr = cagr_from(bench_total, len(dates))
    rows = []
    for m in methods + [bench]:
        mid = m["method_id"]
        base = {"method_id": mid, "label": m.get("label"), "kind": m.get("kind")}
        bad = cohort_mismatch(m.get("cohort", {}), bench["cohort"])
        if m.get("kind") == "reference":
            continue
        if m.get("status") != "ok":
            out["not_ranked"].append({**base, "status": m.get("status"), "reason": m.get("reason")})
            continue
        if [k for k in bad if k not in ("start", "end")]:
            out["incompatible"].append({**base, "status": "incompatible", "reason": "differs in " + ", ".join(bad)})
            continue
        if bad:
            out["short_coverage"].append({**base, "status": "short_coverage",
                                          "reason": "covers only " + f"{m['cohort'].get('start')} to {m['cohort'].get('end')}",
                                          **_short_summary(m, bench, key)})
            continue
        vals, why = _series_values(m, dates, key)
        if vals is None:
            out["not_ranked"].append({**base, "status": "unavailable", "reason": why})
            continue
        total = compound(vals)
        final = capital * total
        c = cagr_from(total, len(dates))
        rows.append({**base, "status": "ranked", "final_value": final, "net_profit": final - capital,
                     "net_profit_cents": int(round((final - capital) * 100)), "total_return": total - 1,
                     "cagr": c, "delta_cagr_vs_vas": (c - bench_cagr) if c is not None and bench_cagr is not None else None,
                     "mdd_daily": m.get("full", {}).get("mdd_daily"), "mdd_month_end": m.get("full", {}).get("mdd"),
                     "costs_aud": m.get("costs_aud"), "excess_vs_bench_arith": m.get("full", {}).get("excess_vs_bench")})
    rows.sort(key=lambda r: (-r["net_profit_cents"], r["method_id"]))
    for r, k in zip(rows, competition_ranks([(-r["net_profit_cents"], r["method_id"]) for r in rows])):
        r["rank"] = k
    out["rows"] = rows
    out["all_lost"] = bool(rows) and all(r["net_profit_cents"] < 0 for r in rows)
    out["state"] = "ranked" if rows else "empty"
    out["units"] = {"net_profit": "AUD", "final_value": "AUD", "total_return": "fraction", "cagr": "fraction/yr",
                    "delta_cagr_vs_vas": "fraction/yr (method CAGR - VAS CAGR, same interval)",
                    "mdd_daily": "fraction (daily marked-to-market)", "mdd_month_end": "fraction (month-end, gate e)",
                    "costs_aud": "AUD recorded brokerage + slippage (already inside the net returns)",
                    "excess_vs_bench_arith": "fraction/yr (annualised arithmetic mean monthly excess; NOT delta_cagr)"}
    return out


def _short_summary(m: dict, bench: dict, key: str) -> dict:
    """Own-interval numbers next to VAS over the SAME interval (never ranked against full-interval methods)."""
    s = m.get(key) or {}
    dates = sorted(s)
    b = bench.get(key) or {}
    if not dates or any(not _finite(s[d]) for d in dates) or any(d not in b for d in dates):
        return {"own_interval": None}
    t, bt = compound([s[d] for d in dates]), compound([b[d] for d in dates])
    return {"own_interval": {"start": dates[0], "end": dates[-1], "months": len(dates), "total_return": t - 1,
                             "vas_total_return": bt - 1}}


# ---- evidence score -------------------------------------------------------------------------

def components(e: dict) -> dict:
    """Score v1 components from per-method evidence inputs. Missing inputs -> component None + reason."""
    out, missing = {}, []
    W, L = e.get("W") or {}, e.get("L") or {}
    vW, vL = e.get("VAS_W") or {}, e.get("VAS_L") or {}
    need = lambda d, k: _finite(d.get(k))
    if all(need(d, "cagr") for d in (W, L, vW, vL)):
        out["R"] = 0.6 * g(W["cagr"] - vW["cagr"]) + 0.4 * g(L["cagr"] - vL["cagr"])
    else:
        missing.append("R: CAGR for W/L (method or VAS) unavailable")
    if all(need(d, "mdd_daily") for d in (W, L, vW, vL)):
        gap = lambda m, v: abs(v["mdd_daily"]) - abs(m["mdd_daily"])
        out["D"] = 0.6 * g(gap(W, vW)) + 0.4 * g(gap(L, vL))
    else:
        missing.append("D: daily drawdown for W/L unavailable (month-end drawdown is not substituted)")
    nb = e.get("neighbours") or {}
    dsr = e.get("dsr")
    if _finite(dsr) and nb.get("evaluated") == COVERAGE_RULES["neighbours_required"]:
        out["B"] = 100.0 * (0.5 * dsr + 0.5 * nb["passed"] / nb["evaluated"])
    else:
        missing.append("B: DSR undefined or registered neighbours incomplete")
    years = e.get("years") or []
    if len(years) >= COVERAGE_RULES["min_common_years"]:
        out["T"] = 100.0 * sum(1 for y in years if y["method"] > y["vas"]) / len(years)
    else:
        missing.append(f"T: fewer than {COVERAGE_RULES['min_common_years']} complete common calendar years")
    if need(W, "cost_drag") and need(L, "cost_drag"):
        out["C"] = clip(100.0 * (1 - (0.6 * W["cost_drag"] + 0.4 * L["cost_drag"]) / 0.03))
    else:
        missing.append("C: cost drag for W/L unavailable")
    if (W.get("months") or 0) < COVERAGE_RULES["W_min_months"]:
        missing.append(f"coverage: W has {W.get('months') or 0} months (needs {COVERAGE_RULES['W_min_months']})")
    if (L.get("months") or 0) < COVERAGE_RULES["L_min_months"]:
        missing.append(f"coverage: L has {L.get('months') or 0} months (needs {COVERAGE_RULES['L_min_months']})")
    return {"components": out, "missing": missing}


def score(comp: dict, weights: Sequence[float] = WEIGHTS["default"]) -> Optional[dict]:
    """Weighted score, or None when any component is missing (no zero-fill, no renormalisation)."""
    if any(k not in comp for k in COMPONENTS):
        return None
    contrib = {k: w * comp[k] for k, w in zip(COMPONENTS, weights)}
    return {"score": sum(contrib.values()), "contributions": contrib}


def evidence_ranking(methods: List[dict], pbo: Optional[dict] = None) -> dict:
    """Qualification first (group from the engine's assessment), evidence score second.

    Each method: method_id, status, qualification {"group": passed|paper|exploratory|None, "assessment":
    "failed"|"not_assessed"|"passed"|"paper"}, evidence {W, L, VAS_W, VAS_L, dsr, neighbours, years}."""
    scored, not_scored = [], []
    for m in methods:
        base = {"method_id": m["method_id"], "label": m.get("label")}
        q = m.get("qualification") or {}
        if m.get("status") != "ok" or q.get("group") is None:
            not_scored.append({**base, "reason": m.get("reason") or "not compatible with this scenario", "missing": []})
            continue
        c = components(m.get("evidence") or {})
        s = score(c["components"])
        coverage_fail = [x for x in c["missing"] if x.startswith("coverage")]
        if s is None or coverage_fail:
            not_scored.append({**base, "reason": "insufficient evidence for a score", "missing": c["missing"],
                               "components": c["components"]})
            continue
        alt = {k: score(c["components"], w)["score"] for k, w in WEIGHTS.items()}
        scored.append({**base, "group": q["group"], "assessment": q.get("assessment"), "components": c["components"],
                       "contributions": s["contributions"], "score": s["score"], "alt_scores": alt,
                       "dsr": (m.get("evidence") or {}).get("dsr_detail"),
                       "neighbours": (m.get("evidence") or {}).get("neighbours")})
    groups = []
    for gid, label in GROUPS:
        rows = [r for r in scored if r["group"] == gid]
        rows.sort(key=lambda r: (-r["score"], r["method_id"]))
        for i, r in enumerate(rows):
            r["group_rank"] = i + 1
            r["near_tie"] = any(abs(r["score"] - o["score"]) < NEAR_TIE for o in rows if o is not r)
        leaders = {}
        for wk in WEIGHTS:
            order = sorted(rows, key=lambda r: (-r["alt_scores"][wk], r["method_id"]))
            for i, r in enumerate(order):
                r.setdefault("sensitivity_ranks", {})[wk] = i + 1
            leaders[wk] = order[0]["method_id"] if order else None
        for r in rows:
            v = r["sensitivity_ranks"].values()
            r["sensitivity_rank_range"] = [min(v), max(v)]
        groups.append({"id": gid, "label": label, "rows": rows,
                       "leader_changes": len(set(x for x in leaders.values() if x)) > 1, "leaders": leaders})
    qualified = any(r["group"] in ("passed", "paper") for r in scored)
    return {"score_manifest": score_manifest(), "groups": groups, "not_scored": not_scored,
            "no_stock_method_qualifies": not any(r["group"] == "passed" for r in scored),
            "any_paper_or_passed": qualified,
            "shared_pbo": {**(pbo or {}), "scope": "selection procedure over the registered candidate cohort "
                                                   "(not a per-method probability of loss)"}}


def score_manifest() -> dict:
    return {"version": SCORE_VERSION, "components": list(COMPONENTS), "weights": {k: list(v) for k, v in WEIGHTS.items()},
            "anchors": ANCHORS, "coverage_rules": COVERAGE_RULES, "near_tie_points": NEAR_TIE,
            "basis": "before tax, after modelled trading costs (fixed; the tax overlay never changes this score)",
            "label": "Evidence score /100: a declared heuristic display score, not a chance of profit, a predicted "
                     "return or a validated forecasting model. Weights are interpretability defaults, not fitted."}


# ---- explanations ----------------------------------------------------------------------------

def _fact(text: str, metric: str, value, window: str, source: str) -> dict:
    return {"text": text, "metric": metric, "value": value, "window": window, "source": source}


def _aud(x: float) -> str:
    return f"{'-' if x < 0 else ''}A${abs(x):,.0f}"


def _pp(x: float) -> str:
    return f"{x * 100:+.1f} percentage points"


def historical_explanation(row: Optional[dict], vas_row: Optional[dict], years: List[dict], cohort_id: str,
                           interval: dict) -> dict:
    """Deterministic facts from the ranked row; every number is a cited metric. No causal or macro claims."""
    if row is None or vas_row is None:
        return {"available": False, "reason": "no ranked result for this method in this scenario", "facts": []}
    win = f"{interval['start']}..{interval['end']}"
    src = f"historical_ranking:{cohort_id}"
    start = row["final_value"] - row["net_profit"]
    f = [_fact(f"Net profit {_aud(row['net_profit'])} on {_aud(start)} invested ({interval['start']} to {interval['end']}), "
               f"versus {_aud(vas_row['net_profit'])} for VAS over the same dates.", "net_profit", row["net_profit"], win, src)]
    if _finite(row.get("delta_cagr_vs_vas")):
        f.append(_fact(f"Annualised return {_pp(row['delta_cagr_vs_vas'])} versus VAS.", "delta_cagr_vs_vas",
                       row["delta_cagr_vs_vas"], win, src))
    complete = [y for y in years if y.get("complete")]
    if complete:
        best = max(complete, key=lambda y: y["method"] - y["vas"])
        worst = min(complete, key=lambda y: y["method"] - y["vas"])
        f.append(_fact(f"Strongest complete year against VAS: {best['year']} ({best['method'] * 100:+.1f}% vs "
                       f"{best['vas'] * 100:+.1f}%).", "calendar_year_return", best["method"], best["year"], src))
        f.append(_fact(f"Weakest complete year against VAS: {worst['year']} ({worst['method'] * 100:+.1f}% vs "
                       f"{worst['vas'] * 100:+.1f}%).", "calendar_year_return", worst["method"], worst["year"], src))
    if _finite(row.get("mdd_daily")) and _finite(vas_row.get("mdd_daily")):
        f.append(_fact(f"Largest fall from a peak (daily): {row['mdd_daily'] * 100:.1f}% versus {vas_row['mdd_daily'] * 100:.1f}% "
                       "for VAS.", "mdd_daily", row["mdd_daily"], win, src))
    if _finite(row.get("costs_aud")):
        f.append(_fact(f"Recorded trading costs (brokerage + modelled slippage): {_aud(row['costs_aud'])}. These are already "
                       "inside the net result and are not deducted again.", "costs_aud", row["costs_aud"], win, src))
    f.append(_fact("Which holdings contributed most is not available: the history records holding names, not a "
                   "reconciled contribution ledger.", "attribution", None, win, "not_available"))
    return {"available": True, "facts": f}


def rank_explanation(row: Optional[dict], qualification_reasons: Sequence[str], missing: Sequence[str],
                     cohort_id: str) -> dict:
    if row is None:
        return {"available": False, "reason": "not scored", "missing": list(missing), "facts": []}
    src = f"evidence_ranking:{cohort_id}"
    rel = {k: row["contributions"][k] - w * 50.0 for k, w in zip(COMPONENTS, WEIGHTS["default"])}
    names = {"R": "net return vs VAS", "D": "downside vs VAS", "B": "robustness (DSR + neighbours)",
             "T": "year-by-year consistency", "C": "trading-cost resilience"}
    best, worst = max(rel, key=rel.get), min(rel, key=rel.get)
    f = [_fact(f"Evidence score {row['score']:.1f}/100 (group rank {row['group_rank']}).", "score", row["score"], "W+L", src),
         _fact(f"Largest positive contribution: {names[best]} ({rel[best]:+.1f} points vs a neutral 50 component).",
               f"component_{best}", row["components"][best], "W+L", src),
         _fact((f"Largest negative contribution: {names[worst]} ({rel[worst]:+.1f} points vs a neutral 50 component)."
                if rel[worst] < 0 else f"No component is below neutral; the weakest is {names[worst]} "
                                       f"({rel[worst]:+.1f} points)."), f"component_{worst}", row["components"][worst], "W+L", src)]
    lo, hi = row["sensitivity_rank_range"]
    f.append(_fact(f"Rank under the two alternative weightings: {lo} to {hi}.", "sensitivity_rank_range", [lo, hi],
                   "W+L", src))
    for reason in qualification_reasons:
        f.append(_fact(f"Research check not passed: {reason}", "qualification", None, "engine assessment", "validation"))
    return {"available": True, "facts": f}


def calendar_years(method: Dict[str, float], vas: Dict[str, float], dates: Sequence[str]) -> List[dict]:
    """Same-year compounded net returns; complete = all 12 months present for both."""
    by: Dict[str, dict] = {}
    for d in dates:
        y = d[:4]
        b = by.setdefault(y, {"year": y, "m": [], "v": []})
        if d in method and d in vas and _finite(method[d]) and _finite(vas[d]):
            b["m"].append(method[d])
            b["v"].append(vas[d])
    return [{"year": y, "method": compound(b["m"]) - 1, "vas": compound(b["v"]) - 1, "months": len(b["m"]),
             "complete": len(b["m"]) == 12} for y, b in sorted(by.items()) if b["m"]]


def identity(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]
