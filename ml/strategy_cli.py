"""CLI glue for strategy v2: strat-backtest, strat-recommend, strat-publish, tfm-cache."""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import logging
import pathlib
import time
from typing import Dict, List, Optional

import pandas as pd

from ml import validation as val
from ml.portfolio import (DEFAULTS, ENGINE_ID, Variant, after_tax_monthly, build_rank_cache, ew_universe,
                          run_backtest_v2)
from ml.signals import TRADEABLE_SIGNALS, Context, tfm_cache_scores
from ml.timesfm_strategy import load_benchmark, load_panel
from stocktrade import project_path, sanitize_for_json

logger = logging.getLogger(__name__)

RESULTS = project_path("results")
SETTINGS = [(500.0, 0.0), (2000.0, 0.0), (5000.0, 0.0), (5000.0, 0.7), (10000.0, 0.0), (10000.0, 0.6),
            (25000.0, 0.0), (50000.0, 0.0), (50000.0, 0.6)]
BENCH_NOTE = "BENCH_TR = VAS.AX adjusted close (total return, net of fees); ^AORD is a price index and excludes dividends."
FALLBACK_TEXT = "Hold {core} buy-and-hold; add new money in single orders of at least A$2,000."


def add_cli_args(sub) -> None:
    bt = sub.add_parser("strat-backtest", help="Run the strategy v2 grid + validation, write results/.")
    bt.add_argument("--end", help="Last date to include (default: latest data).")
    rec = sub.add_parser("strat-recommend", help="Today's tier and next-open orders for the configured strategy.")
    rec.add_argument("--capital", type=float, help="Override capital.")
    rec.add_argument("--holdings", help="Local CSV/JSON of actual holdings and cash (see ml/accounting.import_holdings).")
    sub.add_parser("strat-publish", help="Merge strategy results into docs/data/latest.json.")
    tc = sub.add_parser("tfm-cache", help="Monthly TimesFM forecast cache for TFM_REL/TFM_RAW research signals.")
    tc.add_argument("--start", default="2016-01-01")
    tc.add_argument("--end")
    tc.add_argument("--transform", choices=["rel", "raw"], default="rel")
    tc.add_argument("--horizon", type=int, default=21)
    tc.add_argument("--checkpoint", default="google/timesfm-2.5-200m-pytorch")
    tc.add_argument("--batch-size", type=int, default=128)
    asic = sub.add_parser("asic-download", help="Download ASIC daily short-position reports (resumable).")
    asic.add_argument("--start", default="2015-06-01")
    sub.add_parser("strat-explain-config", help="Resolved strategy settings: values, origins, units, active/shadowed.")
    st = sub.add_parser("strat-opt-study", help="RESEARCH: matched study of the uncapped optimiser vs registered controls.")
    st.add_argument("--end", help="Last date to include (default: latest data).")
    st.add_argument("--capitals", default="10000,50000", help="Comma-separated satellite-only capitals (core 0).")


def strategy_config(orchestrator) -> dict:
    import yaml

    cfg = json.loads(json.dumps(DEFAULTS))
    path = getattr(orchestrator, "config_path", None)
    if path and pathlib.Path(path).exists() and str(path).endswith((".yaml", ".yml")):
        raw = yaml.safe_load(pathlib.Path(path).read_text()) or {}
        cfg.update(raw.get("strategy") or {})
    return cfg


def build_context(orchestrator, cfg: dict) -> Context:
    from ml.dataset_builder import _resolve_symbol_list
    from ml import data_asic

    config = orchestrator.config
    symbols = _resolve_symbol_list(None, config, orchestrator.ingestor, None)
    core = cfg["core_ticker"].replace(".AX", "")
    symbols = list(dict.fromkeys(list(symbols) + [core, "A200"]))
    t0 = time.monotonic()
    panel = load_panel(orchestrator.ingestor, symbols)
    for attempt in range(4):  # the benchmark must exist; fetch it directly if a batch dropped it (Yahoo rate-limits)
        if core in panel["adj_close"].columns:
            break
        orchestrator.ingestor.exclusions.pop(core, None)
        try:
            orchestrator.ingestor.fetch_ohlcv(core, refresh=True)
        except ValueError as err:
            logger.warning("Benchmark %s fetch attempt %d failed: %s", core, attempt + 1, err)
            time.sleep(30 * (attempt + 1))
            continue
        panel = load_panel(orchestrator.ingestor, symbols)
    if core not in panel["adj_close"].columns:
        raise ValueError(f"Benchmark ETF {core} could not be downloaded; cannot run strategy research.")
    aord = load_benchmark("^AORD", config.data.base_path, config.data.start_date)
    asic_dir = project_path(str(config.data.base_path), "asic_short")
    si = None
    cov = data_asic.coverage(asic_dir)
    if cov["files"]:
        si = data_asic.load_si(asic_dir, cfg["asic_lag_days"])
    ctx = Context(panel, aord, cfg, tfm_dir=project_path(str(config.data.base_path), "tfm_cache"), asic=si)
    ctx.asic_coverage = cov
    # Stocks only: drop ETFs and managed funds that ever pass the liquidity filters.
    candidates = [t for t in ctx.base.columns if ctx.base[t].any()]
    types = instrument_types(candidates, RESULTS / "instrument_types.json")
    funds = sorted(t for t in candidates if types.get(t, "EQUITY") != "EQUITY")
    ctx.base.loc[:, funds] = False
    logger.info("Excluded %d ETFs/funds: %s", len(funds), " ".join(funds))
    logger.info("Context: %d tickers x %d days in %.1fs", ctx.A.shape[1], ctx.A.shape[0], time.monotonic() - t0)
    return ctx


def _yahoo_instrument_type(ticker: str) -> Optional[str]:
    import yfinance as yf

    for attempt in range(3):
        try:
            kind = yf.Ticker(f"{ticker}.AX").get_history_metadata().get("instrumentType")
            if kind:
                return str(kind)
        except Exception as err:  # network or rate limit
            logger.debug("instrumentType %s attempt %d: %s", ticker, attempt + 1, err)
        time.sleep(2 * (attempt + 1))
    return None


def instrument_types(tickers, path: pathlib.Path, fetch=_yahoo_instrument_type) -> Dict[str, str]:
    """Yahoo instrument type (EQUITY, ETF, MUTUALFUND...) per ticker, cached in `path`; lookups that fail are retried next run."""
    cache = json.loads(path.read_text()) if path.exists() else {}
    missing = [t for t in tickers if t not in cache]
    for t in missing:
        kind = fetch(t)
        if kind:
            cache[t] = kind
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(cache.items())), indent=0))
        logger.info("Instrument types: looked up %d, %d still unknown", len(missing), sum(t not in cache for t in missing))
    return cache


def run_cli(args, orchestrator) -> int:
    cfg = strategy_config(orchestrator)
    if args.command == "asic-download":
        from ml import data_asic

        data_asic.download(project_path(str(orchestrator.config.data.base_path), "asic_short"), args.start)
        return 0
    if args.command == "strat-publish":   # local JSON only: no market download, no model
        return strat_publish(cfg)
    if args.command == "strat-explain-config":
        return strat_explain_config(cfg, orchestrator)
    ctx = build_context(orchestrator, cfg)
    if args.command == "strat-opt-study":
        return strat_opt_study(ctx, cfg, args)
    if args.command == "tfm-cache":
        return tfm_cache(ctx, cfg, args)
    if args.command == "strat-backtest":
        return strat_backtest(ctx, cfg, args)
    if args.command == "strat-recommend":
        return strat_recommend(ctx, cfg, args)
    return 1


def strat_explain_config(cfg: dict, orchestrator) -> int:
    import yaml

    from ml.policy_spec import explain_config

    path = getattr(orchestrator, "config_path", None)
    raw = {}
    if path and pathlib.Path(path).exists() and str(path).endswith((".yaml", ".yml")):
        raw = (yaml.safe_load(pathlib.Path(path).read_text()) or {}).get("strategy") or {}
    frozen_path = RESULTS / "frozen_choice.json"
    latest = {e["setting"]: e for e in val._registry(frozen_path)["experiments"]} if frozen_path.exists() else {}
    report = explain_config(DEFAULTS, raw, latest)
    for r in report["rows"]:
        print(f"{r['key']:<22} {str(r['value'])[:28]:<28} {r['origin']:<11} {r['units']:<14} {r['status']:<12} {r['note']}")
    for f in report["frozen"]:
        print(f"frozen {f['setting']}: {f['frozen_variant']} ({f['note']})")
    return 0


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _code_commit() -> Optional[str]:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(RESULTS.parent),
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def data_identity(ctx: Context) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(f"{ctx.calendar_id}|{ctx.dates[0]}|{ctx.dates[-1]}|{len(ctx.dates)}|{','.join(ctx.A.columns)}".encode())
    h.update(pd.util.hash_pandas_object(ctx.A.iloc[-260:].fillna(-1), index=True).to_numpy().tobytes())
    return h.hexdigest()[:12]


def strat_opt_study(ctx: Context, cfg: dict, args, results_dir: Optional[pathlib.Path] = None) -> int:
    """Registered matched study: corrected legacy N5/N8, VAS, and the OptPolicy controlled family on the same
    panel, universe, fees, timing and benchmark intervals. RESEARCH ONLY: writes results/opt_study/, never
    recommendation.json, frozen_choice.json or a tier."""
    from ml.policies import run_backtest_opt
    from ml.policy_spec import controlled_family, neighbours

    started = time.time()
    end = args.end
    capitals = [float(c) for c in str(args.capitals).split(",") if c.strip()]
    ranks = build_rank_cache(ctx, ["RAMOM"], end=end, start=None)
    ew_m = ew_universe(ctx, cfg, end)
    family = controlled_family("RAMOM")
    main = next(p for p in family if p.holding_cap is None and p.sizing == "covariance_cost_aware")
    rows, manifest = [], {"schema": "opt_study_v1", "status": "research only - not validated, not for trading",
                          "code_commit": _code_commit(), "data_id": data_identity(ctx), "calendar_id": ctx.calendar_id,
                          "start": cfg["start"], "end": end or str(ctx.dates[-1].date()), "broker": cfg.get("broker"),
                          "price_basis": "adjusted_total_return_synthetic",
                          "policies": [p.to_json() for p in family + neighbours(main)],
                          "legacy_controls": ["RAMOM_N5_noreg_B1", "RAMOM_N8_noreg_B3"],
                          "neighbour_rule": "gamma x0.5, gamma x2, name_limit x0.5 (declared before evaluation)",
                          "capacity_rule_i": "funded_capacity(capital) >= 3 (fee-aware); chosen counts reported separately",
                          "promotion": "blocked: OptPolicy tiers are not computed; owner account policy unset"}
    for capital in capitals:
        setting = f"{int(capital)}_core0"
        bench = run_backtest_v2(ctx, Variant(core_weight=1.0, capital=capital), cfg, ranks, end)
        bench_m = bench["monthly"]
        runs = {"BENCH_TR": bench}
        for v in (Variant(n=5, regime=False, buffer_mult=1, capital=capital), Variant(n=8, regime=False, capital=capital)):
            runs[v.id] = run_backtest_v2(ctx, v, cfg, ranks, end)
        for pol in family + neighbours(main):
            pol = dataclasses.replace(pol, capital=capital)
            t0 = time.monotonic()
            runs[pol.id] = run_backtest_opt(ctx, pol, cfg, ranks, end)
            logger.info("%s %s in %.0fs", setting, pol.id, time.monotonic() - t0)
        for vid, r in runs.items():
            for period in val.PERIODS:
                m = val.metrics(r, bench_m, ew_m, period)
                dec = r.get("decisions")
                extra = {}
                if dec is not None and len(dec):
                    sc = dec["stock_count"].dropna()
                    extra = {"count_min": float(sc.min()) if len(sc) else None, "count_median": float(sc.median()) if len(sc) else None,
                             "count_max": float(sc.max()) if len(sc) else None,
                             "effective_count_mean": float(dec["effective_count"].dropna().mean()) if len(sc) else None,
                             "no_trade_share": float(dec["status"].str.startswith("no_trade").mean()),
                             "failures": int((~dec["status"].isin(["proposal", "no_trade_optimal", "control_targets",
                                                                     "no_trade_forecast_unavailable"])).sum()),
                             "runtime_seconds": r.get("runtime_seconds")}
                rows.append({"setting": setting, "id": vid, "period": period, "funded_capacity": r.get("n_eff"),
                             "sharpe_definition": "zero-rate monthly Sharpe x sqrt(12)", **extra, **m})
    out = (results_dir or RESULTS) / "opt_study"
    manifest["runtime_minutes"] = (time.time() - started) / 60
    _atomic_write(out / "compare.json", json.dumps(sanitize_for_json(rows), indent=1))
    _atomic_write(out / "manifest.json", json.dumps(sanitize_for_json(manifest), indent=1, default=str))
    print(f"opt study written to {out} in {manifest['runtime_minutes']:.1f} min (research only)")
    return 0


# ---- tfm-cache -------------------------------------------------------------------------------

def tfm_cache(ctx: Context, cfg: dict, args) -> int:
    from ml.timesfm_strategy import load_timesfm

    import torch

    torch.manual_seed(42)
    torch.set_num_threads(4)
    forecaster = load_timesfm(args.checkpoint, 512, args.batch_size, positive=args.transform == "raw")
    core = cfg["core_ticker"].replace(".AX", "")
    bench = ctx.A_last[core]
    out_dir = ctx.tfm_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ctx.decision_rows(args.start, args.end)
    todo = [i for i in rows if not (out_dir / f"{args.transform}_{ctx.dates[i]:%Y-%m-%d}.csv").exists()]
    limit = cfg["universe_size"]
    started = time.monotonic()
    for n, i in enumerate(todo, 1):
        names = ctx.eligible(i)[:limit]
        t = time.monotonic()
        df = tfm_cache_scores(ctx, args.transform, i, names, forecaster, bench, horizon=args.horizon)
        df.to_csv(out_dir / f"{args.transform}_{ctx.dates[i]:%Y-%m-%d}.csv", index=False)
        if n == 1:
            projected = (time.monotonic() - t) * len(todo) / 60
            logger.info("First date took %.1fs; projected total %.0f min for %d dates.", time.monotonic() - t,
                        projected, len(todo))
            if projected > 300 and limit > 150:
                limit = 150
                logger.warning("Projection over 300 min: restricting TFM forecasts to the top 150 by MDV60.")
        logger.info("tfm-cache %s %d/%d %s (%d series) %.0fs elapsed", args.transform, n, len(todo),
                    ctx.dates[i].date(), len(df), time.monotonic() - started)
    (out_dir / f"{args.transform}_meta.json").write_text(json.dumps({
        "universe_limit": limit, "horizon": args.horizon, "checkpoint": args.checkpoint, "context": 512,
        "transform": args.transform, "calendar_id": ctx.calendar_id, "score": "median/(q90-q10) at horizon"}))
    return 0


# ---- strat-backtest --------------------------------------------------------------------------

def grid(signals_available: List[str], si_ok: bool) -> List[Variant]:
    vs = [Variant(signal=s, n=n, regime=r) for s in TRADEABLE_SIGNALS for n in (5, 8) for r in (False, True)]
    vs.append(Variant(signal="RAMOM", n=5, regime=False, buffer_mult=1))
    vs.append(Variant(signal="RESMOM", n=5, regime=False, buffer_mult=1))
    vs.append(Variant(signal="RAMOM", n=5, regime=True, si_screen=True))
    for n, b in ((5, 1), (8, 3)):
        vs += [Variant(signal="RAMOM", n=n, regime=False, buffer_mult=b, rebalance=q, tax_aware=t)
               for q, t in (("Q", False), ("M", True), ("Q", True))]
    return vs


TRADING_FIELDS = ("turnover", "orders_per_year", "brokerage_drag", "slippage_drag", "cost_drag", "avg_positions",
                  "avg_holding_days", "pct_disposals_lt_365")


def _period_rows(result, bench_m, ew_m, label, setting, extra=None, engine=True, tax_rate=0.0) -> List[dict]:
    rows = []
    after_tax = dict(result, monthly=after_tax_monthly(result, tax_rate)) if engine else None
    for period in val.PERIODS:
        m = val.metrics(result, bench_m, ew_m, period)
        if not engine:  # index / universe / selection series have no orders of their own
            m.update({k: None for k in TRADING_FIELDS if k in m})
        if engine:
            m["cagr_after_tax"] = val.metrics(after_tax, bench_m, ew_m, period).get("cagr")
        rows.append({"id": label, "setting": setting, "period": period, **(extra or {}), **m})
    return rows


def strat_backtest(ctx: Context, cfg: dict, args) -> int:
    run_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    frozen_path = RESULTS / "frozen_choice.json"   # append-only experiment registry: never deleted or re-frozen
    cfg_hash = config_identity(cfg)
    end = args.end
    rows_needed = ctx.decision_rows(cfg["start"], end)
    tfm_ok = {}
    for sig, tr in (("TFM_REL", "rel"), ("TFM_RAW", "raw")):
        have = sum((ctx.tfm_dir / f"{tr}_{ctx.dates[i]:%Y-%m-%d}.csv").exists() for i in rows_needed)
        meta_path = ctx.tfm_dir / f"{tr}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        compatible = meta.get("horizon") == 21   # manifest must match the monthly engine's horizon before reuse
        tfm_ok[sig] = compatible and have >= len(rows_needed) - 1  # the latest month-end may not be cached yet
        if have and not compatible:
            logger.warning("%s cache manifest missing or horizon != 21 (%s); skipping TFM variants.", sig, meta)
        if 0 < have < len(rows_needed) - 1:
            logger.warning("%s cache covers %d/%d decision dates; skipping TFM variants.", sig, have, len(rows_needed))
    si_ok = ctx.asic is not None and bool(getattr(ctx, "asic_coverage", {}).get("complete"))
    signals = [s for s in TRADEABLE_SIGNALS if not s.startswith("TFM") or tfm_ok[s]]
    t0 = time.monotonic()
    ranks = build_rank_cache(ctx, signals, end=end, start=cfg["start"])
    logger.info("Rank cache for %s in %.1fs", signals, time.monotonic() - t0)

    ew_m = ew_universe(ctx, cfg, end)
    aord_m = ctx.aord.reindex(ctx.dates[ctx.decision_rows(cfg["start"], end)]).pct_change().dropna() \
        if ctx.aord is not None else pd.Series(dtype=float, index=pd.DatetimeIndex([]))  # AORD download can fail (Yahoo 429)

    compare: List[dict] = []
    trials: List[dict] = []
    per_setting: Dict[str, dict] = {}
    for capital, core_w in SETTINGS:
        setting = f"{int(capital)}_core{core_w:g}"
        bench = run_backtest_v2(ctx, Variant(core_weight=1.0, capital=capital), cfg, ranks, end)
        bench_m = bench["monthly"]
        compare += _period_rows(bench, bench_m, ew_m, "BENCH_TR", setting, {"label": "VAS.AX buy-and-hold (total return)"},
                              tax_rate=cfg.get("tax_rate", 0.0))
        for label, m in (("AORD_PRICE", aord_m), ("EW_UNIV", ew_m)):
            fake = {"monthly": m, "equity": (1 + m).cumprod() * capital, "orders": pd.DataFrame(columns=[
                "date", "side", "value", "brokerage", "slippage"]), "holdings": [], "positions": []}
            note = "price index, excludes dividends" if label == "AORD_PRICE" else "equal-weight eligible universe, gross"
            compare += _period_rows(fake, bench_m, ew_m, label, setting, {"label": note}, engine=False)

        results: Dict[str, dict] = {}
        for v in grid(signals, si_ok):
            v = dataclasses.replace(v, capital=capital, core_weight=core_w)
            skip = None
            if v.signal.startswith("TFM") and not tfm_ok[v.signal]:
                skip = "TimesFM forecast cache missing (run tfm-cache)"
            elif v.si_screen and not si_ok:
                skip = "insufficient free ASIC history"
            if skip:
                compare.append({"id": v.id, "setting": setting, "period": "full", "skipped_reason": skip,
                                **_variant_fields(v)})
                continue
            r = run_backtest_v2(ctx, v, cfg, ranks, end)
            results[v.id] = r
            extra = {**_variant_fields(v), "skipped_reason": r["skipped_reason"]}
            compare += _period_rows(r, bench_m, ew_m, v.id, setting, extra, tax_rate=cfg.get("tax_rate", 0.0))
            trials.append({"setting": setting, "variant": v.id, "config_hash": trial_hash(v, cfg_hash),
                           "sharpe_full": val.sharpe(r["monthly"]), "timestamp": run_start})
        per_setting[setting] = {"results": results, "bench": bench, "bench_m": bench_m}

    _append_trials(trials)

    # ---- validation per setting ----
    validation: Dict[str, dict] = {"benchmark_note": BENCH_NOTE, "settings": {}, "engine_id": ENGINE_ID,
                                   "config_id": cfg_hash, "calendar_id": ctx.calendar_id,
                                   "hold_out_note": val.HOLD_OUT_NOTE}
    validation["ic"] = {s: val.rank_ic(ctx, s, ctx.decision_rows(cfg["start"], end), ctx.A_last[cfg["core_ticker"].replace(".AX", "")])
                        for s in signals + ["MOM_12_1"]}
    validation["asic_coverage"] = getattr(ctx, "asic_coverage", None)
    for setting, data in per_setting.items():
        res = {k: r for k, r in data["results"].items() if r["n_eff"]}
        if not res:
            validation["settings"][setting] = {"tier": "FALLBACK", "reasons": ["satellite disabled at this capital"]}
            continue
        monthly = {k: r["monthly"] for k, r in res.items()}
        turnover = lambda vid, cut, res=res: val.turnover_before(res[vid], cut)   # selection-window data only
        hashes = {k: trial_hash(r["variant"], cfg_hash) for k, r in res.items()}
        wf = val.walk_forward_select(monthly, turnover)
        wf["account"] = walk_forward_account(ctx, cfg, res, wf["picks"], ranks, end)
        frozen_info = val.freeze_choice(monthly, turnover, hashes, frozen_path, setting, engine=ENGINE_ID)
        validation["settings"][setting] = _validate_setting(ctx, cfg, setting, res, data, ew_m, wf, frozen_info,
                                                            frozen_path, ranks, end, signals)
        compare += [{**row, "label": "walk-forward selection (one continuous account)"} for row in
                    _period_rows(wf["account"], data["bench_m"], ew_m, "WF_SELECT", setting, engine=False)
                    if row["period"] == "walk_forward"]

    _write_compare(compare)
    _universe_snapshot(ctx)
    _write_history(per_setting, aord_m, ew_m, cfg.get("tax_rate", 0.0))
    (RESULTS / "validation.json").write_text(json.dumps(sanitize_for_json(validation), indent=1))
    logger.info("strat-backtest finished in %.1f min", (time.time() - run_start) / 60)
    _print_summary(compare, validation, cfg)
    return 0


def _universe_snapshot(ctx: Context) -> None:
    """Dated record of today's eligible universe (F18). Past universes cannot be rebuilt from current listings;
    these snapshots accumulate forward. Tickers with no price in the last 10 sessions are listed separately
    (download failure or delisting: not distinguished without an exchange log)."""
    i = len(ctx.dates) - 1
    names = ctx.eligible(i)
    out = RESULTS / "universe" / f"eligible_{ctx.dates[i]:%Y-%m-%d}.csv"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": names, "mdv60": [float(ctx.MDV60.iloc[i][t]) for t in names]}).to_csv(out, index=False)
    gone = [t for t in ctx.A.columns if ctx.age.iloc[i][t] >= 10 and ctx.age.iloc[i][t] < 260]
    (out.parent / f"unpriced_{ctx.dates[i]:%Y-%m-%d}.txt").write_text("\n".join(sorted(gone)))


def config_identity(cfg: dict) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def trial_hash(v: Variant, cfg_hash: str) -> str:
    """Trial identity: variant + resolved config + engine semantics (identical reruns hash identically)."""
    import hashlib

    return hashlib.sha1(f"{v.config_hash()}|{cfg_hash}|{ENGINE_ID}".encode()).hexdigest()[:12]


def walk_forward_account(ctx: Context, cfg: dict, res: Dict[str, dict], picks: List[dict], ranks, end) -> dict:
    """Investable walk-forward: ONE account from the first selection date, switching rules at each 1 July to the
    variant chosen on data before that date; holdings carry over and every transition trade is costed."""
    if not picks:
        return {"monthly": pd.Series(dtype=float), "equity": pd.Series(dtype=float), "orders": pd.DataFrame(
            columns=["date", "side", "value", "brokerage", "slippage"]), "holdings": [], "positions": []}
    sched = [(pd.Timestamp(p["from"]), res[p["variant"]]["variant"]) for p in picks]
    start = ctx.dates[ctx.dates < sched[0][0]][-1]
    r = run_backtest_v2(ctx, sched[0][1], dict(cfg, start=str(start.date())), ranks, end, schedule=sched)
    r["schedule"] = [(d.date().isoformat(), x.id) for d, x in sched]
    return r


def _write_history(per_setting: Dict[str, dict], aord_m: pd.Series, ew_m: pd.Series, tax_rate: float) -> None:
    """Month-by-month net returns and holdings for every variant, so the dashboard can replay any start date."""
    out = {"note": "Monthly net returns after brokerage and slippage; holdings are what each strategy held "
                   "during the month ending on that date. returns_after_tax also deducts estimated capital "
                   "gains tax at tax_rate.", "tax_rate": tax_rate, "settings": {}}
    for setting, data in per_setting.items():
        dates = data["bench_m"].index
        ser = lambda m: [round(float(x), 6) if pd.notna(x) else None for x in m.reindex(dates)]
        strategies = {}
        for vid, r in data["results"].items():
            held = r["books"].shift(1).reindex(dates)  # decided at the previous month-end, held through this month
            strategies[vid] = {"returns": ser(r["monthly"]),
                               "returns_after_tax": ser(after_tax_monthly(r, tax_rate)),
                               "holdings": [h if isinstance(h, str) else "" for h in held]}
        out["settings"][setting] = {
            "dates": [d.date().isoformat() for d in dates],
            "baselines": {"BENCH_TR": ser(data["bench_m"]), "AORD_PRICE": ser(aord_m), "EW_UNIV": ser(ew_m),
                          "BENCH_TR_AFTER_TAX": ser(after_tax_monthly(data["bench"], tax_rate))},
            "strategies": strategies,
        }
    (RESULTS / "history.json").write_text(json.dumps(out, separators=(",", ":")))


def _variant_fields(v: Variant) -> dict:
    return {"signal": v.signal, "n": v.n, "regime": v.regime, "buffer_mult": v.buffer_mult,
            "core_weight": v.core_weight, "capital": v.capital, "si_screen": v.si_screen}


def _validate_setting(ctx, cfg, setting, res, data, ew_m, wf, frozen_info, frozen_path, ranks, end, signals) -> dict:
    bench_m = data["bench_m"]
    bench = data["bench"]
    # Hold-out is only evaluated after the frozen choice exists on disk.
    vid = val.hold_out_gate(frozen_path, setting, time.time(), frozen_info.get("experiment_id"))
    r = res[vid]
    v: Variant = r["variant"]
    met = {p: val.metrics(r, bench_m, ew_m, p) for p in val.PERIODS}
    bmet = {p: val.metrics(bench, bench_m, ew_m, p) for p in val.PERIODS}
    wf_met = val.metrics(wf["account"], bench_m, ew_m, "walk_forward")
    wf_spliced = val.metrics({"monthly": wf["monthly"], "equity": (1 + wf["monthly"]).cumprod(), "orders": pd.DataFrame(
        columns=["date", "side", "value", "brokerage", "slippage"]), "holdings": [], "positions": []},
        bench_m, ew_m, "walk_forward")

    # Trial statistics up to 2024-06 (selection data only).
    cut = pd.Timestamp("2024-07-01")
    selection = {k: x["monthly"].loc[x["monthly"].index < cut] for k, x in res.items()}
    trial_srs = [s.mean() / s.std() for s in selection.values() if len(s) > 12 and s.std() > 0]
    n_trials = max(_trial_count(setting), len(trial_srs), 22)
    dsr_frozen = val.dsr(r["monthly"].to_numpy(), trial_srs, n_trials)
    dsr_wf = val.dsr(val.window(wf["account"]["monthly"], "walk_forward").to_numpy(), trial_srs, n_trials)
    M = pd.DataFrame(selection).dropna()
    pbo = val.pbo_cscv(M.to_numpy(), S=10)

    # Neighbours: differ in exactly one of N, regime, buffer.
    neighbours = [dataclasses.replace(v, n=8 if v.n == 5 else 5), dataclasses.replace(v, regime=not v.regime),
                  dataclasses.replace(v, buffer_mult=1 if v.buffer_mult != 1 else 3)]
    nb_rows, extra_trials = [], []
    for nv in neighbours:
        nr = res.get(nv.id) or run_backtest_v2(ctx, nv, cfg, ranks, end)
        if nv.id not in res:
            extra_trials.append({"setting": setting, "variant": nv.id, "config_hash": trial_hash(nv, config_identity(cfg)),
                                 "sharpe_full": val.sharpe(nr["monthly"]), "timestamp": time.time()})
        # full criterion (a) for a fixed-policy neighbour: its own walk-forward window AND later window beat VAS
        beats = lambda m, b: m.get("cagr", -1) > b.get("cagr", 0) and m.get("sharpe", -9) >= b.get("sharpe", 9)
        nm = val.metrics(nr, bench_m, ew_m, "walk_forward")
        nh = val.metrics(nr, bench_m, ew_m, "hold_out")
        ok = beats(nm, bmet["walk_forward"]) and beats(nh, bmet["hold_out"])
        nb_rows.append({"variant": nv.id, "passes_a_wf": bool(ok), "passes_a_full": bool(ok), "cagr": nm.get("cagr"),
                        "sharpe": nm.get("sharpe"), "cagr_later": nh.get("cagr"), "sharpe_later": nh.get("sharpe"),
                        "window": "fixed-policy walk_forward and hold_out windows"})
    _append_trials(extra_trials)

    stress = run_backtest_v2(ctx, dataclasses.replace(v, slip_mult=cfg["cost_stress_mult"]), cfg, ranks, end)
    stress_met = val.metrics(stress, bench_m, ew_m, "full")
    yearly = pd.DataFrame({"strategy": (1 + r["monthly"]).groupby(r["monthly"].index.year).prod() - 1,
                           "bench": (1 + bench_m).groupby(bench_m.index.year).prod() - 1})
    windows = {}
    for name, (a, b) in {"covid_2020_02_04": ("2020-02-01", "2020-04-30"), "year_2022": ("2022-01-01", "2022-12-31")}.items():
        sl = lambda s: float((1 + s.loc[a:b]).prod() - 1)
        windows[name] = {"strategy": sl(r["monthly"]), "bench": sl(bench_m)}
    decision = val.acceptance(met["hold_out"], met["full"], wf_met, bmet["walk_forward"], bmet["hold_out"],
                              bmet["full"], dsr_frozen["dsr"], pbo["pbo"], met["full"].get("cost_drag", 1.0),
                              sum(n["passes_a_wf"] for n in nb_rows), r["n_eff"])
    return {
        "frozen": {**frozen_info, "variant": vid}, "walk_forward_picks": wf["picks"],
        "metrics": met, "bench_metrics": bmet, "wf_select": wf_met, "wf_spliced_diagnostic": wf_spliced,
        "hold_out_note": val.HOLD_OUT_NOTE, "cost_stress_note": "printed sensitivity, not a gate",
        "capacity": {"funded_n_eff": r["n_eff"], "nominal_n": r.get("n_nominal"),
                     "chosen_avg_positions": met["full"].get("avg_positions")},
        "dsr_frozen": dsr_frozen, "dsr_wf_select": dsr_wf, "pbo": {"pbo": pbo["pbo"], "combinations": pbo["combinations"],
                                                                   "logits": pbo["logits"]},
        "neighbours": nb_rows, "cost_stress": stress_met,
        "survivorship": {"excess_full": met["full"].get("excess_vs_bench"),
                         "minus_1_5pct": (met["full"].get("excess_vs_bench") or 0) - 0.015,
                         "minus_3pct": (met["full"].get("excess_vs_bench") or 0) - 0.03},
        "calendar_years": {str(y): row for y, row in yearly.to_dict("index").items()},
        "stress_windows": windows, **decision,
    }


def _trial_count(setting: str) -> int:
    path = RESULTS / "trials_log.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    return int(df.loc[df["setting"] == setting, "config_hash"].nunique())


def _append_trials(rows: List[dict]) -> None:
    if not rows:
        return
    path = RESULTS / "trials_log.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["timestamp", "setting", "variant", "config_hash", "sharpe_full"])
        if new:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in w.fieldnames})


def _write_compare(rows: List[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "strategy_compare.csv", index=False)
    (RESULTS / "strategy_compare.json").write_text(json.dumps(sanitize_for_json(rows), indent=1))


def _print_summary(compare: List[dict], validation: dict, cfg: dict) -> None:
    df = pd.DataFrame(compare)
    for period in ("walk_forward", "hold_out"):
        for setting in ("10000_core0", "50000_core0", "50000_core0.6"):
            sub = df[(df["period"] == period) & (df["setting"] == setting) & df["cagr"].notna()]
            if sub.empty:
                continue
            print(f"\n== {period} | {setting} ==")
            cols = ["id", "cagr", "sharpe", "mdd", "turnover", "cost_drag", "excess_vs_bench", "excess_ci_lo"]
            print(sub[[c for c in cols if c in sub]].sort_values("sharpe", ascending=False).to_string(
                index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nRank IC:", json.dumps(sanitize_for_json(validation.get("ic")), indent=1))
    for s, v in validation["settings"].items():
        print(f"\n{s}: tier={v.get('tier')} frozen={v.get('frozen', {}).get('variant')} "
              f"DSR={v.get('dsr_frozen', {}).get('dsr')} PBO={v.get('pbo', {}).get('pbo')}")
        for reason in v.get("reasons", []):
            print("   fails", reason)


# ---- strat-recommend / publish ---------------------------------------------------------------

def _configured_setting(cfg: dict) -> Optional[str]:
    """The researched setting for the configured account, or None: no nearest-capital inheritance (F22)."""
    for cap, core in SETTINGS:
        if abs(cap - cfg["capital"]) < 0.5 and abs(core - cfg["core_weight"]) < 1e-9:
            return f"{int(cap)}_core{core:g}"
    return None


def _with_pending_day(ctx: Context) -> Context:
    """Append a placeholder next business day (prices = last close) so the engine emits next-open orders."""
    nxt = ctx.dates[-1] + pd.offsets.BDay(1)
    for name in ("A", "C", "V", "AO", "A_last", "MDV60", "base", "VOL_252", "VOL_63", "MOM_12_1", "RAMOM", "RESMOM", "LOWVOL",
                 "age"):
        f = getattr(ctx, name)
        row = ctx.A_last.iloc[[-1]] if name in ("AO",) else f.iloc[[-1]]
        if name == "base":
            row = f.iloc[[-1]] & False
        setattr(ctx, name, pd.concat([f, row.set_axis([nxt])]))
    if ctx.aord is not None:
        ctx.aord = pd.concat([ctx.aord, ctx.aord.iloc[[-1]].set_axis([nxt])])
    ctx.dates = ctx.dates.append(pd.DatetimeIndex([nxt]))
    return ctx


def strat_recommend(ctx: Context, cfg: dict, args, today: Optional[pd.Timestamp] = None) -> int:
    """Tier, hypothetical paper book and paper proposals. Paper proposals are estimates for the research book;
    execution orders are separate and are never produced while prices are adjusted/synthetic (F16/F22/F23)."""
    from ml.accounting import import_holdings

    if getattr(args, "capital", None):
        cfg["capital"] = args.capital
    validation_path = RESULTS / "validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    setting = _configured_setting(cfg)
    sv = validation.get("settings", {}).get(setting, {}) if setting else {}
    tier = sv.get("tier", "FALLBACK")
    reasons = list(sv.get("reasons", ["no research results yet: run the research workflow (strat-backtest)"]))
    if setting is None:
        tier, reasons = "FALLBACK", [f"capital A${cfg['capital']:,.0f} / core {cfg['core_weight']:g} is not a researched "
                                      "setting; results from the nearest setting are not inherited"]
    if validation and validation.get("engine_id") != ENGINE_ID:
        tier, reasons = "FALLBACK", reasons + ["validation.json predates the current engine (manifest mismatch); rerun research"]
    vid = sv.get("frozen", {}).get("variant")
    core = cfg["core_ticker"]
    last = ctx.dates[-1]
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    data_age = len(pd.bdate_range(last, today)) - 1
    data_state = "current" if data_age <= 3 else "stale"
    holdings_path = getattr(args, "holdings", None) or cfg.get("holdings_file")
    account = import_holdings(holdings_path) if holdings_path else None
    proposals, book, paper = [], {}, None
    is_decision_day = (last + pd.offsets.BDay(1)).month != last.month
    if tier in ("LIVE", "PAPER") and vid and data_state == "current":
        from ml.policy_spec import decode

        v = dataclasses.replace(decode(vid.split("_slipx")[0]), capital=cfg["capital"], core_weight=cfg["core_weight"])
        r = run_backtest_v2(ctx, v, cfg)    # hypothetical research book, valued through the latest session
        book = {k: int(q) for k, q in r["final_book"].items() if q}
        paper = {"date": r["valuation_date"].date().isoformat(), "nav": float(r["equity"].iloc[-1]), "book": book}
        if is_decision_day:
            rp = run_backtest_v2(_with_pending_day(copy.copy(ctx)), v, cfg)   # copy: the real ctx keeps no fake row
            o = rp["orders"]
            o = o[o["date"] > last] if len(o) else o
            proposals = [{"ticker": f"{x.ticker}.AX", "side": x.side, "shares": int(x.shares), "est_price": float(x.price),
                          "est_cost": float(x.brokerage + x.slippage), "status": "estimate",
                          "basis": "hypothetical research book; placeholder open = last close; adjusted prices"}
                         for x in o.itertuples()]
        if tier == "PAPER":
            _paper_append(vid, r["valuation_date"], r)
    blockers = ["price basis is adjusted_total_return_synthetic: no executable share quantities"]
    if tier != "LIVE":
        blockers.append(f"validation tier {tier} does not authorise trading")
    if account is None:
        blockers.append("no actual holdings/cash imported (holdings_file); proposals refer to a hypothetical book")
    if data_state != "current":
        blockers.append(f"market data is {data_age} business days old")
    if tier == "LIVE":
        action = (f"{vid}: paper proposals below are estimates for the research book, not orders." if proposals
                  else f"{vid}: no proposals today; the next review is the last trading day of the month.")
    else:
        action = FALLBACK_TEXT.format(core=core)
        if tier == "PAPER":
            action += f" {vid} is being paper-tracked as a candidate."
    reconciliation = None
    if account is not None and book:
        names = sorted(set(book) | set(account["holdings"]))
        reconciliation = [{"ticker": t, "actual": account["holdings"].get(t, 0), "research_book": book.get(t, 0)}
                          for t in names if account["holdings"].get(t, 0) != book.get(t, 0)]
    rec = {"schema": "recommendation_v2", "tier": tier, "validation_tier": tier, "strategy_id": vid,
           "action_text": action, "as_of": last.date().isoformat(), "data_state": data_state,
           "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "reasons": reasons, "setting": setting,
           "broker_execution_state": "none: this program places no orders",
           "paper_proposals": proposals, "orders_next_open": proposals, "execution_orders": [],
           "execution_blockers": blockers, "paper_book": paper,
           "account": {k: account[k] for k in ("source", "account_hash", "as_of", "cash")} if account else None,
           "reconciliation": reconciliation, "engine_id": ENGINE_ID}
    _survivorship_log(ctx, cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _atomic_write(RESULTS / "recommendation.json", json.dumps(sanitize_for_json(rec), indent=1))
    print(f"\nTier: {tier}\n{action}")
    for o in proposals:
        print(f"  (estimate) {o['side']:<4} {o['shares']:>6} {o['ticker']:<8} ~${o['est_price']:.2f}")
    for reason in reasons + blockers:
        print("  -", reason)
    return 0


def _paper_append(vid: str, as_of: pd.Timestamp, r: dict) -> None:
    path = RESULTS / "paper" / f"{vid}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and as_of.date().isoformat() in path.read_text():
        return
    eq = r["equity"]
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "equity", "positions"])
        w.writerow([as_of.date().isoformat(), f"{float(eq.loc[as_of]):.2f}", ";".join(sorted(t for t, q in r["final_book"].items() if q))])


def _survivorship_log(ctx: Context, cfg: dict) -> None:
    """Forward-looking delisting record: tickers whose last price is more than 10 trading days old."""
    path = RESULTS / "survivorship" / "vanished.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set(pd.read_csv(path)["ticker"]) if path.exists() else set()
    rows = []
    n = len(ctx.dates)
    for t in ctx.A.columns:
        lv = ctx.A[t].last_valid_index()
        if lv is None or t in seen:
            continue
        if n - 1 - ctx.dates.get_loc(lv) > cfg["stale_days"] and lv >= ctx.dates[-260]:
            rows.append({"ticker": t, "last_date": lv.date().isoformat(), "last_close": float(ctx.C[t].loc[lv])})
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def strat_publish(cfg: dict, results_dir: pathlib.Path = RESULTS, out: Optional[pathlib.Path] = None) -> int:
    out = out or project_path("docs", "data", "latest.json")
    RESULTS = results_dir  # noqa: N806 - local override for tests
    payload = json.loads(out.read_text()) if out.exists() else {}
    rec_path, cmp_path, val_path = RESULTS / "recommendation.json", RESULTS / "strategy_compare.json", RESULTS / "validation.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    validation = json.loads(val_path.read_text()) if val_path.exists() else {}
    compare = json.loads(cmp_path.read_text()) if cmp_path.exists() else []
    setting_status = {s: v for s, v in validation.get("settings", {}).items()}
    rows = []
    for row in compare:
        sv = setting_status.get(row.get("setting"), {})
        passes = sv.get("passes") if sv.get("frozen", {}).get("variant") == row.get("id") else None
        rows.append({**row, "passes": passes, "is_frozen": passes is not None})
    cache_files = sorted(project_path("data", "tfm_cache").glob("*_*.csv"))
    if not (rec or compare or validation) and payload.get("strategy_comparison"):
        print(f"No strategy results in {RESULTS}; keeping the last good publication ({payload.get('strategy_published_at')}).")
        return 0
    payload.update({
        "recommendation": {k: rec.get(k) for k in ("tier", "strategy_id", "action_text", "as_of", "reasons", "setting",
                                                   "data_state", "validation_tier", "broker_execution_state",
                                                   "execution_blockers", "generated_at")}
        if rec else {"tier": "FALLBACK", "action_text": FALLBACK_TEXT.format(core=cfg["core_ticker"]), "reasons": []},
        "strategy_comparison": rows,
        "orders_next_open": rec.get("paper_proposals", rec.get("orders_next_open", [])),
        "paper_proposals": rec.get("paper_proposals", rec.get("orders_next_open", [])),
        "execution_orders": rec.get("execution_orders", []),
        "strategy_published_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "benchmark_note": BENCH_NOTE,
        "broker": cfg.get("broker"),
        "validation_summary": {s: {k: v.get(k) for k in ("tier", "frozen", "dsr_frozen", "pbo", "neighbours",
                                                         "walk_forward_picks", "survivorship", "stress_windows")}
                               for s, v in setting_status.items()},
        "signal_ic": validation.get("ic"),
        "history": json.loads((RESULTS / "history.json").read_text()) if (RESULTS / "history.json").exists() else None,
        "data_flags": {
            "asic_stale": not bool((validation.get("asic_coverage") or {}).get("complete")),
            "survivorship_note": "Backtests use today's listings only; delisted companies are missing, so results are biased "
                                 "upward. The 1.5-point hurdle in test (b) is a sensitivity, not a survivorship correction.",
            "tfm_cache_date": cache_files[-1].stem.split("_")[-1] if cache_files else None,
        },
    })
    _atomic_write(out, json.dumps(sanitize_for_json(payload), indent=1))
    print(f"Merged strategy results into {out}")
    return 0
