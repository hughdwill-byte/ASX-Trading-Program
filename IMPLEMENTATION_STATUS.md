# Implementation status (ASX_IMPLEMENT.md, then ASX_UX_IMPLEMENT.md)

## UX / rankings (ASX_UX_IMPLEMENT.md, 2026-10-20) — on top of main 475221c

Display/publication contract `asx_research_display_v1`; ranking artifact `dashboard_v1`; score `evidence_score_v1`.
No trading rule, threshold, frozen experiment, objective or tier changed (`ENGINE_ID` unchanged).

| Area | Status | Where |
|---|---|---|
| Ranking/explanation module | done | `ml/dashboard_rankings.py` (pure): past-profit ranking (integer cents, competition ranks, loss-only wording, cohort/coverage checks, references excluded, after-tax basis only from computed series), evidence score v1 (exact formula/weights/anchors, null on missing, no renormalisation, near ties, group-local ranks, fixed-group sensitivity), method registry from structured policy, deterministic cited explanation facts |
| Research producer | done | `strategy_cli.dashboard_setting` -> `results/dashboard.json` during `strat-backtest` from in-memory results. Per-method DSR (same trial context as the engine, shared per setting), each method's own registered neighbours (factored `registered_neighbours`/`evaluate_neighbours`, unchanged semantics, extra runs registered as trials), daily drawdowns, W/L windows from `validation.PERIODS`; shared PBO labelled procedure-level; WF_SELECT as separate short-coverage account |
| Publisher | done | `strat_publish`: `display_schema`, `publication` manifest (`state` for research validity, `rankings_state` for the artifact; mixed engine/config or unknown schema -> needs refresh, rankings withheld), per-setting `scenarios` (exact match, no inheritance), richer `validation_summary`, recommendation freshness fields from the producer (`price_date`, `data_age_business_days`, `next_review`), actual research-book `allocation` (weights incl. cash; `strategy_cli.paper_allocation`) |
| Dashboard UI | done | `docs/index.html` rewritten: ASX Research; Today / Performance / Research / How it works with hash URLs (`#/performance?s=&m=&b=&t=`), Back/Forward/reload; global Research scenario; Today result + reason + limitation + 3 status cards + "This program places no trades."; Past profitability (top 10 + Show all, search without re-ranking, sortable header buttons with aria-sort, focus kept, live announcements) and Evidence for future use (groups, fallback notice, score parts); method inspector (What it does / Why this historical result / Why this evidence rank, "Inspecting another method"); history chart follows inspected method (method + VAS default, optional references, dash patterns, year ticks, data table, keyboard slider, gaps never zero-filled); Research diagnostics + legacy archive; plain-language glossary; distinct load/missing/schema/network/render states; contrast-safe tokens both themes |
| Workflows | done | research commits `results/dashboard.json`; `test_dashboard.py` in both self-checks |

Checks (all pass): the seven earlier test files, `python3 ml/test_dashboard.py` (11 tests, ~15 s), pyflakes clean on
`ml/*.py stocktrade.py tools/*.py`, and browser checks `python3 tools/build_ui_fixture.py <dir> && node tools/ui_check.mjs
<dir>/site` (42/42 in headless Chromium: Today content, nav/Back/reload, inspection leaves Today unchanged, keyboard sort
focus/aria-sort/announcement, rank numbers stable, evidence groups/fallback, contrast >= 4.5:1 for all text in both themes
on all views, no page-level horizontal scroll at 320 CSS px (=400% zoom) and with 200% text, 44 px targets, landmarks,
chart table alternative, missing/schema/network states). Needs a dev-only `playwright` install; skips otherwise.

Pending / not verified: screen-reader and real-device testing; human usability validation; Windows high-contrast.
Change summary vs a previous snapshot is not shown (no compatible previous snapshot is stored). The real ranking artifact
does not exist until the next research run: until then the live page shows "Method rankings are waiting for the next
research run" (real-results smoke check done locally into the scratchpad, not committed). OptPolicy matched study is
still unpublished, so the optimiser is shown as research-only with "study pending". Broker/tax controls only select
computed results (one broker modelled; tax overlay uses the computed after-tax series).

Spec: `ASX_IMPLEMENT.md` (audit 2026-10-20, audited commit 4f6ccbe). Checkout base: 613dd5d (adds RESMOM on top of
the audited commit; it resolves no F-item). Branch `claude/laughing-hawking-c5e85r`. **Commits are local only:
nothing was pushed** (spec: no push/merge/deploy/dispatch; a push to `claude/**` also starts both workflows).

No real market data was available in this container (Yahoo/Hugging Face/weblink blocked), so **no market study was
run** and nothing was written to `results/` or `docs/data/`. All tests use synthetic fixtures in temp dirs.

## Checks (all pass at the last commit)

```sh
python3 ml/test_strategy_v2.py        # existing, 3 tests corrected (see below)            ~1 s
python3 ml/test_timesfm_strategy.py   # existing, updated for end boundary/open periods     <1 s
python3 ml/test_engine_repairs.py     # F01-F05, F11, F12, F20, F22/F23, F25, calibration   ~18 s
python3 ml/test_optimizer.py          # F14/F15/F19 optimiser, >20-name fixture, oracle      ~5 s
python3 ml/test_accounting.py         # ledger: partial fills, settlement, splits, dividends <1 s
python3 ml/test_policies.py           # null-cap spec, explain-config, policy backtest, study ~30 s
python3 ml/test_ml_causal.py          # F07-F10 labels, as-of, fold preprocessing, artifacts <1 s
python3 -m pyflakes ml/*.py stocktrade.py   # clean
```
Last run: all seven print "passed"/"ok"; heuristic vs exhaustive oracle gap 0.00e+00 on 12 random tiny cases.
Environment: scipy, scikit-learn, pyflakes, pyyaml, joblib were missing from the container and were pip-installed
(all already in `requirements.txt`/used by the repo; no new project dependency). torch is not installed; the sklearn
prediction path no longer needs it (lazy import).

## Requirement status

| ID | Status | Where / what |
|---|---|---|
| F01 | done | `portfolio.affordable_qty` (exact debit incl. fee at final value + slippage <= min(order budget, cash), fee discontinuities); no parcel round-up past budget; overdraft assert in `trade()`. Fixture: A$3,030/3 names/CMC now spends <= A$3,030 (was A$3,040.50). Same-ticker netting is inherent in the optimiser (one net order per name); legacy engine never buys and sells one ticker on one day. CMC "first buy per security per day" holds because each name gets at most one buy per day. |
| F02 | done | Missing execution open -> `rejected` record, no fill/cash; `requested`/`filled`/`partial` statuses; stale holdings frozen and haircut in valuation only. |
| F03 | done | `signals.session_calendar`: optional versioned session CSV (`session_calendar`), else causal 20-row trailing-median rule; `calendar_id` recorded. Fixture: 6 early + 30 later tickers keeps all sessions (old rule erased 100). |
| F04 | done | Risk exits (regime OFF, 2 consecutive ineligible/unscored months) monthly; rank exits on alpha cadence; missing rank != rank sale; `result["cadence"]` names both ("monthly decision close", not daily). |
| F05 | done | Equity through latest session <= end; completed-month returns only; `valuation_date`/`last_decision_date`/`last_execution_date`; paper NAV dated by valuation session; decisions executing after `end` stay pending. |
| F06 | done (legacy classifier) | `FeatureEngineer` target = next session open->close (known label only); `ModelTrainer.oos_predict` chronological OOS for `run_backtest`; in-sample report labelled; `evaluate_yesterday` open_to_close predicts from the prior close. |
| F07 | done | `dataset_builder.make_labels` (forward traded return + `label_end` first, unknown = NaN dropped); raw features returned; `fit_preprocessor`/`apply_preprocessor` fit per training fold (missing-column rule inside the fold). |
| F08 | done | Short/recommendation snapshot stats only on rows dated >= availability; weblink snapshot features (`stocktrade.SNAPSHOT_FEATURES`) excluded from model inputs (no history exists). |
| F09 | partial | Tabular: `classifier_v2` artifact versions schema + feature order + preprocessor + target; incompatible artifacts rejected; payload-over-history precedence; horizon H = cumulative open(t+1)->close(t+H). Sequence model (`train_sequence_model.py`/LSTM path) NOT changed. The committed `ml/model.pkl` is pre-v2 and is now rejected (legacy `recommend` silently skips the global ML blend) - retrain to use it. |
| F10 | partial | `purged_split`: date-grouped chronological split, purge of overlapping labels, threshold from purged inner OOF fold; cutoffs saved in artifact. Sequence-model folds unchanged. |
| F11 | done | `frozen_choice.json` is an append-only registry (`frozen_registry_v1`, legacy file migrated on read); same candidate set + cut + `ENGINE_ID` reuses its choice; never deleted/re-frozen; `HOLD_OUT_NOTE` labels Jul 2024+ as examined/diagnostic; `prospective_from` recorded. Thresholds unchanged. |
| F12 | done | `run_backtest_v2(schedule=...)` one continuous account switching rules each 1 July; `walk_forward_account` feeds criterion (a); spliced series kept only as `wf_spliced_diagnostic`; tie-break turnover uses data before each cut (`validation.turnover_before`). |
| F13 | partial | Trial hash = variant + resolved-config hash + `ENGINE_ID`; TFM cache manifest (horizon) checked before reuse and extended (checkpoint/context/calendar). Not done: model/checkpoint/fold hashes for the ML artifacts beyond the schema fields. |
| F14/F15/F19 | done (research) | `ml/optimizer.py`, `ml/risk_model.py`, `ml/forecasts.py`, `ml/policies.py`, `ml/policy_spec.py` - see section below. |
| F16 | partial | Engine results labelled `price_basis="adjusted_total_return_synthetic"`, `executable_quantities=False`; recommendations never emit execution orders. `ml/accounting.Ledger` implements raw-share accounting with explicit splits/dividends, but no corporate-action data source exists, so no raw-price backtest mode was built. |
| F17 | done | `metrics`: `mdd` (month-end, gate e) + `mdd_daily`; Sortino downside over all months vs 0; `sharpe` (zero-rate), `sharpe_cash_excess`, `ir`; `nav_invalid`. Initial NAV is the first month's base. |
| F18 | partial | Dated eligible-universe snapshots + unpriced list (`results/universe/`, real runs only); 1.5-pt hurdle relabelled as a sensitivity. No historical universe/delisting reconstruction (needs data). |
| F20 | done | Recent 404s retried (only > `retry_days` old cached as missing); `downloads.csv` log; `data_asic.as_of` union-index as-of join; stale SI (> `si_max_age_days`) = NaN; enabled SI screen rejects missing. Publication time itself is unknown (usable = report + lag). |
| F21 | done | Labels match tests; (g) uses the full criterion (a) on each neighbour's fixed-policy WF and later windows; DSR/PBO windows named; cost stress marked "not a gate". |
| F22 | done | Exact-setting match (no nearest-capital inheritance); `--holdings`/`holdings_file` CSV/JSON import with account hash + reconciliation vs research book; `paper_proposals` vs `execution_orders` (always empty, blockers listed); `engine_id` manifest check -> FALLBACK if validation predates the engine. |
| F23 | partial | Data freshness state (stale data -> no proposals); placeholder-open proposals labelled `estimate` and computed on a copied context (no fake fill in NAV/paper); missing optimiser forecasts -> no-trade status (not liquidation), unscored held names get neutral market forecast and no buys. Not done: TFM/ASIC cache regeneration orchestration in workflows. |
| F24 | partial | Weekly TFM: requested end respected; final period marked open at end; fixed fees in edge; recommendation sizing = backtest allocation with integer fee-aware qty; "sharpe" rank renamed `interval` (alias kept). Not migrated to the shared engine; benchmark is still close->close vs strategy open->close. |
| F25 | partial | Overlay keeps "ESTIMATED OVERLAY" label; brokerage in cost base/proceeds; ATO calendar 12-month rule (`accounting.discount_eligible`); future payments not pulled into the last month (`attrs["unpaid_tax"]`). Not done: cash re-simulation after payments, distributions, account tax-status handling. |
| F26 | done/partial | `Tuple` fixed; pyflakes clean on `ml/*.py stocktrade.py`; lazy torch; `strat-publish`/`strat-explain-config` run without market downloads; atomic writes; last-good publication kept when results are missing; broker text matches modelled fees; tier/freshness/execution state separate. Not done: dependency pinning, licence review. |
| F27 | done | `rank_ic` label = next open -> close +21 (traded horizon); no benchmark subtraction (doesn't change ranks); NW t kept. |

### Optimiser / uncapped policy (F14/F15/F19)
- Objective/constraints exactly as the spec (pre-trade-NAV units, cash residual, reserve, name limits, risk budget,
  turnover, participation, purchase-ineligible, frozen). SLSQP continuous relaxation with d = u - v and auxiliary
  |x - b| variables; convex spread + |d|^1.5 impact (impact = stress assumption, default 0) via inequality.
- Discrete stage: integer shares, opening parcel vs economic order floor, full closures always allowed, old small
  holdings kept, exact per-order fees, reserve/settlement rule, limits re-checked; deterministic
  add/drop/resize/transfer search (transfers ordered by conditional marginal contribution) from both the current
  book and the rounded relaxation, then re-solve on the active set; compared with no-trade. Labelled heuristic.
- Statuses: `proposal`, `no_trade_optimal` (valid no purchase), `invalid_input`, `no_feasible_proposal`,
  `solver_failed` (returns current book, never all-cash/sell-everything), `no_trade_forecast_unavailable`.
- `revalidate_at_execution`: sells first, buys resized down at the actual open, missing open rejected.
- Forecasts: `forecasts.calibrate` - pooled, shrunk rank->horizon-return slope + shrunk market level, training only
  on labels complete by the decision close; separate mean-estimation SE. Covariance: LedoitWolf daily x 21, PSD,
  thin/constant handling, includes held names and VAS.
- `OptPolicy` (`opt_policy_v1`): `mode=endogenous`, `holding_cap=null`, `count_target=null`,
  `sizing=covariance_cost_aware`, `weight_basis=pre_trade_total_nav`; nulls round-trip as JSON null; hash/id; legacy
  ids decode to `Variant`. Registered controlled family (cap-5 EW rank, cap-5 variable, no-cap EW subset, no-cap
  variable main, inverse-vol) + legacy N5/N8 + VAS; registered neighbours gamma x0.5/x2, name_limit x0.5.
- Gate (i) mapping for OptPolicy: `funded_capacity(capital) >= 3` (fee-aware), chosen counts reported separately.
  Promotion of OptPolicy is blocked (no tier computed; owner policy unset).
- Fixture results: 24-name A$200k case funds all 24 (2 x 84 + 22 x 83 shares, fees A$72, adverse A$498.50, cash
  A$29.50); unattractive extra candidates get zero; one/two-asset analytical references match; zero oracle gap.
- Entry point: `python stocktrade.py strat-opt-study --capitals 10000,50000` -> `results/opt_study/{compare,manifest}.json`.
  **Not run** (no market data here). Runtime on the full universe is unmeasured; it was not added to `research.yml`.

### Corrected old tests (they encoded defects)
- `test_min_parcel`: nominal n_eff 4 -> funded 3 at A$2,000 (four A$500 parcels + fees overdraw).
- `test_effective_n_and_stale`: stale "sale" at last close x 0.70 -> frozen holding marked at the haircut, no cash.
- `test_buffer_and_costs` min_order case: a disabled satellite now holds the core ETF.
- `test_timesfm_strategy`: final period now marked open at the end instead of dropped.

## Decisions
- Legacy engine stays the labelled control; quarterly variants now get monthly risk exits (behaviour change).
- Buys are planned at the decision close and resized down at the open (declared execution rule).
- Settlement in the research optimiser backtest: same-day sale proceeds fund buys (declared; `Ledger(settlement_days=0)`);
  `Problem.sale_proceeds_fund_buys=False` and `Ledger(settlement_days=2)` are available.
- `ENGINE_ID = monthly_v2_repaired_f01_f05`: old validation.json/frozen entries no longer authorise PAPER/LIVE until
  research is rerun (recommend falls back to VAS until then).
- Research presets (`OptPolicy` defaults: gamma 4, name limit 20%, participation 5% of MDV60, no reserve/risk budget)
  are labelled research presets, not owner policy and not claimed optimal.

## Blockers / needs owner
- Push/PR: not done (spec). Pushing `claude/**` triggers `research.yml` and `pages.yml`.
- Real-data study and research rerun need GitHub Actions or network access to Yahoo.
- Broker tariffs (Stake/CMC) remain UNVERIFIED; no primary-source check was possible offline.
- Raw-price/corporate-action data source for executable quantities (F16) - none free/available identified.
- Owner account policy (risk budget, reserve, core rule) is unset by design; OptPolicy cannot be promoted.

## Next actions
1. Push the branch (owner approval), let `research.yml` rerun on real data under the repaired engine, review tiers.
2. Run `strat-opt-study` on real data; check runtime (consider `max_evals`/time limit per decision) and outcomes.
3. Sequence-model F09/F10 (temporal/static ordering, purged folds); F24 migration of weekly TFM to the shared engine.
4. F25 cash re-simulation of tax payments; F13 ML artifact hashes; F18 historical universe data if a free source exists.
