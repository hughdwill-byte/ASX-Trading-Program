# Implementation status (ASX_IMPLEMENT.md)

Spec: `ASX_IMPLEMENT.md` (audit 2026-10-20, audited commit 4f6ccbe). Checkout base: 613dd5d (adds RESMOM on top of the
audited commit; nothing in it resolves an F-item). Branch: `claude/laughing-hawking-c5e85r`.

## Checks (run all before every push)

```sh
python3 ml/test_strategy_v2.py
python3 ml/test_timesfm_strategy.py
python3 ml/test_engine_repairs.py
python3 -m pyflakes ml/portfolio.py ml/strategy_cli.py ml/signals.py ml/test_strategy_v2.py ml/test_engine_repairs.py
```

Environment note: the container lacked scipy/scikit-learn/pyflakes although `requirements.txt` lists scipy-dependent
code; installed with pip (no new project dependency).

## Completed

### Stage 1 — engine correctness (ml/portfolio.py, ml/signals.py)
- F01 `affordable_qty`: largest integer quantity whose debit (fill incl. adverse slippage + brokerage at that exact
  value) fits min(order budget, sleeve cash); handles fee discontinuities (CMC A$1,000, Stake A$30,000). Parcel
  round-up beyond the budget removed (a sub-parcel order is rejected, never forced). `trade()` asserts no overdraft.
- F01/(i) `funded_capacity`: fee-aware capacity = sat capital // (floor x (1 + worst stock slippage) + buy fee).
  Results report `n_nominal` (old `capital // floor`) and `n_eff` (funded). A$1,500 nominal-3 -> 2 (disabled);
  A$4,000 nominal-8 -> 7; A$2,000 nominal-4 -> 3.
- F02: no fill from a previous close: missing execution open = order `rejected` (recorded in `result["rejected"]`),
  no cash. Orders carry `decision_date`, `requested`, `status` (filled/partial). Buys are planned at the decision
  close and re-sized down at the open (declared rule) - tomorrow's open is no longer used to size today's request.
  Stale holdings are frozen on the book and marked at last close x (1 - stale_haircut); no sale cash.
- F03 `session_calendar`: optional versioned session CSV (`session_calendar` cfg key), else causal fallback
  (count >= 30% of the median of the previous 20 rows). `ctx.calendar_id` recorded.
- F04: risk exits (regime OFF, 2 consecutive ineligible/unscored months) run at every monthly decision; rank exits
  only on the alpha cadence. Missing rank no longer equals an immediate rank sale. `result["cadence"]` names both.
- F05: equity valued through the latest session <= end; `monthly` = completed month-end returns only;
  `valuation_date`, `last_decision_date`, `last_execution_date` returned. Decisions whose execution falls after
  `end` stay pending.
- F16 (partial): results carry `price_basis="adjusted_total_return_synthetic"`, `executable_quantities=False`.
- Enabled SI screen now rejects missing short interest instead of silently passing (F20 part).

Old tests corrected (they encoded defects): `test_min_parcel` (nominal n_eff 4 -> funded 3),
`test_effective_n_and_stale` (fictitious stale sale -> frozen holding), `test_buffer_and_costs` min_order case
(disabled satellite now holds the core ETF).

## Next
Stage 2: accounting ledger, risk model, optimiser (F14/F15/F19). Then policy spec / CLI, then F11/F20/F22/F23/F17/F21.
