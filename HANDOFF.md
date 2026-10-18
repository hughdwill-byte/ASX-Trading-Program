# Handoff: next improvements (strategy v3)

For a fresh Claude session. Read this whole file before changing anything. The owner may also attach
`ASX_Program_Improvement_Research.txt` (an AI-generated research report). Use it for background only:
**where it disagrees with this file, this file wins.** Some of its tickers, file paths and numbers are wrong
(see "Do not build" below).

## Current state (October 2026)

- Repo `hughdwill-byte/ASX-Trading-Program`; work on branch `claude/kind-mccarthy-qbr8wd`, merge to `main` via PR.
- Dashboard: https://hughdwill-byte.github.io/ASX-Trading-Program/ (`docs/index.html`, data in `docs/data/latest.json`).
- Workflows: `.github/workflows/pages.yml` (daily site build) and `research.yml` (monthly research; also on push to
  `claude/**` and manual dispatch). Research with warm caches takes ~7–10 min; results are committed to `results/` on main.
  A push to `claude/**` starts both workflows on the branch; cancel those if you merge straight away.
- The sandbox cannot reach Yahoo or Hugging Face. Test locally with synthetic data and run real data on GitHub Actions.
- Self-checks (run before every push): `python3 ml/test_strategy_v2.py` and `python3 ml/test_timesfm_strategy.py`,
  plus `python3 -m pyflakes ml/portfolio.py ml/strategy_cli.py ml/signals.py ml/test_strategy_v2.py`.
- Key code:
  - `ml/signals.py` `Context`: universe, plus signals `RAMOM`, `LOWVOL`, `COMBO`, `TFM_*`, built in `_build()`.
    `TRADEABLE_SIGNALS` drives the grid.
  - `ml/portfolio.py`:
    - `DEFAULTS` (brokerage 6.50 per order, `min_order` 1000, `min_position` 2500);
    - `Variant` dataclass, whose `id` is e.g. `RAMOM_N5_noreg_B1`;
    - `run_backtest_v2` (monthly loop: sells → trims → buys; orders fill at the next open);
    - `after_tax_monthly` (FIFO CGT estimate).
  - `ml/strategy_cli.py`:
    - `SETTINGS` capital list, `grid()` variants, `_validate_setting` (neighbours, DSR, PBO, acceptance);
    - `_write_history`;
    - `instrument_types` (ETF exclusion cache in `results/instrument_types.json`).
  - `ml/validation.py` `acceptance()`: nine tests (a–i) → LIVE, PAPER or FALLBACK. **Do not loosen these tests.**
  - `docs/index.html`: `strategyLabel()` parses variant ids (regex on `SIGNAL_N#_reg|noreg_B#`); `SIGNAL_NAMES`; `settingLabel()`.
- Latest results (ETFs excluded), Apr 2016–Sep 2026, VAS ×2.58:
  - The best strategy is `RAMOM_N5_noreg_B1` at A$50k: ×5.25 pre-tax, ×3.49 after tax, −38% max drawdown, DSR 0.72, PBO 0.35.
  - Every setting is FALLBACK.
  - Biggest drags: about 50 trades a year, so almost no gains get the 50% CGT discount; cost drag about 2.2% a year; fragile to rule changes.

## Build these, in order (one PR each; run the self-checks; merge; dispatch research.yml on main)

### 1. Broker fee models, more capital levels, A$500 minimum parcel (small)
- `ml/portfolio.py`: add `brokerage_fee(cfg, side, value) -> float` and use it inside `trade()` and `buy_amount()`.
  These currently use the flat `cfg["brokerage"]`. `buy_amount` must solve for qty with the fee for that order size
  (iterate once: estimate qty with the fee at `amount`, then recompute).
  - `"flat"`: `cfg["brokerage"]` (6.50), the old behaviour.
  - `"stake"`: 3.00 if value ≤ 30,000, else 0.0001 × value.
  - `"cmc"`: BUY and value ≤ 1,000 → 0.00; otherwise max(11.00, 0.001 × value). (The engine places at most
    one buy per ticker per day, so the "first buy of the day" condition always holds.)
  - These fees come from the research report and could not be verified. Add a comment telling the owner to check
    hellostake.com/au/pricing and cmcmarkets.com/en-au/stockbroking/pricing.
- New `DEFAULTS` / `config.example.yaml` keys: `broker: "stake"`, `min_parcel: 500.0`. Set `min_position` default
  to 500.0 and `min_order` to 500.0. These two drive `n_eff = min(n, sat_capital // min_position)`.
- `SETTINGS` in `ml/strategy_cli.py` becomes:
  `[(500,0),(2000,0),(5000,0),(5000,0.7),(10000,0),(10000,0.6),(25000,0),(50000,0),(50000,0.6)]`.
  - The 5000/0.7 setting is the report's 70% VAS + 30% stocks.
  - Settings with fewer than 3 positions already fall back to VAS through `skipped_reason` and test (i). Keep that.
- Keep a cost-stress run: it already multiplies slippage. Also scale brokerage by `cost_stress_mult`.
- Tests in `ml/test_strategy_v2.py`:
  - fee values for each model at 500 / 1,000 / 1,001 / 30,000 / 40,000;
  - a small-capital backtest (A$2,000, 5 stocks) gives `n_eff == 4`, and no buy is under `min_parcel`.
- Dashboard: the capital selector lists the new settings automatically. Show the broker name in the settings section.

### 2. Tax-aware holding and quarterly rebalancing (medium)
- New `Variant` fields:
  - `rebalance: str = "M"`; "Q" means trade only on decisions in March, June, September and December.
    In other months, still run the stale-price exits and still append the monthly snapshot, so `monthly` stays monthly.
  - `tax_aware: bool = False`.
  - Id suffixes: `_Q` and `_TAX`. Update `strategyLabel()` (add "quarterly" and "tax-aware") and the `id` property.
- Tax-aware rule in the sell loop (rank sells only; never block stale, regime or ineligible exits):
  - track `entry_px[t]` (fill price) alongside `entry[t]`;
  - if `330 <= (date_j - entry) days < 365` and `px_i[t] > entry_px[t]` and
    `rank <= tax_defer_exit_pct * len(ranks)` and `px_i[t] >= entry_px[t] * (1 - tax_defer_max_dd)`, then
    **defer** the sale to a later month;
  - config: `tax_defer_days: 35`, `tax_defer_exit_pct: 0.8`, `tax_defer_max_dd: 0.15`.
  - (The drawdown check is redundant while price > entry; keep it for clarity.)
- Grid additions: `RAMOM_N5_noreg_B1_Q`, `RAMOM_N5_noreg_B1_TAX`, `RAMOM_N5_noreg_B1_Q_TAX`, and the same three for
  `RAMOM_N8_noreg_B3`. More variants means more trials, so DSR gets stricter. That is intended.
- Selection stays on the pre-tax metrics. Also put `after_tax_monthly` CAGR in `strategy_compare.json` as `cagr_after_tax`
  for every engine row, and add a sortable "After tax" column to the comparison table.
- Tests:
  - a deferral case: held 340 days, in profit, rank just outside K → not sold; at 366 days → sold;
  - a loss case: the same situation at a loss → sold immediately;
  - quarterly: no orders in non-quarter months except stale exits.

### 3. Residual (market-neutral) momentum signal `RESMOM` (medium; watch for look-ahead)
- In `Context._build()`, with `r` = daily log returns (already computed) and `m = r[core]` (VAS):
  - `beta = r.rolling(756, min_periods=500).cov(m) / m.rolling(756, min_periods=500).var()`, then `beta = beta.shift(1)`
    (beta known before day t);
  - `e = r - beta.mul(m, axis=0)` (daily residual return);
  - `self.RESMOM = e.shift(21).rolling(231, min_periods=200).sum() / (e.shift(21).rolling(231, min_periods=200).std() * sqrt(231))`.
- Add `RESMOM` to `SIGNALS`, `TRADEABLE_SIGNALS`, the `scores()` branch with `RAMOM`, and `SIGNAL_NAMES` in
  `docs/index.html` ("Residual momentum"). Also add `RESMOM_N5_noreg_B1` to `grid()`.
- Market only: there are no ASX sector ETF tickers in the data. The report's `XJR.AX` and `XFJ.AX` do not exist.
- Tests:
  - extend `test_no_lookahead`: changing prices after row i must not change `RESMOM` at row i;
  - on synthetic data where stock = 1.5 × market + noise, `RESMOM` is uncorrelated with market return (|corr| < 0.1).

After each PR: dispatch `research.yml` on main, then `pages.yml`, and report pre-tax and after-tax growth versus VAS
plus each setting's tier.

## Do not build (decided)
- **Gross profitability / quality screen.** Yahoo has only about 4 years of annual financials, so 2016–2026 backtests would use
  figures that were not known at the time. Revisit only with point-in-time fundamentals, which need a paid source and owner approval.
- **ASIC short-interest filter.** It already exists (`si_screen`, `RAMOM_N5_reg_B3_SI`) and made results worse.
- **After-tax reporting.** It already exists (`after_tax_monthly`, the dashboard Tax toggle).
- **LightGBM ranker, polars, "6 h → 45 min" workflow work.** Large, unproven on this data, and runs already take about 10 minutes.
- **Deleting TimesFM.** Keep it; it is cached and cheap, and the grid simply shows it losing.
- The report's file names (`tests/…`, `daily_build.yml`, `monthly_research.yml`) do not exist. Use the real ones above.

## Rules
- Do not loosen `acceptance()` or the hold-out gate. The dashboard recommends a strategy only if it passes.
- Stop and ask the owner before adding a dependency, a paid data source, or anything that needs a sign-up.
- Do not commit `results/` or `docs/data/` from local runs (synthetic data). The workflows commit real results.
- Commit messages end with the attribution lines the session provides.
