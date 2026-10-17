# Strategy v2 implementation notes

Written before coding, per the research spec (Part B). Deviations are listed at the end.

## How the code works today

**Panel storage.** `ingest` downloads Yahoo daily bars for every symbol from the ASX company
directory (fallback: `available_symbols.json`, 1,848 tickers) into `data/{TICKER}_ohlcv.csv`
(columns `date, open, high, low, close, adj_close, volume, symbol`; ticker without the `.AX`
suffix; `close` unadjusted, `adj_close` split- and dividend-adjusted). `ml.timesfm_strategy.load_panel`
pivots these into wide date x ticker frames `open, close, adj_close, volume` plus
`adj_open = open * adj_close / close`. `data/` is not committed; on GitHub Actions it is rebuilt
every run except `data/timesfm_cache` (restored with `actions/cache`).

**Index series.** `load_benchmark` downloads `^AORD` (Yahoo `auto_adjust=True`, a price index
with no dividends) and caches it as `data/benchmark_AORD.csv`.

**latest.json.** `stocktrade.py tfm-publish` -> `ml.timesfm_strategy.publish` writes
`docs/data/latest.json` (picks, weekly TimesFM backtest vs universe and ^AORD, momentum
baseline, recent trades, settings). `docs/index.html` fetches it client-side.

**Daily workflow (`pages.yml`).** Install CPU torch + requirements (~40 s), restore cache, self-check
(2 s), `ingest` (~3.5 min), `tfm-publish` (~8 min cold, faster with cache), upload `docs/`,
deploy on `main`. Total ~13 min cold.

## What was built

| Spec section | Where |
|---|---|
| 2 Universe (`eligibility_v2`) | `ml/signals.py` `Context.eligible` (base filters precomputed for all dates) |
| 3 Signals | `ml/signals.py` (`MOM_12_1`, `VOL_252`, `VOL_63`, `RAMOM`, `LOWVOL`, `COMBO`, `TFM_REL`, `TFM_RAW`, SI) |
| 4 Engine (`run_backtest_v2`) | `ml/portfolio.py` |
| 5-6 Grid, baselines, validation | `ml/strategy_cli.py` (`strat-backtest`), `ml/validation.py` |
| 1.4 ASIC | `ml/data_asic.py` (`asic-download`) |
| 7 TimesFM cache | `stocktrade.py tfm-cache`, `.github/workflows/research.yml` |
| 8-9 Tiers, outputs, dashboard | `strat-recommend`, `strat-publish`, `docs/index.html` |
| 10 Tests | `ml/test_strategy_v2.py` (8 test functions covering the 12 checks, ~1.5 s) |
| 11 Workflows | `pages.yml` (daily), `research.yml` (monthly + manual) |

## Deviations from the spec

1. **ETF exclusion**: `exclude_tickers` plus a Yahoo `instrumentType` lookup for every ticker that ever passes the
   liquidity filters (anything not EQUITY is dropped). Results are cached in `results/instrument_types.json`, so only
   new tickers are looked up. (The first version relied on the config list alone and let IVV, IOO, IAA, IEM, MVW and
   QOZ into the universe.)
2. **Presence test** (A on 95% of the last 252 sessions) uses the panel's own trading calendar
   (days when at least 30% of the usual number of stocks traded) rather than the ^AORD calendar.
3. **ASIC URL**: the ASIC index page could not be inspected from the development sandbox (network blocked).
   The downloader uses the file pattern already in `update_short_positions_csv.py`; `coverage()` decides whether
   the history is complete enough (first file by 2016-03-31 and 80%+ of business days). Otherwise SI variants are skipped.
4. **ASX delisted-entities page** is not archived (needs HTML scraping of an unverified page). The forward-looking
   log of vanished tickers is implemented (`results/survivorship/vanished.csv`, committed by the daily job).
5. **TimesFM cache** is CSV (`data/tfm_cache/{rel,raw}_YYYY-MM-DD.csv`), not parquet, to avoid a pyarrow dependency.
   `forecast_on` was left unchanged; the new `signals.tfm_cache_scores` does the transformed/horizon-21 forecast and
   `load_timesfm` gained a `positive` flag (log-relative series can be negative).
6. **Daily TimesFM**: the legacy weekly `tfm-publish` still runs daily (labelled LEGACY on the dashboard; cached
   forecasts keep it to a few minutes). Today's TFM scores are not computed by `strat-recommend`, so if a TFM
   variant ever reaches LIVE/PAPER it needs a daily `tfm-cache` step added first.
7. **LEGACY baseline** is not re-run over 2016-2026 (no weekly forecast history); its live six-month result stays in
   its own dashboard section.
8. **Core/satellite July rebalance** tops up the core only from satellite cash (no forced stock sales); an
   overweight core is trimmed into satellite cash.
9. **Share counts** are integers computed on adjusted prices, so historical position sizes are approximate.
10. **N=5 vs N=8 at A$10k** are identical (effective N = 4 at the A$2,500 floor) but still counted as separate
    trials, which is conservative for the deflated Sharpe ratio.
11. **Trial count** for DSR = unique config hashes per capital setting in `results/trials_log.csv`, so repeated
    research runs do not inflate it.
12. **Next-open orders** come from simulating the frozen variant from 2016 to today (a paper portfolio), not from
    the user's real holdings.
13. `results/frozen_choice.json` is deleted and re-written each research run; the hold-out gate checks that the file
    predates every hold-out evaluation in that run.

14. **After-tax estimate** (`after_tax_monthly`): FIFO lots, losses offset non-discount gains first, 50% discount
    after 12 months, tax paid each 31 October at `tax_rate` (default 32%). Dividend tax and franking are ignored for
    both the strategies and VAS. Shown as a toggle on the dashboard; not used for selection or validation.

## Persistence

- Research results (`results/*.json`, `*.csv`) are committed to `main` by `research.yml`.
- Monthly forecast cache, ASIC reports and model weights live in the Actions cache (`research-*` key).
- Paper-tracking and delisting logs are committed by the daily job (`pages.yml` has `contents: write`).
