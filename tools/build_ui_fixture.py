"""Build a SYNTHETIC demo site for browser checks: synthetic prices -> strat-backtest -> strat-recommend ->
strat-publish into <out>/site (flagged demo). Never writes results/ or docs/data/.

Usage: python tools/build_ui_fixture.py <out-dir> && node tools/ui_check.mjs <out-dir>/site
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np, pandas as pd
from ml import strategy_cli as sc
from ml.portfolio import DEFAULTS
from ml.signals import Context
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(9)
idx = pd.bdate_range("2015-06-01", "2026-09-30")
mkt = rng.normal(0.0003, 0.008, len(idx))
cols = {f"T{i:02d}": 20 * np.exp(np.cumsum(mkt + rng.normal(0.00005 * (i % 5), 0.013, len(idx)))) for i in range(14)}
cols["VAS"] = 50 * np.exp(np.cumsum(mkt))
adj = pd.DataFrame(cols, index=idx); op = adj.shift(1).fillna(adj.iloc[0])
panel = {"adj_close": adj, "close": adj, "open": op, "adj_open": op, "volume": adj*0+2e6}
cfg = dict(DEFAULTS, min_mdv60=0.0, exclude_tickers=[], min_history_days=260, min_presence=0.0, broker="stake")
ctx = Context(panel, None, cfg); ctx.tfm_dir = out
sc.RESULTS, sc.SETTINGS = out, [(2000.0, 0.0), (25000.0, 0.0), (50000.0, 0.6)]
sc.strat_backtest(ctx, cfg, SimpleNamespace(end=None))
# recommendation (force PAPER on 25000 to exercise allocation/proposals), then publish into site/data
v = json.loads((out / "validation.json").read_text())
v["settings"]["25000_core0"]["tier"] = "PAPER"
(out / "validation.json").write_text(json.dumps(v))
cfg2 = dict(cfg, capital=25000.0, core_weight=0.0)
sc.strat_recommend(ctx, cfg2, SimpleNamespace(capital=None), today=ctx.dates[-1])
site = out / "site"; (site / "data").mkdir(parents=True, exist_ok=True)
import shutil; shutil.copy(pathlib.Path(__file__).resolve().parents[1] / "docs" / "index.html", site / "index.html")
lat = site / "data" / "latest.json"; lat.write_text(json.dumps({"demo": True}))
sc.strat_publish(cfg2, results_dir=out, out=lat)
