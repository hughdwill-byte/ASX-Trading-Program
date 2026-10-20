"""Versioned policy specifications for the research optimiser, and the explain-config report.

`OptPolicy` fields `holding_cap` and `count_target` are `None` for the endogenous policy: None means "no preset
holding maximum / no count target" and must round-trip as JSON null (never 5, 0 or Infinity). Legacy
fixed-count ids (`RAMOM_N5_noreg_B1`...) keep decoding to `portfolio.Variant`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Dict, List, Optional

POLICY_SCHEMA = "opt_policy_v1"
LEGACY_RE = re.compile(r"^(?P<signal>[A-Z_]+?)_N(?P<n>\d+)_(?P<reg>reg|noreg)_B(?P<b>\d+)(?P<rest>(_[A-Za-z0-9.]+)*)$")


@dataclasses.dataclass(frozen=True)
class OptPolicy:
    signal: str = "RAMOM"
    mode: str = "endogenous"                  # endogenous | capped_control
    holding_cap: Optional[int] = None         # None = no preset holding maximum
    count_target: Optional[int] = None        # None = no count target
    sizing: str = "covariance_cost_aware"     # covariance_cost_aware | equal_weight | inverse_vol
    selection: str = "optimizer"              # optimizer | rank (controls)
    weight_basis: str = "pre_trade_total_nav"
    gamma: float = 4.0                        # risk aversion, horizon-return units
    name_limit: float = 0.20                  # max pre-trade-NAV weight per stock (core ETF exempt)
    cash_reserve: float = 0.0                 # fraction of pre-trade NAV
    risk_budget: Optional[float] = None       # horizon st.dev of the book; None = not set
    turnover_limit: Optional[float] = None    # one-way fraction of NAV per review; None = not set
    participation: float = 0.05               # max trade / lagged MDV60
    impact: float = 0.0                       # |d|^1.5 coefficient: a STRESS assumption, not a measured ASX cost
    horizon: int = 21                         # sessions; forecasts and covariance in the same units
    train_decisions: int = 60                 # calibration window (decisions)
    alpha_cadence: str = "M"
    core_policy: str = "none"                 # core ETF is an ordinary asset; core changes are separate experiments
    calibration_id: str = "pooled_rank_slope_shrunk_v1"
    risk_model_id: str = "ledoitwolf_daily_x_horizon_v1"
    preset: str = "research_default"          # labelled research preset, NOT an owner-approved account policy
    capital: float = 10000.0
    slip_mult: float = 1.0

    def __post_init__(self) -> None:
        for k in ("holding_cap", "count_target"):
            v = getattr(self, k)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                raise ValueError(f"{k} must be null or a positive integer, got {v!r}")
        if self.mode == "endogenous" and (self.holding_cap is not None or self.count_target is not None):
            raise ValueError("endogenous mode has no holding cap or count target (use mode='capped_control')")

    @property
    def id(self) -> str:
        cap = "nocap" if self.holding_cap is None else f"cap{self.holding_cap}"
        size = {"covariance_cost_aware": "var", "equal_weight": "ew", "inverse_vol": "ivol"}[self.sizing]
        tag = f"OPT_{self.signal}_{cap}_{size}"
        if self.selection == "rank":
            tag += "_rank"
        if self.slip_mult != 1.0:
            tag += f"_slipx{self.slip_mult:g}"
        return f"{tag}_{self.config_hash()[:6]}"

    @property
    def setting(self) -> str:
        return f"{int(self.capital)}_core0"

    def to_json(self) -> dict:
        return {"schema": POLICY_SCHEMA, **dataclasses.asdict(self)}

    @classmethod
    def from_json(cls, d: dict) -> "OptPolicy":
        if d.get("schema") != POLICY_SCHEMA:
            raise ValueError(f"unsupported policy schema {d.get('schema')!r}")
        return cls(**{k: v for k, v in d.items() if k != "schema"})

    def config_hash(self) -> str:
        body = {k: v for k, v in dataclasses.asdict(self).items() if k not in ("capital", "slip_mult")}
        return hashlib.sha1(json.dumps({"schema": POLICY_SCHEMA, **body}, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()[:12]


def controlled_family(signal: str = "RAMOM") -> List[OptPolicy]:
    """Registered controls, fixed BEFORE evaluation. The legacy corrected N5/N8 and VAS buy-and-hold come from
    portfolio.run_backtest_v2. Optional caps 8/12/20 are controls only, never the endogenous ceiling."""
    return [
        OptPolicy(signal=signal, mode="capped_control", holding_cap=5, sizing="equal_weight", selection="rank"),
        OptPolicy(signal=signal, mode="capped_control", holding_cap=5),
        OptPolicy(signal=signal, sizing="equal_weight"),
        OptPolicy(signal=signal),                                   # main candidate
        OptPolicy(signal=signal, sizing="inverse_vol", selection="rank"),
    ]


def neighbours(p: OptPolicy) -> List[OptPolicy]:
    """Registered neighbours for criterion (g) on continuous/uncapped policies (declared before evaluation):
    risk aversion halved and doubled, and the per-name limit halved."""
    return [dataclasses.replace(p, gamma=p.gamma / 2), dataclasses.replace(p, gamma=p.gamma * 2),
            dataclasses.replace(p, name_limit=p.name_limit / 2)]


def decode(policy_id_or_json):
    """Legacy integer-N ids -> portfolio.Variant; JSON dict -> OptPolicy."""
    if isinstance(policy_id_or_json, dict):
        return OptPolicy.from_json(policy_id_or_json)
    from ml.portfolio import Variant

    m = LEGACY_RE.match(policy_id_or_json)
    if not m:
        raise ValueError(f"cannot decode {policy_id_or_json!r} (OPT ids need the policy JSON from the manifest)")
    rest = m["rest"].split("_")[1:]
    return Variant(signal=m["signal"], n=int(m["n"]), regime=m["reg"] == "reg", buffer_mult=int(m["b"]),
                   si_screen="SI" in rest, rebalance="Q" if "Q" in rest else "M", tax_aware="TAX" in rest)


# ---- explain-config --------------------------------------------------------------------------

# key: (units, range, category, status note). status: active | shadowed | inactive.
REGISTRY: Dict[str, tuple] = {
    "capital": ("AUD", ">0", "account", "active: picks the validation setting; must match a researched setting exactly"),
    "core_weight": ("fraction", "[0,1]", "account", "active for legacy controls; must match a researched setting"),
    "core_ticker": ("ticker", "", "data", "active: core ETF and total-return benchmark"),
    "signal": ("name", "", "strategy", "shadowed: the grid / frozen variant decides the signal"),
    "n_positions": ("count", ">=1", "strategy", "shadowed: the grid / frozen variant decides N (legacy only)"),
    "buffer_mult": ("x N", ">=1", "strategy", "shadowed: the grid / frozen variant decides the buffer"),
    "regime_filter": ("bool", "", "strategy", "shadowed: the grid / frozen variant decides the regime flag"),
    "regime_band": ("fraction", "[0,0.2]", "strategy", "active (legacy regime variants)"),
    "regime_sma": ("sessions", ">=20", "strategy", "active (legacy regime variants)"),
    "min_position": ("AUD", ">=0", "owner", "active: legacy slot floor and fee-aware capacity"),
    "min_order": ("AUD", ">=0", "owner", "active: economic order floor"),
    "min_parcel": ("AUD", ">=500", "factual", "active: ASX minimum marketable parcel for opening positions"),
    "broker": ("name", "flat|stake|cmc", "factual", "active: fee schedule (UNVERIFIED; check broker sites)"),
    "brokerage": ("AUD/order", ">=0", "factual", "active only with broker: flat"),
    "slippage_tiers": ("fraction/side", "", "assumption", "active: adverse execution by liquidity tier"),
    "slippage_mdv_high": ("AUD", "", "assumption", "active"),
    "slippage_mdv_mid": ("AUD", "", "assumption", "active"),
    "cost_stress_mult": ("x", ">=1", "assumption", "active: printed stress metric only, not a gate"),
    "stale_days": ("sessions", ">=1", "data", "active: holdings without a price this long are frozen and haircut"),
    "stale_haircut": ("fraction", "[0,1]", "data", "active: valuation haircut only (no cash)"),
    "cash_rate_annual": ("fraction/yr", ">=0", "assumption", "INACTIVE in legacy engine (cash earns 0); used as r_cash by the optimiser"),
    "core_drift_band": ("fraction", "[0,1]", "owner", "active (legacy core variants, July review)"),
    "asic_lag_days": ("business days", ">=0", "data", "active: SI usable date"),
    "si_screen": ("bool", "", "strategy", "shadowed: grid variant decides"),
    "si_max": ("fraction", "[0,1]", "strategy", "active for SI variants; missing SI fails the screen"),
    "universe_size": ("count", ">=1", "data", "active: liquid candidate universe (not a holding count)"),
    "min_history_days": ("sessions", ">=1", "data", "active"),
    "min_presence": ("fraction", "[0,1]", "data", "active"),
    "min_price": ("AUD", ">=0", "data", "active"),
    "min_mdv60": ("AUD/day", ">=0", "data", "active"),
    "max_zero_volume_days": ("sessions/20", ">=0", "data", "active"),
    "max_daily_move": ("fraction", ">0", "data", "active: bad-data filter"),
    "exclude_tickers": ("list", "", "data", "active"),
    "start": ("date", "", "research", "active: first decision date"),
    "tax_defer_days": ("days", "", "strategy", "active for _TAX variants"),
    "tax_defer_exit_pct": ("fraction", "", "strategy", "active for _TAX variants"),
    "tax_defer_max_dd": ("fraction", "", "strategy", "active for _TAX variants"),
    "tax_rate": ("fraction", "[0,0.47]", "owner", "active: estimated after-tax overlay only"),
    "session_calendar": ("path", "", "data", "active when set: versioned ASX session file"),
    "holdings_file": ("path", "", "account", "active for strat-recommend: actual holdings/cash import"),
    "owner_policy": ("dict", "", "owner", "UNSET by default: blocks executable orders/promotion for OPT policies"),
}


def explain_config(defaults: dict, yaml_strategy: Optional[dict], frozen: Optional[dict] = None) -> Dict[str, object]:
    yaml_strategy = yaml_strategy or {}
    rows = []
    for key in sorted(set(defaults) | set(yaml_strategy) | set(REGISTRY)):
        units, rng, cat, note = REGISTRY.get(key, ("?", "", "unknown", "unregistered key: no code reads it"))
        origin = "config.yaml" if key in yaml_strategy else "default" if key in defaults else "unset"
        value = yaml_strategy.get(key, defaults.get(key))
        status = note.split(":")[0].split(" ")[0].lower()
        rows.append({"key": key, "value": value, "origin": origin, "units": units, "range": rng, "category": cat,
                     "status": status, "note": note})
    diffs = []
    if frozen:
        for setting, f in frozen.items():
            if isinstance(f, dict) and f.get("variant"):
                diffs.append({"setting": setting, "frozen_variant": f["variant"],
                              "note": "the frozen variant, not strategy.signal/n_positions/buffer_mult/regime_filter, "
                                      "drives strat-recommend"})
    return {"schema": "explain_config_v1", "rows": rows, "frozen": diffs}
