from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
import pathlib
import time
import sys
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
import hashlib
from threading import Lock

import numpy as np
import pandas as pd
import requests
import io
from pandas.tseries.offsets import BDay
from pandas.api.types import is_numeric_dtype
from requests.exceptions import HTTPError

try:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, roc_auc_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.calibration import CalibratedClassifierCV
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "scikit-learn is required. Install with `pip install scikit-learn`."
    ) from exc

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency
    yf = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None

try:
    from colorama import Fore, Style, init as colorama_init
except ImportError:  # pragma: no cover - optional dependency
    Fore = None
    Style = None
    colorama_init = None

try:  # pragma: no cover - optional dependency
    from ml.predict import predict_next_day  # type: ignore
except ImportError:
    predict_next_day = None  # type: ignore

if TYPE_CHECKING:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SentimentAnalyzerType
else:  # pragma: no cover - runtime only
    SentimentAnalyzerType = Any


# -------- Configuration -----------------------------------------------------------------


@dataclasses.dataclass
class DataSourceConfig:
    eodhd_api_key: str
    alpha_vantage_key: Optional[str] = None
    base_path: pathlib.Path = pathlib.Path("data")
    symbols: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    start_date: str = "2015-01-01"
    end_date: Optional[str] = None
    cache: bool = True
    exchange: str = "AU"
    allowed_security_types: Tuple[str, ...] = ("Common Stock",)
    asx_directory_url: str = (
        "http://www.asx.com.au/asx/research/ASXListedCompanies.csv"
    )
    data_source: str = "yfinance"
    yfinance_suffix: str = ".AX"

    def __post_init__(self) -> None:
        if not isinstance(self.base_path, pathlib.Path):
            self.base_path = pathlib.Path(self.base_path)
        if isinstance(self.symbols, str):
            tokens = self.symbols.strip()
            if tokens.lower() == "all":
                self.symbols = tuple()
            else:
                split_symbols = [part.strip().upper() for part in tokens.split(",") if part.strip()]
                self.symbols = tuple(split_symbols)
        if isinstance(self.symbols, list):
            formatted = []
            for symbol in self.symbols:
                text = str(symbol).strip().upper()
                if text:
                    formatted.append(text)
            self.symbols = tuple(formatted)
        if isinstance(self.allowed_security_types, list):
            self.allowed_security_types = tuple(self.allowed_security_types)


@dataclasses.dataclass
class FeatureConfig:
    rsi_window: int = 14
    short_ma: int = 5
    long_ma: int = 20
    bollinger_window: int = 20
    bollinger_std: float = 2.0


@dataclasses.dataclass
class ModelConfig:
    algorithm: str = "hist_gradient_boosting"
    n_estimators: int = 250
    max_depth: Optional[int] = 8
    max_leaf_nodes: Optional[int] = 31
    min_samples_leaf: int = 10
    learning_rate: float = 0.05
    subsample: float = 1.0
    test_splits: int = 5
    prediction_threshold: float = 0.55
    retrain_weeks: int = 1
    random_state: int = 42
    scale_features: bool = False
    calibrate: bool = False
    calibration_method: str = "sigmoid"
    calibration_cv: int = 3
    scoring_metric: str = "roc_auc"
    class_weight: Optional[str] = "balanced"
    trading_mode: str = "standard"
    dynamic_threshold: bool = True
    dynamic_threshold_quantile: float = 0.85
    dynamic_threshold_min: float = 0.5
    dynamic_threshold_max: float = 0.95
    volatility_threshold: float = 0.015
    volatility_threshold_adjust: float = 0.0
    day_of_week_adjustments: Tuple[float, ...] = dataclasses.field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0)
    )
    hyperparameter_grid: Tuple[Dict[str, Any], ...] = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if isinstance(self.day_of_week_adjustments, list):
            self.day_of_week_adjustments = tuple(float(x) for x in self.day_of_week_adjustments)
        if isinstance(self.hyperparameter_grid, list):
            converted: List[Dict[str, Any]] = []
            for item in self.hyperparameter_grid:
                if isinstance(item, dict):
                    converted.append(dict(item))
            self.hyperparameter_grid = tuple(converted)


@dataclasses.dataclass
class RiskConfig:
    max_positions: int = 5
    allocation_per_trade: float = 0.02
    min_cash_buffer: float = 0.15
    max_volume_multiple: float = 3.0
    max_spread_ratio: float = 0.03
    restrict_open_minutes: int = 15
    restrict_close_minutes: int = 15
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.5
    min_liquidity_ratio: float = 0.5


@dataclasses.dataclass
class FilterConfig:
    min_trading_days: int = 750
    min_price: float = 0.25
    min_avg_dollar_volume: float = 500_000.0
    avg_volume_window: int = 60
    enabled: bool = True


@dataclasses.dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    trade_cost: float = 6.5
    slippage_bps: float = 10.0  # basis points
    report_path: pathlib.Path = pathlib.Path("reports") / "backtest.json"

    def __post_init__(self) -> None:
        if not isinstance(self.report_path, pathlib.Path):
            self.report_path = pathlib.Path(self.report_path)


@dataclasses.dataclass
class ExecutionConfig:
    broker: str = "paper"
    api_host: Optional[str] = None
    api_key: Optional[str] = None


@dataclasses.dataclass
class AppConfig:
    data: DataSourceConfig
    features: FeatureConfig = dataclasses.field(default_factory=FeatureConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    risk: RiskConfig = dataclasses.field(default_factory=RiskConfig)
    filters: FilterConfig = dataclasses.field(default_factory=FilterConfig)
    backtest: BacktestConfig = dataclasses.field(default_factory=BacktestConfig)
    execution: ExecutionConfig = dataclasses.field(default_factory=ExecutionConfig)

    @staticmethod
    def load(path: pathlib.Path) -> "AppConfig":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() in {".yaml", ".yml"}:
                if yaml is None:
                    raise RuntimeError("Install PyYAML to load YAML config files.")
                raw = yaml.safe_load(handle)
            else:
                raw = json.load(handle)
        return AppConfig(
            data=DataSourceConfig(**raw["data"]),
            features=FeatureConfig(**raw.get("features", {})),
            model=ModelConfig(**raw.get("model", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            filters=FilterConfig(**raw.get("filters", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            execution=ExecutionConfig(**raw.get("execution", {})),
        )


# -------- Utilities ---------------------------------------------------------------------


def project_path(*parts: str) -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parent
    return root.joinpath(*parts)


def ensure_directory(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv_if_exists(path: pathlib.Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"], index_col="date")
    return None


def to_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    ensure_directory(path.parent)
    df.to_csv(path, index=True, date_format="%Y-%m-%d")


def dataclass_to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: dataclass_to_serializable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: dataclass_to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [dataclass_to_serializable(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def rolling_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=window, adjust=False).mean()
    roll_down = down.ewm(span=window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def trading_minutes(timestamp: pd.Timestamp) -> int:
    market_open = timestamp.normalize() + dt.timedelta(hours=10) + dt.timedelta(minutes=0)
    delta = timestamp - market_open
    return max(0, int(delta.total_seconds() // 60))


def highlight_text(text: str, highlight: bool) -> str:
    if not highlight:
        return text
    if Fore and Style:
        return f"{Fore.GREEN}{text}{Style.RESET_ALL}"
    return f"{text} (ASX200)"


WEBLINK_CACHE_TTL_SECONDS = 24 * 60 * 60  # refresh once per day
WEBLINK_RETRY_ATTEMPTS = 3


def _is_cache_valid(path: pathlib.Path, ttl: int) -> bool:
    if ttl <= 0 or not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age <= ttl


def load_json_file(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_file(data: Any, path: pathlib.Path) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        if math.isnan(val):
            return None
        return val
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def fetch_json_with_cache(
    session: requests.Session,
    url: str,
    cache_path: pathlib.Path,
    ttl: int = WEBLINK_CACHE_TTL_SECONDS,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    if ttl > 0 and _is_cache_valid(cache_path, ttl):
        try:
            return load_json_file(cache_path)
        except Exception as err:
            logging.debug("Failed to read cached WebLink payload from %s: %s", cache_path, err)

    last_error: Optional[Exception] = None
    for attempt in range(WEBLINK_RETRY_ATTEMPTS):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            save_json_file(payload, cache_path)
            return payload
        except Exception as exc:
            last_error = exc
            if isinstance(exc, HTTPError):
                status = getattr(exc.response, "status_code", None)
                if status == 404:
                    logging.debug("WebLink returned 404 for %s; returning empty payload.", url)
                    if cache_path.exists():
                        try:
                            return load_json_file(cache_path)
                        except Exception as cache_err:  # pragma: no cover - defensive
                            logging.debug("Failed to load cached 404 payload from %s: %s", cache_path, cache_err)
                    return {}
                if status in {400, 401, 403, 422}:
                    logging.warning("WebLink request for %s failed with status %s; not retrying.", url, status)
                    break
            sleep_seconds = min(5, 2 ** attempt)
            logging.debug(
                "WebLink request failed (attempt %s/%s) for %s: %s; retrying in %s s",
                attempt + 1,
                WEBLINK_RETRY_ATTEMPTS,
                url,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    if cache_path.exists():
        logging.warning("Using stale WebLink cache for %s due to error: %s", url, last_error)
        return load_json_file(cache_path)

    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to fetch WebLink data for {url} with no cached fallback.")

# -------- Data Ingestion ----------------------------------------------------------------


class DataIngestor:
    def __init__(self, config: DataSourceConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.cache_ttl = WEBLINK_CACHE_TTL_SECONDS
        self.exclusions: Dict[str, str] = {}
        self._load_exclusions()
        self.latest_skipped: List[Dict[str, str]] = []
        ensure_directory(project_path(str(config.base_path)))

    def _asx_directory_cache_path(self) -> pathlib.Path:
        return project_path(self.config.base_path, "asx_directory.csv")

    def _exclusions_path(self) -> pathlib.Path:
        return project_path(self.config.base_path, "yfinance_exclusions.json")

    def _load_exclusions(self) -> None:
        path = self._exclusions_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.exclusions = {k.upper(): str(v) for k, v in data.items()}
                elif isinstance(data, list):
                    self.exclusions = {str(item).upper(): "no_data" for item in data}
            except Exception as err:  # pragma: no cover - defensive
                logging.warning("Failed to load exclusions file %s: %s", path, err)

    def _save_exclusions(self) -> None:
        path = self._exclusions_path()
        ensure_directory(path.parent)
        try:
            path.write_text(json.dumps(self.exclusions, indent=2), encoding="utf-8")
        except Exception as err:  # pragma: no cover - defensive
            logging.warning("Failed to save exclusions file %s: %s", path, err)

    def fetch_short_positions(
        self,
        eval_date: Optional[pd.Timestamp] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        base_dir = project_path(self.config.base_path, "short_positions")

        def load_from_path(path: pathlib.Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
            if not path.exists():
                return {}, {}
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as err:  # pragma: no cover - defensive
                logging.warning("Failed to load short-position data (%s): %s", path, err)
                return {}, {}

            entries: Iterable[Any]
            payload_date: Optional[str] = None
            if isinstance(payload, dict):
                entries = payload.get("entries") or payload.get("short_positions") or payload.get("data") or []
                metadata = payload.get("metadata", {})
                payload_date = payload.get("date")
                if payload_date:
                    metadata.setdefault("date", payload_date)
            else:
                entries = payload
                metadata = {}

            result_map: Dict[str, Dict[str, Any]] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                symbol = str(
                    entry.get("symbol")
                    or entry.get("E_Symbol")
                    or entry.get("asx_code")
                    or entry.get("ASX Code")
                    or ""
                ).strip().upper()
                if not symbol:
                    continue
                result_map[symbol] = {
                    "rank": entry.get("rank") or entry.get("E_Rank"),
                    "percent": entry.get("percent") or entry.get("E_Percent"),
                    "short_positions": entry.get("short_positions") or entry.get("E_Short_Positions"),
                    "total": entry.get("total") or entry.get("E_Total"),
                    "name": entry.get("name") or entry.get("E_Name"),
                    "date": entry.get("date") or entry.get("E_Date"),
                }
            if not metadata.get("date"):
                if payload_date:
                    try:
                        metadata["date"] = pd.Timestamp(payload_date).date().isoformat()
                    except Exception:
                        metadata["date"] = payload_date
                elif path.stem.isdigit():
                    metadata["date"] = dt.datetime.strptime(path.stem, "%Y%m%d").date().isoformat()
            metadata.setdefault("source", str(path))
            return result_map, metadata

        if eval_date is not None:
            date_str = pd.Timestamp(eval_date).strftime("%Y%m%d")
            dated_dir = base_dir / "dated"
            candidate = dated_dir / f"{date_str}.json"
            if not candidate.exists() and dated_dir.exists():
                files = sorted(dated_dir.glob("*.json"))
                for path in reversed(files):
                    if path.stem <= date_str:
                        candidate = path
                        break
            data_map, meta = load_from_path(candidate)
            if data_map:
                return data_map, meta

        latest_path = base_dir / "latest.json"
        data_map, meta = load_from_path(latest_path)
        if data_map:
            return data_map, meta

        url = "https://webservices.weblink.com.au/api/shortposition/"
        cache_file = self._weblink_cache_path("__market__", "shortposition")
        try:
            payload = fetch_json_with_cache(self.session, url, cache_file, ttl=self.cache_ttl)
        except Exception as err:  # pragma: no cover - network/runtime
            logging.warning("Failed to fetch short positions: %s", err)
            return {}, {}

        entries: Iterable[Any]
        if isinstance(payload, dict):
            entries = payload.get("short_positions") or payload.get("data") or payload.get("items") or []
        else:
            entries = payload or []

        api_map: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("E_Symbol") or entry.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            api_map[symbol] = {
                "rank": entry.get("E_Rank") or entry.get("rank"),
                "percent": entry.get("E_Percent") or entry.get("percent"),
                "short_positions": entry.get("E_Short_Positions") or entry.get("short_positions"),
                "total": entry.get("E_Total") or entry.get("total"),
                "name": entry.get("E_Name") or entry.get("name"),
                "date": entry.get("E_Date") or entry.get("date"),
            }
        metadata = {"source": url, "date": dt.datetime.now().isoformat()}
        return api_map, metadata

    def write_short_positions_report(
        self,
        entries: Iterable[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        timestamp = metadata.get("timestamp") or dt.datetime.now().isoformat()
        slug = (
            str(timestamp)
            .replace(":", "")
            .replace("-", "")
            .replace("T", "_")
            .replace(".", "")
        )
        report_dir = project_path(self.config.base_path, "short_reports")
        ensure_directory(report_dir)
        path = report_dir / f"short_report_{slug}.json"
        payload = {
            "metadata": sanitize_for_json(metadata),
            "entries": sanitize_for_json(list(entries)),
        }
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            logging.info("Short positions report written to %s", path)
        except Exception as err:  # pragma: no cover - defensive
            logging.warning("Failed to write short positions report %s: %s", path, err)

    def _write_available_symbols(self, symbols: Iterable[str]) -> None:
        path = project_path("available_symbols.json")
        try:
            payload = list(dict.fromkeys(symbols))
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as err:  # pragma: no cover - defensive
            logging.warning("Failed to write available symbols file %s: %s", path, err)

    def _load_cached_asx_directory(self, max_symbols: Optional[int]) -> Optional[pd.DataFrame]:
        cache_path = self._asx_directory_cache_path()
        if not cache_path.exists():
            return None
        ttl_seconds = 12 * 60 * 60  # refresh twice daily
        age = time.time() - cache_path.stat().st_mtime
        if not self.config.cache:
            return None
        if age > ttl_seconds:
            return None
        try:
            df = pd.read_csv(cache_path)
            if max_symbols is not None and len(df) < max_symbols:
                return None
            return df
        except Exception as exc:
            logging.warning("Failed to read cached ASX directory (%s); refetching. %s", cache_path, exc)
            return None

    def _fetch_asx_csv_symbols(self) -> Optional[pd.DataFrame]:
        url = self.config.asx_directory_url
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ASXScanner/1.0)",
            "Referer": "https://www2.asx.com.au/",
        }
        try:
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as err:  # pragma: no cover - network runtime
            logging.warning("Failed to download ASX CSV directory: %s", err)
            return None

        try:
            content = response.content.decode("utf-8-sig")
            # Skip the first two lines (title + blank line) before the header row.
            data_io = io.StringIO(content)
            df = pd.read_csv(data_io, skiprows=2)
        except Exception as exc:
            logging.warning("Failed to parse ASX CSV directory: %s", exc)
            return None

        columns = {col.lower().strip(): col for col in df.columns}
        code_col = columns.get("asx code")
        if code_col is None:
            logging.warning("ASX CSV directory missing 'ASX code' column.")
            return None

        df = df.rename(columns={code_col: "symbol"})
        if "company name" in columns:
            df = df.rename(columns={columns["company name"]: "company_name"})
        if "gics industry group" in columns:
            df = df.rename(columns={columns["gics industry group"]: "industry"})
        if "trading status" in columns:
            df = df.rename(columns={columns["trading status"]: "trading_status"})
        elif "status" in columns:
            df = df.rename(columns={columns["status"]: "trading_status"})

        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df = df[df["symbol"].str.len() > 0]
        df = df[~df["symbol"].str.contains(r"\s", regex=True)]

        if "trading_status" in df.columns:
            df = df[
                ~df["trading_status"]
                .astype(str)
                .str.upper()
                .str.contains("DELIST", na=False)
            ]
            df = df[
                ~df["trading_status"]
                .astype(str)
                .str.upper()
                .str.contains("SUSPEND", na=False)
            ]

        df = df.drop_duplicates(subset="symbol").sort_values("symbol").reset_index(drop=True)
        logging.info("Fetched %s ASX symbols from CSV directory.", len(df))
        return df

    def _fetch_asx_directory_symbols(self, max_symbols: Optional[int] = None) -> List[str]:
        cached = self._load_cached_asx_directory(max_symbols)
        if cached is not None and not cached.empty:
            symbols = cached["symbol"].dropna().astype(str).str.upper().tolist()
            if max_symbols is not None:
                symbols = symbols[:max_symbols]
            logging.info(
                "Loaded %s ASX symbols from cache (max=%s).",
                len(symbols),
                max_symbols or "all",
            )
            return symbols

        csv_df = self._fetch_asx_csv_symbols()
        if csv_df is not None and not csv_df.empty:
            if self.config.cache:
                ensure_directory(self._asx_directory_cache_path().parent)
                csv_df.to_csv(self._asx_directory_cache_path(), index=False)
            symbols = csv_df["symbol"].tolist()
            if max_symbols is not None:
                symbols = symbols[:max_symbols]
            return symbols

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ASXScanner/1.0)",
            "Referer": "https://www2.asx.com.au/",
        }
        url = "https://asx.api.markitdigital.com/asx-research/1.0/companies/directory"
        rows: List[Dict[str, Any]] = []
        page = 1
        while True:
            params = {"page": page, "size": 25}
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
            except Exception as err:  # pragma: no cover - network runtime
                logging.warning("Failed to fetch ASX directory page %s: %s", page, err)
                break
            payload = response.json().get("data", {})
            items = payload.get("items", []) or []
            if not items:
                break
            for entry in items:
                symbol = (entry.get("symbol") or "").strip().upper()
                if not symbol or len(symbol) > 6 or not symbol.replace("-", "").isalnum():
                    continue
                status = (entry.get("statusCode") or "").strip().upper()
                market_cap = entry.get("marketCap")
                if isinstance(market_cap, str) and market_cap.upper().startswith("SUSP"):
                    continue
                if status and any(flag in status for flag in ("RE", "SUSP", "DEL")):
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "name": entry.get("displayName"),
                        "industry": entry.get("industry"),
                        "status": status,
                        "date_listed": entry.get("dateListed"),
                    }
                )
                if max_symbols and len(rows) >= max_symbols:
                    break
            if max_symbols and len(rows) >= max_symbols:
                break
            page += 1

        if not rows:
            return []

        df = pd.DataFrame(rows).drop_duplicates(subset="symbol")
        if max_symbols is None and self.config.cache:
            ensure_directory(self._asx_directory_cache_path().parent)
            df.to_csv(self._asx_directory_cache_path(), index=False)
        symbols = df["symbol"].tolist()
        if max_symbols is not None:
            symbols = symbols[:max_symbols]
        logging.info(
            "Fetched %s ASX symbols from MarkitDigital directory (max=%s).",
            len(symbols),
            max_symbols or "all",
        )
        return symbols

    def fetch_exchange_symbols(
        self,
        exchange: Optional[str] = None,
        max_symbols: Optional[int] = None,
        allowed_types: Optional[Tuple[str, ...]] = None,
    ) -> List[str]:
        exchange_code = (exchange or self.config.exchange).upper()
        allowed_types = (
            allowed_types if allowed_types is not None else self.config.allowed_security_types
        )

        if exchange_code in {"AU", "ASX"}:
            symbols = self._fetch_asx_directory_symbols(max_symbols=max_symbols)
            if symbols:
                return symbols
            logging.info("ASX directory lookup returned no symbols; falling back to EODHD.")
        page_size = 500
        collected: List[str] = []
        offset = 0

        while True:
            remaining = None if max_symbols is None else max_symbols - len(collected)
            if remaining is not None and remaining <= 0:
                break
            limit = page_size if remaining is None else min(page_size, remaining)
            params = {
                "api_token": self.config.eodhd_api_key,
                "fmt": "json",
                "limit": limit,
                "offset": offset,
            }
            url = f"https://eodhd.com/api/exchange-symbol-list/{exchange_code}"
            response = self.session.get(url, params=params, timeout=30)
            try:
                response.raise_for_status()
            except HTTPError as err:
                status = getattr(err.response, "status_code", None)
                if status == 402:
                    logging.warning(
                        "API returned 402 Payment Required while fetching symbols "
                        "(offset=%s). Stopping pagination; received %s symbols so far.",
                        offset,
                        len(collected),
                    )
                    break
                raise
            batch = response.json()
            if not batch:
                break

            start_len = len(collected)
            for entry in batch:
                if allowed_types and entry.get("Type") not in allowed_types:
                    continue
                if entry.get("Exchange", "").upper() not in {exchange_code, "AU", "ASX"}:
                    continue
                if not entry.get("Isin"):
                    continue
                code = entry.get("Code")
                if not code:
                    continue
                base = code.split(".")[0].strip().upper()
                if not base or len(base) > 6 or not base.replace("-", "").isalnum():
                    continue
                collected.append(base)

            if len(collected) == start_len:
                break
            if len(batch) < limit:
                break
            offset += limit

        unique_symbols = list(dict.fromkeys(collected))
        if self.exclusions:
            unique_symbols = [sym for sym in unique_symbols if sym not in self.exclusions]
        if max_symbols is not None:
            unique_symbols = unique_symbols[:max_symbols]

        logging.info(
            "Fetched %s symbols from exchange %s (max=%s).",
            len(unique_symbols),
            exchange_code,
            max_symbols or "all",
        )
        return unique_symbols

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        upper_symbol = symbol.upper()
        if upper_symbol in self.exclusions:
            raise ValueError(f"{upper_symbol} is excluded: {self.exclusions[upper_symbol]}")
        cache_path = project_path(self.config.base_path, f"{symbol}_ohlcv.csv")
        if self.config.cache and (cached := load_csv_if_exists(cache_path)) is not None:
            logging.info("Loaded %s from cache.", symbol)
            return cached

        source = self.config.data_source.lower()
        if source == "weblink":
            try:
                df = self._fetch_from_weblink(symbol)
            except Exception as err:
                logging.warning(
                    "WebLink fetch failed for %s; falling back to yfinance. %s", symbol, err
                )
                df = self._fetch_from_yfinance(symbol)
            else:
                if df.empty:
                    logging.warning(
                        "WebLink returned no rows for %s; falling back to yfinance.", symbol
                    )
                    df = self._fetch_from_yfinance(symbol)
        elif source == "yfinance":
            df = self._fetch_from_yfinance(symbol)
        elif source == "alpha_vantage":
            df = self._fetch_from_alpha_vantage(symbol)
        else:
            df = self._fetch_from_eodhd(symbol)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} from {self.config.data_source}.")

        if self.config.cache:
            to_csv(df, cache_path)
        logging.info("Fetched %s rows for %s.", len(df), symbol)
        return df

    def _weblink_cache_path(self, symbol: str, name: str) -> pathlib.Path:
        clean_symbol = symbol.upper()
        return project_path(self.config.base_path, "weblink_cache", clean_symbol, f"{name}.json")

    def _fetch_from_weblink(self, symbol: str, period: str = "daily") -> pd.DataFrame:
        """
        Fetch OHLCV data from WebLink API.
        Example: https://webservices.weblink.com.au/api/StockHist?symbol=BHP&type=daily
        """
        url = f"https://webservices.weblink.com.au/api/StockHist?symbol={symbol}&type={period}"
        cache_file = self._weblink_cache_path(symbol, f"StockHist_{period.lower()}")
        payload = fetch_json_with_cache(self.session, url, cache_file, ttl=self.cache_ttl)
        if not payload:
            return pd.DataFrame()
        df = pd.DataFrame(payload)
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"])
        df.rename(columns={"Date": "date"}, inplace=True)
        df.set_index("date", inplace=True)
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )
        df["adj_close"] = df["close"]
        df["symbol"] = symbol
        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        df = df.sort_index()
        df.index.name = "date"
        return df

    def fetch_index_symbols(self, index_code: str = "XJO") -> List[str]:
        index_code = index_code.upper()
        cache_symbol = f"__index__{index_code}"
        cache_file = self._weblink_cache_path(cache_symbol, "IndexList")
        url = f"https://webservices.weblink.com.au/api/IndexList?index={index_code}"
        payload = fetch_json_with_cache(self.session, url, cache_file, ttl=self.cache_ttl)
        entries: Iterable[Any]
        if isinstance(payload, dict):
            entries = [payload]
        else:
            entries = payload or []
        symbols: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_list = entry.get("list") or entry.get("symbols")
            if isinstance(raw_list, str):
                symbols.extend(token.strip() for token in raw_list.split(","))
            elif isinstance(raw_list, (list, tuple, set)):
                symbols.extend(str(token) for token in raw_list)
        cleaned: List[str] = []
        for token in symbols:
            ticker = str(token).strip().upper()
            if not ticker:
                continue
            if ticker.endswith(".AX"):
                ticker = ticker[:-3]
            cleaned.append(ticker)
        unique = sorted(set(cleaned))
        if not unique:
            logging.warning("Index %s returned no constituents.", index_code)
        return unique

    def _fetch_from_yfinance(self, symbol: str) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is not installed. Install with `pip install yfinance`.")
        ticker = symbol
        suffix = self.config.yfinance_suffix or ""
        if suffix and not ticker.endswith(suffix):
            ticker = f"{symbol}{suffix}"
        start = self.config.start_date
        end = self.config.end_date
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        if "price" in df.columns:
            df = df.drop(columns=["price"])
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "adj close": "adj_close",
                "volume": "volume",
            }
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        df = df[numeric_cols]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        df["volume"] = df["volume"].fillna(0)
        df["symbol"] = symbol
        df.index.name = "date"
        return df

    def _fetch_from_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        api_key = self.config.alpha_vantage_key
        if not api_key:
            raise RuntimeError("alpha_vantage_key is required for alpha_vantage data source.")
        ticker = symbol
        suffix = self.config.yfinance_suffix or ""
        if suffix and not ticker.endswith(suffix):
            ticker = f"{symbol}{suffix}"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": "30min",
            "outputsize": "full",
            "apikey": api_key,
        }
        url = "https://www.alphavantage.co/query"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        time_series_key = next((key for key in payload.keys() if "Time Series" in key), None)
        if time_series_key is None:
            raise RuntimeError(f"Alpha Vantage intraday payload missing time-series data: {payload}")
        records = payload.get(time_series_key, {})
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(records, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        rename_map = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
        df = df.rename(columns=rename_map)
        for col in rename_map.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df_daily = df.resample("1D").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        df_daily["adj_close"] = df_daily["close"]
        df_daily["symbol"] = symbol
        df_daily.index = df_daily.index.tz_localize(None)
        start = pd.to_datetime(self.config.start_date) if self.config.start_date else None
        end = pd.to_datetime(self.config.end_date) if self.config.end_date else None
        if start is not None:
            df_daily = df_daily[df_daily.index >= start]
        if end is not None:
            df_daily = df_daily[df_daily.index <= end]
        df_daily = df_daily.dropna(subset=["close", "volume"])
        df_daily.index.name = "date"
        return df_daily

    def _fetch_from_eodhd(self, symbol: str) -> pd.DataFrame:
        params = {
            "api_token": self.config.eodhd_api_key,
            "fmt": "json",
            "from": self.config.start_date,
            "to": self.config.end_date or dt.date.today().isoformat(),
            "period": "d",
            "order": "a",
        }
        url = f"https://eodhd.com/api/eod/{symbol}.AU"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return pd.DataFrame()

        df = pd.DataFrame(payload)
        df.rename(
            columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "adjClose": "adj_close",
            },
            inplace=True,
        )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        if "adj_close" not in df:
            df["adj_close"] = df["close"]
        df["symbol"] = symbol
        return df

    def ingest_all(
        self,
        symbols: Optional[Iterable[str]] = None,
        exchange: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        if symbols is not None:
            symbol_list = [str(sym).strip().upper() for sym in symbols if str(sym).strip()]
        else:
            symbol_list = list(self.config.symbols)

        if not symbol_list:
            symbol_list = self.fetch_exchange_symbols(exchange=exchange, max_symbols=limit)
        elif limit is not None:
            symbol_list = symbol_list[:limit]

        filtered_symbols = [
            sym for sym in symbol_list if sym not in self.exclusions
        ]
        excluded_existing = len(symbol_list) - len(filtered_symbols)
        if excluded_existing > 0:
            logging.info("Skipping %s symbols previously excluded.", excluded_existing)

        results: Dict[str, pd.DataFrame] = {}
        self.latest_skipped = []
        exclusions_updated = False

        for symbol in filtered_symbols:
            try:
                results[symbol] = self.fetch_ohlcv(symbol)
            except ValueError as err:
                reason = str(err)
                logging.warning("Skipping %s: %s", symbol, reason)
                self.exclusions[symbol] = reason
                self.latest_skipped.append({"symbol": symbol, "reason": reason})
                exclusions_updated = True
            except Exception as err:  # pragma: no cover - network dependent
                logging.exception("Failed to ingest %s: %s", symbol, err)
                self.latest_skipped.append({"symbol": symbol, "reason": str(err)})
        if exclusions_updated:
            self._save_exclusions()
        if results:
            self._write_available_symbols(sorted(results.keys()))
        return results


# -------- Feature Engineering ------------------------------------------------------------


class FeatureEngineer:
    def __init__(
        self,
        config: FeatureConfig,
        data_base_path: Optional[pathlib.Path] = None,
    ) -> None:
        self.config = config
        base = data_base_path or pathlib.Path("data")
        if not isinstance(base, pathlib.Path):
            base = pathlib.Path(base)
        self.base_path = project_path(str(base))
        self.session = requests.Session()
        self.cache_ttl = WEBLINK_CACHE_TTL_SECONDS
        self.sentiment_analyzer: Optional[SentimentAnalyzerType] = None
        if SentimentIntensityAnalyzer is not None:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as err:  # pragma: no cover - runtime guard
                logging.warning("Failed to initialise sentiment analyzer: %s", err)
        else:
            logging.info(
                "vaderSentiment not installed; headline sentiment will default to 0. "
                "Install with `pip install vaderSentiment` for richer signals."
            )
        self._benchmark_cache: Dict[str, pd.Series] = {}
        self._benchmark_symbol: str = "^AXJO"

    def transform(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        data = df.copy()
        resolved_symbol = (symbol or self._infer_symbol(df) or "").upper()
        if data.empty:
            return data
        non_numeric_cols = [col for col in data.columns if not is_numeric_dtype(data[col])]
        if non_numeric_cols:
            data = data.drop(columns=non_numeric_cols)
        data["return_1d"] = data["adj_close"].pct_change()
        safe_returns = data["return_1d"].replace([np.inf, -np.inf], np.nan)
        safe_returns = safe_returns.clip(lower=-0.999999, upper=None)
        data["log_return"] = np.log1p(safe_returns)
        data["rsi"] = rolling_rsi(data["adj_close"], self.config.rsi_window)
        data["ma_short"] = data["adj_close"].rolling(self.config.short_ma).mean()
        data["ma_long"] = data["adj_close"].rolling(self.config.long_ma).mean()
        data["ma_ratio"] = data["ma_short"] / data["ma_long"]

        std = data["adj_close"].rolling(self.config.bollinger_window).std()
        mean = data["adj_close"].rolling(self.config.bollinger_window).mean()
        data["bollinger_upper"] = mean + self.config.bollinger_std * std
        data["bollinger_lower"] = mean - self.config.bollinger_std * std
        data["bollinger_pct"] = (data["adj_close"] - data["bollinger_lower"]) / (
            data["bollinger_upper"] - data["bollinger_lower"]
        )

        data["volume_z"] = (
            data["volume"] / data["volume"].rolling(30).mean()
        ).replace([np.inf, -np.inf], np.nan)
        data["volume_avg_20"] = data["volume"].rolling(20).mean()
        data["volume_relative"] = (
            data["volume"] / data["volume_avg_20"]
        ).replace([np.inf, -np.inf], np.nan)
        data["atr"] = self._average_true_range(data)
        data["intraday_volatility"] = (
            (data["high"] - data["low"]) / data["open"]
        ).replace([np.inf, -np.inf], np.nan)
        data["gap_percent"] = (
            (data["open"] - data["adj_close"].shift(1)) / data["adj_close"].shift(1)
        )
        data["gap_direction"] = np.sign(data["gap_percent"]).replace([np.inf, -np.inf], np.nan)
        data["prev_close_to_open_return"] = (
            (data["open"] - data["adj_close"].shift(1)) / data["adj_close"].shift(1)
        )
        data["O2_30min_return"] = (
            (data["open"].shift(-1) - data["open"]) / data["open"]
        ).replace([np.inf, -np.inf], np.nan)
        data["day_of_week"] = data.index.dayofweek.astype(float)

        self._augment_weblink_features(data, resolved_symbol)
        data["news_sentiment_lag1"] = data["news_sentiment"].shift(1)

        index_returns = self._load_index_returns(
            resolved_symbol,
            data.index.min(),
            data.index.max(),
        )
        if not index_returns.empty:
            aligned_index = index_returns.reindex(data.index).ffill().fillna(0.0)
        else:
            aligned_index = pd.Series(0.0, index=data.index)
        data["index_return_1d"] = aligned_index
        data["index_volatility_5d"] = (
            data["index_return_1d"].rolling(5).std().fillna(0.0)
        )

        data["target"] = (data["close"] > data["open"]).astype(int)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        return data

    @staticmethod
    def _average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["adj_close"].shift(1)).abs()
        lc = (df["low"] - df["adj_close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _augment_weblink_features(self, data: pd.DataFrame, symbol: str) -> None:
        # Fundamentals (ratios & dividends)
        fundamentals: Dict[str, Any] = {}
        dividends: List[Dict[str, Any]] = []
        if symbol:
            try:
                fund_payload = self._load_fundamentals(symbol)
                fundamentals = fund_payload.get("ratios", {}) or {}
                dividends = fund_payload.get("dividends", []) or []
            except Exception as err:
                logging.warning("Fundamentals unavailable for %s: %s", symbol, err)
        data["pe_ratio"] = self._to_float(fundamentals.get("PE"))
        div_yield = fundamentals.get("DivYeild") or fundamentals.get("DividendYield")
        data["div_yield"] = self._to_float(div_yield)
        data["eps"] = self._to_float(fundamentals.get("EPS"))
        data["market_cap"] = self._to_float(fundamentals.get("MarketCap"))
        data["last_dividend"] = self._extract_last_dividend(dividends)

        # Volume strength and VWAP snapshot
        quote_data: Dict[str, Any] = {}
        if symbol:
            try:
                quote_payload = self._load_weblink_json(
                    f"https://webservices.weblink.com.au/api/StockQuote?extended=vol_ma20,rvol&symbol={symbol}",
                    symbol,
                    "StockQuote_vol",
                )
                if isinstance(quote_payload, list) and quote_payload:
                    quote_data = quote_payload[0]
                elif isinstance(quote_payload, dict):
                    quote_data = quote_payload
            except Exception as err:
                logging.warning("Quote enrichment failed for %s: %s", symbol, err)
        rvol = self._to_float(quote_data.get("RVOL"))
        if rvol == 0.0 and "volume_z" in data.columns and not data["volume_z"].dropna().empty:
            rvol = float(data["volume_z"].fillna(0).iloc[-1])
        data["rvol"] = rvol
        vwap = self._to_float(quote_data.get("VWAP"))
        if vwap <= 0 and "adj_close" in data.columns:
            vwap = float(data["adj_close"].ffill().dropna().iloc[-1])
        data["vwap"] = vwap
        vol_ma20 = self._to_float(
            quote_data.get("VOL_MA20")
            or quote_data.get("vol_ma20")
            or quote_data.get("VOLMA20")
            or quote_data.get("volma20")
        )
        if vol_ma20 > 0:
            volume_strength = (data["volume"] / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            volume_strength = data.get("volume_z", pd.Series(0.0, index=data.index)).fillna(0.0)
        data["volume_strength"] = volume_strength.astype(float)

        # News sentiment
        sentiment_score = 0.0
        if symbol:
            try:
                headlines_payload = self._load_weblink_json(
                    f"https://webservices.weblink.com.au/api/headline/{symbol}?year>=2023",
                    symbol,
                    "headlines_since_2023",
                )
                sentiment_score = self._compute_sentiment(headlines_payload)
            except Exception as err:
                logging.warning("News sentiment unavailable for %s: %s", symbol, err)
        data["news_sentiment"] = float(sentiment_score)

        # Short interest & market depth
        short_interest = 0.0
        if symbol:
            try:
                short_interest = self._compute_short_interest(symbol)
            except Exception as err:
                logging.warning("Short interest unavailable for %s: %s", symbol, err)
        data["short_interest"] = float(short_interest)

        order_imbalance = 0.0
        if symbol:
            try:
                order_imbalance = self._compute_order_imbalance(symbol)
            except Exception as err:
                logging.warning("Depth quote unavailable for %s: %s", symbol, err)
        data["order_imbalance"] = float(order_imbalance)

        # Global & commodity context
        data["global_trend"] = float(self._compute_global_trend())
        data["commodity_trend"] = float(self._compute_commodity_trend())

    def _load_index_returns(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        if yf is None:
            return pd.Series(dtype=float)
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        benchmark = self._resolve_benchmark_symbol(symbol)
        if not benchmark:
            return pd.Series(dtype=float)
        cached = self._benchmark_cache.get(benchmark)
        needs_refresh = True
        if cached is not None and not cached.empty:
            cached_start = cached.index.min()
            cached_end = cached.index.max()
            if cached_start <= start and cached_end >= end:
                needs_refresh = False
        if needs_refresh:
            download_start = (start - pd.Timedelta(days=45)).to_pydatetime()
            download_end = (end + pd.Timedelta(days=5)).to_pydatetime()
            df = yf.download(
                benchmark,
                start=download_start,
                end=download_end,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if df.empty:
                return pd.Series(dtype=float)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close_col = None
            for candidate in ("Adj Close", "Close"):
                if candidate in df.columns:
                    close_col = candidate
                    break
            if close_col is None:
                return pd.Series(dtype=float)
            closes = pd.to_datetime(df.index).tz_localize(None)
            series = pd.Series(df[close_col].values, index=closes)
            series = series.ffill()
            returns = series.pct_change().fillna(0.0)
            self._benchmark_cache[benchmark] = returns
        else:
            returns = cached
        mask = (returns.index >= start) & (returns.index <= end)
        return returns.loc[mask]

    def _resolve_benchmark_symbol(self, symbol: str) -> Optional[str]:
        return self._benchmark_symbol

    def _load_weblink_json(
        self,
        url: str,
        symbol: Optional[str],
        cache_name: str,
        ttl: Optional[int] = None,
    ) -> Any:
        cache_file = self._weblink_cache_file(symbol, cache_name)
        try:
            return fetch_json_with_cache(
                self.session,
                url,
                cache_file,
                ttl=self.cache_ttl if ttl is None else ttl,
            )
        except HTTPError as err:
            status = getattr(err.response, "status_code", None)
            if status == 404:
                logging.debug("WebLink endpoint returned 404 for %s (%s); returning empty payload.", url, cache_name)
                if cache_file.exists():
                    try:
                        return load_json_file(cache_file)
                    except Exception as cache_err:  # pragma: no cover - defensive
                        logging.debug("Failed to read cached payload for %s after 404: %s", cache_name, cache_err)
                return {}
            raise

    def _weblink_cache_file(self, symbol: Optional[str], name: str) -> pathlib.Path:
        cache_symbol = (symbol or "__shared__").upper()
        return self.base_path / "weblink_cache" / cache_symbol / f"{name}.json"

    def _fundamentals_cache_path(self, symbol: str) -> pathlib.Path:
        return self.base_path / "fundamentals" / f"{symbol}.json"

    def _load_fundamentals(self, symbol: str) -> Dict[str, Any]:
        cache_path = self._fundamentals_cache_path(symbol)
        if _is_cache_valid(cache_path, self.cache_ttl):
            try:
                return load_json_file(cache_path)
            except Exception as err:
                logging.debug("Failed to read cached fundamentals for %s: %s", symbol, err)
        ratios = self._load_weblink_json(
            f"https://webservices.weblink.com.au/api/SecDetails?symbol={symbol}&type=ratios",
            symbol,
            "SecDetails_ratios",
        )
        dividends = self._load_weblink_json(
            f"https://webservices.weblink.com.au/api/SecDetails?symbol={symbol}&type=dividends",
            symbol,
            "SecDetails_dividends",
        )
        payload = {
            "ratios": ratios,
            "dividends": dividends,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        save_json_file(payload, cache_path)
        return payload

    @staticmethod
    def _infer_symbol(df: pd.DataFrame) -> Optional[str]:
        if "symbol" in df.columns:
            series = df["symbol"].dropna().astype(str)
            if not series.empty:
                return series.iloc[-1]
        return None

    @staticmethod
    def _extract_last_dividend(dividends: Iterable[Dict[str, Any]]) -> float:
        latest_amount = 0.0
        latest_date = ""
        for entry in dividends or []:
            if not isinstance(entry, dict):
                continue
            ex_date = str(entry.get("ExDate") or entry.get("ExDividendDate") or "")
            amount = entry.get("Amount") or entry.get("Dividend")
            value = FeatureEngineer._to_float(amount)
            if ex_date > latest_date and value != 0.0:
                latest_date = ex_date
                latest_amount = value
        return latest_amount

    def _compute_sentiment(self, headlines: Any) -> float:
        if not headlines or self.sentiment_analyzer is None:
            return 0.0
        texts: List[str] = []
        for item in headlines:
            if not isinstance(item, dict):
                continue
            for key in ("Text", "Headline", "Title", "Summary"):
                text_value = item.get(key)
                if text_value:
                    texts.append(str(text_value))
                    break
        if not texts:
            return 0.0
        combined = " ".join(texts).strip()
        if not combined:
            return 0.0
        try:
            return float(self.sentiment_analyzer.polarity_scores(combined)["compound"])
        except Exception as err:  # pragma: no cover - defensive
            logging.debug("Sentiment scoring failed: %s", err)
            return 0.0

    def _compute_short_interest(self, symbol: str) -> float:
        payload = self._load_weblink_json(
            "https://webservices.weblink.com.au/api/shortposition/",
            "__market__",
            "shortposition",
        )
        frame = pd.DataFrame(payload)
        if frame.empty or "E_Symbol" not in frame.columns:
            return 0.0
        mask = frame["E_Symbol"].astype(str).str.upper() == symbol.upper()
        if not mask.any():
            return 0.0
        values = pd.to_numeric(frame.loc[mask, "E_Percent"], errors="coerce").dropna()
        if values.empty:
            return 0.0
        return float(values.mean())

    def _compute_order_imbalance(self, symbol: str) -> float:
        payload = self._load_weblink_json(
            f"https://webservices.weblink.com.au/api/DepthQuote?symbol={symbol}",
            symbol,
            "DepthQuote",
        )
        if not isinstance(payload, list) or not payload:
            return 0.0
        bid_vol = 0.0
        ask_vol = 0.0
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            side = entry.get("bidask")
            quantity = self._to_float(entry.get("quan"))
            if int(self._to_float(side)) == 1:
                bid_vol += quantity
            elif int(self._to_float(side)) == 2:
                ask_vol += quantity
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _compute_global_trend(self) -> float:
        payload = self._load_weblink_json(
            "https://webservices.weblink.com.au/api/Global",
            "__macro__",
            "Global",
        )
        iterable: Iterable[Any]
        if isinstance(payload, dict):
            iterable = payload.get("data") or payload.get("items") or []
        else:
            iterable = payload or []
        values = []
        for entry in iterable:
            if not isinstance(entry, dict):
                continue
            value = self._to_float(entry.get("MovementPercent"), default=np.nan)
            if not np.isnan(value):
                values.append(value)
        if not values:
            return 0.0
        return float(np.mean(values))

    def _compute_commodity_trend(self) -> float:
        payload = self._load_weblink_json(
            "https://webservices.weblink.com.au/api/commodities",
            "__macro__",
            "commodities",
        )
        iterable: Iterable[Any]
        if isinstance(payload, dict):
            iterable = payload.get("data") or payload.get("items") or []
        else:
            iterable = payload or []
        values = []
        for entry in iterable:
            if not isinstance(entry, dict):
                continue
            raw_value = entry.get("MovementPercent")
            if raw_value is None:
                raw_value = entry.get("ChangePercent")
            value = self._to_float(raw_value, default=np.nan)
            if not np.isnan(value):
                values.append(value)
        if not values:
            return 0.0
        return float(np.mean(values))

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, str):
                cleaned = value.strip().replace(",", "").replace("$", "")
                if cleaned.upper() in {"N/A", "NA", ""}:
                    return default
                return float(cleaned)
            return float(value)
        except (TypeError, ValueError):
            return default
# -------- Modeling ----------------------------------------------------------------------


class ModelTrainer:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model: Optional[Pipeline] = None

    def train(self, data: pd.DataFrame) -> Pipeline:
        features = self._feature_columns(data)
        target = data["target"]
        tscv = TimeSeriesSplit(n_splits=self.config.test_splits)

        param_sets: List[Optional[Dict[str, Any]]] = [None]
        if self.config.hyperparameter_grid:
            param_sets.extend(self.config.hyperparameter_grid)

        best_model: Optional[Pipeline] = None
        best_override: Optional[Dict[str, Any]] = None
        best_score = -math.inf
        total_param_sets = len(param_sets)
        folds_per_param = max(1, self.config.test_splits)
        total_tasks = max(1, total_param_sets * folds_per_param)
        completed_tasks = 0
        start_time = time.time()
        last_report_time = start_time

        for grid_idx, overrides in enumerate(param_sets, start=0):
            label = "base" if overrides is None else f"grid-{grid_idx}"
            fold_scores: List[float] = []
            for fold, (train_idx, test_idx) in enumerate(tscv.split(data), start=1):
                X_train = data.iloc[train_idx][features]
                y_train = target.iloc[train_idx]
                X_test = data.iloc[test_idx][features]
                y_test = target.iloc[test_idx]

                model = self._build_pipeline(overrides)
                model.fit(X_train, y_train)
                score = self._score_model(model, X_test, y_test)
                logging.info(
                    "%s fold %s %s %.3f",
                    label,
                    fold,
                    self.config.scoring_metric.upper(),
                    score,
                )
                fold_scores.append(score)
                completed_tasks += 1
                now = time.time()
                if now - last_report_time >= 30:
                    percent_complete = 100.0 * completed_tasks / total_tasks
                    logging.info(
                        "Model training progress: %.1f%% (%s/%s folds evaluated)",
                        percent_complete,
                        completed_tasks,
                        total_tasks,
                    )
                    last_report_time = now
            if not fold_scores:
                continue
            avg_score = float(np.nanmean(fold_scores))
            logging.info("%s average %s %.3f", label, self.config.scoring_metric.upper(), avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_override = overrides if overrides is not None else None
                best_model = self._build_pipeline(best_override)

        if best_model is None:
            raise RuntimeError("Training failed: no model selected.")

        features_frame = data[features]
        best_model.fit(features_frame, target)
        self.model = best_model
        selected_algorithm = (
            (best_override or {}).get("algorithm") or self.config.algorithm
        )
        if completed_tasks < total_tasks:
            logging.info(
                "Model training progress: 100.0%% (%s/%s folds evaluated)",
                total_tasks,
                total_tasks,
            )
        logging.info(
            "Selected model algorithm=%s score=%.3f grid=%s",
            selected_algorithm,
            best_score,
            best_override or "base",
        )
        return best_model

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained.")

        feature_frame = data[self._feature_columns(data)]
        proba_matrix = self.model.predict_proba(feature_frame)
        named_steps = getattr(self.model, "named_steps", {})
        estimator = named_steps.get("model") if isinstance(named_steps, dict) else None
        if estimator is None or not hasattr(estimator, "classes_"):
            classes = np.arange(proba_matrix.shape[1])
        else:
            classes = estimator.classes_

        if proba_matrix.shape[1] == 1:
            single_class = int(classes[0])
            if single_class == 1:
                proba = np.ones(len(feature_frame))
            else:
                proba = np.zeros(len(feature_frame))
        else:
            try:
                positive_idx = list(classes).index(1)
            except ValueError:
                positive_idx = 0
            proba = proba_matrix[:, positive_idx]

        return pd.Series(proba, index=data.index, name="prob_up")

    def _build_pipeline(self, overrides: Optional[Dict[str, Any]] = None) -> Pipeline:
        overrides = overrides or {}
        steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        scale_features = overrides.get("scale_features", self.config.scale_features)
        if scale_features:
            steps.append(("scaler", StandardScaler()))

        estimator = self._build_estimator(overrides)
        calibrate_flag = overrides.get("calibrate", self.config.calibrate)
        if calibrate_flag:
            method = overrides.get("calibration_method", self.config.calibration_method)
            calibration_cv = overrides.get("calibration_cv", self.config.calibration_cv)
            cv = max(2, int(calibration_cv))
            estimator = CalibratedClassifierCV(
                estimator=estimator,
                method=method,
                cv=cv,
                n_jobs=-1,
            )

        steps.append(("model", estimator))
        return Pipeline(steps)

    def _build_estimator(self, overrides: Optional[Dict[str, Any]] = None) -> Any:
        overrides = overrides or {}
        algo = (overrides.get("algorithm") or self.config.algorithm or "random_forest").strip().lower()
        if algo in {"hist_gradient_boosting", "hgb", "hgbt"}:
            params = {
                "max_iter": overrides.get("n_estimators", self.config.n_estimators),
                "learning_rate": overrides.get("learning_rate", self.config.learning_rate),
                "max_depth": overrides.get("max_depth", self.config.max_depth),
                "max_leaf_nodes": overrides.get("max_leaf_nodes", self.config.max_leaf_nodes),
                "min_samples_leaf": overrides.get("min_samples_leaf", self.config.min_samples_leaf),
                "random_state": overrides.get("random_state", self.config.random_state),
                "early_stopping": True,
            }
            if hasattr(HistGradientBoostingClassifier, "subsample"):
                params["subsample"] = overrides.get("subsample", self.config.subsample)
            estimator = HistGradientBoostingClassifier(**params)
        else:
            estimator = RandomForestClassifier(
                n_estimators=overrides.get("n_estimators", self.config.n_estimators),
                max_depth=overrides.get("max_depth", self.config.max_depth),
                min_samples_leaf=overrides.get("min_samples_leaf", self.config.min_samples_leaf),
                random_state=overrides.get("random_state", self.config.random_state),
                n_jobs=-1,
            )
            class_weight = overrides.get("class_weight", self.config.class_weight)
            if class_weight:
                estimator.set_params(class_weight=class_weight)
        return estimator

    def _score_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
    ) -> float:
        metric = (self.config.scoring_metric or "accuracy").lower()
        if metric == "roc_auc":
            try:
                proba = model.predict_proba(X_test)[:, 1]
                if np.isnan(proba).any():
                    raise ValueError("probabilities contain NaN")
                if np.isclose(proba.min(), proba.max()):
                    raise ValueError("constant probabilities")
                return roc_auc_score(y_test, proba)
            except ValueError:
                metric = "accuracy"
        if metric == "f1":
            preds = model.predict(X_test)
            return f1_score(y_test, preds)
        if metric == "accuracy":
            return model.score(X_test, y_test)
        preds = model.predict(X_test)
        return f1_score(y_test, preds)

    @staticmethod
    def _feature_columns(data: pd.DataFrame) -> List[str]:
        exclude = {"target"}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        ordered = [
            "return_1d",
            "log_return",
            "rsi",
            "ma_short",
            "ma_long",
            "ma_ratio",
            "bollinger_upper",
            "bollinger_lower",
            "bollinger_pct",
            "volume_z",
            "volume_avg_20",
            "volume_relative",
            "intraday_volatility",
            "gap_percent",
            "gap_direction",
            "volume_strength",
            "atr",
            "rvol",
            "vwap",
            "pe_ratio",
            "div_yield",
            "eps",
            "market_cap",
            "last_dividend",
            "news_sentiment",
            "short_interest",
            "order_imbalance",
            "global_trend",
            "commodity_trend",
            "O2_30min_return",
            "prev_close_to_open_return",
            "day_of_week",
            "news_sentiment_lag1",
            "index_return_1d",
            "index_volatility_5d",
        ]
        features = [col for col in ordered if col in numeric_cols and col not in exclude]
        for col in numeric_cols:
            if col not in exclude and col not in features:
                features.append(col)
        return features


# -------- Risk Management ---------------------------------------------------------------


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def evaluate_signal(
        self,
        row: pd.Series,
        ATR_COL: str = "atr",
        VOLUME_Z_COL: str = "volume_z",
        PROB_COL: str = "prob_up",
    ) -> bool:
        if pd.isna(row.get(PROB_COL, np.nan)):
            return False
        if row.get(VOLUME_Z_COL, 0) > self.config.max_volume_multiple:
            return False
        min_ratio = self.config.min_liquidity_ratio
        volume = row.get("volume", np.nan)
        volume_avg = row.get("volume_avg_20", np.nan)
        if (
            min_ratio > 0
            and pd.notna(volume)
            and pd.notna(volume_avg)
            and volume_avg > 0
            and volume < min_ratio * volume_avg
        ):
            return False
        if row.get("approx_spread_ratio", 0.0) > self.config.max_spread_ratio:
            return False
        timestamp = row.name
        if isinstance(timestamp, pd.Timestamp):
            if not (
                timestamp.hour == 0
                and timestamp.minute == 0
                and timestamp.second == 0
                and timestamp.tzinfo is None
            ):
                minutes = trading_minutes(timestamp)
                if minutes < self.config.restrict_open_minutes:
                    return False
                if minutes > (6 * 60 - self.config.restrict_close_minutes):
                    return False
        if row.get(ATR_COL, 0) <= 0:
            return False
        return True

    def position_size(
        self,
        capital: float,
        prob_up: Optional[float] = None,
        threshold: Optional[float] = None,
        remaining_slots: Optional[int] = None,
    ) -> float:
        capital = float(max(0.0, capital))
        if capital <= 0:
            return 0.0
        base = capital * self.config.allocation_per_trade
        slots = remaining_slots or self.config.max_positions
        if slots:
            slots = max(1, int(slots))
            base = min(base, capital / slots)
        allocation = base
        if prob_up is not None and threshold is not None and threshold < 1:
            if prob_up <= threshold:
                return 0.0
            edge = prob_up - threshold
            scale = min(1.0, edge / max(1e-6, 1 - threshold))
            allocation *= 0.5 + 0.5 * scale
        allocation = min(allocation, capital)
        if allocation < 1e-6:
            return 0.0
        return allocation


# -------- Backtesting -------------------------------------------------------------------


class Backtester:
    def __init__(
        self,
        config: BacktestConfig,
        risk: RiskManager,
        prediction_threshold: float,
        trading_mode: str,
    ) -> None:
        self.config = config
        self.risk = risk
        self.prediction_threshold = prediction_threshold
        self.trading_mode = (trading_mode or "standard").lower()

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.trading_mode == "open_to_close":
            return self._run_open_to_close(df)
        return self._run_standard(df)

    def _run_standard(self, df: pd.DataFrame) -> Dict[str, Any]:
        capital = self.config.initial_capital
        equity_curve = []
        trades: List[Dict[str, Any]] = []
        position = 0.0
        entry_price = 0.0
        entry_allocation = 0.0
        atr = df["atr"]
        win_trades = 0
        loss_trades = 0

        for current, nxt in zip(df.iterrows(), df.iloc[1:].iterrows()):
            date, row = current
            next_date, next_row = nxt
            equity_curve.append({"date": date, "equity": capital + position * row["adj_close"]})

            prob = row.get("prob_up", np.nan)
            if position == 0.0:
                if prob >= self.prediction_threshold and self.risk.evaluate_signal(row):
                    available_capital = capital * (1 - self.risk.config.min_cash_buffer)
                    allocation = self.risk.position_size(
                        available_capital,
                        prob_up=prob,
                        threshold=self.prediction_threshold,
                        remaining_slots=1,
                    )
                    if allocation <= 0:
                        continue
                    entry_price = row["adj_close"] * (1 + self.config.slippage_bps / 10_000)
                    qty = allocation / entry_price
                    position = qty
                    capital -= allocation + self.config.trade_cost
                    entry_allocation = allocation
                    trades.append(
                        {
                            "date": date.isoformat(),
                            "action": "BUY",
                            "price": float(entry_price),
                            "qty": float(qty),
                            "capital": float(capital),
                        }
                    )
            else:
                exit_price = next_row["adj_close"] * (1 - self.config.slippage_bps / 10_000)
                stop_price = entry_price - self.risk.config.stop_loss_atr * atr.loc[date]
                target_price = entry_price + self.risk.config.take_profit_atr * atr.loc[date]

                should_exit = False
                reason = "RULE"
                if exit_price <= stop_price:
                    exit_price = stop_price
                    should_exit = True
                    reason = "STOP"
                elif exit_price >= target_price:
                    exit_price = target_price
                    should_exit = True
                    reason = "TARGET"
                elif prob < self.prediction_threshold:
                    should_exit = True

                if should_exit:
                    proceeds = position * exit_price
                    capital += proceeds - self.config.trade_cost
                    net_pnl = proceeds - entry_allocation - 2 * self.config.trade_cost
                    if net_pnl > 0:
                        win_trades += 1
                    elif net_pnl < 0:
                        loss_trades += 1
                    trades.append(
                        {
                            "date": next_date.isoformat(),
                            "action": "SELL",
                            "price": float(exit_price),
                            "qty": float(position),
                            "capital": float(capital),
                            "reason": reason,
                            "pnl": float(net_pnl),
                        }
                    )
                    position = 0.0
                    entry_allocation = 0.0

        equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
        return self._build_report(equity_series, trades, win_trades, loss_trades)

    def _run_open_to_close(self, df: pd.DataFrame) -> Dict[str, Any]:
        capital = self.config.initial_capital
        equity_curve = []
        trades: List[Dict[str, Any]] = []
        win_trades = 0
        loss_trades = 0
        atr = df["atr"]
        for date, row in df.iterrows():
            equity_curve.append({"date": date, "equity": capital})
            prob = row.get("prob_up", np.nan)
            if pd.isna(prob):
                continue
            if prob < self.prediction_threshold or not self.risk.evaluate_signal(row):
                continue
            available_capital = capital * (1 - self.risk.config.min_cash_buffer)
            allocation = self.risk.position_size(
                available_capital,
                prob_up=prob,
                threshold=self.prediction_threshold,
                remaining_slots=self.risk.config.max_positions,
            )
            if allocation <= 0:
                continue
            entry_price = row.get("open", np.nan)
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            entry_price *= 1 + self.config.slippage_bps / 10_000
            qty = allocation / entry_price
            capital -= allocation + self.config.trade_cost
            trades.append(
                {
                    "date": date.isoformat(),
                    "action": "BUY",
                    "price": float(entry_price),
                    "qty": float(qty),
                    "capital": float(capital),
                }
            )
            exit_raw_price = row.get("adj_close", np.nan)
            if pd.isna(exit_raw_price) or exit_raw_price <= 0:
                exit_raw_price = entry_price
            reason = "EOD"
            atr_value = atr.loc[date] if date in atr.index else np.nan
            if not pd.isna(atr_value) and atr_value > 0:
                stop_price = entry_price - self.risk.config.stop_loss_atr * atr_value
                target_price = entry_price + self.risk.config.take_profit_atr * atr_value
                day_low = row.get("low", entry_price)
                day_high = row.get("high", entry_price)
                if day_low <= stop_price:
                    exit_raw_price = stop_price
                    reason = "STOP"
                elif day_high >= target_price:
                    exit_raw_price = target_price
                    reason = "TARGET"
            exit_slippage = max(0.0, 1 - self.config.slippage_bps / 10_000)
            exit_price = exit_raw_price * exit_slippage
            proceeds = qty * exit_price
            capital += proceeds - self.config.trade_cost
            net_pnl = proceeds - allocation - 2 * self.config.trade_cost
            if net_pnl > 0:
                win_trades += 1
            elif net_pnl < 0:
                loss_trades += 1
            trades.append(
                {
                    "date": date.isoformat(),
                    "action": "SELL",
                    "price": float(exit_price),
                    "qty": float(qty),
                    "capital": float(capital),
                    "reason": reason,
                    "pnl": float(net_pnl),
                }
            )
            equity_curve[-1]["equity"] = capital
        if not equity_curve:
            equity_curve.append({"date": df.index[-1], "equity": capital})
        equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
        return self._build_report(equity_series, trades, win_trades, loss_trades)

    def _build_report(
        self,
        equity_series: pd.Series,
        trades: List[Dict[str, Any]],
        win_trades: int,
        loss_trades: int,
    ) -> Dict[str, Any]:
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        daily_returns = equity_series.pct_change().dropna()
        sharpe = (
            math.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if not daily_returns.empty
            else float("nan")
        )
        drawdown = ((equity_series / equity_series.cummax()) - 1).min()
        total_closed_trades = win_trades + loss_trades
        success_rate = (win_trades / total_closed_trades) if total_closed_trades else float("nan")

        report = {
            "initial_capital": self.config.initial_capital,
            "ending_capital": float(equity_series.iloc[-1]),
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown),
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "success_rate": float(success_rate),
            "trades": trades,
        }

        ensure_directory(project_path(self.config.report_path.parent))
        with project_path(self.config.report_path).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logging.info("Backtest report saved to %s", self.config.report_path)
        return report


# -------- Execution (Stub) --------------------------------------------------------------


class ExecutionEngine:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        logging.info(
            "Order %s %s %s @ %.2f (broker=%s) meta=%s",
            side,
            quantity,
            symbol,
            price,
            self.config.broker,
            metadata,
        )


# -------- Orchestrator ------------------------------------------------------------------


class TradingOrchestrator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ingestor = DataIngestor(config.data)
        self.engineer = FeatureEngineer(config.features, config.data.base_path)
        self.trainer = ModelTrainer(config.model)
        self.risk = RiskManager(config.risk)
        self.filters = config.filters
        self.backtester = Backtester(
            config.backtest,
            self.risk,
            config.model.prediction_threshold,
            config.model.trading_mode,
        )
        self.executor = ExecutionEngine(config.execution)
        self._index_cache: Dict[str, Set[str]] = {}

    def run_ingest(
        self,
        exchange: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        if self.config.data.symbols:
            symbols = list(self.config.data.symbols)
            if limit is not None:
                symbols = symbols[:limit]
            logging.info("Starting ingestion for %s configured symbols.", len(symbols))
        else:
            exchange_code = (exchange or self.config.data.exchange).upper()
            symbols = self.ingestor.fetch_exchange_symbols(exchange=exchange_code, max_symbols=limit)
            logging.info(
                "Starting ingestion for %s symbols from exchange %s.",
                len(symbols),
                exchange_code,
            )
        data = self.ingestor.ingest_all(symbols=symbols)
        logging.info("Ingestion finished.")
        return data

    def _get_index_symbols(self, index_code: str) -> Set[str]:
        code = index_code.upper()
        if code in self._index_cache:
            return self._index_cache[code]
        try:
            symbols = set(self.ingestor.fetch_index_symbols(code))
        except Exception as err:  # pragma: no cover - network dependent
            logging.debug("Unable to load constituents for %s: %s", code, err)
            symbols = set()
        if not symbols:
            logging.info("Index %s returned no constituents.", code)
        self._index_cache[code] = symbols
        return symbols

    def _adjust_threshold(self, probabilities: List[float], context: Dict[str, Any]) -> float:
        base_threshold = self.config.model.prediction_threshold
        cfg = self.config.model
        if not probabilities:
            return base_threshold

        threshold = base_threshold
        if cfg.dynamic_threshold:
            try:
                quantile_val = float(np.quantile(probabilities, cfg.dynamic_threshold_quantile))
            except Exception:
                quantile_val = base_threshold
            threshold = max(threshold, quantile_val)
            threshold = max(cfg.dynamic_threshold_min, min(threshold, cfg.dynamic_threshold_max))
        index_vol = context.get("index_volatility")
        if index_vol is not None and not math.isnan(index_vol) and cfg.volatility_threshold_adjust != 0:
            if index_vol >= cfg.volatility_threshold:
                threshold += cfg.volatility_threshold_adjust
            else:
                threshold -= cfg.volatility_threshold_adjust
        day_of_week = context.get("day_of_week")
        adjustments = cfg.day_of_week_adjustments or ()
        if (
            day_of_week is not None
            and isinstance(day_of_week, int)
            and 0 <= day_of_week < len(adjustments)
        ):
            threshold += adjustments[day_of_week]
        threshold = max(cfg.dynamic_threshold_min, min(threshold, cfg.dynamic_threshold_max))
        logging.debug(
            "Effective threshold %.3f (base=%.3f, context=%s)",
            threshold,
            base_threshold,
            context,
        )
        return float(threshold)

    def run_training(self, symbol: str) -> Tuple[RandomForestClassifier, pd.DataFrame]:
        logging.info("Training model for %s.", symbol)
        df = self.ingestor.fetch_ohlcv(symbol)
        engineered = self.engineer.transform(df, symbol)
        trainer = ModelTrainer(self.config.model)
        model = trainer.train(engineered)
        logging.info("Training completed for %s.", symbol)
        logging.debug(
            "\n%s",
            classification_report(
                engineered["target"],
                model.predict(engineered[trainer._feature_columns(engineered)]),
            ),
        )
        self.trainer = trainer
        return model, engineered

    def run_backtest(self, symbol: str) -> Dict[str, Any]:
        _, engineered = self.run_training(symbol)
        proba = self.trainer.predict(engineered)
        engineered = engineered.assign(prob_up=proba)
        engineered["approx_spread_ratio"] = (engineered["high"] - engineered["low"]) / engineered["adj_close"]
        report = self.backtester.run(engineered)
        logging.info("Backtest completed for %s.", symbol)
        return report

    def evaluate_yesterday(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: Optional[int] = None,
        evaluation_date: Optional[str] = None,
        symbols_override: Optional[Iterable[str]] = None,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
        engineered_cache: Optional[Dict[str, pd.DataFrame]] = None,
        top_n: Optional[int] = 5,
    ) -> Dict[str, Any]:
        """
        Reconstruct the recommendations that would have been issued on a given
        evaluation date (defaulting to the most recent trading day before the
        latest observation) and compare the outcome using the following
        session's closing price.
        """
        if symbols_override is not None:
            symbols = [str(s).strip().upper() for s in symbols_override if str(s).strip()]
        elif symbol:
            symbols = [symbol.upper()]
        elif self.config.data.symbols:
            symbols = list(self.config.data.symbols)
            if limit is not None:
                symbols = symbols[:limit]
        else:
            symbols = self.ingestor.fetch_exchange_symbols(exchange=exchange, max_symbols=limit)
        if not symbols:
            raise ValueError("No symbols available for evaluation.")

        eval_ts_input = pd.to_datetime(evaluation_date).normalize() if evaluation_date else None
        resolved_eval_ts: Optional[pd.Timestamp] = eval_ts_input
        resolved_comp_ts: Optional[pd.Timestamp] = None

        candidate_records: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        considered = 0
        skipped_count = 0
        short_data_map: Dict[str, Dict[str, Any]] = {}
        short_meta: Dict[str, Any] = {}
        short_loaded = False

        asx200_symbols: Set[str] = self._get_index_symbols("XJO")

        for idx, sym in enumerate(symbols, start=1):
            try:
                if preloaded_data is not None:
                    df_obj = preloaded_data.get(sym)
                    if df_obj is None:
                        raise ValueError("preloaded_missing")
                    df = df_obj
                else:
                    df = self.ingestor.fetch_ohlcv(sym).sort_index()
            except Exception as err:
                skipped.append({"symbol": sym, "reason": f"data_fetch_failed: {err}"})
                skipped_count += 1
                continue

            if len(df) < max(self.config.model.test_splits * 2, 50):
                skipped.append({"symbol": sym, "reason": "insufficient_history"})
                skipped_count += 1
                continue

            dates = df.index.unique()
            if len(dates) < 2:
                skipped.append({"symbol": sym, "reason": "insufficient_dates"})
                skipped_count += 1
                continue

            # Determine evaluation/comparison dates
            if eval_ts_input is not None:
                possible = dates[dates <= eval_ts_input]
                if len(possible) == 0:
                    skipped.append({"symbol": sym, "reason": "no_data_on_or_before_date"})
                    skipped_count += 1
                    continue
                eval_ts = possible[-1]
                if resolved_eval_ts is None:
                    resolved_eval_ts = eval_ts
            else:
                if resolved_eval_ts is None:
                    resolved_eval_ts = dates[-2]
                eval_ts = resolved_eval_ts

            if eval_ts not in dates:
                skipped.append({"symbol": sym, "reason": "missing_evaluation_date"})
                skipped_count += 1
                continue

            next_candidates = dates[dates > eval_ts]
            if len(next_candidates) == 0:
                skipped.append({"symbol": sym, "reason": "missing_follow_up_session"})
                skipped_count += 1
                continue
            comp_ts = next_candidates[0]

            if resolved_comp_ts is None and (self.config.model.trading_mode or "standard").lower() != "open_to_close":
                resolved_comp_ts = comp_ts

            if not short_loaded:
                fetch_target = eval_ts
                short_data_map, short_meta = self.ingestor.fetch_short_positions(fetch_target)
                if not short_meta.get("date") and short_data_map:
                    dates = [entry.get("date") for entry in short_data_map.values() if entry.get("date")]
                    if dates:
                        short_meta["date"] = sorted(dates)[-1]
                short_loaded = True

            df_for_filters = df[df.index <= eval_ts]
            passes_filters, _, reason = self._evaluate_filters(sym, df_for_filters)
            if not passes_filters:
                skipped.append({"symbol": sym, "reason": f"prefilter_fail: {reason}"})
                skipped_count += 1
                continue

            source_engineered = None
            if engineered_cache is not None:
                source_engineered = engineered_cache.get(sym)
            if source_engineered is not None:
                engineered = source_engineered.loc[source_engineered.index <= comp_ts].copy()
            else:
                df_for_features = df[df.index <= comp_ts]
                engineered = self.engineer.transform(df_for_features, sym)
            if eval_ts not in engineered.index:
                skipped.append({"symbol": sym, "reason": "no_engineered_row"})
                skipped_count += 1
                continue

            train_frame = engineered[engineered.index < eval_ts]
            if len(train_frame) <= self.config.model.test_splits:
                skipped.append({"symbol": sym, "reason": "insufficient_training_rows"})
                skipped_count += 1
                continue

            evaluation_frame = engineered.loc[[eval_ts]]

            trainer = ModelTrainer(self.config.model)
            try:
                trainer.train(train_frame)
            except ValueError as err:
                skipped.append({"symbol": sym, "reason": f"training_error: {err}"})
                skipped_count += 1
                continue

            prob = float(trainer.predict(evaluation_frame).iloc[0])
            evaluation_row = evaluation_frame.iloc[0].copy()
            evaluation_row["prob_up"] = prob
            high = df.loc[eval_ts, "high"]
            low = df.loc[eval_ts, "low"]
            close = df.loc[eval_ts, "adj_close"]
            evaluation_row["approx_spread_ratio"] = (high - low) / close if close else 0.0

            considered += 1
            trading_mode = (self.config.model.trading_mode or "standard").lower()
            comparison_ts = eval_ts if trading_mode == "open_to_close" else comp_ts
            if resolved_comp_ts is None:
                resolved_comp_ts = comparison_ts

            exit_price = float(df.loc[comp_ts, "adj_close"])
            entry_price = float(close)
            exit_reason = "FOLLOW_UP"
            if trading_mode == "open_to_close":
                open_price = float(df.loc[eval_ts, "open"])
                day_close = float(df.loc[eval_ts, "adj_close"])
                day_high = float(df.loc[eval_ts, "high"])
                day_low = float(df.loc[eval_ts, "low"])
                atr_value = float(evaluation_row.get("atr", float("nan")))
                entry_price = open_price
                slippage_factor = 1 + self.config.backtest.slippage_bps / 10_000
                entry_price *= slippage_factor
                exit_raw_price = day_close
                if not math.isnan(atr_value) and atr_value > 0:
                    stop_price = entry_price - self.risk.config.stop_loss_atr * atr_value
                    target_price = entry_price + self.risk.config.take_profit_atr * atr_value
                    if day_low <= stop_price:
                        exit_raw_price = stop_price
                        exit_reason = "STOP"
                    elif day_high >= target_price:
                        exit_raw_price = target_price
                        exit_reason = "TARGET"
                    else:
                        exit_reason = "EOD"
                else:
                    exit_reason = "EOD"
                exit_slippage = max(0.0, 1 - self.config.backtest.slippage_bps / 10_000)
                exit_price = exit_raw_price * exit_slippage
            return_pct = (exit_price - entry_price) / entry_price if entry_price else float("nan")
            actual_up = exit_price > entry_price
            short_info = short_data_map.get(sym.upper()) if short_data_map else None
            try:
                short_percent = float(short_info.get("percent")) if short_info and short_info.get("percent") is not None else None
            except (TypeError, ValueError):
                short_percent = None

            risk_ok = self.risk.evaluate_signal(evaluation_row)
            candidate_record = {
                "symbol": sym,
                "prob_up": prob,
                "entry_price": entry_price,
                "next_price": exit_price,
                "return_pct": return_pct,
                "success": bool(return_pct > 0 if not math.isnan(return_pct) else False),
                "actual_up": actual_up,
                "evaluation_date": eval_ts.isoformat(),
                "comparison_date": comparison_ts.isoformat(),
                "is_asx200": sym in asx200_symbols,
                "short_percent": short_percent,
                "exit_reason": exit_reason,
                "risk_ok": risk_ok,
                "index_volatility": float(evaluation_row.get("index_volatility_5d", float("nan"))),
                "day_of_week": int(eval_ts.dayofweek),
            }
            candidate_records.append(candidate_record)
            if not risk_ok:
                skipped.append(
                    {
                        "symbol": sym,
                        "reason": "risk_rejected",
                        "prob_up": prob,
                    }
                )
                skipped_count += 1

            if limit and idx >= limit:
                break
        risk_candidates = [item for item in candidate_records if item.get("risk_ok")]
        probabilities = [item["prob_up"] for item in risk_candidates]
        vol_values = [
            item.get("index_volatility")
            for item in risk_candidates
            if item.get("index_volatility") is not None and not math.isnan(item.get("index_volatility"))
        ]
        avg_volatility = float(np.mean(vol_values)) if vol_values else float("nan")
        eval_day_of_week: Optional[int] = None
        if resolved_eval_ts is not None:
            eval_day_of_week = int(resolved_eval_ts.dayofweek)
        elif risk_candidates:
            eval_day_of_week = risk_candidates[0].get("day_of_week")
        quantile_val = float(np.quantile(probabilities, self.config.model.dynamic_threshold_quantile)) if probabilities else float("nan")
        effective_threshold = self._adjust_threshold(
            probabilities,
            {
                "index_volatility": avg_volatility,
                "day_of_week": eval_day_of_week,
            },
        )
        logging.info(
            "Effective threshold %.3f (quantile %.3f, avg_index_vol %.4f) for %s",
            effective_threshold,
            quantile_val if not math.isnan(quantile_val) else float("nan"),
            avg_volatility if not math.isnan(avg_volatility) else float("nan"),
            resolved_eval_ts.date() if resolved_eval_ts is not None else "N/A",
        )
        risk_candidates.sort(key=lambda item: item["prob_up"], reverse=True)
        capital_budget = max(
            0.0,
            self.config.backtest.initial_capital * (1 - self.risk.config.min_cash_buffer),
        )
        max_positions = max(1, self.risk.config.max_positions)
        recommended: List[Dict[str, Any]] = []
        allocated_total = 0.0
        for candidate in risk_candidates:
            if top_n is not None and top_n > 0 and len(recommended) >= top_n:
                break
            if math.isnan(candidate.get("return_pct", float("nan"))):
                continue
            remaining_capital = capital_budget - allocated_total
            if remaining_capital <= 0:
                break
            remaining_slots = max_positions - len(recommended)
            if remaining_slots <= 0:
                break
            if candidate["prob_up"] < effective_threshold:
                skipped.append(
                    {
                        "symbol": candidate["symbol"],
                        "reason": "below_threshold",
                        "prob_up": candidate["prob_up"],
                        "threshold": effective_threshold,
                    }
                )
                skipped_count += 1
                continue
            allocation = self.risk.position_size(
                remaining_capital,
                prob_up=candidate["prob_up"],
                threshold=effective_threshold,
                remaining_slots=remaining_slots,
            )
            if allocation <= 0:
                continue
            pnl_amount = allocation * candidate["return_pct"]
            item = dict(candidate)
            item["allocation"] = float(allocation)
            item["pnl"] = float(pnl_amount)
            item["score"] = candidate["prob_up"] - effective_threshold
            recommended.append(item)
            allocated_total += allocation

        recommended_success = sum(1 for item in recommended if item["success"])
        recommended_count = len(recommended)
        recommended_failure = recommended_count - recommended_success
        accuracy = (
            recommended_success / recommended_count if recommended_count else float("nan")
        )
        total_invested = float(sum(item.get("allocation", 0.0) for item in recommended))
        total_pnl = float(sum(item.get("pnl", 0.0) for item in recommended))
        roi = total_pnl / total_invested if total_invested else float("nan")

        return {
            "evaluation_date": (
                resolved_eval_ts.isoformat() if resolved_eval_ts is not None else None
            ),
            "comparison_date": (
                resolved_comp_ts.isoformat() if resolved_comp_ts is not None else None
            ),
            "recommended": recommended,
            "skipped": skipped,
            "summary": {
                "considered": considered,
                "recommended": recommended_count,
                "wins": recommended_success,
                "losses": recommended_failure,
                "skipped": skipped_count,
                "accuracy": accuracy,
                "invested": total_invested,
                "pnl": total_pnl,
                "roi": roi,
                "short_date": short_meta.get("date"),
                "short_source": short_meta.get("source"),
                "short_entries": len(short_data_map),
                "threshold": effective_threshold,
                "prob_quantile": quantile_val if probabilities else float("nan"),
                "avg_index_volatility": avg_volatility if not math.isnan(avg_volatility) else float("nan"),
            },
            "short_meta": short_meta,
        }

    def evaluate_period(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: Optional[int] = None,
        top_n: Optional[int] = 5,
        workers: Optional[int] = 1,
    ) -> Dict[str, Any]:
        if not start_date:
            raise ValueError("start_date is required when evaluating over a period.")

        try:
            start_ts = pd.to_datetime(start_date).normalize()
        except Exception as exc:  # pragma: no cover - input validation
            raise ValueError(f"Invalid start_date '{start_date}': {exc}") from exc

        if end_date:
            try:
                end_ts = pd.to_datetime(end_date).normalize()
            except Exception as exc:
                raise ValueError(f"Invalid end_date '{end_date}': {exc}") from exc
            if end_ts < start_ts:
                start_ts, end_ts = end_ts, start_ts
            date_range = pd.bdate_range(start_ts, end_ts)
        else:
            period_days = int(days) if days else 21
            if period_days <= 0:
                raise ValueError("days must be a positive integer.")
            date_range = pd.bdate_range(start_ts, periods=period_days)

        if len(date_range) == 0:
            raise ValueError("No business days found in the specified period.")

        if symbol:
            base_symbols = [symbol.upper()]
        elif self.config.data.symbols:
            base_symbols = list(self.config.data.symbols)
        else:
            base_symbols = self.ingestor.fetch_exchange_symbols(exchange=exchange, max_symbols=limit)

        if limit is not None and not symbol:
            base_symbols = base_symbols[:limit]

        if not base_symbols:
            raise ValueError("No symbols available for evaluation.")

        start_time = time.monotonic()
        daily_rows: List[Dict[str, Any]] = []
        total_recommended = 0
        total_wins = 0
        total_losses = 0
        total_considered = 0
        total_skipped = 0
        return_values: List[float] = []
        total_invested = 0.0
        total_pnl = 0.0

        workers = workers or 1
        symbol_data: Dict[str, pd.DataFrame] = {}
        for sym in base_symbols:
            try:
                symbol_data[sym] = self.ingestor.fetch_ohlcv(sym).sort_index()
            except Exception as err:
                logging.debug("Skipping %s during preload (fetch failed): %s", sym, err)

        engineered_cache: Dict[str, pd.DataFrame] = {}
        for sym, raw_df in symbol_data.items():
            try:
                engineered_cache[sym] = self.engineer.transform(raw_df, sym)
            except Exception as err:
                logging.debug("Skipping engineered preload for %s: %s", sym, err)

        def evaluate_single_day(eval_ts: pd.Timestamp) -> Dict[str, Any]:
            return self.evaluate_yesterday(
                symbol=None if symbol is None else symbol,
                exchange=exchange,
                limit=limit,
                evaluation_date=eval_ts.isoformat(),
                symbols_override=base_symbols,
                preloaded_data=symbol_data,
                engineered_cache=engineered_cache,
                top_n=top_n,
            )

        total_days = len(date_range)
        progress_counter = {"completed": 0}
        progress_lock = Lock()

        def log_progress() -> None:
            with progress_lock:
                progress_counter["completed"] += 1
                done = progress_counter["completed"]
                pct = (done / total_days) * 100 if total_days else 100.0
                logging.info(
                    "Workflow progress: %.1f%% (%d/%d days)",
                    pct,
                    done,
                    total_days,
                )

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures: List[Tuple[int, Any]] = []
                for idx, ts in enumerate(date_range):
                    future = executor.submit(evaluate_single_day, ts)
                    futures.append((idx, future))
                ordered_reports: List[Optional[Dict[str, Any]]] = [None] * total_days
                for idx, future in futures:
                    report = future.result()
                    ordered_reports[idx] = report
                    log_progress()
                daily_reports = [report for report in ordered_reports if report is not None]
        else:
            daily_reports = []
            for ts in date_range:
                daily_reports.append(evaluate_single_day(ts))
                log_progress()

        for day_report in daily_reports:
            summary = day_report["summary"]
            recommended = day_report["recommended"]
            avg_return_pct = float("nan")
            if recommended:
                day_returns = [item["return_pct"] for item in recommended if isinstance(item.get("return_pct"), (int, float))]
                if day_returns:
                    avg_return_pct = float(np.mean(day_returns) * 100)
                    return_values.extend(day_returns)

            accuracy = summary.get("accuracy")
            accuracy_pct = (
                float(accuracy * 100)
                if isinstance(accuracy, (int, float)) and not math.isnan(accuracy)
                else float("nan")
            )

            daily_rows.append(
                {
                    "date": day_report.get("evaluation_date"),
                    "recommended": summary.get("recommended", 0),
                    "wins": summary.get("wins", 0),
                    "losses": summary.get("losses", 0),
                    "skipped": summary.get("skipped", 0),
                    "accuracy_pct": accuracy_pct,
                    "avg_return_pct": avg_return_pct,
                    "invested": float(summary.get("invested", 0.0)),
                    "pnl": float(summary.get("pnl", 0.0)),
                }
            )

            total_recommended += summary.get("recommended", 0)
            total_wins += summary.get("wins", 0)
            total_losses += summary.get("losses", 0)
            total_skipped += summary.get("skipped", 0)
            total_considered += summary.get("considered", 0)
            total_invested += float(summary.get("invested", 0.0))
            total_pnl += float(summary.get("pnl", 0.0))

        aggregate_accuracy = (
            total_wins / total_recommended if total_recommended else float("nan")
        )
        aggregate_avg_return = (
            float(np.mean(return_values) * 100) if return_values else float("nan")
        )
        aggregate_total_return = (
            float(np.sum(return_values) * 100) if return_values else float("nan")
        )
        aggregate_roi = total_pnl / total_invested if total_invested else float("nan")
        elapsed = time.monotonic() - start_time
        logging.info(
            "Workflow completed in %.2f minutes (%.2f seconds).",
            elapsed / 60.0,
            elapsed,
        )

        return {
            "period": {
                "start": date_range[0].isoformat(),
                "end": date_range[-1].isoformat(),
            },
            "daily": daily_rows,
            "aggregate": {
                "considered": total_considered,
                "recommended": total_recommended,
                "wins": total_wins,
                "losses": total_losses,
                "skipped": total_skipped,
                "accuracy": aggregate_accuracy,
                "avg_return_pct": aggregate_avg_return,
                "total_return_pct": aggregate_total_return,
                "invested": total_invested,
                "pnl": total_pnl,
                "roi": aggregate_roi,
            },
        }

    def run_live_once(self, symbol: str) -> None:
        _, engineered = self.run_training(symbol)
        today = dt.date.today()
        recent = engineered.loc[engineered.index >= (pd.Timestamp(today) - 5 * BDay())]
        proba = self.trainer.predict(recent)
        latest = recent.iloc[-1].copy()
        latest["prob_up"] = proba.iloc[-1]
        latest["approx_spread_ratio"] = (latest["high"] - latest["low"]) / latest["adj_close"]
        logging.info("Latest prob_up for %s: %.3f", symbol, latest["prob_up"])
        if latest["prob_up"] >= self.config.model.prediction_threshold and self.risk.evaluate_signal(latest):
            capital = self.config.backtest.initial_capital
            available_capital = capital * (1 - self.risk.config.min_cash_buffer)
            allocation = self.risk.position_size(
                available_capital,
                prob_up=latest["prob_up"],
                threshold=self.config.model.prediction_threshold,
                remaining_slots=self.risk.config.max_positions,
            )
            if allocation <= 0:
                logging.info("Allocation reduced to zero after risk sizing; skipping order.")
                return
            qty = allocation / latest["adj_close"]
            self.executor.place_order(
                symbol=f"{symbol}.AU",
                side="BUY",
                quantity=qty,
                price=latest["adj_close"],
                metadata={"prob_up": latest["prob_up"]},
            )
        else:
            logging.info("Signal rejected or below threshold for %s.", symbol)

    def _evaluate_filters(self, symbol: str, df: pd.DataFrame) -> Tuple[bool, Dict[str, float], Optional[str]]:
        filters = self.filters
        stats = {
            "history_days": float(len(df)),
            "avg_volume": float("nan"),
            "avg_dollar_volume": float("nan"),
            "latest_price": float("nan"),
        }
        if not filters.enabled:
            return True, stats, None

        if len(df) == 0:
            return False, stats, "no_data"

        latest_price = float(df["close"].iloc[-1])
        stats["latest_price"] = latest_price

        if len(df) < filters.min_trading_days:
            return False, stats, "insufficient_history"

        window = max(1, filters.avg_volume_window)
        recent = df.tail(window)
        avg_volume = float(recent["volume"].mean(skipna=True))
        avg_price = float(recent["close"].mean(skipna=True))
        avg_dollar_volume = avg_volume * avg_price
        stats.update(
            {
                "avg_volume": avg_volume,
                "avg_dollar_volume": avg_dollar_volume,
            }
        )

        if latest_price < filters.min_price:
            return False, stats, "price_below_threshold"

        if avg_dollar_volume < filters.min_avg_dollar_volume:
            return False, stats, "liquidity_below_threshold"

        return True, stats, None

    def run_recommendations(
        self,
        capital: Optional[float] = None,
        exchange: Optional[str] = None,
        limit: Optional[int] = None,
        top: int = 5,
        data_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        available_capital = capital or self.config.backtest.initial_capital
        min_cash_buffer = available_capital * self.risk.config.min_cash_buffer

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                numeric = float(value)
                if math.isnan(numeric):
                    return default
                return numeric
            except (TypeError, ValueError):
                return default

        original_source = self.ingestor.config.data_source
        if data_source:
            self.ingestor.config.data_source = data_source

        effective_exchange = exchange or self.config.data.exchange
        cache_key: Optional[Dict[str, Any]] = None
        result: Dict[str, Any] = {}
        short_report_map: Dict[str, Dict[str, Any]] = {}
        short_data_map: Dict[str, Dict[str, Any]] = {}
        short_meta: Dict[str, Any] = {}

        try:
            symbols: List[str]
            if exchange:
                symbols = self.ingestor.fetch_exchange_symbols(exchange=exchange, max_symbols=limit)
                if not symbols and self.config.data.symbols:
                    logging.warning(
                        "Falling back to configured symbols because exchange lookup returned none."
                    )
                    symbols = list(self.config.data.symbols)
            elif self.config.data.symbols:
                symbols = list(self.config.data.symbols)
            else:
                symbols = self.ingestor.fetch_exchange_symbols(max_symbols=limit)
            if not symbols:
                raise ValueError("No symbols provided for recommendation.")

            short_data_map, short_meta = self.ingestor.fetch_short_positions()
            if not short_meta.get("date") and short_data_map:
                dates = [entry.get("date") for entry in short_data_map.values() if entry.get("date")]
                if dates:
                    short_meta["date"] = sorted(dates)[-1]
            short_meta.setdefault("source", "local_latest")

            cache_key = {
                "date": dt.date.today().isoformat(),
                "exchange": effective_exchange,
                "limit": limit,
                "top": top,
                "data_source": self.ingestor.config.data_source,
                "threshold": self.config.model.prediction_threshold,
                "capital": float(available_capital),
                "symbols": list(symbols),
                "short_date": short_meta.get("date"),
                "short_source": short_meta.get("source"),
            }
            cached_result = self._load_recommendation_cache(cache_key)
            if cached_result is not None:
                logging.info("Using cached recommendations for %s.", cache_key["date"])
                return cached_result

            asx200_symbols: Set[str] = set()
            try:
                asx200_symbols = self._get_index_symbols("XJO")
            except Exception as err:
                logging.warning("Failed to load ASX200 constituents: %s", err)

            candidates: List[Dict[str, Any]] = []
            processed_symbols: List[Dict[str, Any]] = []
            for idx, symbol in enumerate(symbols, start=1):
                try:
                    df = self.ingestor.fetch_ohlcv(symbol)
                except Exception as err:
                    logging.debug("Skipping %s: data fetch failed (%s)", symbol, err)
                    processed_symbols.append(
                        {
                            "symbol": symbol,
                            "is_asx200": symbol in asx200_symbols,
                            "status": "data_fetch_failed",
                            "reason": str(err),
                        }
                    )
                    continue
                passes_filters, stats, reason = self._evaluate_filters(symbol, df)
                base_info = {"symbol": symbol, "is_asx200": symbol in asx200_symbols, **stats}
                if not passes_filters:
                    processed_symbols.append({**base_info, "status": "prefilter_fail", "reason": reason})
                    continue
                report_entry = short_report_map.setdefault(
                    symbol,
                    {
                        "symbol": symbol,
                        "status": "prefilter_pass",
                    },
                )
                short_info = short_data_map.get(symbol.upper())
                if short_info:
                    report_entry.update(
                        {
                            "short_percent": short_info.get("percent"),
                            "short_rank": short_info.get("rank"),
                            "short_positions": short_info.get("short_positions"),
                            "float_total": short_info.get("total"),
                            "name": short_info.get("name"),
                            "reported_date": short_info.get("date"),
                        }
                    )
                engineered = self.engineer.transform(df, symbol)
                if len(engineered) <= self.config.model.test_splits:
                    logging.debug("Skipping %s: insufficient engineered rows (%s)", symbol, len(engineered))
                    processed_symbols.append({**base_info, "status": "insufficient_history", "rows": len(engineered)})
                    entry = short_report_map.get(symbol)
                    if entry:
                        entry["status"] = "insufficient_history"
                    continue
                trainer = ModelTrainer(self.config.model)
                try:
                    trainer.train(engineered)
                except ValueError as err:
                    logging.debug("Skipping %s: training error (%s)", symbol, err)
                    processed_symbols.append({**base_info, "status": "training_error", "reason": str(err)})
                    entry = short_report_map.get(symbol)
                    if entry:
                        entry["status"] = "training_error"
                    continue

                proba_series = trainer.predict(engineered)
                engineered = engineered.assign(prob_up=proba_series)
                engineered["approx_spread_ratio"] = (engineered["high"] - engineered["low"]) / engineered["adj_close"]
                latest = engineered.iloc[-1].copy()
                base_prob = float(latest["prob_up"])
                ml_prob: Optional[float] = None
                ml_result: Optional[Dict[str, Any]] = None
                if predict_next_day is not None:
                    ml_payload: Dict[str, Any] = latest.to_dict()
                    ml_payload.setdefault("symbol", symbol)
                    if isinstance(latest.name, pd.Timestamp):
                        ml_payload.setdefault("date", latest.name.isoformat())
                    if short_info:
                        ml_payload.setdefault("short_percent_cached", _safe_float(short_info.get("percent"), float("nan")))
                        ml_payload.setdefault("short_rank_cached", _safe_float(short_info.get("rank"), float("nan")))
                        ml_payload.setdefault("short_positions_cached", _safe_float(short_info.get("short_positions"), float("nan")))
                        short_report_date = short_info.get("date")
                        ml_payload.setdefault("short_report_score", 0.0)
                        if short_report_date:
                            try:
                                short_ts = pd.Timestamp(short_report_date).tz_localize(None)
                                if isinstance(latest.name, pd.Timestamp):
                                    ml_payload["days_since_short_report"] = float((latest.name.normalize() - short_ts.normalize()).days)
                            except Exception:
                                ml_payload.setdefault("days_since_short_report", float("nan"))
                        else:
                            ml_payload.setdefault("days_since_short_report", float("nan"))
                    try:
                        ml_result = predict_next_day(ml_payload, history=engineered)
                        ml_prob = float(ml_result.get("prob_up", float("nan")))
                        if math.isnan(ml_prob):
                            ml_prob = None
                    except FileNotFoundError:
                        logging.debug("Global ML model unavailable; skipping ensemble probability for %s.", symbol)
                    except Exception as err:
                        logging.debug("Global ML prediction failed for %s: %s", symbol, err)
                blended_prob = base_prob
                if ml_prob is not None:
                    blended_prob = float(0.5 * base_prob + 0.5 * ml_prob)
                latest["prob_up"] = blended_prob

                if not self.risk.evaluate_signal(latest):
                    processed_symbols.append(
                        {
                            **base_info,
                            "status": "filtered_out",
                            "prob_up": blended_prob,
                            "prob_model": base_prob,
                            "prob_ml": ml_prob,
                            "volume_z": float(latest.get("volume_z", float("nan"))),
                            "spread_ratio": float(latest.get("approx_spread_ratio", float("nan"))),
                        }
                    )
                    entry = short_report_map.get(symbol)
                    if entry:
                        entry["status"] = "filtered_out"
                        entry["prob_up"] = blended_prob
                        entry["prob_model"] = base_prob
                        entry["prob_ml"] = ml_prob
                    continue

                price = float(latest["adj_close"])
                atr = float(latest["atr"])
                rsi = float(latest["rsi"])
                rvol = _safe_float(latest.get("rvol"), _safe_float(latest.get("volume_z"), 0.0))
                news_sent = _safe_float(latest.get("news_sentiment"))
                short_interest = _safe_float(latest.get("short_interest"))
                order_imbalance = _safe_float(latest.get("order_imbalance"))
                vwap = _safe_float(latest.get("vwap"), price)
                volume_strength = _safe_float(
                    latest.get("volume_strength"),
                    _safe_float(latest.get("volume_z"), 0.0),
                )
                global_trend = _safe_float(latest.get("global_trend"))
                commodity_trend = _safe_float(latest.get("commodity_trend"))
                score = blended_prob - self.config.model.prediction_threshold
                index_volatility = _safe_float(latest.get("index_volatility_5d"))
                day_of_week = int(pd.Timestamp(latest.name).dayofweek) if isinstance(latest.name, pd.Timestamp) else None
                entry = short_report_map.get(symbol)
                if entry:
                    entry.update(
                        {
                            "status": "candidate",
                            "prob_up": blended_prob,
                            "prob_model": base_prob,
                            "prob_ml": ml_prob,
                            "score": score,
                            "price": price,
                            "rsi": rsi,
                        }
                    )
                candidate_record = {
                    **base_info,
                    "prob_up": blended_prob,
                    "prob_model": base_prob,
                    "prob_ml": ml_prob,
                    "price": price,
                    "atr": atr,
                    "rsi": rsi,
                    "rvol": rvol,
                    "news_sentiment": news_sent,
                    "short_interest": short_interest,
                    "order_imbalance": order_imbalance,
                    "vwap": vwap,
                    "volume_strength": volume_strength,
                    "global_trend": global_trend,
                    "commodity_trend": commodity_trend,
                    "score": score,
                    "is_asx200": base_info["is_asx200"],
                    "index_volatility": index_volatility,
                    "day_of_week": day_of_week,
                    "ml_explanation": ml_result.get("explanation") if ml_result else None,
                }
                candidates.append(candidate_record)
                processed_symbols.append(
                    {
                        **base_info,
                        "status": "candidate",
                        "prob_up": blended_prob,
                        "prob_model": base_prob,
                        "prob_ml": ml_prob,
                        "price": price,
                    }
                )
                if limit and idx >= limit:
                    break
    
            if not candidates:
                logging.warning("No qualifying candidates found.")
                result = {
                    "capital": available_capital,
                    "remaining_capital": available_capital,
                    "recommendations": [],
                    "processed": processed_symbols,
                    "asx200": sorted(asx200_symbols),
                    "threshold": self.config.model.prediction_threshold,
                }
            else:
                probabilities = [item["prob_up"] for item in candidates]
                vol_values = [
                    item.get("index_volatility")
                    for item in candidates
                    if item.get("index_volatility") is not None and not math.isnan(item.get("index_volatility"))
                ]
                avg_volatility = float(np.mean(vol_values)) if vol_values else float("nan")
                day_of_week_context = next(
                    (item.get("day_of_week") for item in candidates if item.get("day_of_week") is not None),
                    None,
                )
                effective_threshold = self._adjust_threshold(
                    probabilities,
                    {
                        "index_volatility": avg_volatility,
                        "day_of_week": day_of_week_context,
                    },
                )
                for cand in candidates:
                    cand["score"] = cand["prob_up"] - effective_threshold
                logging.info(
                    "Recommendation threshold %.3f (avg_index_vol %.4f)",
                    effective_threshold,
                    avg_volatility if not math.isnan(avg_volatility) else float("nan"),
                )
                candidates.sort(key=lambda item: item["score"], reverse=True)
                recommendations: List[Dict[str, Any]] = []
                remaining_capital = available_capital
                trade_cost = self.config.backtest.trade_cost

                for candidate in candidates:
                    if len(recommendations) >= top:
                        break
                    if candidate["prob_up"] < effective_threshold:
                        continue
                    deployable_capital = max(0.0, remaining_capital - min_cash_buffer)
                    trade_budget = self.risk.position_size(
                        deployable_capital,
                        prob_up=candidate.get("prob_up"),
                        threshold=effective_threshold,
                        remaining_slots=max(1, top - len(recommendations)),
                    )
                    trade_budget = min(trade_budget, deployable_capital)
                    if trade_budget <= 0:
                        break
                    price = candidate["price"]
                    qty = math.floor(trade_budget / price)
                    if qty <= 0:
                        continue
                    allocation = qty * price
                    total_cost = allocation + trade_cost
                    if remaining_capital - total_cost < min_cash_buffer:
                        continue
                    if total_cost > remaining_capital:
                        continue
                    candidate.update({"qty": qty, "allocation": allocation, "trade_cost": trade_cost})
                    remaining_capital -= total_cost
                    recommendations.append(candidate)

                result = {
                    "capital": available_capital,
                    "remaining_capital": remaining_capital,
                    "recommendations": recommendations,
                    "processed": processed_symbols,
                    "asx200": sorted(asx200_symbols),
                    "threshold": effective_threshold,
                }
                for rec in recommendations:
                    entry = short_report_map.get(rec["symbol"])
                    if entry:
                        entry.update(
                            {
                                "status": "recommended",
                                "allocation": rec.get("allocation"),
                                "qty": rec.get("qty"),
                                "score": rec.get("score"),
                                "prob_up": rec.get("prob_up"),
                                "prob_model": rec.get("prob_model"),
                                "prob_ml": rec.get("prob_ml"),
                            }
                        )
                for entry in short_report_map.values():
                    if entry.get("status") == "candidate":
                        entry["status"] = "candidate_not_selected"
        finally:
            self.ingestor.config.data_source = original_source

        result.setdefault("short_meta", short_meta)

        if short_report_map:
            metadata = {
                "timestamp": dt.datetime.now().isoformat(),
                "exchange": effective_exchange,
                "limit": limit,
                "top": top,
                "data_source": data_source or original_source,
                "total_entries": len(short_report_map),
                "recommended": sum(
                    1 for entry in short_report_map.values() if entry.get("status") == "recommended"
                ),
                "short_date": short_meta.get("date"),
                "short_source": short_meta.get("source"),
                "short_entries": len(short_data_map),
            }
            self.ingestor.write_short_positions_report(short_report_map.values(), metadata)

        if cache_key:
            self._save_recommendation_cache(cache_key, result)
        return result

    def _recommendation_cache_dir(self) -> pathlib.Path:
        path = project_path(self.config.data.base_path, "recommendations")
        ensure_directory(path)
        return path

    def _recommendation_cache_path(self, key: Dict[str, Any]) -> pathlib.Path:
        raw = json.dumps(key, sort_keys=True).encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()
        filename = f"{key.get('date', 'day')}_{digest}.json"
        return self._recommendation_cache_dir() / filename

    def _load_recommendation_cache(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        path = self._recommendation_cache_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as err:
            logging.warning("Failed to read recommendation cache %s: %s", path, err)
            return None
        if payload.get("metadata") != key:
            return None
        result = payload.get("result")
        if not isinstance(result, dict):
            return None
        logging.info("Loaded cached recommendations from %s.", path.name)
        return result

    def _save_recommendation_cache(self, key: Dict[str, Any], result: Dict[str, Any]) -> None:
        path = self._recommendation_cache_path(key)
        payload = {
            "metadata": key,
            "result": sanitize_for_json(result),
        }
        try:
            ensure_directory(path.parent)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logging.info("Saved recommendation cache to %s.", path.name)
        except Exception as err:  # pragma: no cover - defensive
            logging.warning("Failed to write recommendation cache %s: %s", path, err)


# -------- CLI ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASX trading assistant for data ingestion, training, and backtesting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python stocktrade.py ingest --config config.yaml
              python stocktrade.py backtest --config config.yaml --symbol BHP
              python stocktrade.py live --config config.yaml --symbol BHP
            """
        ),
    )
    parser.add_argument("--config", required=False, default="config.yaml", help="Path to config file (yaml/json).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Download and cache historical data.")
    ingest_parser.add_argument(
        "--exchange",
        help="Exchange code to source symbols from when config symbols are not provided.",
    )
    ingest_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of symbols ingested (useful for large universes).",
    )

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest for a symbol.")
    backtest_parser.add_argument("--symbol", required=True, help="Ticker without .AU suffix.")

    live_parser = subparsers.add_parser("live", help="Evaluate latest signal and simulate order.")
    live_parser.add_argument("--symbol", required=True, help="Ticker without .AU suffix.")

    subparsers.add_parser("generate-config", help="Write a sample configuration file next to this script.")

    recommend_parser = subparsers.add_parser("recommend", help="Scan symbols and suggest top candidates.")
    recommend_parser.add_argument("--capital", type=float, help="Override capital amount for allocation decisions.")
    recommend_parser.add_argument(
        "--exchange",
        help="Exchange code to pull symbols from (e.g., AU for ASX). If omitted, uses config symbols.",
    )
    recommend_parser.add_argument("--limit", type=int, help="Limit number of symbols fetched from exchange.")
    recommend_parser.add_argument("--top", type=int, default=5, help="Number of recommendations to display.")
    recommend_parser.add_argument(
        "--data-source",
        help="Override data source for this recommendation run (e.g., weblink, yfinance, eodhd).",
    )

    yesterday_parser = subparsers.add_parser(
        "yesterday",
        help="Compare yesterday's predictions with today's outcomes.",
    )
    yesterday_parser.add_argument("--symbol", help="Evaluate a single symbol.")
    yesterday_parser.add_argument(
        "--exchange",
        help="Exchange code to pull symbols from when config symbols are absent.",
    )
    yesterday_parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of symbols processed (useful with large symbol lists).",
    )
    yesterday_parser.add_argument(
        "--date",
        help="ISO date (YYYY-MM-DD) to evaluate (default uses most recent session).",
    )
    yesterday_parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD) for multi-day evaluation.",
    )
    yesterday_parser.add_argument(
        "--end-date",
        help="Optional end date (YYYY-MM-DD) for multi-day evaluation.",
    )
    yesterday_parser.add_argument(
        "--days",
        type=int,
        help="Number of business days to evaluate from the start date (defaults to 21).",
    )
    yesterday_parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Limit number of recommendations per day (default 5).",
    )
    yesterday_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for multi-day evaluation (default 1).",
    )

    return parser


def write_sample_config(path: pathlib.Path) -> None:
    sample = {
        "data": {
            "eodhd_api_key": "YOUR_EODHD_KEY",
            "alpha_vantage_key": None,
            "base_path": "data",
            "symbols": ["BHP", "CBA", "FMG", "CSL"],
            "start_date": "2016-01-01",
            "end_date": None,
            "cache": True,
            "exchange": "AU",
            "allowed_security_types": ["Common Stock"],
            "asx_directory_url": "http://www.asx.com.au/asx/research/ASXListedCompanies.csv",
            "data_source": "weblink",
            "yfinance_suffix": ".AX",
        },
        "features": dataclass_to_serializable(FeatureConfig()),
        "model": dataclass_to_serializable(ModelConfig()),
        "risk": dataclass_to_serializable(RiskConfig()),
        "filters": dataclass_to_serializable(FilterConfig()),
        "backtest": dataclass_to_serializable(BacktestConfig()),
        "execution": dataclass_to_serializable(ExecutionConfig()),
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(sample, handle) if path.suffix in {".yaml", ".yml"} and yaml else json.dump(sample, handle, indent=2)
    logging.info("Sample config written to %s", path)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    config_path = project_path(args.config)
    if args.command == "generate-config":
        sample_path = config_path if config_path.suffix else config_path.with_suffix(".yaml")
        write_sample_config(sample_path)
        return 0

    app_config = AppConfig.load(config_path)
    orchestrator = TradingOrchestrator(app_config)

    if args.command == "ingest":
        orchestrator.run_ingest(exchange=args.exchange, limit=args.limit)
        skipped = getattr(orchestrator.ingestor, "latest_skipped", [])
        if skipped:
            print(f"Skipped {len(skipped)} symbols with no yfinance data:")
            for entry in skipped[:20]:
                reason = entry.get("reason", "")
                print(f"  {entry.get('symbol', '?'):<6} {reason}")
            if len(skipped) > 20:
                print(f"  ... {len(skipped) - 20} more omitted")
    elif args.command == "backtest":
        orchestrator.run_backtest(args.symbol)
    elif args.command == "live":
        orchestrator.run_live_once(args.symbol)
    elif args.command == "recommend":
        result = orchestrator.run_recommendations(
            capital=args.capital,
            exchange=args.exchange,
            limit=args.limit,
            top=args.top,
            data_source=args.data_source,
        )
        recs = result["recommendations"]
        processed = result.get("processed", [])
        if colorama_init:
            try:
                colorama_init()
            except Exception:  # pragma: no cover - defensive
                pass
        if processed:
            print("\nProcessed symbols:")
            for item in processed:
                status = item.get("status", "unknown")
                symbol = item.get("symbol", "?")
                details = []
                if "prob_up" in item:
                    details.append(f"prob={item['prob_up']:.3f}")
                if "price" in item:
                    details.append(f"price={item['price']:.2f}")
                if "reason" in item:
                    details.append(f"reason={item['reason']}")
                if "rows" in item:
                    details.append(f"rows={item['rows']}")
                if "volume_z" in item:
                    details.append(f"vol_z={item['volume_z']:.2f}")
                if "spread_ratio" in item:
                    details.append(f"spread={item['spread_ratio']:.3f}")
                info = " | ".join(details)
                line = f"  {symbol:<8} {status:<18} {info}".rstrip()
                print(highlight_text(line, bool(item.get("is_asx200"))))
        if not recs:
            logging.info("No recommendations generated.")
        else:
            print("\nTop recommendations:")
            header = (
                f"{'Symbol':<8} {'ProbUp':>7} {'Price':>8} {'RSI':>6} "
                f"{'RVOL':>6} {'NewsSent':>9} {'Short%':>7} {'Score':>7}"
            )
            print(header)
            print("-" * len(header))
            for item in recs:
                row = (
                    f"{item['symbol']:<8} "
                    f"{item['prob_up']:>7.2f} "
                    f"{item['price']:>8.2f} "
                    f"{item['rsi']:>6.1f} "
                    f"{item.get('rvol', 0.0):>6.2f} "
                    f"{item.get('news_sentiment', 0.0):>+9.2f} "
                    f"{item.get('short_interest', 0.0):>7.2f} "
                    f"{item.get('score', 0.0):>+7.2f}"
                )
                print(highlight_text(row, bool(item.get("is_asx200"))))
            print("\nSizing:")
            for item in recs:
                line = (
                    f"  {item['symbol']:<8} qty={item['qty']} spend={item['allocation']:.2f} "
                    f"vwap={item.get('vwap', item['price']):.2f} ob={item.get('order_imbalance', 0.0):+.2f}"
                )
                print(highlight_text(line, bool(item.get("is_asx200"))))
            print(f"\nCapital used: {result['capital'] - result['remaining_capital']:.2f}")
            print(f"Capital remaining: {result['remaining_capital']:.2f}")
    elif args.command == "yesterday":
        if args.start_date or args.end_date or args.days:
            if not args.start_date:
                raise SystemExit("Start date is required when evaluating over a period.")
            period_report = orchestrator.evaluate_period(
                start_date=args.start_date,
                end_date=args.end_date,
                days=args.days,
                symbol=args.symbol,
                exchange=args.exchange,
                limit=args.limit,
                top_n=args.top,
                workers=args.workers,
            )
            period_info = period_report["period"]
            print(
                f"\nHistorical performance | "
                f"{period_info['start'][:10]} -> {period_info['end'][:10]}"
            )
            daily = period_report.get("daily", [])
            if not daily:
                print("No trading days evaluated in the specified range.")
            else:
                print(
                    f"{'Date':<12} {'Recs':>6} {'Wins':>6} {'Losses':>7} "
                    f"{'Skipped':>8} {'Accuracy%':>10} {'AvgRet%':>9} {'Invested':>11} {'PnL':>11}"
                )
                print("-" * 110)
                for row in daily:
                    accuracy_pct = row["accuracy_pct"]
                    accuracy_str = (
                        f"{accuracy_pct:.2f}%"
                        if isinstance(accuracy_pct, (int, float)) and not math.isnan(accuracy_pct)
                        else "N/A"
                    )
                    avg_ret = row["avg_return_pct"]
                    avg_ret_str = (
                        f"{avg_ret:.2f}%"
                        if isinstance(avg_ret, (int, float)) and not math.isnan(avg_ret)
                        else "N/A"
                    )
                    invested_val = float(row.get("invested", 0.0))
                    pnl_val = float(row.get("pnl", 0.0))
                    print(
                        f"{(row['date'] or '')[:10]:<12} "
                        f"{row['recommended']:>6} {row['wins']:>6} {row['losses']:>7} "
                        f"{row['skipped']:>8} {accuracy_str:>10} {avg_ret_str:>9} "
                        f"{invested_val:>11.2f} {pnl_val:>11.2f}"
                    )

            aggregate = period_report.get("aggregate", {})
            if aggregate:
                agg_accuracy = aggregate.get("accuracy")
                agg_accuracy_str = (
                    f"{agg_accuracy * 100:.2f}%"
                    if isinstance(agg_accuracy, (int, float)) and not math.isnan(agg_accuracy)
                    else "N/A"
                )
                agg_ret = aggregate.get("avg_return_pct")
                agg_ret_str = (
                    f"{agg_ret:.2f}%"
                    if isinstance(agg_ret, (int, float)) and not math.isnan(agg_ret)
                    else "N/A"
                )
                agg_total = aggregate.get("total_return_pct")
                agg_total_str = (
                    f"{agg_total:.2f}%"
                    if isinstance(agg_total, (int, float)) and not math.isnan(agg_total)
                    else "N/A"
                )
                agg_invested = float(aggregate.get("invested", 0.0))
                agg_pnl = float(aggregate.get("pnl", 0.0))
                agg_roi = aggregate.get("roi")
                agg_roi_str = (
                    f"{agg_roi * 100:.2f}%"
                    if isinstance(agg_roi, (int, float)) and not math.isnan(agg_roi)
                    else "N/A"
                )
                print(
                    f"\nTotals | Considered: {aggregate.get('considered', 0)} | "
                    f"Recommended: {aggregate.get('recommended', 0)} | "
                    f"Wins: {aggregate.get('wins', 0)} | Losses: {aggregate.get('losses', 0)} | "
                    f"Skipped: {aggregate.get('skipped', 0)} | "
                    f"Accuracy: {agg_accuracy_str} | Avg Return: {agg_ret_str} | Total Return: {agg_total_str} | "
                    f"Invested: {agg_invested:.2f} | PnL: {agg_pnl:.2f} | ROI: {agg_roi_str}"
                )
        else:
            report = orchestrator.evaluate_yesterday(
                symbol=args.symbol,
                exchange=args.exchange,
                limit=args.limit,
                evaluation_date=args.date,
                top_n=args.top,
            )
            eval_date = report.get("evaluation_date")
            comp_date = report.get("comparison_date")
            header = "\nHistorical recommendation replay"
            if eval_date:
                header += f" | Evaluation: {eval_date[:10]}"
            if comp_date:
                header += f" -> Next: {comp_date[:10]}"
            print(header)

            recommended = report.get("recommended", [])
            skipped = report.get("skipped", [])
            summary = report.get("summary", {})

            if recommended:
                print(f"{'Symbol':<8} {'ProbUp':>7} {'Entry':>10} {'Next':>10} {'Return%':>8} {'Success':>8}")
                print("-" * 60)
                for item in recommended:
                    ret_pct = item["return_pct"] * 100
                    success_flag = "YES" if item["success"] else "NO"
                    line = (
                        f"{item['symbol']:<8} {item['prob_up']:>7.2f} "
                        f"{item['entry_price']:>10.2f} {item['next_price']:>10.2f} "
                        f"{ret_pct:>8.2f} {success_flag:>8}"
                    )
                    print(highlight_text(line, bool(item.get("is_asx200"))))
            else:
                print("No symbols met the recommendation criteria for the selected date.")

            if skipped:
                print(f"\nSkipped ({min(len(skipped), 10)} shown of {len(skipped)}):")
                for entry in skipped[:10]:
                    symbol = entry.get("symbol", "?")
                    reason = entry.get("reason", "unknown")
                    prob = entry.get("prob_up")
                    extra = f" prob={prob:.2f}" if isinstance(prob, float) else ""
                    print(f"  {symbol:<8} {reason}{extra}")

            if summary:
                accuracy = summary.get("accuracy")
                accuracy_str = (
                    f"{accuracy * 100:.2f}%"
                    if isinstance(accuracy, (int, float)) and not math.isnan(accuracy)
                    else "N/A"
                )
                invested = float(summary.get("invested", 0.0))
                pnl = float(summary.get("pnl", 0.0))
                roi = summary.get("roi")
                roi_str = (
                    f"{roi * 100:.2f}%"
                    if isinstance(roi, (int, float)) and not math.isnan(roi)
                    else "N/A"
                )
                print(
                    f"\nConsidered: {summary.get('considered', 0)} | "
                    f"Recommended: {summary.get('recommended', 0)} | "
                    f"Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)} | "
                    f"Skipped: {summary.get('skipped', 0)} | Accuracy: {accuracy_str}"
                )
                print(f"Invested: {invested:.2f} | PnL: {pnl:.2f} | ROI: {roi_str}")
    else:  # pragma: no cover - defensive branch
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


# -------- CLI Cheat Sheet -------------------------------------------------------------

# Quick reference for common workflows (copy + paste ready)
#
# 1. Refresh cached data before any analysis (full exchange by default):
#    python stocktrade.py --config config.yaml ingest --exchange AU
#    # Add --limit 100 for a faster partial update.
#
# 2. Train or refresh the global ML models:
#    python ml/train_model.py --config config.yaml --algorithm auto           # tabular baseline -> ml/model.pkl
#    python ml/train_sequence_model.py --config config.yaml --sequence-length 60 --folds 5  # LSTM fusion -> ml/model.pt
#    # Optional: append --limit 150 or --no-save for quicker dry runs.
#    # Processed features are exported to data/processed/ml_next_day_dataset.csv for inspection.
#
# 3. Sanity-check the saved model on a single engineered row:
#    python -c "from ml.predict import predict_next_day; from stocktrade import AppConfig, DataIngestor, FeatureEngineer; import pathlib; cfg = AppConfig.load(pathlib.Path('config.yaml')); ing = DataIngestor(cfg.data); fe = FeatureEngineer(cfg.features, cfg.data.base_path); row = fe.transform(ing.fetch_ohlcv('BHP'), 'BHP').iloc[-1]; print(predict_next_day(row))"
#
# 4. Generate current recommendations (top N defaults to 5):
#    python stocktrade.py --config config.yaml recommend --exchange AU --data-source yfinance
#    # ProbUp now blends the legacy per-symbol trainer with ml/model.pkl (equal weights).
#
# 5. Replay a single day’s recommendations (requires ingest after the following session):
#    python stocktrade.py --config config.yaml yesterday --date 2025-10-20 --exchange AU
#    # Optional: --top 5 to change the number of picks.
#
# 6. Multi-day replay table (adds daily stats; supports parallel workers):
#    python stocktrade.py --config config.yaml yesterday --start-date 2025-09-01 --end-date 2025-09-30 --exchange AU --workers 12
#
# 7. Backtest a specific ticker with the current configuration:
#    python stocktrade.py --config config.yaml backtest --symbol BHP
#
# 8. Generate a sample config file next to the script:
#    python stocktrade.py --config my_config.yaml generate-config
#
# 9. Daily recommendations and order simulation (live mode for one symbol):
#    python update_short_positions_csv.py   # pull latest ASIC short report
#    python stocktrade.py --config config.yaml ingest --exchange AU
#    python stocktrade.py --config config.yaml recommend --exchange AU --limit 100


# -------- Recommendation Output Cheat Sheet -------------------------------------------
#
# Columns shown in "Top recommendations":
#   Symbol    : Ticker (without suffix) that passed filters and risk checks.
#   ProbUp    : Blended probability (50% in-symbol trainer, 50% ml/model.pkl inference).
#   Price     : Latest adjusted close used for sizing.
#   RSI       : Relative Strength Index; <30 oversold, >70 overbought.
#   RVOL      : Relative volume vs 20-day average (1.0 = typical volume).
#   NewsSent  : Most recent VADER news sentiment score (range -1 to +1).
#   Short%    : Reported short-interest percentage (if available).
#   Score     : Ranking helper equal to ProbUp minus the dynamic threshold.
#
# Extra diagnostics (printed in the “Processed symbols” list and JSON caches):
#   ProbModel : Probability from the legacy per-symbol trainer (pre-blend).
#   ProbML    : Probability from ml/model.pkl (global lookahead model).
#   MLExplain : Top SHAP/weight contributors when available.
#
# Sizing lines show the suggested quantity, notional spend, VWAP reference,
# and order-book imbalance (positive bid-heavy, negative ask-heavy). Treat
# these as guidance—adjust or skip trades that conflict with your own risk
# tolerance or market context.

