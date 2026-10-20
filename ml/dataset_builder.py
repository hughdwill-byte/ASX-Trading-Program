from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from stocktrade import (
    SNAPSHOT_FEATURES,
    AppConfig,
    DataIngestor,
    FeatureEngineer,
    ensure_directory,
    project_path,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetArtifacts:
    symbols: List[str]
    feature_names: List[str]
    dates: Tuple[pd.Timestamp, pd.Timestamp]
    recommendation_stats: Dict[str, Dict[str, Any]]
    short_report_stats: Dict[str, Dict[str, Any]]


@dataclass
class SequenceSampleMeta:
    symbol: str
    window_end: pd.Timestamp
    target_reference: pd.Timestamp
    horizon: int


@dataclass
class SequenceDataset:
    sequences: np.ndarray
    static_features: np.ndarray
    targets: np.ndarray
    metadata: List[SequenceSampleMeta]
    temporal_features: List[str]
    static_feature_names: List[str]
    feature_names: List[str]
    preprocessor: Dict[str, Any]
    window: int
    horizon: int


def build_dataset(
    config_path: Optional[Union[str, pathlib.Path]] = None,
    symbols: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    normalise: bool = True,
    save_processed: bool = True,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct a RAW feature matrix X and target vector y for H-session direction prediction.

    Timing (F06/F07): features use data up to the close of t; the decision is made after that close; the
    traded target is the cumulative return from the open of t+1 to the close of t+H (> 0 = 1). Rows whose
    label is not yet observable are dropped. `X.attrs["label_end"]` holds each row's label end date for
    purging. X is NOT imputed or scaled: fit `fit_preprocessor` on a training fold only and apply it with
    `apply_preprocessor` (`normalise` is recorded for that step).

    Parameters
    ----------
    config_path:
        Path to the application's configuration file. Defaults to ``config.yaml`` in the project root.
    symbols:
        Optional iterable of tickers to restrict the build to. When omitted, uses symbols from config or cache.
    limit:
        Optional maximum number of symbols to load (useful for quick experiments).
    start_date / end_date:
        Optional temporal bounds applied to each symbol's history.
    normalise:
        When True, median-impute and z-score scale the feature matrix.
    save_processed:
        When True, persist the fully prepared dataset under ``data/processed/`` for debugging.

    Returns
    -------
    (X, y):
        X is a DataFrame indexed by (symbol, date) containing numeric features.
        y is a Series aligned with X containing the binary next-day movement target.
        X.attrs carries the fitted ``imputer`` and ``scaler`` along with metadata.
    """

    resolved_config_path = _resolve_config_path(config_path)
    config = AppConfig.load(resolved_config_path)
    ingestor = DataIngestor(config.data)
    engineer = FeatureEngineer(config.features, config.data.base_path)

    symbol_list = _resolve_symbol_list(symbols, config, ingestor, limit)
    if not symbol_list:
        raise ValueError("No symbols available to build the dataset.")

    start_ts = _coerce_timestamp(start_date) if start_date is not None else None
    end_ts = _coerce_timestamp(end_date) if end_date is not None else None

    logger.info("Building ML dataset for %s symbols.", len(symbol_list))
    frames: List[pd.DataFrame] = []
    for sym in symbol_list:
        try:
            raw = ingestor.fetch_ohlcv(sym)
        except Exception as err:
            logger.warning("Skipping %s (fetch failed): %s", sym, err)
            continue
        df = raw.sort_index().copy()
        if start_ts is not None:
            df = df.loc[df.index >= start_ts]
        if end_ts is not None:
            df = df.loc[df.index <= end_ts]
        if df.empty:
            logger.debug("No rows remaining for %s after date filtering.", sym)
            continue

        try:
            engineered = engineer.transform(df, sym)
        except Exception as err:
            logger.warning("Feature engineering failed for %s: %s", sym, err)
            continue
        if engineered.empty or engineered.shape[0] < 2:
            logger.debug("Insufficient engineered rows for %s.", sym)
            continue

        features = make_labels(engineered.copy(), horizon)
        features.dropna(subset=["target"], inplace=True)   # unknown (terminal) labels are excluded, never 0
        if features.empty:
            continue
        features["target"] = features["target"].astype(int)
        features["symbol"] = sym.upper()
        features.reset_index(inplace=True)
        features.rename(columns={"index": "date"}, inplace=True)
        frames.append(features)

    if not frames:
        raise RuntimeError("Dataset build failed: no usable engineered frames were produced.")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    le = pd.to_datetime(combined["label_end"])
    combined["label_end"] = le.dt.tz_localize(None) if le.dt.tz is not None else le
    combined.sort_values(["date", "symbol"], inplace=True)
    combined.set_index(["symbol", "date"], inplace=True)

    recommendation_stats = _load_recommendation_features(config.data.base_path)
    short_report_stats = _load_short_report_features(config.data.base_path)

    combined = _augment_with_recommendations(combined, recommendation_stats)
    combined = _augment_with_short_reports(combined, short_report_stats)

    combined.replace([np.inf, -np.inf], np.nan, inplace=True)

    if "target" not in combined.columns:
        raise RuntimeError("Combined dataset missing target column after aggregation.")

    full_dataset = combined.copy()
    processed_dir = project_path(config.data.base_path, "processed")
    if save_processed:
        ensure_directory(processed_dir)
        snapshot = full_dataset.reset_index()
        snapshot_path = processed_dir / "ml_next_day_dataset.csv"
        try:
            snapshot.to_csv(snapshot_path, index=False)
            logger.info("Processed dataset saved to %s", snapshot_path)
        except Exception as err:
            logger.warning("Failed to save processed dataset snapshot: %s", err)

    y = full_dataset.pop("target").astype(int)
    label_end = full_dataset.pop("label_end")
    full_dataset = full_dataset.drop(columns=["fwd_return"] + [c for c in SNAPSHOT_FEATURES if c in full_dataset])
    feature_frame = full_dataset.select_dtypes(include=[np.number]).copy()
    dropped_columns = sorted(set(full_dataset.columns) - set(feature_frame.columns))
    if dropped_columns:
        logger.debug("Dropped non-numeric columns from feature set: %s", dropped_columns)
    feature_names = list(feature_frame.columns)

    if not feature_names:
        raise RuntimeError("No numeric features available after cleaning.")

    X = feature_frame
    X.attrs["preprocessor"] = None   # fitted per training fold (fit_preprocessor), never on all history
    X.attrs["label_end"] = label_end
    X.attrs["target"] = target_definition(horizon)
    X.attrs["normalise"] = bool(normalise)
    X.attrs["artifacts"] = DatasetArtifacts(
        symbols=sorted({sym for sym, _ in X.index}),
        feature_names=feature_names,
        dates=(
            feature_frame.index.get_level_values("date").min(),
            feature_frame.index.get_level_values("date").max(),
        ),
        recommendation_stats=recommendation_stats,
        short_report_stats=short_report_stats,
    )
    X.attrs["metadata"] = {
        "symbols": sorted({sym for sym, _ in X.index}),
        "feature_names": feature_names,
        "rows": int(len(X)),
        "normalised": bool(normalise),
        "date_range": (
            feature_frame.index.get_level_values("date").min().isoformat(),
            feature_frame.index.get_level_values("date").max().isoformat(),
        ),
    }
    logger.info("Dataset ready: %s rows, %s features.", len(X), len(feature_names))
    return X, y


PREPROCESSOR_SCHEMA = "classifier_features_v2"


def target_definition(horizon: int) -> Dict[str, Any]:
    return {"name": "direction_open_t1_to_close_tH", "horizon": int(horizon),
            "definition": f"close(t+{horizon}) / open(t+1) - 1 > 0; features up to close(t)",
            "feature_cutoff": "close of t", "execution": "open of t+1"}


def make_labels(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Forward traded return and label end BEFORE classification. Unknown labels stay NaN."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    px = frame["adj_close"] if "adj_close" in frame else frame["close"]
    if "open" in frame and "close" in frame:
        entry = frame["open"].shift(-1) * (px / frame["close"]).shift(-1)   # adjusted next open
    else:
        entry = px.shift(-1)
    fwd = px.shift(-horizon) / entry - 1
    frame["fwd_return"] = fwd
    frame["label_end"] = pd.Series(frame.index, index=frame.index).shift(-horizon)
    frame["target"] = np.where(fwd.notna(), (fwd > 0).astype(float), np.nan)
    return frame


def fit_preprocessor(X_train: pd.DataFrame, normalise: bool = True) -> Dict[str, Any]:
    """Fit missing-column rules, median imputation and scaling on a training fold only."""
    keep = [c for c in X_train.columns if X_train[c].notna().any()]   # all-missing in the fold: dropped
    imputer = SimpleImputer(strategy="median").fit(X_train[keep])
    scaler = StandardScaler().fit(imputer.transform(X_train[keep])) if normalise else None
    return {"schema": PREPROCESSOR_SCHEMA, "columns": keep, "input_columns": list(X_train.columns),
            "imputer": imputer, "scaler": scaler}


def apply_preprocessor(pp: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Transform with the fitted column order; missing columns become NaN (imputed), extra ones ignored."""
    if pp.get("schema") != PREPROCESSOR_SCHEMA:
        raise ValueError("incompatible preprocessor schema")
    data = pp["imputer"].transform(X.reindex(columns=pp["columns"]))
    return pp["scaler"].transform(data) if pp["scaler"] is not None else data


def purged_split(dates: pd.Index, label_end: pd.Series, test_frac: float = 0.2):
    """Date-grouped chronological split: every symbol of one decision date is on the same side; training rows
    whose label ends on/after the first test date are purged. Returns (train_mask, test_mask, info)."""
    d = pd.DatetimeIndex(dates)
    unique = np.array(sorted(d.unique()))
    if len(unique) < 3:
        raise RuntimeError("Not enough distinct dates for a train/test split.")
    first_test = unique[min(max(1, int((1 - test_frac) * len(unique))), len(unique) - 1)]
    test = d >= first_test
    ends = pd.DatetimeIndex(label_end)
    train = (d < first_test) & (ends < first_test)
    info = {"first_test_date": pd.Timestamp(first_test).isoformat(), "train_cutoff": pd.Timestamp(d[train].max()).isoformat(),
            "label_cutoff": pd.Timestamp(ends[train].max()).isoformat(), "purged_rows": int(((d < first_test) & ~train).sum())}
    return np.asarray(train), np.asarray(test), info


def _resolve_config_path(config_path: Optional[Union[str, pathlib.Path]]) -> pathlib.Path:
    if config_path is None:
        return project_path("config.yaml")
    path = pathlib.Path(config_path)
    if not path.is_absolute():
        path = project_path(str(path))
    return path


def _resolve_symbol_list(
    symbols: Optional[Sequence[str]],
    config: AppConfig,
    ingestor: DataIngestor,
    limit: Optional[int],
) -> List[str]:
    if symbols:
        resolved = [str(sym).strip().upper() for sym in symbols if str(sym).strip()]
    elif config.data.symbols:
        resolved = [sym.strip().upper() for sym in config.data.symbols if str(sym).strip()]
    else:
        resolved = _load_cached_symbols(config.data.base_path)
        if not resolved:
            try:
                resolved = ingestor.fetch_exchange_symbols(
                    exchange=config.data.exchange, max_symbols=limit
                )
            except Exception as err:
                logger.warning("Failed to fetch exchange symbols: %s", err)
                resolved = []
    if limit is not None:
        resolved = resolved[: int(limit)]
    return resolved


def _load_cached_symbols(base_path: Union[str, pathlib.Path]) -> List[str]:
    candidates = [
        project_path("available_symbols.json"),
        project_path(base_path, "available_symbols.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception as err:
                logger.debug("Failed to read cached symbols from %s: %s", path, err)
                continue
            if isinstance(raw, dict):
                items: Iterable[str] = raw.keys()
            else:
                items = raw
            return [str(item).strip().upper() for item in items if str(item).strip()]
    # Fallback to local OHLCV cache names.
    data_dir = project_path(base_path)
    if data_dir.exists():
        symbols = []
        for csv_file in data_dir.glob("*_ohlcv.csv"):
            ticker = csv_file.stem.replace("_ohlcv", "").upper()
            symbols.append(ticker)
        return sorted(symbols)
    return []


def _coerce_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(None)


def _load_recommendation_features(base_path: Union[str, pathlib.Path]) -> Dict[str, Dict[str, Any]]:
    directory = project_path(base_path, "recommendations")
    if not directory.exists():
        return {}
    stats: Dict[str, Dict[str, Any]] = {}
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as err:
            logger.debug("Skipping recommendation cache %s: %s", path, err)
            continue
        metadata = payload.get("metadata", {})
        date_val = metadata.get("date") or metadata.get("timestamp")
        date = _safe_timestamp(date_val)
        result = payload.get("result") or {}
        recs = result.get("recommendations") or []
        processed = result.get("processed") or []

        for entry in recs:
            symbol = str(entry.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            stat = stats.setdefault(
                symbol,
                {
                    "reco_count": 0,
                    "reco_prob_sum": 0.0,
                    "reco_score_sum": 0.0,
                    "reco_processed": 0,
                    "reco_last_prob": np.nan,
                    "reco_last_score": np.nan,
                    "reco_last_date": None,
                },
            )
            stat["reco_count"] += 1
            prob = _safe_float(entry.get("prob_up"))
            score = _safe_float(entry.get("score"))
            if np.isfinite(prob):
                stat["reco_prob_sum"] += prob
            if np.isfinite(score):
                stat["reco_score_sum"] += score
            if date is not None and (
                stat["reco_last_date"] is None or date >= stat["reco_last_date"]
            ):
                stat["reco_last_date"] = date
                stat["reco_last_prob"] = prob
                stat["reco_last_score"] = score

        for entry in processed:
            symbol = str(entry.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            stat = stats.setdefault(
                symbol,
                {
                    "reco_count": 0,
                    "reco_prob_sum": 0.0,
                    "reco_score_sum": 0.0,
                    "reco_processed": 0,
                    "reco_last_prob": np.nan,
                    "reco_last_score": np.nan,
                    "reco_last_date": None,
                },
            )
            stat["reco_processed"] += 1
            if entry.get("status") == "recommended":
                stat["reco_count"] = max(stat["reco_count"], 1)
    for symbol, stat in stats.items():
        count = stat.get("reco_count", 0)
        stat["reco_avg_prob"] = stat["reco_prob_sum"] / count if count else np.nan
        stat["reco_avg_score"] = stat["reco_score_sum"] / count if count else np.nan
    return stats


def _load_short_report_features(base_path: Union[str, pathlib.Path]) -> Dict[str, Dict[str, Any]]:
    directory = project_path(base_path, "short_reports")
    if not directory.exists():
        return {}
    stats: Dict[str, Dict[str, Any]] = {}
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as err:
            logger.debug("Skipping short report %s: %s", path, err)
            continue
        entries = payload.get("entries") or []
        for entry in entries:
            symbol = str(entry.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            reported = _safe_timestamp(entry.get("reported_date"))
            stat = stats.setdefault(
                symbol,
                {
                    "short_percent": np.nan,
                    "short_rank": np.nan,
                    "short_positions": np.nan,
                    "short_report_score": 0.0,
                    "short_report_date": None,
                },
            )
            percent = _safe_float(entry.get("short_percent"))
            rank = _safe_float(entry.get("short_rank"))
            positions = _safe_float(entry.get("short_positions"))
            status = str(entry.get("status") or "").lower()
            if np.isfinite(percent):
                stat["short_percent"] = percent
            if np.isfinite(rank):
                stat["short_rank"] = rank
            if np.isfinite(positions):
                stat["short_positions"] = positions
            stat["short_report_score"] = _score_short_status(status)
            if reported is not None and (
                stat["short_report_date"] is None or reported >= stat["short_report_date"]
            ):
                stat["short_report_date"] = reported
    return stats


def _augment_with_recommendations(
    dataset: pd.DataFrame, stats: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    if not stats:
        dataset["reco_count"] = 0.0
        dataset["reco_avg_prob"] = np.nan
        dataset["reco_last_prob"] = np.nan
        dataset["reco_days_since"] = np.nan
        return dataset
    symbols = dataset.index.get_level_values("symbol")
    dates = dataset.index.get_level_values("date")
    # These are snapshot aggregates (available at their last date), not a point-in-time history: they are used
    # only on rows dated on/after that availability date and are missing earlier (never broadcast backwards).
    avail = _available_mask(symbols, dates, stats, "reco_last_date")
    for col, key, default in (("reco_count", "reco_count", 0.0), ("reco_avg_prob", "reco_avg_prob", np.nan),
                              ("reco_last_prob", "reco_last_prob", np.nan)):
        vals = symbols.map(lambda s: stats.get(s, {}).get(key, default)).astype(float)
        dataset[col] = np.where(avail, vals, np.nan)
    dataset["reco_days_since"] = _days_since(symbols, dates, stats, "reco_last_date")
    return dataset


def _available_mask(symbols, dates, stats, date_key: str) -> np.ndarray:
    out = []
    for sym, dt_val in zip(symbols, dates):
        when = stats.get(sym, {}).get(date_key)
        out.append(when is not None and dt_val >= when)
    return np.array(out, dtype=bool)


def _days_since(symbols, dates, stats, date_key: str) -> np.ndarray:
    out = []
    for sym, dt_val in zip(symbols, dates):
        when = stats.get(sym, {}).get(date_key)
        out.append(float((dt_val - when).days) if when is not None and dt_val >= when else np.nan)
    return np.array(out, dtype=float)


def _augment_with_short_reports(
    dataset: pd.DataFrame, stats: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    if not stats:
        dataset["short_percent_cached"] = np.nan
        dataset["short_rank_cached"] = np.nan
        dataset["short_positions_cached"] = np.nan
        dataset["short_report_score"] = 0.0
        dataset["days_since_short_report"] = np.nan
        return dataset
    symbols = dataset.index.get_level_values("symbol")
    dates = dataset.index.get_level_values("date")
    avail = _available_mask(symbols, dates, stats, "short_report_date")   # as-of: no backwards broadcast
    for col, key in (("short_percent_cached", "short_percent"), ("short_rank_cached", "short_rank"),
                     ("short_positions_cached", "short_positions"), ("short_report_score", "short_report_score")):
        vals = symbols.map(lambda s: stats.get(s, {}).get(key, np.nan)).astype(float)
        dataset[col] = np.where(avail, vals, np.nan)
    dataset["days_since_short_report"] = _days_since(symbols, dates, stats, "short_report_date")
    return dataset


def _safe_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        return pd.Timestamp(value).tz_localize(None)
    except Exception:
        return None


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
        if np.isnan(result):
            return float("nan")
        return result
    except Exception:
        return float("nan")


def _score_short_status(status: str) -> float:
    mapping = {
        "recommended": 1.0,
        "candidate": 0.5,
        "candidate_not_selected": -0.5,
        "filtered_out": -1.0,
        "rejected": -1.0,
        "watch": -0.25,
    }
    return mapping.get(status, 0.0)


def build_sequence_dataset(
    *,
    config_path: Optional[Union[str, pathlib.Path]] = None,
    symbols: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    normalise: bool = True,
    save_processed: bool = True,
    sequence_length: int = 30,
    horizon: int = 1,
    static_features: Optional[Sequence[str]] = None,
    temporal_features: Optional[Sequence[str]] = None,
) -> SequenceDataset:
    """
    Build sliding-window tensors ready for sequence models such as LSTMs or Transformers.

    Drawing on *Deep Learning* (Goodfellow et al., 2016) we capture temporal dynamics using
    recurrent encoders, while *Applied Machine Learning and AI for Engineers* (Anand et al., 2024)
    motivates rigorous rolling-window evaluation. This helper aligns with those practices by
    producing chronological windows that can be fed into temporal models and scored via
    time-series cross-validation.
    """

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    X, y = build_dataset(
        config_path=config_path,
        symbols=symbols,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        normalise=normalise,
        save_processed=save_processed,
    )
    if X.empty or y.empty:
        raise RuntimeError("Cannot build sequence dataset from empty feature matrix.")

    columns = list(X.columns)
    resolved_static = _resolve_static_columns(columns, static_features)
    resolved_temporal = _resolve_temporal_columns(columns, temporal_features, resolved_static)
    if not resolved_temporal:
        raise RuntimeError("No temporal features selected for sequence modelling.")

    (
        sequences,
        static_matrix,
        targets,
        metadata,
    ) = _construct_temporal_windows(
        X,
        y,
        resolved_temporal,
        resolved_static,
        sequence_length,
        horizon,
    )

    if not sequences:
        raise RuntimeError(
            "Sequence construction failed; try reducing `sequence_length` or ensure symbols have enough history."
        )

    temporal_tensor = np.stack(sequences).astype(np.float32)
    if resolved_static:
        static_tensor = np.stack(static_matrix).astype(np.float32)
    else:
        static_tensor = np.zeros((len(sequences), 0), dtype=np.float32)
    target_array = np.array(targets, dtype=np.float32)

    preprocessor = X.attrs.get("preprocessor", {})

    return SequenceDataset(
        sequences=temporal_tensor,
        static_features=static_tensor,
        targets=target_array,
        metadata=metadata,
        temporal_features=resolved_temporal,
        static_feature_names=resolved_static,
        feature_names=list(resolved_temporal) + list(resolved_static),
        preprocessor=preprocessor,
        window=sequence_length,
        horizon=horizon,
    )


def _construct_temporal_windows(
    X: pd.DataFrame,
    y: pd.Series,
    temporal_cols: Sequence[str],
    static_cols: Sequence[str],
    window: int,
    horizon: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[SequenceSampleMeta]]:
    sequences: List[np.ndarray] = []
    static_matrix: List[np.ndarray] = []
    targets: List[float] = []
    metadata: List[SequenceSampleMeta] = []

    grouped = X.groupby(level="symbol", sort=False)
    for symbol, frame in grouped:
        frame_sorted = frame.sort_index()
        if isinstance(frame_sorted.index, pd.MultiIndex):
            frame_sorted.index = frame_sorted.index.get_level_values("date")
        frame_sorted.index = _ensure_naive_datetime_index(frame_sorted.index)
        target_series = y.loc[(symbol, slice(None))].sort_index()
        if isinstance(target_series.index, pd.MultiIndex):
            target_series = target_series.droplevel(0)
        target_series.index = _ensure_naive_datetime_index(target_series.index)
        target_series = target_series.reindex(frame_sorted.index)
        if len(frame_sorted) < window + (horizon - 1):
            continue
        upper_bound = len(frame_sorted) - horizon + 1
        for end_idx in range(window - 1, upper_bound):
            target_idx = end_idx + horizon - 1
            target_raw = target_series.iloc[target_idx]
            if pd.isna(target_raw):
                continue
            target_value = float(target_raw)
            seq_slice = frame_sorted.iloc[end_idx - window + 1 : end_idx + 1]
            temporal_values = seq_slice[temporal_cols].to_numpy(dtype=np.float32)
            if static_cols:
                static_snapshot = seq_slice.iloc[-1][list(static_cols)].to_numpy(dtype=np.float32)
            else:
                static_snapshot = np.empty((0,), dtype=np.float32)
            sequences.append(temporal_values)
            static_matrix.append(static_snapshot)
            targets.append(target_value)
            metadata.append(
                SequenceSampleMeta(
                    symbol=str(symbol),
                    window_end=_as_naive_timestamp(seq_slice.index[-1]),
                    target_reference=_as_naive_timestamp(target_series.index[target_idx]),
                    horizon=horizon,
                )
            )

    # Ensure chronological order for folds
    if metadata:
        order = np.argsort([meta.window_end.value for meta in metadata])
        sequences = [sequences[idx] for idx in order]
        static_matrix = [static_matrix[idx] for idx in order]
        targets = [targets[idx] for idx in order]
        metadata = [metadata[idx] for idx in order]

    return sequences, static_matrix, targets, metadata


def _resolve_static_columns(
    columns: Sequence[str], explicit: Optional[Sequence[str]]
) -> List[str]:
    if explicit:
        valid = [col for col in explicit if col in columns]
        return list(dict.fromkeys(valid))
    keywords = (
        "pe",
        "dividend",
        "eps",
        "market_cap",
        "reco_",
        "short_percent",
        "short_rank",
        "short_positions",
        "short_report",
        "days_since_short_report",
        "beta",
        "sector",
        "industry",
        "fundamental",
    )
    derived = [
        col for col in columns if any(keyword in col.lower() for keyword in keywords)
    ]
    return list(dict.fromkeys(derived))


def _resolve_temporal_columns(
    columns: Sequence[str],
    explicit: Optional[Sequence[str]],
    static_cols: Sequence[str],
) -> List[str]:
    if explicit:
        temporal = [col for col in explicit if col in columns]
    else:
        temporal = [col for col in columns if col not in static_cols]
    return list(dict.fromkeys(temporal))


def _as_naive_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _ensure_naive_datetime_index(index: Any) -> pd.DatetimeIndex:
    dt_index = pd.to_datetime(index)
    try:
        tz = getattr(dt_index, "tz", None)
    except Exception:  # pragma: no cover - defensive
        tz = None
    if tz is not None:
        dt_index = dt_index.tz_localize(None)
    return pd.DatetimeIndex(dt_index)
