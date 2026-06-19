from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys
from typing import Any, Dict, Optional, Sequence, Union

if __package__ is None or __package__ == "":
    # Allow running as a script via ``python ml/train_model.py`` by pushing the project root onto sys.path.
    current_path = pathlib.Path(__file__).resolve()
    project_root = current_path.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    lgb = None
    LIGHTGBM_AVAILABLE = False

from ml.dataset_builder import build_dataset
from stocktrade import ensure_directory, project_path

logger = logging.getLogger(__name__)


def train_model(
    config_path: Optional[Union[str, pathlib.Path]] = None,
    symbols: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    normalise: bool = True,
    save_processed: bool = True,
    algorithm: str = "auto",
    model_path: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """
    Train a supervised classifier to predict next-day price direction.

    The resulting artifact (model + preprocessing metadata) is persisted to ``ml/model.pkl``
    unless ``model_path`` is overridden.
    """

    X, y = build_dataset(
        config_path=config_path,
        symbols=symbols,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        normalise=normalise,
        save_processed=save_processed,
    )

    if y.empty or X.empty:
        raise RuntimeError("Dataset is empty; cannot train model.")

    preprocessor = X.attrs.get("preprocessor", {})
    scaler = preprocessor.get("scaler")
    imputer = preprocessor.get("imputer")
    if imputer is None:
        raise RuntimeError("Dataset build did not supply an imputer for preprocessing.")

    feature_names = list(X.columns)
    index_dates = X.index.get_level_values("date")
    unique_dates = np.array(sorted(index_dates.unique()))
    if unique_dates.shape[0] < 2:
        raise RuntimeError("Not enough distinct trading days to perform a train/test split.")

    split_idx = max(1, int(0.8 * len(unique_dates)))
    if split_idx >= len(unique_dates):
        split_idx = len(unique_dates) - 1
    split_date = unique_dates[split_idx]

    date_mask = pd.Series(index_dates <= split_date, index=X.index)
    train_mask = date_mask
    test_mask = ~train_mask
    if not test_mask.any():
        # Ensure at least the final day is held out.
        last_day = unique_dates[-1]
        test_mask = pd.Series(index_dates == last_day, index=X.index)
        train_mask = ~test_mask

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    if X_train.empty or X_test.empty:
        raise RuntimeError("Train/test split failed; one of the partitions is empty.")

    algorithm = (algorithm or "auto").lower()
    estimator, algorithm_used = _build_estimator(algorithm)
    logger.info(
        "Training %s on %s rows with %s features.",
        algorithm_used,
        len(X_train),
        len(feature_names),
    )

    estimator.fit(X_train.to_numpy(), y_train.to_numpy())

    train_proba = _predict_proba(estimator, X_train.to_numpy())
    test_proba = _predict_proba(estimator, X_test.to_numpy())

    threshold = _optimise_threshold(y_train.to_numpy(), train_proba)
    train_preds = (train_proba >= threshold).astype(int)
    test_preds = (test_proba >= threshold).astype(int)

    metrics = _compute_metrics(
        y_train.to_numpy(),
        train_proba,
        train_preds,
        y_test.to_numpy(),
        test_proba,
        test_preds,
    )

    logger.info("Test ROC-AUC: %.4f | Test F1: %.4f", metrics["test"]["roc_auc"], metrics["test"]["f1"])
    logger.debug(
        "Classification report (test):\n%s",
        classification_report(y_test, test_preds, digits=4),
    )

    artifact = {
        "model": estimator,
        "feature_names": feature_names,
        "preprocessor": {"imputer": imputer, "scaler": scaler},
        "threshold": float(threshold),
        "algorithm": algorithm_used,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metrics": metrics,
        "rows": {"train": int(len(X_train)), "test": int(len(X_test)), "total": int(len(X))},
        "normalised": bool(normalise),
    }

    target_path = pathlib.Path(model_path) if model_path else project_path("ml", "model.pkl")
    ensure_directory(target_path.parent)
    joblib.dump(artifact, target_path)
    logger.info("Model artifact saved to %s", target_path)
    return artifact


def _build_estimator(algorithm: str) -> Tuple[Any, str]:
    if algorithm == "lightgbm" or (algorithm == "auto" and LIGHTGBM_AVAILABLE):
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM requested but the package is not installed.")
        estimator = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        return estimator, "lightgbm"

    if algorithm not in {"auto", "logistic"}:
        raise ValueError(f"Unsupported algorithm selection: {algorithm}")
    estimator = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return estimator, "logistic_regression"


def _predict_proba(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(features)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(features)
        return 1.0 / (1.0 + np.exp(-decision))
    preds = estimator.predict(features)
    return preds.astype(float)


def _optimise_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if thresholds.size == 0:
        return 0.5
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    return max(0.01, min(best_threshold, 0.99))


def _compute_metrics(
    y_train: np.ndarray,
    proba_train: np.ndarray,
    preds_train: np.ndarray,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    preds_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    result = {
        "train": _single_metrics(y_train, proba_train, preds_train),
        "test": _single_metrics(y_test, proba_test, preds_test),
    }
    return result


def _single_metrics(y_true: np.ndarray, proba: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true, proba)
    metrics["accuracy"] = accuracy_score(y_true, preds)
    metrics["f1"] = f1_score(y_true, preds)
    metrics["brier"] = brier_score_loss(y_true, proba)
    return metrics


def _safe_metric(func, y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(func(y_true, proba))
    except ValueError:
        return float("nan")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and persist the next-day movement classification model."
    )
    parser.add_argument("--config", type=str, help="Path to config file (defaults to config.yaml).")
    parser.add_argument("--symbols", nargs="+", help="Optional explicit set of symbols to include.")
    parser.add_argument("--limit", type=int, help="Maximum number of symbols to load.")
    parser.add_argument("--start-date", type=str, help="Lower bound for historical data (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="Upper bound for historical data (YYYY-MM-DD).")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["auto", "lightgbm", "logistic"],
        default="auto",
        help="Model algorithm to use (auto tries LightGBM first).",
    )
    parser.add_argument(
        "--no-normalise",
        action="store_true",
        help="Disable z-score scaling (median imputation still applies).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the processed dataset snapshot to data/processed/.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Optional override for the output model artifact path.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train_model(
        config_path=args.config,
        symbols=args.symbols,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        normalise=not args.no_normalise,
        save_processed=not args.no_save,
        algorithm=args.algorithm,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
