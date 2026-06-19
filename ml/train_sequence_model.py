from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
import pathlib
import sys
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore

if __package__ is None or __package__ == "":
    current_path = pathlib.Path(__file__).resolve()
    project_root = current_path.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from ml.dataset_builder import SequenceDataset, SequenceSampleMeta, build_sequence_dataset
from ml.sequence_model import SequenceFusionModel, SequenceModelConfig
from stocktrade import ensure_directory, project_path

logger = logging.getLogger(__name__)


class TemporalTensorDataset(Dataset):
    """
    Torch dataset wrapping the sliding windows created by ``build_sequence_dataset``.
    """

    def __init__(self, data: SequenceDataset) -> None:
        self.sequences = torch.from_numpy(data.sequences).float()
        if data.static_features.size:
            self.static = torch.from_numpy(data.static_features).float()
        else:
            self.static = torch.zeros((len(data.sequences), 0), dtype=torch.float32)
        self.targets = torch.from_numpy(data.targets).float()

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.static[idx], self.targets[idx]


def train_sequence_model(
    *,
    config_path: Optional[pathlib.Path] = None,
    symbols: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    normalise: bool = True,
    save_processed: bool = True,
    sequence_length: int = 60,
    horizon: int = 1,
    static_features: Optional[Sequence[str]] = None,
    temporal_features: Optional[Sequence[str]] = None,
    epochs: int = 25,
    batch_size: int = 64,
    hidden_size: int = 128,
    temporal_layers: int = 2,
    dropout: float = 0.2,
    attention_heads: int = 4,
    feedforward_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    grad_clip: float = 1.0,
    folds: int = 4,
    test_fraction: float = 0.1,
    val_fraction: float = 0.1,
    mlflow_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    model_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """
    Train the sequence fusion model and persist the resulting artifact.

    The routine follows the guidance from *Applied Machine Learning and AI for Engineers* by
    adopting rolling ``TimeSeriesSplit`` folds, while optional MLflow logging reflects
    the MLOps practices promoted in the *LLM Engineer's Handbook*.
    """

    sequence_data = build_sequence_dataset(
        config_path=config_path,
        symbols=symbols,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        normalise=normalise,
        save_processed=save_processed,
        sequence_length=sequence_length,
        horizon=horizon,
        static_features=static_features,
        temporal_features=temporal_features,
    )

    dataset = TemporalTensorDataset(sequence_data)
    num_samples = len(dataset)
    if num_samples < 10:
        raise RuntimeError("Not enough samples to train the sequence model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    config = SequenceModelConfig(
        input_size=sequence_data.sequences.shape[-1],
        static_size=sequence_data.static_features.shape[-1],
        hidden_size=hidden_size,
        temporal_layers=temporal_layers,
        dropout=dropout,
        attention_heads=attention_heads,
        feedforward_size=feedforward_size,
    )

    indices = np.arange(num_samples)
    test_count = _resolve_holdout_count(num_samples, test_fraction, minimum=1)
    train_val_cutoff = num_samples - test_count
    if train_val_cutoff <= max(sequence_length, 5):
        raise RuntimeError("Insufficient history remaining after allocating the test set.")

    train_val_indices = indices[:train_val_cutoff]
    test_indices = indices[train_val_cutoff:]
    logger.info("Dataset split: %s train/val, %s test sequences.", len(train_val_indices), len(test_indices))

    mlflow_run = _initialise_mlflow(mlflow_uri, mlflow_experiment)
    if mlflow_run is not None and mlflow is not None:
        mlflow.log_params(
            {
                "sequence_length": sequence_length,
                "horizon": horizon,
                "hidden_size": hidden_size,
                "temporal_layers": temporal_layers,
                "dropout": dropout,
                "attention_heads": attention_heads,
                "feedforward_size": feedforward_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "patience": patience,
                "grad_clip": grad_clip,
                "folds": folds,
                "test_fraction": test_fraction,
                "val_fraction": val_fraction,
            }
        )

    fold_metrics: List[Dict[str, Dict[str, float]]] = []
    if folds >= 2 and len(train_val_indices) > folds:
        time_split = TimeSeriesSplit(n_splits=folds)
        for fold_id, (inner_train, inner_val) in enumerate(time_split.split(train_val_indices), start=1):
            actual_train = train_val_indices[inner_train]
            actual_val = train_val_indices[inner_val]
            logger.info(
                "Fold %s/%s -> %s training sequences, %s validation sequences.",
                fold_id,
                folds,
                len(actual_train),
                len(actual_val),
            )
            fold_result = _fit_single_model(
                dataset=dataset,
                train_indices=actual_train,
                val_indices=actual_val,
                config=config,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                grad_clip=grad_clip,
                fold_prefix=f"fold{fold_id}",
                mlflow_active=mlflow_run is not None,
            )
            fold_metrics.append(fold_result["metrics"])
    else:
        logger.warning("Skipping cross-validation; not enough samples for %s folds.", folds)

    val_count = _resolve_holdout_count(len(train_val_indices), val_fraction, minimum=1)
    if val_count >= len(train_val_indices):
        val_count = max(1, len(train_val_indices) // 5)
    final_train_indices = train_val_indices[:-val_count]
    final_val_indices = train_val_indices[-val_count:]

    final_result = _fit_single_model(
        dataset=dataset,
        train_indices=final_train_indices,
        val_indices=final_val_indices,
        config=config,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        grad_clip=grad_clip,
        fold_prefix="final",
        mlflow_active=mlflow_run is not None,
        capture_state=True,
    )

    final_state = final_result["state_dict"]
    threshold = final_result["threshold"]

    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)
    test_loss, test_probs, test_targets = _evaluate_model(
        final_state,
        config,
        dataset,
        test_loader,
        device,
    )
    test_preds = (test_probs >= threshold).astype(int)
    test_metrics = _compute_metrics(
        y_true=test_targets,
        y_pred=test_preds,
        y_proba=test_probs,
    )
    test_metrics["loss"] = float(test_loss)
    logger.info("Test ROC-AUC: %.4f | Accuracy: %.4f", test_metrics["roc_auc"], test_metrics["accuracy"])

    artifact_path = model_path or project_path("ml", "model.pt")
    ensure_directory(artifact_path.parent)

    metadata = _build_artifact_metadata(sequence_data.metadata, train_val_indices, test_indices)
    artifact = {
        "model_type": "sequence_lstm",
        "state_dict": final_state,
        "config": asdict(config),
        "sequence_length": sequence_length,
        "horizon": horizon,
        "temporal_features": sequence_data.temporal_features,
        "static_features": sequence_data.static_feature_names,
        "preprocessor": sequence_data.preprocessor,
        "threshold": float(threshold),
        "metrics": {
            "cross_validation": fold_metrics,
            "train": final_result["metrics"]["train"],
            "validation": final_result["metrics"]["validation"],
            "test": test_metrics,
        },
        "metadata": metadata,
        "training_timestamp": dt.datetime.utcnow().isoformat(),
        "ml_references": {
            "sequence_modelling": "Deep Learning (Goodfellow et al., 2016)",
            "evaluation": "Applied Machine Learning and AI for Engineers (Anand et al., 2024)",
            "mlops": "LLM Engineer's Handbook (Savinov et al., 2024)",
        },
    }

    torch.save(artifact, artifact_path)
    logger.info("Persisted sequence model to %s", artifact_path)

    if mlflow_run is not None and mlflow is not None:
        summary_metrics = _summarise_folds(fold_metrics)
        for metric, stats in summary_metrics.items():
            mlflow.log_metric(f"cv_{metric}_mean", stats["mean"])
            mlflow.log_metric(f"cv_{metric}_std", stats["std"])
        for split, split_metrics in artifact["metrics"].items():
            if isinstance(split_metrics, dict):
                for key, value in split_metrics.items():
                    if isinstance(value, (float, int)):
                        mlflow.log_metric(f"{split}_{key}", float(value))
        mlflow.log_artifact(str(artifact_path))
        mlflow.end_run()

    return artifact


def _fit_single_model(
    *,
    dataset: TemporalTensorDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config: SequenceModelConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    grad_clip: float,
    fold_prefix: str,
    mlflow_active: bool,
    capture_state: bool = False,
) -> Dict[str, Any]:
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

    model = SequenceFusionModel(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -np.inf
    best_metrics: Dict[str, Dict[str, float]] = {}
    best_threshold = 0.5
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_loss, val_probs, val_targets = _evaluate_loader(model, val_loader, criterion, device)
        val_threshold = 0.5  # placeholder before optimisation
        val_preds = (val_probs >= val_threshold).astype(int)
        val_metrics = _compute_metrics(val_targets, val_preds, val_probs)
        val_metrics["loss"] = float(val_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": float(val_metrics["roc_auc"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )

        roc_value = val_metrics["roc_auc"]
        if math.isnan(roc_value):
            roc_value = -np.inf
        if roc_value > best_val_auc + 1e-4 or not best_metrics:
            best_val_auc = roc_value
            best_state = copy.deepcopy(model.state_dict())
            train_loss_eval, train_probs_eval, train_targets_eval = _evaluate_loader(
                model, train_loader, criterion, device
            )
            threshold = _optimise_threshold(train_targets_eval, train_probs_eval)
            train_preds_eval = (train_probs_eval >= threshold).astype(int)
            train_metrics = _compute_metrics(train_targets_eval, train_preds_eval, train_probs_eval)
            train_metrics["loss"] = float(train_loss_eval)
            val_preds_eval = (val_probs >= threshold).astype(int)
            val_metrics_refined = _compute_metrics(val_targets, val_preds_eval, val_probs)
            val_metrics_refined["loss"] = float(val_loss)
            best_metrics = {"train": train_metrics, "validation": val_metrics_refined}
            best_threshold = float(threshold)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("%s early stopping at epoch %s", fold_prefix, epoch)
                break

    model.load_state_dict(best_state)

    if mlflow_active and mlflow is not None:
        for record in history:
            epoch_idx = int(record["epoch"])
            mlflow.log_metric(f"{fold_prefix}_val_auc", record["val_auc"], step=epoch_idx)
            mlflow.log_metric(f"{fold_prefix}_train_loss", record["train_loss"], step=epoch_idx)
            mlflow.log_metric(f"{fold_prefix}_val_loss", record["val_loss"], step=epoch_idx)

    result = {
        "metrics": best_metrics,
        "threshold": best_threshold,
    }
    if capture_state:
        cpu_state = {key: value.detach().cpu() for key, value in best_state.items()}
        result["state_dict"] = cpu_state
    return result


def _train_epoch(
    model: SequenceFusionModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    for temporal, static, target in loader:
        temporal = temporal.to(device)
        static = static.to(device) if static.numel() else None
        target = target.to(device)
        optimizer.zero_grad()
        logits, _ = model(temporal, static)
        loss = criterion(logits, target)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * temporal.size(0)
    return total_loss / max(1, len(loader.dataset))


def _evaluate_loader(
    model: SequenceFusionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    probabilities: List[float] = []
    targets: List[float] = []
    with torch.no_grad():
        for temporal, static, target in loader:
            temporal = temporal.to(device)
            static = static.to(device) if static.numel() else None
            target = target.to(device)
            logits, _ = model(temporal, static)
            loss = criterion(logits, target)
            total_loss += loss.item() * temporal.size(0)
            probs = torch.sigmoid(logits)
            probabilities.extend(probs.detach().cpu().numpy().tolist())
            targets.extend(target.detach().cpu().numpy().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, np.asarray(probabilities, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _evaluate_model(
    state_dict: Dict[str, Any],
    config: SequenceModelConfig,
    dataset: TemporalTensorDataset,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model = SequenceFusionModel(config).to(device)
    model.load_state_dict(state_dict)
    criterion = nn.BCEWithLogitsLoss()
    return _evaluate_loader(model, loader, criterion, device)


def _optimise_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if thresholds.size == 0:
        return 0.5
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    return max(0.05, min(best_threshold, 0.95))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true, y_proba)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["brier"] = brier_score_loss(y_true, y_proba)
    return {key: float(value) for key, value in metrics.items()}


def _safe_metric(func, y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(func(y_true, proba))
    except ValueError:
        return float("nan")


def _resolve_holdout_count(total: int, fraction: float, minimum: int) -> int:
    if total <= 0:
        return 0
    if fraction <= 0:
        return max(minimum, 1)
    if fraction < 1:
        count = int(round(total * fraction))
    else:
        count = int(fraction)
    count = max(minimum, count)
    return min(count, total - minimum)


def _build_artifact_metadata(
    metadata: Sequence[SequenceSampleMeta],
    train_val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Dict[str, Any]:
    if not metadata:
        return {}
    train_meta = [metadata[idx] for idx in train_val_indices]
    test_meta = [metadata[idx] for idx in test_indices]
    symbols = sorted({meta.symbol for meta in metadata})
    return {
        "symbols": symbols,
        "train_range": (
            str(train_meta[0].window_end.date()) if train_meta else None,
            str(train_meta[-1].window_end.date()) if train_meta else None,
        ),
        "test_range": (
            str(test_meta[0].window_end.date()) if test_meta else None,
            str(test_meta[-1].window_end.date()) if test_meta else None,
        ),
        "sample_counts": {
            "train_val": len(train_meta),
            "test": len(test_meta),
        },
    }


def _summarise_folds(fold_metrics: Sequence[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    if not fold_metrics:
        return {}
    aggregate: Dict[str, List[float]] = {}
    for metrics in fold_metrics:
        validation_metrics = metrics.get("validation", {})
        for key, value in validation_metrics.items():
            aggregate.setdefault(key, []).append(float(value))
    summary = {}
    for key, values in aggregate.items():
        arr = np.array(values, dtype=np.float32)
        summary[key] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
        }
    return summary


def _initialise_mlflow(uri: Optional[str], experiment: Optional[str]):
    if not uri or mlflow is None:
        return None
    try:
        mlflow.set_tracking_uri(uri)
        if experiment:
            mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=f"sequence_model_{dt.datetime.utcnow().isoformat()}")
        return run
    except Exception as err:  # pragma: no cover - fails only when mlflow misconfigured
        logger.warning("MLflow initialisation failed: %s", err)
        return None


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the deep sequence model for stock movement prediction."
    )
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument("--symbols", nargs="+", help="Optional subset of tickers.")
    parser.add_argument("--limit", type=int, help="Maximum symbols to load.")
    parser.add_argument("--start-date", type=str, help="Lower bound for price history (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="Upper bound for price history (YYYY-MM-DD).")
    parser.add_argument("--no-normalise", action="store_true", help="Disable z-score scaling.")
    parser.add_argument("--no-save", action="store_true", help="Skip saving processed dataset snapshot.")
    parser.add_argument("--sequence-length", type=int, default=60, help="Sequence length (timesteps).")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in days.")
    parser.add_argument("--static-features", nargs="+", help="Explicit static feature columns.")
    parser.add_argument("--temporal-features", nargs="+", help="Explicit temporal feature columns.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per fit.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size.")
    parser.add_argument("--temporal-layers", type=int, default=2, help="Number of stacked LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--attention-heads", type=int, default=4, help="Number of attention heads (0 to disable).")
    parser.add_argument("--feedforward-size", type=int, default=128, help="Hidden width of fusion MLP.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularisation strength.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--folds", type=int, default=4, help="Number of time-series CV folds.")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of samples held out for test.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of train+val reserved for validation.")
    parser.add_argument("--mlflow-uri", type=str, help="MLflow tracking URI.")
    parser.add_argument("--mlflow-experiment", type=str, help="MLflow experiment name.")
    parser.add_argument("--model-path", type=str, help="Override output path for the model artifact.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_path = pathlib.Path(args.model_path) if args.model_path else None
    config_path = pathlib.Path(args.config) if args.config else None

    train_sequence_model(
        config_path=config_path,
        symbols=args.symbols,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        normalise=not args.no_normalise,
        save_processed=not args.no_save,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        static_features=args.static_features,
        temporal_features=args.temporal_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        temporal_layers=args.temporal_layers,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
        feedforward_size=args.feedforward_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        folds=args.folds,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
