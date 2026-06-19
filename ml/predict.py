from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd

from stocktrade import project_path

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

from ml.sequence_model import SequenceFusionModel, SequenceModelConfig

_ARTIFACT_CACHE: Dict[pathlib.Path, Dict[str, Any]] = {}
DEFAULT_MODEL_CANDIDATES = (
    project_path("ml", "model.pt"),
    project_path("ml", "model.pkl"),
)


def load_model_artifact(model_path: Optional[Union[str, pathlib.Path]] = None) -> Dict[str, Any]:
    """
    Load and cache the persisted model artifact produced by ``ml.train_model``.
    """
    resolved_path: Optional[pathlib.Path]
    if model_path:
        resolved_path = pathlib.Path(model_path)
    else:
        resolved_path = None
        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                resolved_path = candidate
                break
    if resolved_path is None:
        raise FileNotFoundError(
            f"No model artifact found. Expected one of {[str(p) for p in DEFAULT_MODEL_CANDIDATES]}."
        )
    path = resolved_path
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Train the model first.")
    cached = _ARTIFACT_CACHE.get(path)
    if cached is not None:
        return cached

    if path.suffix.lower() == ".pt":
        if torch is None:
            raise RuntimeError("PyTorch is required to load sequence model artifacts.")
        artifact = torch.load(path, map_location="cpu")
    else:
        artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Unexpected artifact format at {path}: expected dict, got {type(artifact)}")
    _ARTIFACT_CACHE[path] = artifact
    logger.info("Loaded model artifact from %s", path)
    return artifact


def predict_next_day(
    input_features: Union[pd.Series, pd.DataFrame, Mapping[str, Any]],
    model_path: Optional[Union[str, pathlib.Path]] = None,
    history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Generate a next-day direction prediction for a single feature row or sliding window.

    Parameters
    ----------
    input_features:
        Feature values as a mapping/Series/DataFrame. Column names must align with the training feature set.
        When the artefact is a sequence model, this may be ignored if ``history`` is provided.
    model_path:
        Optional override path for the model artifact. Defaults to the first available candidate.
    history:
        Chronologically ordered feature DataFrame. Required for sequence models so that the final
        ``sequence_length`` timesteps can be extracted.
    """

    artifact = load_model_artifact(model_path)
    model_type = artifact.get("model_type", "sklearn")

    if model_type == "sequence_lstm":
        if history is None and isinstance(input_features, pd.DataFrame):
            history = input_features
        if history is None:
            raise ValueError(
                "Sequence model inference requires a `history` DataFrame containing at least "
                f"{artifact.get('sequence_length')} timesteps."
            )
        return _predict_sequence_model(artifact, history)

    model = artifact["model"]
    feature_names: Sequence[str] = artifact["feature_names"]
    preprocessor: Mapping[str, Any] = artifact.get("preprocessor", {})
    imputer = preprocessor.get("imputer")
    scaler = preprocessor.get("scaler")
    if imputer is None:
        raise RuntimeError("Loaded artifact does not include an imputer.")

    if history is not None and isinstance(history, pd.DataFrame) and len(history) >= 1:
        latest_row = history.iloc[[-1]].copy()
        for col in feature_names:
            if col not in latest_row.columns:
                latest_row[col] = np.nan
        features_df = _coerce_features(latest_row, feature_names)
    else:
        features_df = _coerce_features(input_features, feature_names)

    data = imputer.transform(features_df)
    if scaler is not None:
        data = scaler.transform(data)
    prob_matrix = _predict_proba(model, data)
    prob_up = float(prob_matrix[0])
    threshold = float(artifact.get("threshold", 0.5))
    predicted_class = int(prob_up >= threshold)

    explanation = _build_explanation(model, data, feature_names)

    return {
        "prob_up": prob_up,
        "predicted_class": predicted_class,
        "explanation": explanation,
        "threshold": threshold,
    }


def _coerce_features(
    features: Union[pd.Series, pd.DataFrame, Mapping[str, Any]],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        df = features.copy()
        if len(df) == 0:
            raise ValueError("predict_next_day received an empty DataFrame.")
        if len(df) > 1:
            df = df.tail(1)
    elif isinstance(features, pd.Series):
        df = features.to_frame().T
    else:
        df = pd.DataFrame([dict(features)])
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    df = df.reindex(columns=feature_names)
    return df.astype(float, errors="ignore")


def _predict_proba(model: Any, data: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(data)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(data)
        return 1.0 / (1.0 + np.exp(-decision))
    preds = model.predict(data)
    return preds.astype(float)


def _build_explanation(
    model: Any,
    data: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    try:
        import shap  # type: ignore

        if hasattr(model, "booster_") or model.__class__.__name__.lower().startswith("lgbm"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            contributions = shap_values[0]
            ranked = np.argsort(np.abs(contributions))[::-1][:5]
            top = [
                {"feature": feature_names[idx], "contribution": float(contributions[idx])}
                for idx in ranked
            ]
            return {"method": "shap", "top_features": top}
    except ImportError:
        logger.debug("SHAP not installed; falling back to linear/importance explanation.")
    except Exception as err:
        logger.debug("SHAP explanation failed: %s", err)

    coef = getattr(model, "coef_", None)
    if coef is not None:
        weights = coef[0]
        contributions = data[0] * weights
        ranked = np.argsort(np.abs(contributions))[::-1][:5]
        top = [
            {
                "feature": feature_names[idx],
                "weight": float(weights[idx]),
                "contribution": float(contributions[idx]),
            }
            for idx in ranked
        ]
        return {"method": "linear_weights", "top_features": top}

    importance = getattr(model, "feature_importances_", None)
    if importance is not None:
        ranked = np.argsort(importance)[::-1][:5]
        top = [
            {"feature": feature_names[idx], "importance": float(importance[idx])}
            for idx in ranked
        ]
        return {"method": "feature_importance", "top_features": top}

    return {"method": "none", "detail": "Explanation unavailable for this model type."}


def _predict_sequence_model(artifact: Dict[str, Any], history: pd.DataFrame) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to run the sequence model.")
    required = {"state_dict", "config", "temporal_features", "sequence_length"}
    missing = required - set(artifact.keys())
    if missing:
        raise ValueError(f"Sequence artifact missing keys: {sorted(missing)}")

    temporal_features: Sequence[str] = artifact["temporal_features"]
    static_features: Sequence[str] = artifact.get("static_features", [])
    sequence_length = int(artifact["sequence_length"])
    preprocessor: Mapping[str, Any] = artifact.get("preprocessor", {})
    imputer = preprocessor.get("imputer")
    scaler = preprocessor.get("scaler")

    ordered_columns = list(temporal_features) + list(static_features)
    frame = history.copy()
    frame = frame.sort_index()
    for col in ordered_columns:
        if col not in frame.columns:
            frame[col] = np.nan
    frame = frame[ordered_columns]
    if len(frame) < sequence_length:
        raise ValueError(
            f"Insufficient history: need {sequence_length} rows but received {len(frame)}."
        )
    window = frame.tail(sequence_length)
    numeric_matrix = window.apply(pd.to_numeric, errors="coerce").to_numpy()
    if imputer is not None:
        numeric_matrix = imputer.transform(numeric_matrix)
    if scaler is not None:
        numeric_matrix = scaler.transform(numeric_matrix)

    temporal_count = len(temporal_features)
    temporal_array = numeric_matrix[:, :temporal_count].astype(np.float32)
    if static_features:
        static_array = numeric_matrix[-1, temporal_count:].astype(np.float32)
    else:
        static_array = np.zeros((0,), dtype=np.float32)

    temporal_tensor = torch.from_numpy(temporal_array).unsqueeze(0)
    static_tensor = torch.from_numpy(static_array).unsqueeze(0) if static_features else None

    config = SequenceModelConfig(**artifact["config"])
    model = artifact.get("_model_instance")
    if model is None:
        model = SequenceFusionModel(config)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        artifact["_model_instance"] = model
    else:
        model.eval()

    with torch.no_grad():
        logits, _ = model(temporal_tensor, static_tensor)
        prob_up = float(torch.sigmoid(logits)[0].cpu().item())

    threshold = float(artifact.get("threshold", 0.5))
    predicted_class = int(prob_up >= threshold)

    explanation = _build_sequence_explanation(window, temporal_features, static_features)

    return {
        "prob_up": prob_up,
        "predicted_class": predicted_class,
        "threshold": threshold,
        "explanation": explanation,
    }


def _build_sequence_explanation(
    window: pd.DataFrame,
    temporal_features: Sequence[str],
    static_features: Sequence[str],
) -> Dict[str, Any]:
    window_start = window.index[0]
    window_end = window.index[-1]
    latest = window.iloc[-1]
    deltas = (latest - window.iloc[0]).fillna(0.0)
    magnitudes = deltas[temporal_features].abs().sort_values(ascending=False)
    top_keys = magnitudes.index[:5]
    top_features = [
        {
            "feature": key,
            "delta": float(deltas.get(key, 0.0)),
            "latest": float(latest.get(key, 0.0)),
        }
        for key in top_keys
    ]
    static_snapshot = {key: float(latest.get(key, 0.0)) for key in static_features}
    return {
        "method": "temporal_delta",
        "window_start": str(window_start),
        "window_end": str(window_end),
        "top_changes": top_features,
        "static_snapshot": static_snapshot,
    }
