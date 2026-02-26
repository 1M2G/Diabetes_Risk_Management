"""
Inference bundle: full preprocessing + model for saving/loading and prediction.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config.schema import DataSchema
from ..data_processing.pipeline import PipelineResult

logger = logging.getLogger(__name__)

# Default path for the saved best model used by the system
DEFAULT_BEST_MODEL_DIR = Path("outputs/best_model")
BUNDLE_FILENAME = "inference_bundle.joblib"
METADATA_FILENAME = "metadata.json"


class InferenceBundle:
    """
    Fitted preprocessing pipeline + model for inference.
    Accepts raw DataFrames (same schema as training) and returns predictions.
    """

    def __init__(
        self,
        pipeline_result: PipelineResult,
        model: Any,
        model_name: str,
        metric_name: str = "f1_weighted",
        metric_value: float = 0.0,
    ) -> None:
        self._schema = DataSchema()
        self._imputer = pipeline_result.imputer
        self._outlier_handler = pipeline_result.outlier_handler
        self._feature_engineer = pipeline_result.feature_engineer
        self._encoder = pipeline_result.encoder
        self._scaler = pipeline_result.scaler
        self._feature_selector = pipeline_result.feature_selector
        self._model = model
        self._feature_names = list(pipeline_result.feature_names)
        self._model_name = model_name
        self._metric_name = metric_name
        self._metric_value = float(metric_value)
        self._classes_ = self._get_classes()

    def _get_classes(self) -> np.ndarray:
        est = self._model
        if hasattr(est, "named_steps") and "clf" in getattr(est, "named_steps", {}):
            est = est.named_steps["clf"]
        if hasattr(est, "classes_"):
            return np.asarray(est.classes_)
        return np.array([])

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def classes_(self) -> np.ndarray:
        return self._classes_

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def metric_value(self) -> float:
        return self._metric_value

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Run full preprocessing: impute -> outlier -> feature_engineer -> encode -> scale -> select.
        Returns feature matrix X (n_samples, n_features) for model input.
        """
        if self._imputer is None:
            raise RuntimeError("InferenceBundle not properly initialized (missing imputer)")
        out = self._imputer.transform(df)
        if self._outlier_handler is not None:
            out = self._outlier_handler.transform(out)
        if self._feature_engineer is not None:
            out = self._feature_engineer.transform(out)
        out = self._encoder.transform(out)
        out = self._scaler.transform(out)
        exclude = {self._schema.PATIENT_ID, self._schema.TARGET, "_outlier_flag"}
        feat_cols = [c for c in out.columns if c not in exclude]
        X = out[feat_cols]
        if self._feature_selector is not None:
            X = self._feature_selector.transform(X)
        return np.asarray(X, dtype=np.float64)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels. X: raw DataFrame (schema as training) or already transformed array."""
        if isinstance(X, pd.DataFrame):
            X = self.transform(X)
        return self._model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities. X: raw DataFrame or transformed array."""
        if isinstance(X, pd.DataFrame):
            X = self.transform(X)
        if hasattr(self._model, "predict_proba"):
            return np.asarray(self._model.predict_proba(X))
        return np.zeros((len(X), len(self._classes_)), dtype=np.float64)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "metric_name": self._metric_name,
            "metric_value": self._metric_value,
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names,
            "classes": self._classes_.tolist(),
        }


def save_best_model(
    bundle: InferenceBundle,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save the inference bundle (preprocessors + model) to disk.
    Writes inference_bundle.joblib and metadata.json.
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required for saving the model. Install with: pip install joblib")

    out_dir = Path(output_dir) if output_dir else DEFAULT_BEST_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full bundle with joblib (model + preprocessors)
    bundle_path = out_dir / BUNDLE_FILENAME
    joblib.dump(bundle, bundle_path)
    logger.info("Saved inference bundle to %s", bundle_path)

    meta_path = out_dir / METADATA_FILENAME
    meta = bundle.to_metadata()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    return out_dir


def load_best_model(
    model_dir: Optional[Path] = None,
) -> InferenceBundle:
    """
    Load the inference bundle from disk (default: outputs/best_model).
    Returns InferenceBundle for predict / predict_proba / transform.
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required for loading the model. Install with: pip install joblib")

    dir_path = Path(model_dir) if model_dir else DEFAULT_BEST_MODEL_DIR
    bundle_path = dir_path / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {bundle_path}. "
            "Run evaluation first (e.g. python run_evaluation.py) to train and save the best model."
        )
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, InferenceBundle):
        raise TypeError(f"Expected InferenceBundle, got {type(bundle)}")
    logger.info("Loaded inference bundle from %s (model: %s)", bundle_path, bundle.model_name)
    return bundle
