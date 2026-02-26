"""
Model training with hyperparameter tuning, stratified CV, and class imbalance handling.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from ..config.schema import ModelConfig
from ..exceptions import DataValidationError
from .definitions import get_model_definitions

logger = logging.getLogger(__name__)


def _is_xgboost(estimator: Any) -> bool:
    """True if estimator is XGBClassifier (does not support class_weight)."""
    try:
        import xgboost as xgb
        return isinstance(estimator, xgb.XGBClassifier)
    except ImportError:
        return False


@dataclass
class TrainingResult:
    """Result of training a single model."""

    model_name: str
    best_estimator: Any
    best_params: Dict[str, Any]
    best_cv_score: float
    cv_results: Optional[Dict[str, Any]] = None
    classes_: Optional[np.ndarray] = None


def _make_estimator_with_imbalance(
    base_estimator: Any,
    imbalance_strategy: str,
    random_state: int,
    smote_k_neighbors: int = 5,
) -> Tuple[Any, bool]:
    """Wrap estimator in a Pipeline with optional SMOTE; return (estimator, is_pipeline)."""
    is_pipeline = False
    if imbalance_strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            smote = SMOTE(random_state=random_state, k_neighbors=min(smote_k_neighbors, 5))
            pipe = ImbPipeline([("smote", smote), ("clf", base_estimator)])
            return pipe, True
        except ImportError:
            logger.warning("imbalanced-learn not installed; falling back to class_weight")
            imbalance_strategy = "class_weight"
    if imbalance_strategy == "class_weight" and hasattr(base_estimator, "set_params"):
        if not _is_xgboost(base_estimator):
            try:
                base_estimator.set_params(class_weight="balanced")
            except Exception:
                pass
    return base_estimator, is_pipeline


def _get_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights from class frequencies (for XGBoost etc.)."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_classes = len(classes)
    weight_per_class = n / (n_classes * counts)
    return weight_per_class[np.searchsorted(classes, y)]


def _needs_label_encoding(base_estimator: Any) -> bool:
    """True if the estimator expects integer labels (XGBoost, MLPClassifier with early_stopping)."""
    if _is_xgboost(base_estimator):
        return True
    from sklearn.neural_network import MLPClassifier
    if isinstance(base_estimator, MLPClassifier):
        return True
    return False


class _LabelEncoderWrapper:
    """Wraps an estimator so predict/predict_proba return original labels (inverse of LabelEncoder)."""

    def __init__(self, estimator: Any, label_encoder: LabelEncoder):
        self._estimator = estimator
        self._le = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X: Any) -> np.ndarray:
        pred = self._estimator.predict(X)
        if pred.dtype.kind in ("i", "u") or np.issubdtype(pred.dtype, np.integer):
            return self._le.inverse_transform(pred.astype(int))
        return pred

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._estimator.predict_proba(X)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "_LabelEncoderWrapper":
        return self


class ModelTrainer:
    """
    Trains a single model or all models with hyperparameter tuning,
    stratified CV, and optional SMOTE/class_weight for imbalance.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self._config = config or ModelConfig()
        self._definitions = get_model_definitions()

    def train_single(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> TrainingResult:
        """
        Train one model with tuning. Uses stratified CV; optionally evaluate on val set.
        """
        if model_name not in self._definitions:
            raise DataValidationError(f"Unknown model: {model_name}")
        base_est, param_grid = self._definitions[model_name]
        import sklearn.base
        estimator = sklearn.base.clone(base_est)
        estimator, is_pipeline = _make_estimator_with_imbalance(
            estimator,
            self._config.imbalance_strategy,
            self._config.random_state,
            self._config.smote_k_neighbors,
        )
        if is_pipeline:
            param_grid = {"clf__" + k: v for k, v in param_grid.items()}
        cv = StratifiedKFold(
            n_splits=self._config.cv_folds,
            shuffle=True,
            random_state=self._config.random_state,
        )
        if self._config.search_type == "random":
            search = RandomizedSearchCV(
                estimator,
                param_distributions=param_grid,
                n_iter=min(self._config.random_search_n_iter, self._total_combinations(param_grid)),
                cv=cv,
                scoring=self._config.scoring,
                n_jobs=self._config.n_jobs,
                random_state=self._config.random_state,
                refit=True,
            )
        else:
            search = GridSearchCV(
                estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=self._config.scoring,
                n_jobs=self._config.n_jobs,
                refit=True,
            )
        X = X_train if isinstance(X_train, np.ndarray) else X_train.values
        y = y_train if isinstance(y_train, np.ndarray) else y_train.values
        label_encoder = None
        if _needs_label_encoding(base_est) and (y.dtype == object or y.dtype.kind in ("U", "S") or not np.issubdtype(y.dtype, np.integer)):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        fit_params = {}
        try:
            if "XGB" in type(base_est).__name__ and self._config.imbalance_strategy == "class_weight":
                fit_params["sample_weight"] = _get_sample_weights(y)
            elif is_pipeline and "XGB" in type(base_est).__name__:
                fit_params["clf__sample_weight"] = _get_sample_weights(y)
        except Exception:
            pass
        if fit_params:
            search.fit(X, y, **fit_params)
        else:
            search.fit(X, y)
        best = search.best_estimator_
        if label_encoder is not None:
            best = _LabelEncoderWrapper(best, label_encoder)
        if hasattr(best, "classes_"):
            classes_ = best.classes_
        elif hasattr(best, "named_steps") and "clf" in best.named_steps:
            classes_ = getattr(best.named_steps["clf"], "classes_", np.unique(y_train))
        else:
            classes_ = np.unique(y_train) if hasattr(y_train, "unique") else np.unique(y)
        return TrainingResult(
            model_name=model_name,
            best_estimator=best,
            best_params=search.best_params_,
            best_cv_score=float(search.best_score_),
            cv_results=search.cv_results_,
            classes_=classes_,
        )

    def _total_combinations(self, param_grid: Dict[str, List[Any]]) -> int:
        n = 1
        for v in param_grid.values():
            n *= len(v)
        return n

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        model_names: Optional[List[str]] = None,
    ) -> List[TrainingResult]:
        """Train all (or selected) models and return list of TrainingResult."""
        names = model_names or list(self._definitions.keys())
        results = []
        for name in names:
            if name not in self._definitions:
                continue
            logger.info("Training model: %s", name)
            try:
                res = self.train_single(name, X_train, y_train, X_val, y_val)
                results.append(res)
            except Exception as e:
                logger.exception("Model %s failed: %s", name, e)
        return results
