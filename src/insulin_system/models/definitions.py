"""
Model definitions: baseline and advanced classifiers with default params and search grids.
"""

from typing import Any, Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Baseline
BASELINE_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
]
# Advanced
ADVANCED_MODELS = [
    "gradient_boosting",
    "svm_rbf",
    "mlp",
]

MODEL_NAMES = BASELINE_MODELS + ADVANCED_MODELS


def _lr_params() -> Tuple[Any, Dict[str, List[Any]]]:
    est = LogisticRegression(
        penalty="l2",
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
    )
    grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
    }
    return est, grid


def _dt_params() -> Tuple[Any, Dict[str, List[Any]]]:
    est = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "ccp_alpha": [0.0, 0.001, 0.01, 0.1],
    }
    return est, grid


def _rf_params() -> Tuple[Any, Dict[str, List[Any]]]:
    est = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    return est, grid


def _gb_params() -> Tuple[Any, Dict[str, List[Any]]]:
    try:
        import xgboost as xgb
        kwargs = {"objective": "multi:softmax", "num_class": 4, "random_state": 42}
        if hasattr(xgb.XGBClassifier, "use_label_encoder"):
            kwargs["use_label_encoder"] = False
            kwargs["eval_metric"] = "mlogloss"
        est = xgb.XGBClassifier(**kwargs)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        est = GradientBoostingClassifier(random_state=42)
    grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0] if "colsample_bytree" in dir(est) else [],
    }
    if not hasattr(est, "colsample_bytree"):
        grid = {k: v for k, v in grid.items() if k != "colsample_bytree"}
    return est, grid


def _svm_params() -> Tuple[Any, Dict[str, List[Any]]]:
    est = SVC(kernel="rbf", random_state=42, class_weight="balanced", probability=True)
    grid = {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto", 0.01, 0.1],
    }
    return est, grid


def _mlp_params() -> Tuple[Any, Dict[str, List[Any]]]:
    est = MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
    grid = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    }
    return est, grid


def get_model_definitions() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    """Return dict of model_name -> (estimator, param_grid)."""
    return {
        "logistic_regression": _lr_params(),
        "decision_tree": _dt_params(),
        "random_forest": _rf_params(),
        "gradient_boosting": _gb_params(),
        "svm_rbf": _svm_params(),
        "mlp": _mlp_params(),
    }
