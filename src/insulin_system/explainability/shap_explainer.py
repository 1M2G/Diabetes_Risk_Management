"""
SHAP explainer: global and local explanations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.schema import ExplainabilityConfig

logger = logging.getLogger(__name__)


def _get_estimator_for_shap(estimator: Any) -> Any:
    """Unwrap pipeline to get inner estimator for SHAP (tree vs kernel)."""
    if hasattr(estimator, "named_steps") and "clf" in getattr(estimator, "named_steps", {}):
        return estimator.named_steps["clf"]
    return estimator


def _is_tree_based(estimator: Any) -> bool:
    try:
        import xgboost as xgb
        if isinstance(estimator, xgb.XGBClassifier):
            return True
    except ImportError:
        pass
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    return isinstance(estimator, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier))


@dataclass
class ExplainerResult:
    """Artifacts from running SHAP explainer."""
    model_name: str
    shap_values: Any  # np.ndarray or list of arrays (multi-class)
    base_values: Any
    feature_names: List[str]
    paths: Dict[str, str] = field(default_factory=dict)
    explainer_type: str = ""


class SHAPExplainer:
    """Builds SHAP explainer and produces global + local plots."""

    def __init__(self, config: Optional[ExplainabilityConfig] = None):
        self._cfg = config or ExplainabilityConfig()
        self._explainer = None
        self._shap_values = None
        self._base_values = None

    def fit(
        self,
        estimator: Any,
        X_background: np.ndarray,
        feature_names: List[str],
    ) -> "SHAPExplainer":
        import shap
        est = _get_estimator_for_shap(estimator)
        X_bg = np.asarray(X_background)
        if X_bg.size > self._cfg.background_size * X_bg.shape[1]:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_bg.shape[0], size=min(self._cfg.background_size, X_bg.shape[0]), replace=False)
            X_bg = X_bg[idx]
        if _is_tree_based(est):
            self._explainer = shap.TreeExplainer(est, X_bg, feature_perturbation="interaction")
            self._explainer_type = "tree"
        elif hasattr(est, "coef_") and getattr(est, "coef_", None) is not None:
            try:
                masker = shap.maskers.Independent(X_bg, max_samples=self._cfg.background_size)
                self._explainer = shap.LinearExplainer(est, masker)
                self._explainer_type = "linear"
            except Exception:
                self._explainer = shap.KernelExplainer(
                    est.predict_proba if hasattr(est, "predict_proba") else est.predict,
                    X_bg,
                )
                self._explainer_type = "kernel"
        else:
            self._explainer = shap.KernelExplainer(
                est.predict_proba if hasattr(est, "predict_proba") else est.predict,
                X_bg,
            )
            self._explainer_type = "kernel"
        self._estimator = estimator
        self._feature_names = feature_names
        return self

    def explain(
        self,
        X: np.ndarray,
        model_name: str = "model",
        output_dir: Optional[Path] = None,
        segment_col: Optional[np.ndarray] = None,
    ) -> ExplainerResult:
        import shap
        import matplotlib.pyplot as plt
        out_dir = Path(output_dir) if output_dir else self._cfg.ensure_output_dir()
        out_dir = out_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        X = np.asarray(X)

        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            # Multi-class: stack to (n_samples, n_features, n_classes) or use predicted class
            pred = self._estimator.predict(X)
            classes_ = getattr(self._estimator, "classes_", np.unique(pred))
            if hasattr(classes_, "__len__") and len(classes_) > 1:
                # Use mean abs across classes for global summary; for local use predicted class
                shap_global = np.array(shap_values).transpose(0, 2, 1)  # (n, n_class, n_feat) -> (n, n_feat) mean abs
                shap_mean_abs = np.mean(np.abs(shap_global), axis=1)
            else:
                shap_mean_abs = np.abs(shap_values[0]) if isinstance(shap_values[0], np.ndarray) else np.abs(np.array(shap_values))
            base_values = getattr(self._explainer, "expected_value", None)
        else:
            shap_mean_abs = np.abs(shap_values)
            base_values = getattr(self._explainer, "expected_value", None)

        paths: Dict[str, str] = {}

        # Global: summary (bar)
        try:
            shap.summary_plot(
                shap_values if not isinstance(shap_values, list) else shap_values[0],
                X,
                feature_names=self._feature_names,
                max_display=self._cfg.max_display_features,
                show=False,
            )
            plt.savefig(out_dir / "shap_summary.png", bbox_inches="tight", dpi=120)
            plt.close()
            paths["summary"] = str(out_dir / "shap_summary.png")
        except Exception as e:
            logger.warning("SHAP summary_plot failed: %s", e)

        # Dependence plots for top features (by mean |SHAP|)
        mean_abs = np.mean(np.abs(shap_mean_abs), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][: self._cfg.top_k_features]
        for i, feat_idx in enumerate(top_idx[:5]):
            try:
                shap.dependence_plot(
                    feat_idx,
                    shap_values if not isinstance(shap_values, list) else shap_values[0],
                    X,
                    feature_names=self._feature_names,
                    show=False,
                )
                plt.savefig(out_dir / f"shap_dependence_{self._feature_names[feat_idx].replace(' ', '_')}.png", bbox_inches="tight", dpi=120)
                plt.close()
                paths[f"dependence_{self._feature_names[feat_idx]}"] = str(out_dir / f"shap_dependence_{self._feature_names[feat_idx].replace(' ', '_')}.png")
            except Exception as e:
                logger.debug("Dependence plot failed for %s: %s", self._feature_names[feat_idx], e)

        # Feature interaction (tree only)
        if self._explainer_type == "tree" and hasattr(self._explainer, "shap_interaction_values"):
            try:
                inter = self._explainer.shap_interaction_values(X[: min(200, len(X))])
                if inter is not None:
                    if isinstance(inter, list):
                        inter = inter[0]
                    shap.summary_plot(inter, X[: min(200, len(X))], feature_names=self._feature_names, max_display=self._cfg.top_k_features, show=False)
                    plt.savefig(out_dir / "shap_interaction.png", bbox_inches="tight", dpi=120)
                    plt.close()
                    paths["interaction"] = str(out_dir / "shap_interaction.png")
            except Exception as e:
                logger.debug("SHAP interaction plot failed: %s", e)

        # Cohort-level: by segment
        if segment_col is not None and len(segment_col) == len(X):
            for seg in np.unique(segment_col)[:5]:
                mask = segment_col == seg
                if mask.sum() < 20:
                    continue
                try:
                    sv_seg = self._explainer.shap_values(X[mask])
                    if isinstance(sv_seg, list):
                        sv_seg = sv_seg[0]
                    shap.summary_plot(sv_seg, X[mask], feature_names=self._feature_names, max_display=self._cfg.top_k_features, show=False)
                    plt.savefig(out_dir / f"shap_cohort_segment_{seg}.png", bbox_inches="tight", dpi=120)
                    plt.close()
                    paths[f"cohort_{seg}"] = str(out_dir / f"shap_cohort_segment_{seg}.png")
                except Exception as e:
                    logger.debug("Cohort plot failed for segment %s: %s", seg, e)

        # Local: waterfall for a few samples (SHAP 0.44+ uses plots.waterfall)
        n_w = min(self._cfg.n_waterfall_samples, len(X))
        for idx in range(n_w):
            try:
                exp = self._explainer(X[idx : idx + 1])
                if isinstance(exp, list):
                    exp = exp[0]
                if hasattr(shap, "plots") and hasattr(shap.plots, "waterfall"):
                    shap.plots.waterfall(exp[0], show=False)
                else:
                    shap.waterfall_plot(exp[0] if hasattr(exp, "__getitem__") else exp, X[idx], feature_names=self._feature_names, show=False)
                plt.savefig(out_dir / f"shap_waterfall_{idx}.png", bbox_inches="tight", dpi=120)
                plt.close()
                paths[f"waterfall_{idx}"] = str(out_dir / f"shap_waterfall_{idx}.png")
            except Exception as e:
                logger.debug("Waterfall %s failed: %s", idx, e)

        # Force plot (HTML) for a couple of samples
        ev = getattr(self._explainer, "expected_value", None)
        if isinstance(ev, (list, np.ndarray)) and len(ev) > 0:
            ev = ev[0]
        sv_one = shap_values[0] if isinstance(shap_values, list) else shap_values
        for idx in range(min(self._cfg.n_force_plot_samples, len(X))):
            try:
                f = shap.force_plot(
                    ev,
                    sv_one[idx],
                    X[idx],
                    feature_names=self._feature_names,
                    matplotlib=False,
                )
                if hasattr(shap, "save_html"):
                    shap.save_html(str(out_dir / f"shap_force_{idx}.html"), f)
                else:
                    (out_dir / f"shap_force_{idx}.html").write_text(str(f), encoding="utf-8")
                paths[f"force_{idx}"] = str(out_dir / f"shap_force_{idx}.html")
            except Exception as e:
                logger.debug("Force plot %s failed: %s", idx, e)

        return ExplainerResult(
            model_name=model_name,
            shap_values=shap_values,
            base_values=base_values,
            feature_names=self._feature_names,
            paths=paths,
            explainer_type=self._explainer_type,
        )

    def counterfactual(
        self,
        instance: np.ndarray,
        shap_vals_one: np.ndarray,
        predicted_class: Any,
        classes: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Simple counterfactual: for top |SHAP| features, suggest 'if X were median, prediction might change'."""
        names = feature_names or self._feature_names
        order = np.argsort(np.abs(shap_vals_one))[::-1][:5]
        out = []
        for i in order:
            fname = names[i] if i < len(names) else str(i)
            current_val = instance[i]
            shap_v = float(shap_vals_one[i])
            direction = "increase" if shap_v > 0 else "decrease"
            out.append({
                "feature": fname,
                "current_value": float(current_val),
                "shap_contribution": shap_v,
                "suggestion": f"Changing {fname} could {direction} support for predicted class '{predicted_class}'.",
            })
        return out
