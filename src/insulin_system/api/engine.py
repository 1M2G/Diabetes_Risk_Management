"""
GlucoSense inference engine: prediction, explanation, and recommendation in one place.

Loads the saved model bundle once and exposes run_predict, run_explain, run_recommend
for use by the API. This is the core backend engine for the clinical support system.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..domain.constants import (
    CGM_ERROR_CONFIDENCE_CAP,
    FAST_ACTING_CARBS_GRAMS,
    GLUCOSE_HYPO_ALERT_MGDL,
    HIGH_UNCERTAINTY_CORRECTION_MULTIPLIER,
    TYPICAL_CORRECTION_TDD_FRACTION,
)
from ..config.schema import (
    DashboardConfig,
    RecommendationConfig,
    get_glucose_zone,
    get_glucose_zone_cds,
    _glucose_label_from_zone,
    _trend_display,
)
from ..persistence import load_best_model, InferenceBundle
from .schemas import (
    ExplanationDriver,
    PatientInput,
    PredictionResponse,
    ExplainResponse,
    RecommendationResponse,
    ModelInfoResponse,
    FeatureImportanceResponse,
)

logger = logging.getLogger(__name__)

# Module-level bundle (lazy load)
_bundle: Optional[InferenceBundle] = None
_config: Optional[DashboardConfig] = None
_shap_explainer: Optional[Any] = None
_background_X: Optional[np.ndarray] = None


def get_bundle(model_dir: Optional[Path] = None) -> InferenceBundle:
    """Load and cache the inference bundle."""
    global _bundle, _config
    if _bundle is None:
        cfg = DashboardConfig()
        if model_dir:
            cfg = DashboardConfig(best_model_dir=model_dir)
        _config = cfg
        _bundle = load_best_model(cfg.best_model_dir)
        
        # If the loaded bundle is a Python dict (from old dump), wrap it.
        if isinstance(_bundle, dict):
            # Create an InferenceBundle class from persistence manually since legacy saving bypassed
            from ..persistence.bundle import InferenceBundle
            new_bundle = InferenceBundle.__new__(InferenceBundle)
            new_bundle.__dict__.update(_bundle)
            _bundle = new_bundle
            
        logger.info("Loaded inference bundle: %s", getattr(_bundle, "model_name", "Unknown"))
    return _bundle


def _get_shap_explainer(bundle: InferenceBundle, X_background: np.ndarray) -> Optional[Any]:
    """Lazy-fit SHAP explainer on a background sample (e.g. from reference data)."""
    global _shap_explainer, _background_X
    if _shap_explainer is not None and _background_X is not None and X_background.shape[1] == _background_X.shape[1]:
        return _shap_explainer
    try:
        from ..explainability import SHAPExplainer
        explainer = SHAPExplainer()
        explainer.fit(bundle._model, X_background, bundle.feature_names)
        _shap_explainer = explainer
        _background_X = X_background
        return _shap_explainer
    except Exception as e:
        logger.warning("SHAP explainer not available: %s", e)
        return None


def run_predict(patient: PatientInput, df: pd.DataFrame, bundle: InferenceBundle) -> PredictionResponse:
    """Run prediction only. Returns structured response with confidence and probabilities."""
    X = bundle.transform(df)
    pred = bundle.predict(X)[0]
    proba = bundle.predict_proba(X)[0]
    classes = list(bundle.classes_)
    idx = list(classes).index(pred)
    confidence = float(proba[idx])
    entropy = float(-(proba * np.log(proba + 1e-10)).sum())
    prob_breakdown = {str(c): float(proba[i]) for i, c in enumerate(classes)}
    return PredictionResponse(
        predicted_class=str(pred),
        confidence=confidence,
        uncertainty_entropy=entropy,
        probability_breakdown=prob_breakdown,
        feature_names_used=bundle.feature_names,
    )


def run_explain(
    patient: PatientInput,
    df: pd.DataFrame,
    bundle: InferenceBundle,
    X_background: Optional[np.ndarray] = None,
) -> ExplainResponse:
    """Run prediction and SHAP-based explanation. If X_background provided, uses SHAP; else returns drivers from prob breakdown."""
    X = bundle.transform(df)
    pred = bundle.predict(X)[0]
    proba = bundle.predict_proba(X)[0]
    classes = list(bundle.classes_)
    idx = list(classes).index(pred)
    confidence = float(proba[idx])

    top_drivers: List[ExplanationDriver] = []
    counterfactuals: List[Dict[str, Any]] = []

    explainer = _get_shap_explainer(bundle, X_background) if X_background is not None and len(X_background) > 0 else None
    if explainer is not None:
        try:
            sv_one = explainer.get_local_shap_values(X, sample_idx=0)
            x_row = X[0]
            from ..explainability.clinical_report import CLINICAL_FEATURE_NAMES
            for fname, shap_val in explainer.get_top_drivers(X, bundle.feature_names, top_k=10):
                clinical_name = CLINICAL_FEATURE_NAMES.get(fname, fname)
                feat_idx = bundle.feature_names.index(fname) if fname in bundle.feature_names else 0
                top_drivers.append(ExplanationDriver(
                    feature=fname,
                    value=float(x_row[feat_idx]),
                    shap_value=float(shap_val),
                    clinical_sentence=f"{clinical_name} (value={x_row[feat_idx]:.2f}) contributed to prediction.",
                ))
            counterfactuals = explainer.counterfactual(x_row, sv_one, str(pred), np.array(classes), bundle.feature_names)
        except Exception as e:
            logger.debug("SHAP explanation failed: %s", e)

    if not top_drivers and proba is not None:
        for i, c in enumerate(classes):
            top_drivers.append(ExplanationDriver(
                feature=f"P({c})",
                value=float(proba[i]),
                shap_value=float(proba[i]),
                clinical_sentence=f"Probability of {c}: {proba[i]:.0%}.",
            ))

    return ExplainResponse(
        predicted_class=str(pred),
        confidence=confidence,
        top_drivers=top_drivers[:10],
        counterfactuals=counterfactuals,
    )


def run_recommend(
    patient: PatientInput,
    df: pd.DataFrame,
    bundle: InferenceBundle,
    X_background: Optional[np.ndarray] = None,
) -> RecommendationResponse:
    """Run full recommendation: ML model prediction + config-driven clinical recommendation + explanation."""
    from ..recommendation import RecommendationGenerator
    from ..explainability.clinical_report import CLINICAL_FEATURE_NAMES

    X = bundle.transform(df)
    pred = bundle.predict(X)[0]
    proba = bundle.predict_proba(X)[0]
    try:
        import mlflow
        if mlflow.active_run() or mlflow.get_tracking_uri():
            mlflow.log_metric("recommend_predicted_class_hash", hash(str(pred)) % 1000)
            mlflow.log_metric("recommend_confidence", float(proba[list(bundle.classes_).index(pred)]))
    except Exception:
        pass
    classes = list(bundle.classes_)
    idx = list(classes).index(pred)
    try:
        _c = proba[idx]
        confidence = float(_c) if _c is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    if np.isnan(confidence) or confidence < 0 or confidence > 1:
        confidence = 0.0
    try:
        _e = -(proba * np.log(proba + 1e-10)).sum()
        entropy = float(_e) if _e is not None else 0.0
    except (TypeError, ValueError):
        entropy = 0.0
    if np.isnan(entropy) or entropy < 0:
        entropy = 0.0
    prob_breakdown = {}
    for i, c in enumerate(classes):
        try:
            p = proba[i]
            prob_breakdown[str(c)] = float(p) if p is not None else 0.0
        except (TypeError, ValueError):
            prob_breakdown[str(c)] = 0.0

    patient_dict = dict(patient.to_row_dict()) if patient else {}
    if patient:
        if getattr(patient, "iob", None) is not None:
            patient_dict["iob"] = patient.iob
        if getattr(patient, "anticipated_carbs", None) is not None:
            patient_dict["anticipated_carbs"] = patient.anticipated_carbs
        if getattr(patient, "glucose_trend", None) is not None:
            patient_dict["glucose_trend"] = patient.glucose_trend
        if getattr(patient, "icr", None) is not None:
            patient_dict["icr"] = patient.icr
        if getattr(patient, "isf", None) is not None:
            patient_dict["isf"] = patient.isf
        if getattr(patient, "ketone_level", None) is not None:
            patient_dict["ketone_level"] = patient.ketone_level
        if getattr(patient, "cgm_sensor_error", None) is not None:
            patient_dict["cgm_sensor_error"] = patient.cgm_sensor_error
        if getattr(patient, "typical_daily_insulin", None) is not None:
            patient_dict["typical_daily_insulin"] = patient.typical_daily_insulin
    top_driver_names = []
    rec_gen = RecommendationGenerator()
    rec = rec_gen.generate(str(pred), confidence, entropy, prob_breakdown, patient_dict=patient_dict, top_driver_names=None)
    dosage = rec.dosage_suggestion

    explanation_drivers: List[ExplanationDriver] = []
    cf: List[Dict[str, Any]] = []
    explainer = _get_shap_explainer(bundle, X_background) if X_background is not None and len(X_background) > 0 else None
    if explainer is not None:
        try:
            sv = explainer._explainer.shap_values(X)
            sv_one = sv[0] if isinstance(sv, list) else sv[0]
            x_row = X[0]
            order = np.argsort(np.abs(sv_one))[::-1][:10]
            for i in order:
                fname = bundle.feature_names[i] if i < len(bundle.feature_names) else f"feature_{i}"
                clinical_name = CLINICAL_FEATURE_NAMES.get(fname, fname)
                explanation_drivers.append(ExplanationDriver(
                    feature=fname,
                    value=float(x_row[i]),
                    shap_value=float(sv_one[i]),
                    clinical_sentence=f"{clinical_name} (value={x_row[i]:.2f}).",
                ))
            cf = explainer.counterfactual(x_row, sv_one, str(pred), np.array(classes), bundle.feature_names)
        except Exception:
            pass

    alt_scenarios = [c.get("suggestion", str(c)) for c in cf[:5]] if cf else [
        "If glucose or HbA1c were lower, the system might suggest maintaining or reducing dosage.",
        "If glucose or HbA1c were higher, the system might suggest increasing dosage.",
    ]
    if not explanation_drivers:
        for c, p in list(prob_breakdown.items())[:5]:
            explanation_drivers.append(ExplanationDriver(feature=c, value=p, shap_value=p, clinical_sentence=f"P({c}) = {p:.0%}."))

    # Build UI Recommendation block (Part 3)
    gl = patient_dict.get("glucose_level")
    zone = get_glucose_zone(gl) if gl is not None else None
    gl_label = _glucose_label_from_zone(zone)
    current_reading_display = f"{gl:.0f} mg/dL ({gl_label})" if gl is not None and gl_label else (f"{gl:.0f} mg/dL" if gl is not None else "")
    trend_display = _trend_display(patient_dict.get("glucose_trend"))
    iob_val = patient_dict.get("iob")
    iob_display = f"{iob_val:.3f} mL" if iob_val is not None else "Not provided"

    # What the readings suggest: plain-language narrative (Part 3 UI)
    zid = zone.get("id", "") if zone else ""
    if zid == "hypo":
        system_interpretation = f"Your blood sugar is low. Do not take insulin. Treat with {FAST_ACTING_CARBS_GRAMS}g fast-acting carbs first, then recheck in 15 minutes."
    elif zid in ("mild_hyper", "moderate_high", "severe_high") and iob_val is not None and float(iob_val) > 0 and (dosage.dose_change_units <= 0 or "withhold" in (dosage.context_summary or "").lower() or "IOB" in (dosage.context_summary or "")):
        system_interpretation = "Your blood sugar is high, but you still have active insulin working. Adding more insulin now could cause a low later."
    elif zid == "low_normal":
        system_interpretation = "Your blood sugar is on the lower side. Dose only for the food you eat; reduce the meal dose to avoid going too low."
    elif zid == "target":
        system_interpretation = "Your blood sugar is in a good range. Use your usual dose for food; no correction needed."
    elif dosage.context_summary:
        system_interpretation = dosage.context_summary
    else:
        system_interpretation = dosage.detail or dosage.summary

    # Recommended action: inject X units with explanation
    dose_units = dosage.dose_change_units
    if zone and zone.get("id") == "hypo":
        recommended_action = "Do not inject. Consume 15g fast-acting carbs."
    elif dose_units > 0:
        reduction_note = ""
        if dosage.context_summary and ("IOB" in dosage.context_summary or "reduced" in dosage.context_summary.lower() or "withhold" in dosage.context_summary.lower()):
            reduction_note = f" (Reduced to account for IOB and Trend)"
        recommended_action = f"Inject {dose_units:.1f} Units{reduction_note}."
    elif dose_units < 0:
        recommended_action = f"Reduce dose by {abs(dose_units):.1f} Units."
    elif dosage.action and str(dosage.action).lower() in ("maintain", "none"):
        recommended_action = "Maintain current dose. No change."
    else:
        recommended_action = dosage.summary or "Review recommendation above."

    # CDS Safety Engine: build structured output
    from ..domain.constants import KETONE_HIGH_VALUES
    ketone_level = patient_dict.get("ketone_level")
    ketone_high = ketone_level and str(ketone_level).lower() in KETONE_HIGH_VALUES
    cgm_error = patient_dict.get("cgm_sensor_error") is True
    cds_category = get_glucose_zone_cds(gl, ketone_high=ketone_high)

    risk_flags: List[str] = []
    cds_confidence = confidence
    if cgm_error:
        cds_confidence = min(cds_confidence, CGM_ERROR_CONFIDENCE_CAP)
        risk_flags.append("cgm_error")
    if rec.is_high_risk and "high_uncertainty" not in risk_flags:
        risk_flags.append("high_uncertainty")
    if cds_category in ("level1_hypoglycemia", "level2_hypoglycemia"):
        risk_flags.append("hypoglycemia_alert")
    if ketone_high:
        risk_flags.append("high_ketones")

    # ISF/7-day check: if suggested adjustment >> typical, flag HIGH UNCERTAINTY
    typical_tdd = patient_dict.get("typical_daily_insulin")
    isf_val = patient_dict.get("isf")
    if typical_tdd and isf_val and dose_units > 0:
        try:
            tdd = float(typical_tdd)
            isf = float(isf_val)
            if tdd > 0 and isf > 0:
                typical_correction = tdd * TYPICAL_CORRECTION_TDD_FRACTION
                if dose_units > typical_correction * HIGH_UNCERTAINTY_CORRECTION_MULTIPLIER:
                    risk_flags.append("high_uncertainty")
        except (TypeError, ValueError):
            pass

    status = "rejected" if cds_category in ("level1_hypoglycemia", "level2_hypoglycemia") else "ok"
    urgent_thresh = getattr(RecommendationConfig(), "cds_urgent_validation_threshold", 0.8)
    requires_urgent_validation = cds_confidence < urgent_thresh

    if status == "rejected":
        suggested_action = f"REJECTED: Do not administer insulin. Consume {FAST_ACTING_CARBS_GRAMS}g fast-acting carbs. Manual finger-stick check recommended."
    elif cgm_error:
        suggested_action = "The system suggests withholding insulin until a manual finger-stick confirms glucose. Draft Recommendation."
    elif ketone_high:
        suggested_action = f"The system suggests {recommended_action} Critical Alert: High ketones reported. Verify with finger-stick and ketone check before dosing. Draft Recommendation."
    else:
        suggested_action = f"The system suggests {recommended_action} Draft Recommendation."

    rationale = f"The system suggests {system_interpretation}" if system_interpretation else "The system suggests reviewing the recommendation above."
    if requires_urgent_validation:
        rationale += " Requires Urgent Clinician Validation."

    return RecommendationResponse(
        predicted_class=str(pred),
        confidence=confidence,
        uncertainty_entropy=entropy,
        dosage_action=dosage.action,
        dosage_magnitude=dosage.magnitude,
        adjustment_score=dosage.adjustment_score,
        dose_change_units=dosage.dose_change_units,
        meal_bolus_units=getattr(dosage, "meal_bolus_units", 0.0),
        correction_dose_units=getattr(dosage, "correction_dose_units", 0.0),
        recommendation_summary=dosage.summary,
        recommendation_detail=dosage.detail,
        context_summary=dosage.context_summary,
        current_reading_display=current_reading_display,
        trend_display=trend_display,
        iob_display=iob_display,
        system_interpretation=system_interpretation,
        recommended_action=recommended_action,
        is_high_risk=rec.is_high_risk,
        high_risk_reason=rec.high_risk_reason,
        probability_breakdown=prob_breakdown,
        explanation_drivers=explanation_drivers,
        alternative_scenarios=alt_scenarios,
        status=status,
        category=cds_category,
        suggested_action=suggested_action,
        rationale=rationale,
        confidence_level=cds_confidence,
        risk_flags=risk_flags,
        requires_urgent_validation=requires_urgent_validation,
    )


def get_model_info(bundle: InferenceBundle) -> ModelInfoResponse:
    """Model metadata and performance metrics for GET /model-info."""
    meta = getattr(bundle, "to_metadata", lambda: {})()
    if not meta:
        meta = {
            "model_name": bundle.model_name,
            "metric_name": getattr(bundle, "metric_name", "f1_weighted"),
            "metric_value": getattr(bundle, "metric_value", 0.0),
            "feature_names": bundle.feature_names,
            "classes": list(bundle.classes_),
        }
    return ModelInfoResponse(
        model_name=meta.get("model_name", bundle.model_name),
        metric_name=meta.get("metric_name", "f1_weighted"),
        metric_value=float(meta.get("metric_value", 0)),
        feature_names=meta.get("feature_names", bundle.feature_names),
        classes=meta.get("classes", list(bundle.classes_)),
        n_features=len(meta.get("feature_names", bundle.feature_names)),
    )


def get_feature_importance(bundle: InferenceBundle, evaluation_dir: Path) -> Optional[FeatureImportanceResponse]:
    """Load feature importance from model (built-in) or evaluation artifacts."""
    est = bundle._model
    if hasattr(est, "named_steps") and "clf" in getattr(est, "named_steps", {}):
        est = est.named_steps["clf"]
    importance = None
    if hasattr(est, "feature_importances_"):
        importance = est.feature_importances_.tolist()
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        importance = np.mean(np.abs(coef), axis=0).tolist()
    if importance is not None and len(importance) == len(bundle.feature_names):
        return FeatureImportanceResponse(
            feature_names=bundle.feature_names,
            importance=importance,
            source="builtin",
        )
    return None
