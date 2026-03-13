"""
FastAPI route definitions for GlucoSense Clinical Support API.

Endpoints: POST /predict, POST /explain, POST /recommend, GET /model-info, GET /feature-importance.
Input validation and structured JSON responses with clinical metadata.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..config.schema import AlertConfig, DashboardConfig, GLUCOSE_ZONES, get_glucose_zone
from ..domain.constants import (
    FAST_ACTING_CARBS_GRAMS,
    GLUCOSE_HYPO_ALERT_MGDL,
    GLUCOSE_LOW_FOR_DOSE_REDUCTION_MGDL,
    GLUCOSE_MODERATE_HIGH_ALERT_MIN_MGDL,
    GLUCOSE_MODERATE_HIGH_MAX_MGDL,
    GLUCOSE_SEVERE_HIGH_ALERT_MIN_MGDL,
)

logger = logging.getLogger(__name__)

# API defaults
DEFAULT_RECORDS_LIMIT = 100
CHART_MISSING_GLUCOSE_DEFAULT = 0.0  # For Recharts when value is null
DEFAULT_NOTIFICATIONS_LIMIT = 20
DEFAULT_ALERTS_LIMIT = 50
DEFAULT_GLUCOSE_TRENDS_HOURS = 72
SHAP_BACKGROUND_SAMPLE_SIZE = 100
RANDOM_SEED = 42
COUNTERFACTUAL_TOP_K = 5
PROBABILITY_BREAKDOWN_TOP_K = 5
from ..safety.audit import log_prediction
from ..monitoring import get_monitor
from ..storage import (
    init_db,
    insert_record,
    insert_clinician_feedback,
    get_clinician_feedback,
    get_records,
    get_notifications,
    insert_notification,
    delete_notifications_by_type,
    mark_notifications_read,
    get_glucose_readings,
    insert_glucose_reading,
    insert_dose_event,
    get_alerts,
    insert_alert,
    resolve_alert,
    resolve_all_alerts,
    get_patient_context,
    upsert_patient_context,
    get_setting,
    set_setting,
    run_seed_if_needed,
)


def _check_critical_alerts(
    glucose_level: Optional[float],
    is_high_risk: bool,
    predicted_class: Optional[str],
) -> None:
    """Insert alerts for critical conditions using glucose zone thresholds."""
    try:
        if glucose_level is not None:
            gl = float(glucose_level)
            zone = get_glucose_zone(gl)
            if zone:
                zid = zone.get("id", "")
                sev = zone.get("severity", "normal")
                if zid == "hypo":
                    insert_alert(
                        "critical",
                        "Hypoglycemia",
                        f"Glucose {gl} mg/dL is below {GLUCOSE_HYPO_ALERT_MGDL}. Stop insulin. Consume {FAST_ACTING_CARBS_GRAMS}g fast-acting carbs.",
                    )
                elif zid == "moderate_high":
                    insert_alert(
                        "warning",
                        "Moderate hyperglycemia",
                        f"Glucose {gl} mg/dL ({GLUCOSE_MODERATE_HIGH_ALERT_MIN_MGDL}–{GLUCOSE_MODERATE_HIGH_MAX_MGDL}). Add correction dose. Check hydration/stress.",
                    )
                elif zid == "severe_high":
                    insert_alert(
                        "critical",
                        "Severe hyperglycemia",
                        f"Glucose {gl} mg/dL above {GLUCOSE_SEVERE_HIGH_ALERT_MIN_MGDL}. Add correction. Check ketones if BG high >2 hours.",
                    )
        if is_high_risk:
            insert_alert(
                "warning",
                "High-risk recommendation",
                "Last recommendation was flagged for clinician review (system less certain than usual).",
            )
        if predicted_class and str(predicted_class).lower() == "down" and glucose_level is not None and float(glucose_level) < GLUCOSE_LOW_FOR_DOSE_REDUCTION_MGDL:
            insert_alert(
                "warning",
                "Reduce dose with low glucose",
                "System suggests reducing insulin while glucose is already low. Verify before reducing.",
            )
    except Exception:
        pass


# Person-centric fields to store in records for Reports (assessment context, not model metrics)
_INPUT_SUMMARY_KEYS = (
    "glucose_level", "iob", "anticipated_carbs", "glucose_trend",
    "age", "food_intake", "physical_activity", "weight", "BMI", "HbA1c",
    "icr", "isf",
    "ketone_level", "cgm_sensor_error", "typical_daily_insulin",
)


def _build_input_summary(body: Dict[str, Any]) -> Dict[str, Any]:
    """Build person-centric input summary for Reports (assessed person context)."""
    out: Dict[str, Any] = {}
    for k in _INPUT_SUMMARY_KEYS:
        v = body.get(k)
        if v is not None and (not isinstance(v, str) or str(v).strip() != ""):
            out[k] = v
    return out if out else {"n_fields": len(body)}


def _update_patient_context_from_body(body: Dict[str, Any]) -> None:
    try:
        name = body.get("patient_name") or "Current Patient"
        condition = body.get("condition") or "Type 1 Diabetes"
        gl = body.get("glucose_level")
        carbs = body.get("carbohydrates") or body.get("food_intake")
        activity = body.get("physical_activity")
        if gl is not None or carbs is not None or activity is not None or name or condition:
            upsert_patient_context(
                name=str(name),
                condition=str(condition),
                glucose=int(gl) if gl is not None else None,
                carbohydrates=int(carbs) if carbs is not None and str(carbs).isdigit() else None,
                activity_minutes=int(activity) if activity is not None else None,
            )
    except Exception:
        pass
from .schemas import (
    PatientInput,
    PredictionResponse,
    ExplainResponse,
    RecommendationResponse,
    ModelInfoResponse,
    FeatureImportanceResponse,
)
from .validators import patient_input_to_dataframe, validate_patient_input
from .engine import (
    get_bundle,
    run_predict,
    run_explain,
    run_recommend,
    get_model_info,
    get_feature_importance,
)

router = APIRouter(prefix="/api", tags=["GlucoSense"])

# Ensure database exists and seed data on first use
try:
    init_db()
    run_seed_if_needed()
except Exception:
    pass

# Optional background data for SHAP (loaded on first explain/recommend with reference data)
_background_X: Optional[Any] = None


def _load_background_if_needed():
    global _background_X
    if _background_X is not None:
        return
    try:
        from ..dashboard.data_loader import load_dashboard_data
        import numpy as np
        cfg = DashboardConfig()
        data = load_dashboard_data(cfg, cfg.data_path, run_pipeline_for_reference=True)
        if data.reference_X is not None and data.reference_X.shape[0] > 0:
            n = min(SHAP_BACKGROUND_SAMPLE_SIZE, data.reference_X.shape[0])
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(data.reference_X.shape[0], size=n, replace=False)
            _background_X = data.reference_X[idx]
        else:
            _background_X = None
    except Exception:
        _background_X = None


def _validation_response(errors: list) -> JSONResponse:
    """Return 422 with structured validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation failed", "errors": errors},
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(body: Dict[str, Any]):
    """
    Get insulin dosage prediction for a single patient.
    Request body: patient record (categorical + numeric features). Strict validation on age, gender, food_intake, previous_medications.
    """
    request_id = str(uuid.uuid4())
    try:
        patient, warnings, errors = validate_patient_input(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if errors:
        return _validation_response(errors)
    try:
        bundle = get_bundle()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    df = patient_input_to_dataframe(patient)
    try:
        resp = run_predict(patient, df, bundle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    resp.request_id = request_id
    log_prediction("/predict", request_id, resp.predicted_class, resp.confidence, request_summary={"n_fields": len(body)})
    try:
        insert_record(
            endpoint="predict",
            request_id=request_id,
            predicted_class=resp.predicted_class,
            confidence=resp.confidence,
            input_summary=_build_input_summary(body),
            response_summary={"predicted_class": resp.predicted_class, "confidence": resp.confidence},
        )
    except Exception:
        pass
    return resp


@router.post("/explain", response_model=ExplainResponse)
def explain(body: Dict[str, Any]):
    """
    Get detailed explanation for a prediction (SHAP-based when background data available).
    """
    request_id = str(uuid.uuid4())
    try:
        patient, _, errors = validate_patient_input(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if errors:
        return _validation_response(errors)
    try:
        bundle = get_bundle()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = patient_input_to_dataframe(patient)
    _load_background_if_needed()
    try:
        resp = run_explain(patient, df, bundle, _background_X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {e}")
    resp.request_id = request_id
    log_prediction("/explain", request_id, resp.predicted_class, resp.confidence)
    try:
        insert_record(
            endpoint="explain",
            request_id=request_id,
            predicted_class=resp.predicted_class,
            confidence=resp.confidence,
            input_summary=_build_input_summary(body),
            response_summary={"predicted_class": resp.predicted_class, "confidence": resp.confidence},
        )
    except Exception:
        pass
    return resp


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(body: Dict[str, Any]):
    """
    Get full recommendation with dosage suggestion, reasoning, and explanation components.
    Strict validation: age (0–100), gender (Male/Female), food_intake (Low/Medium/High), previous_medications (None/Insulin/Oral). If Oral, medication_name required.
    """
    request_id = str(uuid.uuid4())
    try:
        patient, _, errors = validate_patient_input(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if errors:
        return _validation_response(errors)
    try:
        bundle = get_bundle()
    except Exception as e:
        logger.error("Model load failed for /recommend: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Run the pipeline first: python run_pipeline.py. Error: {e}"
        )
    df = patient_input_to_dataframe(patient)
    _load_background_if_needed()
    try:
        resp = run_recommend(patient, df, bundle, _background_X)
    except Exception as e:
        logger.exception("Recommendation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}. Check logs for details."
        )
    resp.request_id = request_id
    log_prediction("/recommend", request_id, resp.predicted_class, resp.confidence, resp.is_high_risk)
    try:
        get_monitor().log_prediction(resp.predicted_class, resp.confidence, resp.is_high_risk, "recommend")
    except Exception:
        pass
    try:
        insert_record(
            endpoint="recommend",
            request_id=request_id,
            predicted_class=resp.predicted_class,
            confidence=resp.confidence,
            is_high_risk=resp.is_high_risk,
            input_summary=_build_input_summary(body),
            response_summary={
                "predicted_class": resp.predicted_class,
                "confidence": resp.confidence,
                "dosage_action": resp.dosage_action,
                "is_high_risk": resp.is_high_risk,
            },
        )
    except Exception:
        pass
    _update_patient_context_from_body(body)
    # Record entered glucose as a trend point so chart shows data after assessment (no trend before input)
    try:
        gl = body.get("glucose_level")
        if gl is not None and str(gl).strip() != "":
            insert_glucose_reading(float(gl), is_predicted=False)
    except Exception as e:
        logger.warning("Failed to record glucose for trend: %s", e)
    # Critical-condition alert detection
    try:
        gl_val = body.get("glucose_level")
        gl_float = float(gl_val) if gl_val is not None and str(gl_val).strip() != "" else None
        _check_critical_alerts(gl_float, resp.is_high_risk, resp.predicted_class)
    except Exception:
        pass
    return resp


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Get model performance metrics and metadata."""
    try:
        bundle = get_bundle()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return get_model_info(bundle)


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance():
    """Get global feature importance (built-in from model)."""
    try:
        bundle = get_bundle()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Model not loaded")
    cfg = DashboardConfig()
    out = get_feature_importance(bundle, cfg.evaluation_dir)
    if out is None:
        raise HTTPException(status_code=404, detail="Feature importance not available for this model")
    return out


@router.post("/feedback")
def record_feedback(body: Dict[str, Any]):
    """
    Record clinician override/feedback for model improvement.
    Body: record_id?, request_id?, predicted_class?, clinician_action?, actual_dose_units?, override_reason?, input_summary?
    """
    try:
        fid = insert_clinician_feedback(
            record_id=body.get("record_id"),
            request_id=body.get("request_id"),
            predicted_class=body.get("predicted_class"),
            clinician_action=body.get("clinician_action"),
            actual_dose_units=body.get("actual_dose_units"),
            override_reason=body.get("override_reason"),
            input_summary=body.get("input_summary"),
        )
        return {"ok": True, "feedback_id": fid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback")
def list_feedback(limit: int = 100):
    """List clinician feedback records for analysis."""
    try:
        records = get_clinician_feedback(limit=limit)
        return {"feedback": records, "count": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/stats")
def monitoring_stats(n: int = 100):
    """Get recent prediction stats for monitoring (class distribution, avg confidence, high-risk %)."""
    try:
        return get_monitor().get_recent_stats(n=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/records")
def list_records(limit: int = DEFAULT_RECORDS_LIMIT):
    """List recent prediction/recommendation records from the database."""
    try:
        records = get_records(limit=limit)
        return {"records": records, "count": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    """Health check: API and optional database."""
    try:
        init_db()
        run_seed_if_needed()
        return {"status": "ok", "database": "ready"}
    except Exception as e:
        return {"status": "ok", "database": str(e)}


REPORTS_DOWNLOAD_NOTIFICATION_TYPE = "reports_download"


@router.get("/notifications")
def list_notifications(limit: int = DEFAULT_NOTIFICATIONS_LIMIT):
    """List notifications (from seed or runtime)."""
    try:
        run_seed_if_needed()
        items = get_notifications(limit=limit)
        return {"notifications": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notifications")
def create_notification(body: Dict[str, Any]):
    """Create a notification. For type=reports_download, replaces any existing one."""
    text = body.get("text") or ""
    notification_type = body.get("type") or body.get("notification_type")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    try:
        run_seed_if_needed()
        if notification_type == REPORTS_DOWNLOAD_NOTIFICATION_TYPE:
            delete_notifications_by_type(REPORTS_DOWNLOAD_NOTIFICATION_TYPE)
        insert_notification(text.strip(), notification_type=notification_type)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/notifications/by-type/{notification_type}")
def delete_notifications_by_type_route(notification_type: str):
    """Delete notifications by type (e.g. reports_download)."""
    try:
        delete_notifications_by_type(notification_type)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/notifications/read")
def notifications_mark_read():
    """Mark all notifications as read."""
    try:
        mark_notifications_read()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
def list_alerts(limit: int = DEFAULT_ALERTS_LIMIT, unresolved_only: bool = True):
    """List critical-condition alerts (unresolved by default)."""
    try:
        run_seed_if_needed()
        items = get_alerts(limit=limit, unresolved_only=unresolved_only)
        return {"alerts": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/resolve-all")
def resolve_all_alerts_route():
    """Mark all unresolved alerts as resolved."""
    try:
        count = resolve_all_alerts()
        return {"status": "ok", "resolved": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/resolve")
def resolve_alert_route(body: Dict[str, Any]):
    """Mark a single alert as resolved. Body: { \"id\": 1 }."""
    alert_id = body.get("id")
    if alert_id is None:
        raise HTTPException(status_code=400, detail="id is required")
    try:
        aid = int(alert_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="id must be an integer")
    try:
        ok = resolve_alert(aid)
        if not ok:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient-context")
def patient_context():
    """Current patient context for sidebar (name, condition, recent metrics)."""
    try:
        run_seed_if_needed()
        ctx = get_patient_context()
        if not ctx:
            return {"name": "Current Patient", "condition": "Type 1 Diabetes", "glucose": None, "carbohydrates": None, "activity_minutes": None}
        return ctx
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/glucose-zones")
def glucose_zones():
    """Glucose interpretation & dosage chart (standard reference zones)."""
    return {"zones": GLUCOSE_ZONES}


@router.get("/glucose-zones/interpret")
def interpret_glucose(glucose: Optional[float] = None):
    """Return the zone and action for a given glucose value (mg/dL). Query param: ?glucose=120"""
    if glucose is None:
        return {"glucose": None, "zone": None, "message": "Please provide a glucose value (e.g. ?glucose=120)."}
    try:
        gl = float(glucose)
    except (TypeError, ValueError):
        return {"glucose": glucose, "zone": None, "message": "Invalid glucose value; must be a number."}
    zone = get_glucose_zone(gl)
    if zone is None:
        return {"glucose": gl, "zone": None, "message": "No zone found for this value."}
    return {"glucose": gl, "zone": zone}


@router.get("/glucose-trends")
def glucose_trends(hours: int = DEFAULT_GLUCOSE_TRENDS_HOURS):
    """Glucose readings for chart. Returns series with time, actual, predicted (chronological). Realtime: each assessment adds a point."""
    try:
        try:
            run_seed_if_needed()
        except Exception:
            pass  # non-fatal: trend data does not depend on seed
        rows = get_glucose_readings(hours=hours)
        from datetime import datetime
        seen = {}
        out = []
        for r in rows:
            ts = r.get("reading_at")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
                # Unique label per reading (include seconds) so chart shows each point
                time_label = dt.strftime("%H:%M:%S") if dt else (str(ts)[:19] if ts else "")
            except Exception:
                time_label = str(ts)[:19] if ts else ""
            key = ts or time_label
            if key not in seen:
                seen[key] = len(out)
                out.append({"time": time_label, "actual": None, "predicted": None})
            idx = seen[key]
            val = r.get("value")
            if val is not None:
                try:
                    num_val = float(val)
                except (TypeError, ValueError):
                    num_val = CHART_MISSING_GLUCOSE_DEFAULT
            else:
                num_val = CHART_MISSING_GLUCOSE_DEFAULT
            if r.get("is_predicted"):
                out[idx]["predicted"] = num_val
            else:
                out[idx]["actual"] = num_val
        for row in out:
            if row["predicted"] is None and row["actual"] is not None:
                row["predicted"] = row["actual"]
            elif row["actual"] is None and row["predicted"] is not None:
                row["actual"] = row["predicted"]
            # Ensure numbers for Recharts (no null)
            row["actual"] = row["actual"] if row["actual"] is not None else CHART_MISSING_GLUCOSE_DEFAULT
            row["predicted"] = row["predicted"] if row["predicted"] is not None else CHART_MISSING_GLUCOSE_DEFAULT
        return {"series": out, "count": len(out)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dose")
def record_dose(body: Dict[str, Any]):
    """Record a dose administration event."""
    meal_bolus = body.get("meal_bolus") or body.get("mealBolus")
    correction_dose = body.get("correction_dose") or body.get("correctionDose")
    total_dose = body.get("total_dose") or body.get("totalDose") or body.get("summary")
    request_id = body.get("request_id")
    try:
        mid = insert_dose_event(
            meal_bolus=str(meal_bolus) if meal_bolus is not None else None,
            correction_dose=str(correction_dose) if correction_dose is not None else None,
            total_dose=str(total_dose) if total_dose is not None else None,
            request_id=str(request_id) if request_id is not None else None,
        )
        return {"id": mid, "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings")
def get_settings():
    """Get app settings (units, theme, etc.)."""
    try:
        run_seed_if_needed()
        return {
            "units": get_setting("units") or "mg/dL",
            "theme": get_setting("theme") or "light",
            "notifications_enabled": get_setting("notifications_enabled") != "false",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings")
def put_settings(body: Dict[str, Any]):
    """Update app settings."""
    try:
        if "units" in body:
            set_setting("units", str(body["units"]))
        if "theme" in body:
            set_setting("theme", str(body["theme"]))
        if "notifications_enabled" in body:
            set_setting("notifications_enabled", "true" if body["notifications_enabled"] else "false")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
