"""
Presentation layer: request validation using domain rules.

Calls the business logic (domain) for validation; builds PatientInput from sanitized body.
Returns structured validation errors for 422 responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..domain.constants import MIN_NUMERIC_FEATURES_FOR_RELIABLE_PREDICTION
from ..domain.validation import validate_assessment_input
from .schemas import PatientInput

# Default for missing numeric values in DataFrame
MISSING_NUMERIC_DEFAULT = 0.0


def validate_patient_input(body: Dict[str, Any]) -> Tuple[PatientInput, List[str], List[Dict[str, str]]]:
    """
    Validate using domain rules and coerce to PatientInput.
    Returns (patient, warnings, errors).
    If errors is non-empty, the API should return 422 with structured errors; patient may still be built from sanitized values for optional fields.
    """
    sanitized, errors = validate_assessment_input(body)
    warnings: List[str] = []

    # Require key fields for recommendation (business rule: at least these must be present and valid)
    if not errors:
        numeric_keys = [
            "age",
            "glucose_level",
            "physical_activity",
            "BMI",
            "HbA1c",
            "weight",
            "insulin_sensitivity",
            "sleep_hours",
            "creatinine",
        ]
        provided_numeric = sum(1 for k in numeric_keys if sanitized.get(k) is not None)
        if provided_numeric < MIN_NUMERIC_FEATURES_FOR_RELIABLE_PREDICTION:
            warnings.append("Few numeric features provided; prediction may be less reliable.")

    # Build PatientInput from sanitized dict (only include keys that exist in schema)
    row = {
        "patient_id": sanitized.get("patient_id"),
        "gender": sanitized.get("gender"),
        "family_history": sanitized.get("family_history"),
        "food_intake": sanitized.get("food_intake"),
        "previous_medications": sanitized.get("previous_medications"),
        "age": sanitized.get("age"),
        "glucose_level": sanitized.get("glucose_level"),
        "physical_activity": sanitized.get("physical_activity"),
        "BMI": sanitized.get("BMI"),
        "HbA1c": sanitized.get("HbA1c"),
        "weight": sanitized.get("weight"),
        "insulin_sensitivity": sanitized.get("insulin_sensitivity"),
        "sleep_hours": sanitized.get("sleep_hours"),
        "creatinine": sanitized.get("creatinine"),
        "iob": sanitized.get("iob"),
        "anticipated_carbs": sanitized.get("anticipated_carbs"),
        "glucose_trend": sanitized.get("glucose_trend"),
        "icr": sanitized.get("icr"),
        "isf": sanitized.get("isf"),
        "ketone_level": sanitized.get("ketone_level"),
        "cgm_sensor_error": sanitized.get("cgm_sensor_error"),
        "typical_daily_insulin": sanitized.get("typical_daily_insulin"),
    }
    if sanitized.get("medication_name") is not None:
        row["medication_name"] = sanitized["medication_name"]

    try:
        patient = PatientInput(**row)
    except Exception as e:
        if not errors:
            errors.append({"field": "body", "message": str(e)})
        raise ValueError(f"Invalid patient data: {e}") from e

    return patient, warnings, errors


def patient_input_to_dataframe(patient: PatientInput):
    """Build a single-row DataFrame with numeric columns as float so pipeline never sees str/object."""
    import pandas as pd
    from ..config.schema import DataSchema
    row = patient.to_row_dict()
    schema = DataSchema()
    numeric_cols = list(schema.NUMERIC) + list(getattr(schema, "CONTEXTUAL_IMPUTE", ()))
    for col in numeric_cols:
        if col in row and row[col] is not None:
            try:
                row[col] = float(row[col])
            except (TypeError, ValueError):
                row[col] = MISSING_NUMERIC_DEFAULT
        elif col in row:
            row[col] = MISSING_NUMERIC_DEFAULT
    # Ensure glucose_trend is present for feature engineering
    if "glucose_trend" not in row:
        row["glucose_trend"] = "stable"
    return pd.DataFrame([row])
