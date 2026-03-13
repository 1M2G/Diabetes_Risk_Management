"""
GlucoSense Clinical Support - API backend for the React dashboard.

Run: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Type 1 Diabetes Management: stable API for patient, population, and clinical data.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing insulin_system from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from insulin_system.config.schema import DashboardConfig, DataSchema
from insulin_system.dashboard.data_loader import load_dashboard_data, DashboardData

app = FastAPI(
    title="GlucoSense Clinical Support API",
    description="Type 1 Diabetes Management - Dashboard backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static outputs (images, HTML) for explainability and evaluation
OUTPUTS_DIR = ROOT / "outputs"
if OUTPUTS_DIR.exists():
    app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# In-memory cache for dashboard data (reload on first request or when data_path changes)
_cached_data: Optional[DashboardData] = None
_cached_data_path: Optional[str] = None


def get_dashboard_data(data_path: Optional[str] = None) -> DashboardData:
    global _cached_data, _cached_data_path
    path = data_path or "insulin_dosage_prediction.csv"
    path_obj = ROOT / path if path else ROOT / "insulin_dosage_prediction.csv"
    if _cached_data is None or _cached_data_path != path:
        cfg = DashboardConfig(data_path=path_obj)
        _cached_data = load_dashboard_data(cfg, path_obj, run_pipeline_for_reference=True)
        _cached_data_path = path
    return _cached_data


# --- Response models ---
class HealthResponse(BaseModel):
    status: str


class PatientProfileResponse(BaseModel):
    profile: Dict[str, Any]
    predicted_class: str
    confidence: float
    entropy: float
    probability_breakdown: Dict[str, float]
    recommendation_summary: str
    recommendation_detail: str
    is_high_risk: bool
    high_risk_reason: Optional[str]
    shap_summary_url: Optional[str]
    natural_language: Optional[str]
    alternative_scenarios: List[str]


class PopulationResponse(BaseModel):
    evaluation_summary: List[Dict[str, Any]]
    distribution: Dict[str, int]
    temporal_validation: Optional[List[Dict[str, Any]]]
    cohort_crosstab: Optional[Dict[str, Dict[str, int]]]
    best_model: str
    permutation_importance_url: Optional[str]
    builtin_importance_url: Optional[str]


class SimilarPatient(BaseModel):
    index: int
    distance: float
    outcome: str


class ClinicalResponse(BaseModel):
    accuracy: float
    classification_report: str
    high_risk_count: int
    high_risk_list: List[Dict[str, Any]]
    pathway_steps: List[str]
    model_name: str


# --- Routes ---
@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/api/config")
def get_config():
    """Return default data path and model info."""
    data = get_dashboard_data()
    return {
        "data_path": _cached_data_path or "insulin_dosage_prediction.csv",
        "model_name": data.model_name,
        "classes": data.classes,
        "n_reference": len(data.reference_X) if data.reference_X is not None else 0,
        "has_bundle": data.bundle is not None,
    }


@app.get("/api/patient/{patient_index}", response_model=PatientProfileResponse)
def get_patient(patient_index: int, data_path: Optional[str] = None):
    """Patient-level: profile, prediction, recommendation."""
    data = get_dashboard_data(data_path)
    if not data.bundle:
        raise HTTPException(status_code=503, detail="No model loaded. Run evaluation first.")
    n_ref = len(data.reference_X) if data.reference_X is not None else 0
    if not n_ref or data.reference_df is None:
        raise HTTPException(status_code=404, detail="No reference data. Set data path and ensure pipeline has run.")
    if patient_index < 0 or patient_index >= len(data.reference_df):
        raise HTTPException(status_code=404, detail=f"Patient index must be 0..{n_ref - 1}")

    schema = DataSchema()
    raw_row = data.reference_df.iloc[patient_index]
    profile_cols = [c for c in raw_row.index if c not in (schema.TARGET, "_outlier_flag")]
    profile = {str(k): str(raw_row[k]) for k in profile_cols}

    X_one = data.reference_X[patient_index : patient_index + 1]
    pred = data.bundle.predict(X_one)[0]
    proba = data.bundle.predict_proba(X_one)[0]
    conf = float(proba[list(data.classes).index(pred)])
    entropy = float(-(proba * np.log(proba + 1e-10)).sum())
    prob_breakdown = {str(c): float(proba[i]) for i, c in enumerate(data.classes)}

    from insulin_system.recommendation import RecommendationGenerator
    rec_gen = RecommendationGenerator()
    rec = rec_gen.generate(str(pred), conf, entropy, prob_breakdown)

    rec_list = [r for r in data.recommendations if r.get("patient_index") == patient_index]
    natural_language = rec_list[0].get("natural_language") if rec_list else None
    alternative_scenarios = rec_list[0].get("alternative_scenarios", []) if rec_list else []

    shap_url = None
    model_name = data.model_name or "random_forest"
    summary_path = ROOT / "outputs" / "explainability" / model_name / "shap_summary.png"
    if summary_path.exists():
        shap_url = f"/static/outputs/explainability/{model_name}/shap_summary.png"

    return PatientProfileResponse(
        profile=profile,
        predicted_class=str(pred),
        confidence=conf,
        entropy=entropy,
        probability_breakdown=prob_breakdown,
        recommendation_summary=rec.dosage_suggestion.summary,
        recommendation_detail=rec.dosage_suggestion.detail,
        is_high_risk=rec.is_high_risk,
        high_risk_reason=rec.high_risk_reason,
        shap_summary_url=shap_url,
        natural_language=natural_language,
        alternative_scenarios=alternative_scenarios,
    )


@app.get("/api/population", response_model=PopulationResponse)
def get_population(data_path: Optional[str] = None):
    """Population-level: evaluation summary, distribution, temporal, cohort."""
    data = get_dashboard_data(data_path)
    eval_list = []
    if data.evaluation_summary is not None and not data.evaluation_summary.empty:
        eval_list = data.evaluation_summary.to_dict("records")
        for r in eval_list:
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    r[k] = float(v)

    distribution = {}
    if data.reference_y is not None and len(data.reference_y) > 0:
        for v in data.reference_y:
            s = str(v)
            distribution[s] = distribution.get(s, 0) + 1

    temporal_list = None
    if data.temporal_validation is not None and not data.temporal_validation.empty:
        temporal_list = data.temporal_validation.to_dict("records")

    cohort_ct: Optional[Dict[str, Dict[str, int]]] = None
    if data.bundle and data.reference_X is not None and data.reference_y is not None and len(data.reference_y) > 0:
        preds = data.bundle.predict(data.reference_X)
        from collections import defaultdict
        ct: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for a, p in zip(data.reference_y, preds):
            ct[str(a)][str(p)] += 1
        cohort_ct = {k: dict(v) for k, v in ct.items()}

    best_model = data.evaluation_summary.iloc[0]["model"] if data.evaluation_summary is not None and not data.evaluation_summary.empty else data.model_name or ""
    perm_url = None
    builtin_url = None
    if best_model:
        perm_path = ROOT / "outputs" / "evaluation" / best_model / "feature_importance_permutation.png"
        if perm_path.exists():
            perm_url = f"/static/outputs/evaluation/{best_model}/feature_importance_permutation.png"
        builtin_path = ROOT / "outputs" / "evaluation" / best_model / "feature_importance_builtin.png"
        if builtin_path.exists():
            builtin_url = f"/static/outputs/evaluation/{best_model}/feature_importance_builtin.png"

    return PopulationResponse(
        evaluation_summary=eval_list,
        distribution=distribution,
        temporal_validation=temporal_list,
        cohort_crosstab=cohort_ct,
        best_model=best_model,
        permutation_importance_url=perm_url,
        builtin_importance_url=builtin_url,
    )


@app.get("/api/similar")
def get_similar(query_index: int, k: int = 5, data_path: Optional[str] = None):
    """Similar patient search."""
    data = get_dashboard_data(data_path)
    if data.reference_X is None or data.reference_y is None:
        raise HTTPException(status_code=404, detail="No reference data.")
    from sklearn.neighbors import NearestNeighbors
    k = min(k + 1, len(data.reference_X))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(data.reference_X)
    dists, indices = nn.kneighbors(data.reference_X[query_index : query_index + 1])
    indices = indices[0][1:]
    dists = dists[0][1:]
    return [
        {"index": int(i), "distance": float(d), "outcome": str(data.reference_y[i])}
        for i, d in zip(indices, dists)
    ]


@app.get("/api/clinical", response_model=ClinicalResponse)
def get_clinical(data_path: Optional[str] = None):
    """Clinical tools: accuracy, report, high-risk, pathway."""
    data = get_dashboard_data(data_path)
    if not data.bundle:
        raise HTTPException(status_code=503, detail="No model loaded.")

    accuracy = 0.0
    report = ""
    if data.reference_X is not None and data.reference_y is not None:
        preds = data.bundle.predict(data.reference_X)
        accuracy = float(np.mean(np.array(preds) == np.array(data.reference_y)))
        from sklearn.metrics import classification_report
        report = classification_report(
            data.reference_y, preds,
            target_names=list(data.classes),
            zero_division=0,
        )

    high_risk = [r for r in data.recommendations if r.get("is_high_risk")]
    pathway = [
        "Input: Patient features (glucose, HbA1c, BMI, etc.)",
        "Model: Predicted insulin category (down / up / steady / no)",
        "Recommendation: Dosage suggestion and confidence",
        "Review: High-risk cases flagged for clinician",
        "Action: Adjust dosage per clinical guidelines",
    ]
    return ClinicalResponse(
        accuracy=accuracy,
        classification_report=report,
        high_risk_count=len(high_risk),
        high_risk_list=high_risk,
        pathway_steps=pathway,
        model_name=data.model_name or "",
    )
