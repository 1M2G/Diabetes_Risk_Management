"""
API-First Routes for External Applications
These endpoints are designed for integration with other Type 1 Diabetes management apps
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
from datetime import datetime, timedelta
from app import db, socketio
from app.models import User, PatientProfile, DoctorProfile, DoctorPatientAssignment, PatientData, Alert
from app.auth import role_required, get_current_user
from app.ml_service import ml_service
from app.utils import check_glucose_alerts, get_patient_summary, create_alert
import json

# API Blueprint for external applications
api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'insulin-management-api',
        'version': '2.0.0',
        'ml_model': ml_service.get_active_model_info().get('model_name', 'unknown'),
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@api_bp.route('/ml/predict', methods=['POST'])
@jwt_required(optional=True)
def ml_predict():
    """
    ML Prediction Endpoint for External Apps
    
    Request Body:
    {
        "patient_data": {
            "glucose_level": 150,
            "food_intake": 60,
            "physical_activity": 30,
            ...
        },
        "patient_profile": {
            "age": 30,
            "weight_kg": 70,
            "diabetes_type": "Type 1",
            "carb_ratio": 15,
            "insulin_sensitivity_factor": 50,
            ...
        }
    }
    
    Returns:
    {
        "recommended_dose": 4.5,
        "confidence": 0.85,
        "prediction_type": "bolus",
        "explanation": "...",
        "safety_flags": [],
        "model_info": {...}
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    patient_data = data.get('patient_data', {})
    patient_profile = data.get('patient_profile', {})
    
    if not patient_data.get('glucose_level'):
        return jsonify({'error': 'glucose_level is required in patient_data'}), 400
    
    # Ensure Type 1 diabetes
    if patient_profile.get('diabetes_type') != 'Type 1':
        return jsonify({
            'error': 'This API is designed for Type 1 Diabetes only',
            'diabetes_type': patient_profile.get('diabetes_type', 'unknown')
        }), 400
    
    # Get prediction
    result = ml_service.predict(patient_data, patient_profile)
    
    return jsonify(result), 200

@api_bp.route('/ml/explain', methods=['POST'])
@jwt_required(optional=True)
def ml_explain():
    """
    Get detailed explanation of ML prediction
    
    Same request format as /ml/predict
    Returns detailed explanation with reasoning steps
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    patient_data = data.get('patient_data', {})
    patient_profile = data.get('patient_profile', {})
    
    if not patient_data.get('glucose_level'):
        return jsonify({'error': 'glucose_level is required'}), 400
    
    explanation = ml_service.explain(patient_data, patient_profile)
    
    return jsonify(explanation), 200

@api_bp.route('/ml/analyze-patterns', methods=['POST'])
@jwt_required(optional=True)
def ml_analyze_patterns():
    """
    Analyze patterns in historical glucose data
    
    Request Body:
    {
        "historical_data": [
            {
                "timestamp": "2024-01-01T08:00:00Z",
                "glucose_level": 120,
                "insulin_dosage": 5.0,
                ...
            },
            ...
        ]
    }
    
    Returns Type 1 diabetes specific metrics:
    - Time in Range (TIR)
    - Time Below Range (TBR)
    - Time Above Range (TAR)
    - Glucose Variability (CV)
    - Stability Score
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    historical_data = data.get('historical_data', [])
    
    if not historical_data:
        return jsonify({'error': 'historical_data array required'}), 400
    
    analysis = ml_service.analyze_patterns(historical_data)
    
    return jsonify(analysis), 200

@api_bp.route('/ml/models', methods=['GET'])
@jwt_required(optional=True)
def list_ml_models():
    """List all available ML models"""
    models = ml_service.list_available_models()
    active_model_info = ml_service.get_active_model_info()
    
    return jsonify({
        'available_models': models,
        'active_model': active_model_info
    }), 200

@api_bp.route('/ml/models/<model_id>/switch', methods=['POST'])
@jwt_required()
@role_required('doctor')
def switch_ml_model(model_id: str):
    """Switch active ML model (doctor only)"""
    success = ml_service.switch_model(model_id)
    
    if success:
        return jsonify({
            'message': f'Switched to model: {model_id}',
            'active_model': ml_service.get_active_model_info()
        }), 200
    else:
        return jsonify({'error': f'Model {model_id} not found'}), 404

@api_bp.route('/type1/metrics', methods=['POST'])
@jwt_required(optional=True)
def calculate_type1_metrics():
    """
    Calculate Type 1 Diabetes specific metrics
    
    Request Body:
    {
        "glucose_readings": [120, 150, 140, ...],
        "timestamps": ["2024-01-01T08:00:00Z", ...],
        "target_range": {"min": 70, "max": 180}
    }
    
    Returns:
    {
        "time_in_range": 0.75,
        "time_below_range": 0.05,
        "time_above_range": 0.20,
        "mean_glucose": 145.5,
        "glucose_variability_cv": 28.3,
        "stability_score": 82.5,
        ...
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    glucose_readings = data.get('glucose_readings', [])
    
    if not glucose_readings:
        return jsonify({'error': 'glucose_readings array required'}), 400
    
    # Convert to historical data format
    historical_data = []
    timestamps = data.get('timestamps', [])
    
    for i, glucose in enumerate(glucose_readings):
        entry = {'glucose_level': glucose}
        if i < len(timestamps):
            entry['timestamp'] = timestamps[i]
        historical_data.append(entry)
    
    analysis = ml_service.analyze_patterns(historical_data)
    
    return jsonify(analysis), 200

@api_bp.route('/type1/safety-check', methods=['POST'])
@jwt_required(optional=True)
def safety_check():
    """
    Safety check for insulin dose recommendation
    
    Request Body:
    {
        "recommended_dose": 5.0,
        "patient_profile": {
            "weight_kg": 70,
            "total_daily_dose": 50,
            ...
        },
        "current_glucose": 150
    }
    
    Returns safety assessment
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    recommended_dose = data.get('recommended_dose', 0)
    patient_profile = data.get('patient_profile', {})
    current_glucose = data.get('current_glucose', 120)
    
    # Calculate safety limits
    from app.ml_framework.base_model import BaseInsulinModel
    temp_model = BaseInsulinModel("temp", "1.0.0")
    safety_limits = temp_model.calculate_safety_limits(patient_profile)
    
    # Safety checks
    safety_flags = []
    is_safe = True
    
    if recommended_dose > safety_limits['max_bolus']:
        safety_flags.append({
            'type': 'dose_too_high',
            'severity': 'high',
            'message': f'Dose exceeds maximum safe bolus ({safety_limits["max_bolus"]} units)'
        })
        is_safe = False
    
    if current_glucose < 70 and recommended_dose > 0.5:
        safety_flags.append({
            'type': 'hypoglycemia_risk',
            'severity': 'critical',
            'message': 'Do not give insulin when glucose is low - treat hypoglycemia first'
        })
        is_safe = False
    
    if recommended_dose < 0:
        safety_flags.append({
            'type': 'negative_dose',
            'severity': 'high',
            'message': 'Dose cannot be negative'
        })
        is_safe = False
    
    return jsonify({
        'is_safe': is_safe,
        'safety_flags': safety_flags,
        'safety_limits': safety_limits,
        'recommended_dose': recommended_dose,
        'assessment': 'safe' if is_safe else 'unsafe'
    }), 200

@api_bp.route('/data/submit', methods=['POST'])
@jwt_required(optional=True)
def submit_patient_data():
    """
    Submit patient data from external application
    
    Request Body:
    {
        "patient_id": "external_patient_id",
        "glucose_level": 150,
        "insulin_dosage": 5.0,
        "food_intake": 60,
        "timestamp": "2024-01-01T08:00:00Z",
        ...
    }
    
    Returns:
    {
        "data_id": 123,
        "ml_analysis": {...},
        "alerts": [...]
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    # For external apps, we might not have a user_id
    # In this case, we can create a temporary record or use API key
    user_id = get_jwt_identity() if get_jwt_identity() else None
    
    if not user_id:
        # For external apps without authentication, use patient_id mapping
        external_patient_id = data.get('patient_id')
        if not external_patient_id:
            return jsonify({'error': 'patient_id or authentication required'}), 400
        
        # In production, you'd map external_patient_id to internal user_id
        # For now, return analysis without storing
        patient_data = {
            'glucose_level': data.get('glucose_level'),
            'food_intake': data.get('food_intake', 0),
            'physical_activity': data.get('physical_activity', 0)
        }
        patient_profile = {
            'diabetes_type': 'Type 1',
            'age': data.get('age', 45),
            'weight_kg': data.get('weight_kg', 70)
        }
        
        ml_result = ml_service.predict(patient_data, patient_profile)
        
        return jsonify({
            'ml_analysis': ml_result,
            'message': 'Analysis complete (data not stored - authentication required for storage)'
        }), 200
    
    # If authenticated, store the data
    patient_data_entry = PatientData(
        user_id=user_id,
        glucose_level=data.get('glucose_level'),
        insulin_dosage=data.get('insulin_dosage'),
        food_intake=data.get('food_intake'),
        physical_activity=data.get('physical_activity'),
        activity_intensity=data.get('activity_intensity', 'Medium'),
        notes=data.get('notes')
    )
    
    if data.get('timestamp'):
        patient_data_entry.timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
    
    db.session.add(patient_data_entry)
    db.session.flush()
    
    # Get patient profile for ML
    user = User.query.get(user_id)
    profile_data = user.patient_profile.to_dict() if user and user.patient_profile else {}
    
    # Run ML analysis
    ml_result = ml_service.predict(
        {
            'glucose_level': data.get('glucose_level'),
            'food_intake': data.get('food_intake', 0),
            'physical_activity': data.get('physical_activity', 0)
        },
        profile_data
    )
    
    # Store ML analysis
    patient_data_entry.ml_analysis = json.dumps(ml_result)
    db.session.commit()
    
    # Check for alerts
    alert = check_glucose_alerts(patient_data_entry)
    
    return jsonify({
        'data_id': patient_data_entry.id,
        'ml_analysis': ml_result,
        'alert': alert.to_dict() if alert else None
    }), 201

@api_bp.route('/docs', methods=['GET'])
def api_docs():
    """API Documentation"""
    return jsonify({
        'title': 'Insulin Management API',
        'version': '2.0.0',
        'description': 'API for Type 1 Diabetes insulin management with ML-powered recommendations',
        'endpoints': {
            'GET /api/health': 'Health check',
            'POST /api/ml/predict': 'Get insulin dose recommendation',
            'POST /api/ml/explain': 'Get detailed explanation',
            'POST /api/ml/analyze-patterns': 'Analyze historical patterns',
            'GET /api/ml/models': 'List available ML models',
            'POST /api/type1/metrics': 'Calculate Type 1 diabetes metrics',
            'POST /api/type1/safety-check': 'Safety check for dose',
            'POST /api/data/submit': 'Submit patient data'
        },
        'authentication': 'JWT Bearer token (optional for some endpoints)',
        'target_population': 'Type 1 Diabetes',
        'safety_note': 'All recommendations are advisory only. Always consult healthcare provider.'
    }), 200

