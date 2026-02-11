from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from app import db, socketio
from app.models import User, PatientProfile, DoctorProfile, DoctorPatientAssignment, PatientData, Alert
from app.auth import role_required, get_current_user
from app.ml_service import ml_service
from app.utils import check_glucose_alerts, get_patient_summary, create_alert
import json

# Authentication Blueprint
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['email', 'password', 'role', 'first_name', 'last_name']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400
    
    # Check if user exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    # Validate role
    if data['role'] not in ['patient', 'doctor']:
        return jsonify({'error': 'Invalid role. Must be "patient" or "doctor"'}), 400
    
    # Create user
    user = User(
        email=data['email'],
        role=data['role'],
        first_name=data['first_name'],
        last_name=data['last_name'],
        phone=data.get('phone')
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.flush()
    
    # Create profile based on role
    if data['role'] == 'patient':
        profile = PatientProfile(
            user_id=user.id,
            age=data.get('age'),
            gender=data.get('gender'),
            bmi=data.get('bmi'),
            hba1c=data.get('hba1c'),
            insulin_type=data.get('insulin_type'),
            diabetes_type=data.get('diabetes_type'),
            medical_history=data.get('medical_history'),
            emergency_contact=data.get('emergency_contact')
        )
    else:
        profile = DoctorProfile(
            user_id=user.id,
            license_number=data.get('license_number'),
            specialization=data.get('specialization'),
            hospital_affiliation=data.get('hospital_affiliation'),
            years_experience=data.get('years_experience')
        )
    
    db.session.add(profile)
    db.session.commit()
    
    # Create access token
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'message': 'Registration successful',
        'access_token': access_token,
        'user': user.to_dict()
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is inactive'}), 403
    
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'access_token': access_token,
        'user': user.to_dict()
    }), 200

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user_info():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    response = user.to_dict()
    
    # Add profile data
    if user.role == 'patient' and user.patient_profile:
        response['profile'] = user.patient_profile.to_dict()
    elif user.role == 'doctor' and user.doctor_profile:
        response['profile'] = user.doctor_profile.to_dict()
    
    return jsonify(response), 200

# Patient Blueprint
patient_bp = Blueprint('patient', __name__)

@patient_bp.route('/data', methods=['POST'])
@jwt_required()
@role_required('patient')
def log_patient_data():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Create patient data entry
    patient_data = PatientData(
        user_id=user_id,
        glucose_level=data.get('glucose_level'),
        insulin_dosage=data.get('insulin_dosage'),
        insulin_type_used=data.get('insulin_type_used'),
        food_intake=data.get('food_intake'),
        physical_activity=data.get('physical_activity'),
        activity_intensity=data.get('activity_intensity', 'Medium'),
        notes=data.get('notes'),
        meal_type=data.get('meal_type')
    )
    
    db.session.add(patient_data)
    db.session.flush()
    
    # Get patient profile for ML analysis
    user = User.query.get(user_id)
    profile_data = user.patient_profile.to_dict() if user.patient_profile else {}
    
    # Run ML analysis
    ml_result = ml_service.predict(
        {
            'glucose_level': data.get('glucose_level'),
            'food_intake': data.get('food_intake', 0),
            'physical_activity': data.get('physical_activity', 0),
            'activity_intensity': data.get('activity_intensity', 'Medium')
        },
        profile_data
    )
    
    # Store ML analysis
    patient_data.ml_analysis = json.dumps(ml_result)
    db.session.commit()
    
    # Check for glucose alerts
    alert = check_glucose_alerts(patient_data)
    
    # Emit real-time alert if critical
    if alert and alert.severity == 'critical':
        socketio.emit('critical_alert', {
            'alert_id': alert.id,
            'message': alert.message,
            'patient_id': user_id
        }, room=f'doctor_{user_id}')
    
    return jsonify({
        'message': 'Data logged successfully',
        'data': patient_data.to_dict(),
        'ml_analysis': ml_result
    }), 201

@patient_bp.route('/data', methods=['GET'])
@jwt_required()
@role_required('patient')
def get_patient_data():
    user_id = get_jwt_identity()
    
    # Get query parameters
    days = request.args.get('days', 30, type=int)
    limit = request.args.get('limit', 100, type=int)
    
    end_date = datetime.utcnow()
    start_date = datetime.utcnow().replace(day=1) if days == 0 else end_date - timedelta(days=days)
    
    data_entries = PatientData.query.filter(
        PatientData.user_id == user_id,
        PatientData.timestamp >= start_date
    ).order_by(PatientData.timestamp.desc()).limit(limit).all()
    
    return jsonify({
        'data': [entry.to_dict() for entry in data_entries],
        'count': len(data_entries)
    }), 200

@patient_bp.route('/summary', methods=['GET'])
@jwt_required()
@role_required('patient')
def get_patient_summary_endpoint():
    user_id = get_jwt_identity()
    summary = get_patient_summary(user_id)
    
    # Get recent alerts
    alerts = Alert.query.filter_by(patient_id=user_id, status='active').order_by(Alert.created_at.desc()).limit(5).all()
    
    return jsonify({
        'summary': summary,
        'recent_alerts': [alert.to_dict() for alert in alerts]
    }), 200

@patient_bp.route('/profile', methods=['GET', 'PUT'])
@jwt_required()
@role_required('patient')
def manage_patient_profile():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if request.method == 'GET':
        if not user.patient_profile:
            return jsonify({'error': 'Profile not found'}), 404
        return jsonify(user.patient_profile.to_dict()), 200
    
    # PUT - Update profile
    data = request.get_json()
    profile = user.patient_profile
    
    if not profile:
        profile = PatientProfile(user_id=user_id)
        db.session.add(profile)
    
    # Update fields
    for field in ['age', 'gender', 'bmi', 'hba1c', 'insulin_type', 'diabetes_type', 'medical_history', 'emergency_contact']:
        if field in data:
            setattr(profile, field, data[field])
    
    profile.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify(profile.to_dict()), 200

@patient_bp.route('/alerts', methods=['GET'])
@jwt_required()
@role_required('patient')
def get_patient_alerts():
    user_id = get_jwt_identity()
    
    status = request.args.get('status', 'active')
    limit = request.args.get('limit', 20, type=int)
    
    alerts = Alert.query.filter_by(patient_id=user_id, status=status).order_by(Alert.created_at.desc()).limit(limit).all()
    
    return jsonify({
        'alerts': [alert.to_dict() for alert in alerts]
    }), 200

# Doctor Blueprint
doctor_bp = Blueprint('doctor', __name__)

@doctor_bp.route('/patients', methods=['GET'])
@jwt_required()
@role_required('doctor')
def get_doctor_patients():
    doctor_id = get_jwt_identity()
    doctor = User.query.get(doctor_id)
    
    if not doctor or not doctor.doctor_profile:
        return jsonify({'error': 'Doctor profile not found'}), 404
    
    # Get assigned patients
    assignments = DoctorPatientAssignment.query.filter_by(
        doctor_id=doctor.doctor_profile.id,
        is_active=True
    ).all()
    
    patients_data = []
    for assignment in assignments:
        patient = User.query.get(assignment.patient_id)
        if patient:
            patient_dict = patient.to_dict()
            if patient.patient_profile:
                patient_dict['profile'] = patient.patient_profile.to_dict()
            
            # Get summary
            summary = get_patient_summary(patient.id)
            patient_dict['summary'] = summary
            
            # Get active alerts count
            active_alerts = Alert.query.filter_by(patient_id=patient.id, status='active').count()
            patient_dict['active_alerts_count'] = active_alerts
            
            patients_data.append(patient_dict)
    
    return jsonify({
        'patients': patients_data,
        'count': len(patients_data)
    }), 200

@doctor_bp.route('/patients/<int:patient_id>/data', methods=['GET'])
@jwt_required()
@role_required('doctor')
def get_patient_data_doctor(patient_id):
    doctor_id = get_jwt_identity()
    doctor = User.query.get(doctor_id)
    
    # Verify assignment
    assignment = DoctorPatientAssignment.query.filter_by(
        doctor_id=doctor.doctor_profile.id,
        patient_id=patient_id,
        is_active=True
    ).first()
    
    if not assignment:
        return jsonify({'error': 'Patient not assigned to you'}), 403
    
    days = request.args.get('days', 30, type=int)
    limit = request.args.get('limit', 100, type=int)
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    data_entries = PatientData.query.filter(
        PatientData.user_id == patient_id,
        PatientData.timestamp >= start_date
    ).order_by(PatientData.timestamp.desc()).limit(limit).all()
    
    return jsonify({
        'data': [entry.to_dict() for entry in data_entries],
        'count': len(data_entries)
    }), 200

@doctor_bp.route('/patients/<int:patient_id>/summary', methods=['GET'])
@jwt_required()
@role_required('doctor')
def get_patient_summary_doctor(patient_id):
    doctor_id = get_jwt_identity()
    doctor = User.query.get(doctor_id)
    
    # Verify assignment
    assignment = DoctorPatientAssignment.query.filter_by(
        doctor_id=doctor.doctor_profile.id,
        patient_id=patient_id,
        is_active=True
    ).first()
    
    if not assignment:
        return jsonify({'error': 'Patient not assigned to you'}), 403
    
    summary = get_patient_summary(patient_id)
    
    # Get all alerts
    alerts = Alert.query.filter_by(patient_id=patient_id).order_by(Alert.created_at.desc()).limit(20).all()
    
    # Get recent data for pattern analysis (last 14 days for Type 1 metrics)
    days = request.args.get('days', 14, type=int)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    recent_data = PatientData.query.filter(
        PatientData.user_id == patient_id,
        PatientData.timestamp >= start_date
    ).order_by(PatientData.timestamp.desc()).all()
    
    historical_data = [entry.to_dict() for entry in recent_data]
    
    # Get Type 1 diabetes specific pattern analysis
    pattern_analysis = ml_service.analyze_patterns(historical_data)
    
    # Get patient profile for additional context
    patient = User.query.get(patient_id)
    patient_profile = patient.patient_profile.to_dict() if patient and patient.patient_profile else {}
    
    # Calculate additional Type 1 diabetes metrics
    type1_metrics = {}
    if historical_data:
        glucose_levels = [d.get('glucose_level') for d in historical_data if d.get('glucose_level')]
        if glucose_levels:
            type1_metrics = {
                'total_readings': len(glucose_levels),
                'mean_glucose': sum(glucose_levels) / len(glucose_levels),
                'min_glucose': min(glucose_levels),
                'max_glucose': max(glucose_levels),
                'glucose_std': (sum((x - sum(glucose_levels)/len(glucose_levels))**2 for x in glucose_levels) / len(glucose_levels))**0.5
            }
    
    return jsonify({
        'summary': summary,
        'alerts': [alert.to_dict() for alert in alerts],
        'pattern_analysis': pattern_analysis,
        'type1_metrics': type1_metrics,
        'patient_profile': patient_profile,
        'data_period': {
            'days': days,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'data_points': len(historical_data)
        }
    }), 200

@doctor_bp.route('/alerts', methods=['GET'])
@jwt_required()
@role_required('doctor')
def get_doctor_alerts():
    doctor_id = get_jwt_identity()
    doctor = User.query.get(doctor_id)
    
    if not doctor or not doctor.doctor_profile:
        return jsonify({'error': 'Doctor profile not found'}), 404
    
    # Get alerts for assigned patients
    assignments = DoctorPatientAssignment.query.filter_by(
        doctor_id=doctor.doctor_profile.id,
        is_active=True
    ).all()
    
    patient_ids = [a.patient_id for a in assignments]
    
    status = request.args.get('status', 'active')
    severity = request.args.get('severity')
    
    query = Alert.query.filter(
        Alert.patient_id.in_(patient_ids),
        Alert.status == status
    )
    
    if severity:
        query = query.filter(Alert.severity == severity)
    
    alerts = query.order_by(Alert.created_at.desc()).limit(50).all()
    
    return jsonify({
        'alerts': [alert.to_dict() for alert in alerts]
    }), 200

@doctor_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@jwt_required()
@role_required('doctor')
def acknowledge_alert(alert_id):
    doctor_id = get_jwt_identity()
    alert = Alert.query.get(alert_id)
    
    if not alert:
        return jsonify({'error': 'Alert not found'}), 404
    
    # Verify assignment
    assignment = DoctorPatientAssignment.query.filter_by(
        doctor_id=alert.doctor_id,
        patient_id=alert.patient_id,
        is_active=True
    ).first()
    
    if not assignment and alert.doctor_id != doctor_id:
        return jsonify({'error': 'Not authorized'}), 403
    
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = doctor_id
    alert.status = 'acknowledged'
    
    db.session.commit()
    
    return jsonify(alert.to_dict()), 200

@doctor_bp.route('/assign-patient', methods=['POST'])
@jwt_required()
@role_required('doctor')
def assign_patient():
    doctor_id = get_jwt_identity()
    doctor = User.query.get(doctor_id)
    
    data = request.get_json()
    patient_email = data.get('patient_email')
    
    if not patient_email:
        return jsonify({'error': 'Patient email required'}), 400
    
    patient = User.query.filter_by(email=patient_email, role='patient').first()
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    # Check if already assigned
    existing = DoctorPatientAssignment.query.filter_by(
        doctor_id=doctor.doctor_profile.id,
        patient_id=patient.id
    ).first()
    
    if existing:
        existing.is_active = True
    else:
        assignment = DoctorPatientAssignment(
            doctor_id=doctor.doctor_profile.id,
            patient_id=patient.id
        )
        db.session.add(assignment)
    
    db.session.commit()
    
    return jsonify({'message': 'Patient assigned successfully'}), 200

# ML Blueprint
ml_bp = Blueprint('ml', __name__)

@ml_bp.route('/analyze', methods=['POST'])
@jwt_required()
def analyze_data():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    
    # Get profile data
    profile_data = {}
    if user.role == 'patient' and user.patient_profile:
        profile_data = user.patient_profile.to_dict()
    
    # Run ML analysis
    result = ml_service.predict(data, profile_data)
    
    return jsonify(result), 200

@ml_bp.route('/patterns', methods=['POST'])
@jwt_required()
def analyze_patterns():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    historical_data = data.get('historical_data', [])
    
    result = ml_service.analyze_patterns(historical_data)
    
    return jsonify(result), 200
