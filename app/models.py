from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'patient' or 'doctor'
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    patient_profile = db.relationship('PatientProfile', backref='user', uselist=False, cascade='all, delete-orphan')
    doctor_profile = db.relationship('DoctorProfile', backref='user', uselist=False, cascade='all, delete-orphan')
    patient_data = db.relationship('PatientData', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'role': self.role,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class PatientProfile(db.Model):
    __tablename__ = 'patient_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    bmi = db.Column(db.Float)
    hba1c = db.Column(db.Float)  # HbA1c level
    insulin_type = db.Column(db.String(50))
    diabetes_type = db.Column(db.String(20))  # Type 1 or Type 2
    medical_history = db.Column(db.Text)
    emergency_contact = db.Column(db.String(200))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'age': self.age,
            'gender': self.gender,
            'bmi': self.bmi,
            'hba1c': self.hba1c,
            'insulin_type': self.insulin_type,
            'diabetes_type': self.diabetes_type,
            'medical_history': self.medical_history,
            'emergency_contact': self.emergency_contact,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class DoctorProfile(db.Model):
    __tablename__ = 'doctor_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    license_number = db.Column(db.String(100))
    specialization = db.Column(db.String(100))
    hospital_affiliation = db.Column(db.String(200))
    years_experience = db.Column(db.Integer)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient_assignments = db.relationship('DoctorPatientAssignment', backref='doctor', lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'license_number': self.license_number,
            'specialization': self.specialization,
            'hospital_affiliation': self.hospital_affiliation,
            'years_experience': self.years_experience,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class DoctorPatientAssignment(db.Model):
    __tablename__ = 'doctor_patient_assignments'
    
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor_profiles.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    patient = db.relationship('User', foreign_keys=[patient_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'doctor_id': self.doctor_id,
            'patient_id': self.patient_id,
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'is_active': self.is_active
        }

class PatientData(db.Model):
    __tablename__ = 'patient_data'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Glucose and insulin data
    glucose_level = db.Column(db.Float)
    insulin_dosage = db.Column(db.Float)
    insulin_type_used = db.Column(db.String(50))
    
    # Lifestyle factors
    food_intake = db.Column(db.Float)  # Carbs in grams
    physical_activity = db.Column(db.Float)  # Minutes of exercise
    activity_intensity = db.Column(db.String(20))  # Low, Medium, High
    
    # Additional context
    notes = db.Column(db.Text)
    meal_type = db.Column(db.String(50))  # Breakfast, Lunch, Dinner, Snack
    
    # ML analysis results (stored as JSON)
    ml_analysis = db.Column(db.Text)  # JSON string with predictions and explanations
    
    def to_dict(self):
        ml_data = None
        if self.ml_analysis:
            try:
                ml_data = json.loads(self.ml_analysis)
            except:
                pass
        
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'glucose_level': self.glucose_level,
            'insulin_dosage': self.insulin_dosage,
            'insulin_type_used': self.insulin_type_used,
            'food_intake': self.food_intake,
            'physical_activity': self.physical_activity,
            'activity_intensity': self.activity_intensity,
            'notes': self.notes,
            'meal_type': self.meal_type,
            'ml_analysis': ml_data
        }

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor_profiles.id'), nullable=True)
    alert_type = db.Column(db.String(50), nullable=False)  # 'high_glucose', 'low_glucose', 'pattern_anomaly', etc.
    severity = db.Column(db.String(20), nullable=False)  # 'low', 'medium', 'high', 'critical'
    message = db.Column(db.Text, nullable=False)
    patient_data_id = db.Column(db.Integer, db.ForeignKey('patient_data.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    acknowledged_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    status = db.Column(db.String(20), default='active')  # 'active', 'acknowledged', 'resolved'
    
    # Relationships
    patient = db.relationship('User', foreign_keys=[patient_id])
    doctor = db.relationship('DoctorProfile', foreign_keys=[doctor_id])
    data_entry = db.relationship('PatientData', foreign_keys=[patient_data_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'doctor_id': self.doctor_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'patient_data_id': self.patient_data_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'status': self.status
        }

