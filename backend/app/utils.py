from datetime import datetime, timedelta
from app.models import Alert, PatientData, User
from app import db

def create_alert(patient_id, alert_type, severity, message, patient_data_id=None, doctor_id=None):
    """Helper function to create alerts"""
    alert = Alert(
        patient_id=patient_id,
        doctor_id=doctor_id,
        alert_type=alert_type,
        severity=severity,
        message=message,
        patient_data_id=patient_data_id
    )
    db.session.add(alert)
    db.session.commit()
    return alert

def check_glucose_alerts(patient_data_entry):
    """Check if glucose level requires alert"""
    glucose = patient_data_entry.glucose_level
    
    if glucose is None:
        return None
    
    if glucose > 250:
        return create_alert(
            patient_id=patient_data_entry.user_id,
            alert_type='high_glucose',
            severity='critical',
            message=f'Critical high glucose level detected: {glucose} mg/dL. Immediate attention required.',
            patient_data_id=patient_data_entry.id
        )
    elif glucose < 70:
        return create_alert(
            patient_id=patient_data_entry.user_id,
            alert_type='low_glucose',
            severity='critical',
            message=f'Critical low glucose level detected: {glucose} mg/dL. Immediate attention required.',
            patient_data_id=patient_data_entry.id
        )
    elif glucose > 180:
        return create_alert(
            patient_id=patient_data_entry.user_id,
            alert_type='high_glucose',
            severity='high',
            message=f'Elevated glucose level: {glucose} mg/dL. Review recommended.',
            patient_data_id=patient_data_entry.id
        )
    elif glucose < 90:
        return create_alert(
            patient_id=patient_data_entry.user_id,
            alert_type='low_glucose',
            severity='high',
            message=f'Low glucose level: {glucose} mg/dL. Monitor closely.',
            patient_data_id=patient_data_entry.id
        )
    
    return None

def get_patient_summary(patient_id, days=30):
    """Generate summary statistics for a patient with Type 1 diabetes metrics"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    data_entries = PatientData.query.filter(
        PatientData.user_id == patient_id,
        PatientData.timestamp >= start_date
    ).order_by(PatientData.timestamp.desc()).all()
    
    if not data_entries:
        return {
            'total_entries': 0,
            'average_glucose': None,
            'glucose_range': None,
            'average_insulin': None,
            'recent_trend': 'insufficient_data',
            'time_in_range': None,
            'time_below_range': None,
            'time_above_range': None,
            'glucose_variability': None
        }
    
    glucose_levels = [d.glucose_level for d in data_entries if d.glucose_level is not None]
    insulin_dosages = [d.insulin_dosage for d in data_entries if d.insulin_dosage is not None]
    
    # Calculate Type 1 diabetes specific metrics
    time_in_range = None
    time_below_range = None
    time_above_range = None
    glucose_variability = None
    
    if glucose_levels:
        # Time in Range (70-180 mg/dL)
        in_range = sum(1 for g in glucose_levels if 70 <= g <= 180)
        time_in_range = in_range / len(glucose_levels)
        
        # Time Below Range (<70 mg/dL)
        below_range = sum(1 for g in glucose_levels if g < 70)
        time_below_range = below_range / len(glucose_levels)
        
        # Time Above Range (>180 mg/dL)
        above_range = sum(1 for g in glucose_levels if g > 180)
        time_above_range = above_range / len(glucose_levels)
        
        # Glucose Variability (Coefficient of Variation)
        mean_glucose = sum(glucose_levels) / len(glucose_levels)
        variance = sum((g - mean_glucose) ** 2 for g in glucose_levels) / len(glucose_levels)
        std_dev = variance ** 0.5
        glucose_variability = (std_dev / mean_glucose) * 100 if mean_glucose > 0 else 0
    
    summary = {
        'total_entries': len(data_entries),
        'average_glucose': sum(glucose_levels) / len(glucose_levels) if glucose_levels else None,
        'glucose_range': {
            'min': min(glucose_levels) if glucose_levels else None,
            'max': max(glucose_levels) if glucose_levels else None
        },
        'average_insulin': sum(insulin_dosages) / len(insulin_dosages) if insulin_dosages else None,
        'recent_trend': 'stable',
        'time_in_range': round(time_in_range, 3) if time_in_range is not None else None,
        'time_below_range': round(time_below_range, 3) if time_below_range is not None else None,
        'time_above_range': round(time_above_range, 3) if time_above_range is not None else None,
        'glucose_variability': round(glucose_variability, 1) if glucose_variability is not None else None
    }
    
    # Calculate trend
    if len(glucose_levels) >= 7:
        recent_avg = sum(glucose_levels[:7]) / 7
        older_avg = sum(glucose_levels[7:14]) / 7 if len(glucose_levels) >= 14 else recent_avg
        
        if recent_avg > older_avg * 1.1:
            summary['recent_trend'] = 'increasing'
        elif recent_avg < older_avg * 0.9:
            summary['recent_trend'] = 'decreasing'
    
    return summary

