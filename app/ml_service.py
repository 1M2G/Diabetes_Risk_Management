"""
ML Service - Wrapper for ML Framework
Provides unified interface to ML models
"""

from app.ml_framework.model_registry import model_registry
from app.ml_framework.models import DefaultInsulinModel, AdvancedInsulinModel
import pandas as pd
from typing import Dict, Any, Optional
import os

class MLService:
    """
    Service layer for ML operations
    Manages model registry and provides high-level API
    """
    
    def __init__(self):
        self.active_model = None
        self._initialize_default_models()
        self._load_active_model()
    
    def _initialize_default_models(self):
        """Initialize and register default models"""
        # Models are auto-registered via __init__.py
        # But we can ensure they're available
        if 'default' not in [m for m in model_registry.list_models().keys()]:
            model_registry.register_model(DefaultInsulinModel, 'default')
        if 'advanced' not in [m for m in model_registry.list_models().keys()]:
            model_registry.register_model(AdvancedInsulinModel, 'advanced')
    
    def _load_active_model(self):
        """Load the active model"""
        # Try to load advanced model first, fallback to default
        model_id = os.getenv('ACTIVE_ML_MODEL', 'advanced')
        
        if model_registry.set_active_model(model_id):
            self.active_model = model_registry.get_active_model()
            print(f"Active ML model: {model_id}")
        else:
            # Fallback to default
            if model_registry.set_active_model('default'):
                self.active_model = model_registry.get_active_model()
                print(f"Using default ML model")
            else:
                print("Warning: No ML model available")
    
    def get_active_model_info(self) -> Dict[str, Any]:
        """Get information about the active model"""
        if self.active_model:
            return self.active_model.get_model_info()
        return {'error': 'No active model'}
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models"""
        return model_registry.list_models()
    
    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model"""
        if model_registry.set_active_model(model_id):
            self.active_model = model_registry.get_active_model()
            return True
        return False
    
    def predict(self, patient_data: Dict[str, Any], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make insulin dose prediction
        
        Args:
            patient_data: Current patient data
            patient_profile: Patient profile information
        
        Returns:
            Prediction result with dose recommendation and explanation
        """
        if not self.active_model:
            return self._fallback_prediction(patient_data, patient_profile)
        
        try:
            result = self.active_model.predict(patient_data, patient_profile)
            
            # Add metadata
            result['model_info'] = {
                'model_name': self.active_model.model_name,
                'model_version': self.active_model.model_version,
                'model_id': getattr(self.active_model, 'MODEL_ID', 'unknown')
            }
            
            return result
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self._fallback_prediction(patient_data, patient_profile, error=str(e))
    
    def explain(self, patient_data: Dict[str, Any], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed explanation of prediction
        
        Args:
            patient_data: Current patient data
            patient_profile: Patient profile information
        
        Returns:
            Detailed explanation
        """
        if not self.active_model:
            return {'error': 'No active model available'}
        
        try:
            return self.active_model.explain(patient_data, patient_profile)
        except Exception as e:
            return {'error': f'Explanation failed: {str(e)}'}
    
    def analyze_patterns(self, historical_data: list) -> Dict[str, Any]:
        """
        Analyze patterns in historical patient data
        
        Args:
            historical_data: List of historical data points
        
        Returns:
            Pattern analysis results
        """
        if not historical_data or len(historical_data) < 7:
            return {
                'pattern': 'insufficient_data',
                'message': 'Need at least 7 days of data for pattern analysis.',
                'data_points': len(historical_data) if historical_data else 0
            }
        
        df = pd.DataFrame(historical_data)
        
        # Calculate Type 1 diabetes specific metrics
        if 'glucose_level' not in df.columns:
            return {'error': 'Glucose level data required'}
        
        glucose_levels = df['glucose_level'].dropna()
        
        if len(glucose_levels) == 0:
            return {'error': 'No valid glucose data'}
        
        # Time in Range (TIR) - Target: 70-180 mg/dL
        in_range = ((glucose_levels >= 70) & (glucose_levels <= 180)).sum()
        time_in_range = in_range / len(glucose_levels)
        
        # Time Below Range (TBR) - <70 mg/dL
        below_range = (glucose_levels < 70).sum()
        time_below_range = below_range / len(glucose_levels)
        
        # Time Above Range (TAR) - >180 mg/dL
        above_range = (glucose_levels > 180).sum()
        time_above_range = above_range / len(glucose_levels)
        
        # Glucose Variability (Coefficient of Variation)
        mean_glucose = glucose_levels.mean()
        std_glucose = glucose_levels.std()
        cv = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0
        
        # Mean Glucose
        mean_glucose_value = float(mean_glucose)
        
        # Glucose Management Indicator (GMI) - estimated A1C
        gmi = (mean_glucose_value + 46.7) / 28.7
        
        # Trends
        glucose_trend = 0.0
        if len(glucose_levels) >= 7:
            recent_avg = glucose_levels.tail(7).mean()
            older_avg = glucose_levels.head(max(1, len(glucose_levels) - 7)).mean()
            glucose_trend = float(recent_avg - older_avg)
        
        # Identify patterns
        patterns = []
        
        if time_in_range < 0.7:
            patterns.append(f"Time in Range below target ({time_in_range:.1%} vs 70% target)")
        
        if time_below_range > 0.04:  # >4% is concerning
            patterns.append(f"High time below range ({time_below_range:.1%}) - hypoglycemia risk")
        
        if time_above_range > 0.25:  # >25% is concerning
            patterns.append(f"High time above range ({time_above_range:.1%}) - hyperglycemia risk")
        
        if cv > 36:  # CV >36% indicates high variability
            patterns.append(f"High glucose variability (CV: {cv:.1f}%) - stability concern")
        
        if glucose_trend > 10:
            patterns.append("Glucose levels showing upward trend")
        elif glucose_trend < -10:
            patterns.append("Glucose levels showing downward trend")
        
        # Stability score (0-100)
        stability_score = 100
        stability_score -= (0.7 - time_in_range) * 100 if time_in_range < 0.7 else 0
        stability_score -= min(cv / 2, 30)  # Penalize high variability
        stability_score = max(0, min(100, stability_score))
        
        return {
            'pattern': 'trend_identified' if patterns else 'stable',
            'patterns': patterns,
            'metrics': {
                'time_in_range': round(time_in_range, 3),
                'time_below_range': round(time_below_range, 3),
                'time_above_range': round(time_above_range, 3),
                'mean_glucose': round(mean_glucose_value, 1),
                'glucose_variability_cv': round(cv, 1),
                'glucose_management_indicator': round(gmi, 1),
                'stability_score': round(stability_score, 1),
                'glucose_trend': round(glucose_trend, 1)
            },
            'targets': {
                'time_in_range_target': 0.70,  # 70% TIR is goal
                'time_below_range_limit': 0.04,  # <4% TBR
                'time_above_range_limit': 0.25,  # <25% TAR
                'cv_target': 36,  # CV <36%
                'mean_glucose_target': 154  # ~7% A1C equivalent
            },
            'data_points': len(historical_data),
            'assessment': self._assess_control(time_in_range, time_below_range, time_above_range, cv)
        }
    
    def _assess_control(self, tir: float, tbr: float, tar: float, cv: float) -> Dict[str, Any]:
        """Assess overall diabetes control"""
        # Excellent control
        if tir >= 0.7 and tbr < 0.04 and tar < 0.25 and cv < 36:
            return {
                'level': 'excellent',
                'message': 'Excellent glucose control - maintain current regimen',
                'recommendations': ['Continue current management', 'Maintain regular monitoring']
            }
        
        # Good control
        elif tir >= 0.5 and tbr < 0.1 and tar < 0.4:
            return {
                'level': 'good',
                'message': 'Good glucose control with room for improvement',
                'recommendations': [
                    'Focus on increasing time in range',
                    'Review meal timing and carb counting',
                    'Consider basal rate optimization'
                ]
            }
        
        # Needs improvement
        elif tir < 0.5 or tbr > 0.1 or tar > 0.4:
            return {
                'level': 'needs_improvement',
                'message': 'Glucose control needs improvement - consult healthcare provider',
                'recommendations': [
                    'Schedule appointment with endocrinologist',
                    'Review insulin-to-carb ratios',
                    'Consider CGM for better monitoring',
                    'Review meal planning and timing'
                ]
            }
        
        # Poor control
        else:
            return {
                'level': 'poor',
                'message': 'Poor glucose control - immediate medical attention recommended',
                'recommendations': [
                    'Urgent consultation with healthcare provider',
                    'Review all insulin settings',
                    'Check for ketones if glucose consistently high',
                    'Consider diabetes education program'
                ]
            }
    
    def _fallback_prediction(self, patient_data: Dict, patient_profile: Dict, error: str = None) -> Dict[str, Any]:
        """Fallback rule-based prediction"""
        glucose = patient_data.get('glucose_level', 120)
        food = patient_data.get('food_intake', 0)
        carb_ratio = patient_profile.get('carb_ratio', 15)
        isf = patient_profile.get('insulin_sensitivity_factor', 50)
        
        dose = 0.0
        explanation_parts = []
        
        # Meal bolus
        if food > 0:
            meal_dose = food / carb_ratio
            dose += meal_dose
            explanation_parts.append(f"Meal bolus: {food}g ÷ {carb_ratio} = {meal_dose:.2f} units")
        
        # Correction bolus (only if glucose > target)
        target_glucose = 100
        if glucose > target_glucose:
            correction = (glucose - target_glucose) / isf
            dose += correction
            explanation_parts.append(f"Correction: ({glucose} - {target_glucose}) ÷ {isf} = {correction:.2f} units")
        
        # Safety: no dose if hypoglycemic
        if glucose < 70:
            dose = 0
            explanation_parts = ["No insulin recommended - treat hypoglycemia first"]
        
        explanation = ". ".join(explanation_parts) if explanation_parts else "Rule-based calculation"
        
        if error:
            explanation += f" (Model error: {error})"
        
        return {
            'recommended_dose': round(max(0, dose), 2),
            'confidence': 0.3,
            'prediction_type': 'rule_based_fallback',
            'explanation': explanation,
            'safety_flags': ['low_confidence', 'rule_based'] if error else ['low_confidence'],
            'feature_importance': {},
            'model_info': {
                'model_name': 'Rule-Based Fallback',
                'model_version': '1.0.0',
                'model_id': 'fallback'
            },
            'error': error
        }

# Global ML service instance
ml_service = MLService()
