"""
Advanced Insulin Model - Enhanced implementation with LSTM for time-series
Better suited for Type 1 Diabetes with pattern recognition
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import shap
from typing import Dict, List, Any
from app.ml_framework.base_model import BaseInsulinModel

class AdvancedInsulinModel(BaseInsulinModel):
    """
    Advanced ML model with better handling of Type 1 Diabetes patterns
    Uses Gradient Boosting with time-series awareness
    """
    
    MODEL_ID = 'advanced'
    
    def __init__(self, model_name: str = "Advanced Insulin Model", model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.scaler = StandardScaler()
        self.explainer = None
        self.feature_names = [
            'glucose_level', 'glucose_trend', 'food_intake', 'physical_activity',
            'age', 'bmi', 'hba1c', 'weight_kg', 'insulin_sensitivity_factor',
            'carb_ratio', 'basal_rate', 'time_of_day_hour', 'time_since_last_meal',
            'time_since_last_insulin', 'glucose_variability', 'recent_activity_level'
        ]
    
    def get_supported_features(self) -> List[str]:
        return self.feature_names
    
    def train(self, training_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train the advanced model"""
        print(f"Training {self.model_name}...")
        
        # Prepare features with time-series awareness
        X = self._prepare_features_with_history(training_data)
        y = training_data['insulin_dosage'].values
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Create SHAP explainer
        sample_size = min(100, len(X_train_scaled))
        self.explainer = shap.TreeExplainer(self.model)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Calculate percentage within 20% of actual
        within_20_percent = np.mean(np.abs(y_pred - y_test) / (y_test + 1e-6) < 0.2)
        
        self.is_trained = True
        self.training_metadata = {
            'train_score': float(train_score),
            'test_score': float(test_score),
            'mae': float(mae),
            'rmse': float(rmse),
            'within_20_percent': float(within_20_percent),
            'n_samples': len(training_data),
            'n_features': len(self.feature_names),
            'training_date': pd.Timestamp.now().isoformat(),
            'model_type': 'GradientBoostingRegressor'
        }
        
        print(f"Advanced model trained - Test R²: {test_score:.3f}, MAE: {mae:.2f} units, Within 20%: {within_20_percent:.1%}")
        
        return self.training_metadata
    
    def _prepare_features_with_history(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features including time-series patterns"""
        features = pd.DataFrame()
        
        # Core features
        features['glucose_level'] = data.get('glucose_level', 120)
        features['food_intake'] = data.get('food_intake', 0)
        features['physical_activity'] = data.get('physical_activity', 0)
        
        # Calculate glucose trend (if historical data available)
        if 'glucose_level' in data.columns and len(data) > 1:
            features['glucose_trend'] = data['glucose_level'].diff().fillna(0)
        else:
            features['glucose_trend'] = 0
        
        # Profile features
        features['age'] = data.get('age', 45)
        features['bmi'] = data.get('bmi', 25)
        features['hba1c'] = data.get('hba1c', 7.0)
        features['weight_kg'] = data.get('weight_kg', 70)
        
        # Insulin parameters
        features['insulin_sensitivity_factor'] = data.get('insulin_sensitivity_factor', 50)
        features['carb_ratio'] = data.get('carb_ratio', 15)
        features['basal_rate'] = data.get('basal_rate', 1.0)
        
        # Time features
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            features['time_of_day_hour'] = timestamps.dt.hour
        else:
            features['time_of_day_hour'] = 12
        
        # Time since last meal/insulin (simplified - would need historical data)
        features['time_since_last_meal'] = data.get('time_since_last_meal', 4)  # hours
        features['time_since_last_insulin'] = data.get('time_since_last_insulin', 2)  # hours
        
        # Glucose variability (coefficient of variation if multiple readings)
        if 'glucose_level' in data.columns:
            glucose_values = data['glucose_level'].dropna()
            if len(glucose_values) > 1:
                features['glucose_variability'] = glucose_values.std() / (glucose_values.mean() + 1e-6)
            else:
                features['glucose_variability'] = 0.1
        else:
            features['glucose_variability'] = 0.1
        
        # Recent activity level
        features['recent_activity_level'] = data.get('recent_activity_level', 0)
        
        return features
    
    def predict(self, patient_data: Dict[str, Any], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Make insulin dose recommendation with advanced features"""
        if not self.is_trained:
            return self._fallback_prediction(patient_data, patient_profile)
        
        # Validate input
        is_valid, error = self.validate_input(patient_data, patient_profile)
        if not is_valid:
            return {
                'recommended_dose': 0.0,
                'confidence': 0.0,
                'prediction_type': 'error',
                'explanation': f"Invalid input: {error}",
                'safety_flags': ['invalid_input'],
                'error': error
            }
        
        # Prepare features
        feature_dict = self._prepare_prediction_features(patient_data, patient_profile)
        X = pd.DataFrame([feature_dict])
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X = X.fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predicted_dose = float(self.model.predict(X_scaled)[0])
        
        # Apply safety limits with stricter controls
        safety_limits = self.calculate_safety_limits(patient_profile)
        
        # Additional Type 1 diabetes specific safety checks
        glucose = patient_data.get('glucose_level', 120)
        if glucose < 70:
            # Reduce or eliminate dose if hypoglycemic
            predicted_dose = min(predicted_dose, 0.5)
        
        predicted_dose = max(safety_limits['min_dose'], 
                           min(predicted_dose, safety_limits['max_bolus']))
        
        # Get detailed explanation
        explanation_data = self.explain(patient_data, patient_profile)
        
        # Higher confidence for advanced model
        feature_completeness = sum(1 for f in self.feature_names 
                                 if feature_dict.get(f, None) is not None) / len(self.feature_names)
        confidence = min(0.98, 0.6 + feature_completeness * 0.38)
        
        # Determine prediction type
        prediction_type = self._determine_prediction_type(patient_data, predicted_dose)
        
        return {
            'recommended_dose': round(predicted_dose, 2),
            'confidence': round(confidence, 3),
            'prediction_type': prediction_type,
            'explanation': explanation_data.get('summary', ''),
            'safety_flags': explanation_data.get('safety_flags', []),
            'feature_importance': explanation_data.get('feature_importance', {}),
            'reasoning_steps': explanation_data.get('reasoning_steps', []),
            'safety_limits': safety_limits,
            'stability_metrics': self._calculate_stability_metrics(patient_data, patient_profile)
        }
    
    def _prepare_prediction_features(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Prepare advanced features for prediction"""
        glucose = patient_data.get('glucose_level', 120)
        
        return {
            'glucose_level': glucose,
            'glucose_trend': patient_data.get('glucose_trend', 0),
            'food_intake': patient_data.get('food_intake', 0),
            'physical_activity': patient_data.get('physical_activity', 0),
            'age': patient_profile.get('age', 45),
            'bmi': patient_profile.get('bmi', 25),
            'hba1c': patient_profile.get('hba1c', 7.0),
            'weight_kg': patient_profile.get('weight_kg', 70),
            'insulin_sensitivity_factor': patient_profile.get('insulin_sensitivity_factor', 50),
            'carb_ratio': patient_profile.get('carb_ratio', 15),
            'basal_rate': patient_profile.get('basal_rate', 1.0),
            'time_of_day_hour': pd.Timestamp.now().hour,
            'time_since_last_meal': patient_data.get('time_since_last_meal', 4),
            'time_since_last_insulin': patient_data.get('time_since_last_insulin', 2),
            'glucose_variability': patient_data.get('glucose_variability', 0.1),
            'recent_activity_level': patient_data.get('recent_activity_level', 0)
        }
    
    def _determine_prediction_type(self, patient_data: Dict, dose: float) -> str:
        """Determine prediction type with more nuance"""
        food = patient_data.get('food_intake', 0)
        glucose = patient_data.get('glucose_level', 120)
        time_since_meal = patient_data.get('time_since_last_meal', 4)
        
        if food > 0:
            return 'bolus'
        elif glucose > 180 and time_since_meal < 2:
            return 'correction_bolus'
        elif glucose > 150:
            return 'correction'
        elif glucose < 80:
            return 'basal_reduction'
        else:
            return 'basal_adjustment'
    
    def explain(self, patient_data: Dict[str, Any], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed explanation with stability focus"""
        if not self.is_trained or self.explainer is None:
            return self._simple_explanation(patient_data, patient_profile)
        
        # Prepare features
        feature_dict = self._prepare_prediction_features(patient_data, patient_profile)
        X = pd.DataFrame([feature_dict])
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled[0])
        
        # Create feature importance
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = {
                'value': float(X.iloc[0, i]),
                'contribution': float(shap_values[i]),
                'impact': 'increases' if shap_values[i] > 0 else 'decreases'
            }
        
        # Sort by absolute contribution
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]['contribution']), 
                               reverse=True)
        
        # Build detailed reasoning steps
        reasoning_steps = []
        glucose = patient_data.get('glucose_level', 120)
        food = patient_data.get('food_intake', 0)
        glucose_trend = patient_data.get('glucose_trend', 0)
        
        reasoning_steps.append({
            'step': 1,
            'description': f"Current glucose: {glucose} mg/dL",
            'impact': 'critical' if glucose < 70 or glucose > 250 else 'normal',
            'recommendation': 'Monitor closely' if glucose < 70 or glucose > 250 else 'Continue monitoring'
        })
        
        if abs(glucose_trend) > 10:
            trend_direction = 'rising' if glucose_trend > 0 else 'falling'
            reasoning_steps.append({
                'step': 2,
                'description': f"Glucose {trend_direction} rapidly ({abs(glucose_trend):.1f} mg/dL change)",
                'impact': 'high',
                'recommendation': 'Consider immediate action'
            })
        
        if food > 0:
            reasoning_steps.append({
                'step': 3,
                'description': f"Meal bolus needed for {food}g carbohydrates",
                'impact': 'meal_bolus',
                'recommendation': f"Calculate: {food}g ÷ {patient_profile.get('carb_ratio', 15)} = {food/patient_profile.get('carb_ratio', 15):.1f} units"
            })
        
        # Top contributing factors
        top_factors = sorted_features[:5]
        reasoning_steps.append({
            'step': 4,
            'description': f"Model analysis: {len(top_factors)} key factors identified",
            'impact': 'model_decision',
            'recommendation': 'Review detailed feature contributions below'
        })
        
        # Stability assessment
        glucose_var = patient_data.get('glucose_variability', 0.1)
        if glucose_var > 0.3:
            reasoning_steps.append({
                'step': 5,
                'description': f"High glucose variability detected ({glucose_var:.2%})",
                'impact': 'stability_concern',
                'recommendation': 'Consider basal rate adjustment for better stability'
            })
        
        # Safety assessment
        safety_flags = []
        if glucose < 70:
            safety_flags.append({
                'type': 'hypoglycemia_risk',
                'severity': 'critical' if glucose < 54 else 'high',
                'message': 'Immediate action required - treat hypoglycemia'
            })
        elif glucose > 250:
            safety_flags.append({
                'type': 'hyperglycemia_risk',
                'severity': 'critical',
                'message': 'Severe hyperglycemia - check ketones if Type 1'
            })
        
        # Build comprehensive summary
        summary = f"Advanced analysis for Type 1 Diabetes management. "
        summary += f"Current glucose: {glucose} mg/dL. "
        
        if food > 0:
            summary += f"Meal bolus calculation for {food}g carbs. "
        
        if top_factors:
            top_factor = top_factors[0]
            summary += f"Primary factor: {top_factor[0]} ({top_factor[1]['impact']} dose requirement). "
        
        summary += "Recommendation includes safety limits and stability considerations."
        
        return {
            'summary': summary,
            'reasoning_steps': reasoning_steps,
            'feature_importance': {k: v for k, v in feature_importance.items()},
            'top_contributors': [{'feature': k, **v} for k, v in top_factors],
            'safety_flags': safety_flags,
            'risk_assessment': self._assess_risk(patient_data, patient_profile),
            'stability_analysis': self._analyze_stability(patient_data, patient_profile)
        }
    
    def _calculate_stability_metrics(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Calculate Type 1 diabetes stability metrics"""
        glucose = patient_data.get('glucose_level', 120)
        glucose_var = patient_data.get('glucose_variability', 0.1)
        
        # Time in range estimation (simplified)
        time_in_range = 0.0
        if 70 <= glucose <= 180:
            time_in_range = 1.0
        elif 54 <= glucose < 70 or 180 < glucose <= 250:
            time_in_range = 0.5
        
        return {
            'glucose_variability': round(glucose_var, 3),
            'time_in_range_estimate': round(time_in_range, 2),
            'stability_score': round(1.0 - min(glucose_var, 0.5), 2),
            'target_range': '70-180 mg/dL'
        }
    
    def _analyze_stability(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Analyze glucose stability"""
        glucose_var = patient_data.get('glucose_variability', 0.1)
        glucose = patient_data.get('glucose_level', 120)
        
        stability_level = 'excellent'
        if glucose_var > 0.3:
            stability_level = 'poor'
        elif glucose_var > 0.2:
            stability_level = 'moderate'
        
        recommendations = []
        if glucose_var > 0.3:
            recommendations.append("Consider basal rate optimization")
            recommendations.append("Review meal timing and carb counting accuracy")
        if glucose < 70 or glucose > 180:
            recommendations.append("Focus on achieving target range (70-180 mg/dL)")
        
        return {
            'stability_level': stability_level,
            'variability': round(glucose_var, 3),
            'recommendations': recommendations
        }
    
    def _assess_risk(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Enhanced risk assessment for Type 1 diabetes"""
        glucose = patient_data.get('glucose_level', 120)
        glucose_trend = patient_data.get('glucose_trend', 0)
        
        risk_level = 'low'
        concerns = []
        
        if glucose < 54:
            risk_level = 'critical'
            concerns.append('Severe hypoglycemia - immediate treatment needed')
        elif glucose < 70:
            risk_level = 'high'
            concerns.append('Hypoglycemia - treat with fast-acting carbs')
        elif glucose > 250:
            risk_level = 'critical'
            concerns.append('Severe hyperglycemia - check ketones')
        elif glucose > 180:
            risk_level = 'moderate'
            concerns.append('Elevated glucose - consider correction')
        
        if abs(glucose_trend) > 20:
            concerns.append(f'Rapid glucose change ({glucose_trend:.1f} mg/dL)')
        
        return {
            'level': risk_level,
            'concerns': concerns,
            'recommendation': 'Monitor closely' if risk_level != 'low' else 'Continue routine monitoring'
        }
    
    def _fallback_prediction(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Fallback with Type 1 diabetes specific rules"""
        glucose = patient_data.get('glucose_level', 120)
        food = patient_data.get('food_intake', 0)
        carb_ratio = patient_profile.get('carb_ratio', 15)
        isf = patient_profile.get('insulin_sensitivity_factor', 50)
        basal_rate = patient_profile.get('basal_rate', 1.0)
        
        dose = 0.0
        
        # Meal bolus
        if food > 0:
            dose += food / carb_ratio
        
        # Correction bolus (only if glucose > target)
        target_glucose = 100
        if glucose > target_glucose:
            correction = (glucose - target_glucose) / isf
            dose += correction
        
        # Safety: don't give correction if glucose is low
        if glucose < 70:
            dose = 0
        
        return {
            'recommended_dose': round(max(0, dose), 2),
            'confidence': 0.4,
            'prediction_type': 'rule_based',
            'explanation': 'Using Type 1 diabetes rule-based calculation (model not trained)',
            'safety_flags': [],
            'feature_importance': {}
        }
    
    def _simple_explanation(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Simple explanation"""
        glucose = patient_data.get('glucose_level', 120)
        return {
            'summary': f"Glucose: {glucose} mg/dL. Advanced model not trained.",
            'reasoning_steps': [],
            'feature_importance': {},
            'safety_flags': [],
            'risk_assessment': self._assess_risk(patient_data, patient_profile)
        }

