"""
Default Insulin Model - Baseline implementation
Suitable for Type 1 Diabetes management
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import shap
from typing import Dict, List, Any
from app.ml_framework.base_model import BaseInsulinModel

class DefaultInsulinModel(BaseInsulinModel):
    """
    Default ML model for insulin dose recommendation
    Uses Random Forest for dose prediction
    """
    
    MODEL_ID = 'default'
    
    def __init__(self, model_name: str = "Default Insulin Model", model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.scaler = StandardScaler()
        self.explainer = None
        self.feature_names = [
            'glucose_level', 'food_intake', 'physical_activity',
            'age', 'bmi', 'hba1c', 'weight_kg', 'insulin_sensitivity_factor',
            'carb_ratio', 'basal_rate', 'time_of_day_hour'
        ]
    
    def get_supported_features(self) -> List[str]:
        return self.feature_names
    
    def train(self, training_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        print(f"Training {self.model_name}...")
        
        # Prepare features
        X = self._prepare_features(training_data)
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
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Create SHAP explainer
        sample_size = min(100, len(X_train_scaled))
        sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Calculate MAE
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.is_trained = True
        self.training_metadata = {
            'train_score': float(train_score),
            'test_score': float(test_score),
            'mae': float(mae),
            'rmse': float(rmse),
            'n_samples': len(training_data),
            'n_features': len(self.feature_names),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        print(f"Model trained - Test R²: {test_score:.3f}, MAE: {mae:.2f} units")
        
        return self.training_metadata
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix"""
        features = pd.DataFrame()
        
        # Core features
        features['glucose_level'] = data.get('glucose_level', 120)
        features['food_intake'] = data.get('food_intake', 0)
        features['physical_activity'] = data.get('physical_activity', 0)
        
        # Profile features (use defaults if missing)
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
            features['time_of_day_hour'] = 12  # Default to noon
        
        return features
    
    def predict(self, patient_data: Dict[str, Any], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Make insulin dose recommendation"""
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
        
        # Apply safety limits
        safety_limits = self.calculate_safety_limits(patient_profile)
        predicted_dose = max(safety_limits['min_dose'], 
                           min(predicted_dose, safety_limits['max_bolus']))
        
        # Get explanation
        explanation_data = self.explain(patient_data, patient_profile)
        
        # Calculate confidence based on feature completeness
        feature_completeness = sum(1 for f in self.feature_names 
                                 if feature_dict.get(f, None) is not None) / len(self.feature_names)
        confidence = min(0.95, 0.5 + feature_completeness * 0.45)
        
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
            'safety_limits': safety_limits
        }
    
    def _prepare_prediction_features(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Prepare features for prediction"""
        return {
            'glucose_level': patient_data.get('glucose_level', 120),
            'food_intake': patient_data.get('food_intake', 0),
            'physical_activity': patient_data.get('physical_activity', 0),
            'age': patient_profile.get('age', 45),
            'bmi': patient_profile.get('bmi', 25),
            'hba1c': patient_profile.get('hba1c', 7.0),
            'weight_kg': patient_profile.get('weight_kg', 70),
            'insulin_sensitivity_factor': patient_profile.get('insulin_sensitivity_factor', 50),
            'carb_ratio': patient_profile.get('carb_ratio', 15),
            'basal_rate': patient_profile.get('basal_rate', 1.0),
            'time_of_day_hour': pd.Timestamp.now().hour
        }
    
    def _determine_prediction_type(self, patient_data: Dict, dose: float) -> str:
        """Determine if this is a basal, bolus, or correction dose"""
        food = patient_data.get('food_intake', 0)
        glucose = patient_data.get('glucose_level', 120)
        
        if food > 0:
            return 'bolus'  # Meal bolus
        elif glucose > 180:
            return 'correction'  # Correction bolus
        else:
            return 'basal'  # Basal adjustment
    
    def explain(self, patient_data: Dict[str, Any], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed explanation"""
        if not self.is_trained or self.explainer is None:
            return self._simple_explanation(patient_data, patient_profile)
        
        # Prepare features
        feature_dict = self._prepare_prediction_features(patient_data, patient_profile)
        X = pd.DataFrame([feature_dict])
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled[0])
        
        # Create feature importance dictionary
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
        
        # Build reasoning steps
        reasoning_steps = []
        glucose = patient_data.get('glucose_level', 120)
        food = patient_data.get('food_intake', 0)
        
        reasoning_steps.append({
            'step': 1,
            'description': f"Current glucose level: {glucose} mg/dL",
            'impact': 'high' if glucose > 180 or glucose < 70 else 'normal'
        })
        
        if food > 0:
            reasoning_steps.append({
                'step': 2,
                'description': f"Meal detected: {food}g carbohydrates",
                'impact': 'meal_bolus_required'
            })
        
        # Top contributing factors
        top_factors = sorted_features[:3]
        reasoning_steps.append({
            'step': 3,
            'description': f"Primary factors: {', '.join([f[0] for f in top_factors])}",
            'impact': 'model_decision'
        })
        
        # Safety assessment
        safety_flags = []
        if glucose < 70:
            safety_flags.append('hypoglycemia_risk')
        elif glucose > 250:
            safety_flags.append('hyperglycemia_risk')
        
        # Build summary
        summary = f"Recommended dose based on glucose level ({glucose} mg/dL)"
        if food > 0:
            summary += f" and meal ({food}g carbs)"
        summary += ". "
        
        if top_factors:
            top_factor = top_factors[0]
            summary += f"Primary factor: {top_factor[0]} ({top_factor[1]['impact']} dose)."
        
        return {
            'summary': summary,
            'reasoning_steps': reasoning_steps,
            'feature_importance': {k: v for k, v in feature_importance.items()},
            'top_contributors': [{'feature': k, **v} for k, v in top_factors],
            'safety_flags': safety_flags,
            'risk_assessment': self._assess_risk(patient_data, patient_profile)
        }
    
    def _assess_risk(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Assess risk level"""
        glucose = patient_data.get('glucose_level', 120)
        
        if glucose < 54:
            return {'level': 'critical', 'message': 'Severe hypoglycemia risk'}
        elif glucose < 70:
            return {'level': 'high', 'message': 'Hypoglycemia risk'}
        elif glucose > 250:
            return {'level': 'critical', 'message': 'Severe hyperglycemia risk'}
        elif glucose > 180:
            return {'level': 'moderate', 'message': 'Elevated glucose'}
        else:
            return {'level': 'low', 'message': 'Glucose in target range'}
    
    def _fallback_prediction(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Fallback prediction when model not trained"""
        glucose = patient_data.get('glucose_level', 120)
        food = patient_data.get('food_intake', 0)
        carb_ratio = patient_profile.get('carb_ratio', 15)
        isf = patient_profile.get('insulin_sensitivity_factor', 50)
        
        # Simple rule-based calculation
        dose = 0.0
        if food > 0:
            dose += food / carb_ratio
        
        if glucose > 150:
            correction = (glucose - 150) / isf
            dose += correction
        
        return {
            'recommended_dose': round(max(0, dose), 2),
            'confidence': 0.3,
            'prediction_type': 'rule_based',
            'explanation': 'Using rule-based calculation (model not trained)',
            'safety_flags': [],
            'feature_importance': {}
        }
    
    def _simple_explanation(self, patient_data: Dict, patient_profile: Dict) -> Dict:
        """Simple explanation when model not available"""
        glucose = patient_data.get('glucose_level', 120)
        return {
            'summary': f"Glucose level: {glucose} mg/dL. Model not trained.",
            'reasoning_steps': [],
            'feature_importance': {},
            'safety_flags': [],
            'risk_assessment': self._assess_risk(patient_data, patient_profile)
        }

