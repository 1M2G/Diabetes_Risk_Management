"""
Base ML Model Interface for Insulin Management System
All custom models must inherit from this base class
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class BaseInsulinModel(ABC):
    """
    Abstract base class for all insulin management ML models.
    
    Developers must implement:
    - train(): Train the model
    - predict(): Make predictions
    - explain(): Provide explanations
    - get_model_info(): Return model metadata
    """
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize the model
        
        Args:
            model_name: Unique name for the model
            model_version: Version of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.is_trained = False
        self.training_metadata = {}
        
    @abstractmethod
    def train(self, training_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data
        
        Args:
            training_data: DataFrame with columns:
                - Required: glucose_level, insulin_dosage
                - Optional: age, bmi, hba1c, food_intake, physical_activity, etc.
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary with training metrics (accuracy, loss, etc.)
        """
        pass
    
    @abstractmethod
    def predict(self, patient_data: Dict[str, Any], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make insulin dose recommendation
        
        Args:
            patient_data: Current patient data (glucose, food, activity, etc.)
            patient_profile: Patient profile (age, bmi, hba1c, diabetes_type, etc.)
        
        Returns:
            Dictionary with:
                - recommended_dose: Recommended insulin dose in units
                - confidence: Confidence score (0-1)
                - prediction_type: Type of prediction (basal, bolus, correction)
                - explanation: Human-readable explanation
                - safety_flags: List of safety concerns
                - feature_importance: Feature contributions
        """
        pass
    
    @abstractmethod
    def explain(self, patient_data: Dict[str, Any], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide detailed explanation of the prediction
        
        Args:
            patient_data: Current patient data
            patient_profile: Patient profile
        
        Returns:
            Dictionary with detailed explanation including:
                - reasoning_steps: Step-by-step reasoning
                - contributing_factors: Factors that influenced the decision
                - risk_assessment: Risk level and concerns
                - alternative_scenarios: What-if scenarios
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and information
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'model_type': self.__class__.__name__,
            'target_population': 'Type 1 Diabetes',
            'supported_features': self.get_supported_features()
        }
    
    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """
        Return list of features this model supports
        
        Returns:
            List of feature names
        """
        pass
    
    def validate_input(self, patient_data: Dict, patient_profile: Dict):
        """
        Validate input data
        
        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        required_fields = ['glucose_level']
        for field in required_fields:
            if field not in patient_data or patient_data[field] is None:
                return False, f"Missing required field: {field}"
        
        # Validate glucose level range
        glucose = patient_data.get('glucose_level')
        if glucose < 20 or glucose > 600:
            return False, f"Glucose level out of valid range (20-600 mg/dL): {glucose}"
        
        # Check diabetes type
        if patient_profile.get('diabetes_type') != 'Type 1':
            return False, "This model is designed for Type 1 diabetes only"
        
        return True, None
    
    def calculate_safety_limits(self, patient_profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate safety limits for insulin dosing
        
        Returns:
            Dictionary with min_dose, max_dose, max_bolus, etc.
        """
        # Base safety limits (can be customized per model)
        weight = patient_profile.get('weight_kg', 70)  # Default 70kg
        tdd = patient_profile.get('total_daily_dose', 50)  # Total daily dose
        
        return {
            'min_dose': 0.0,
            'max_bolus': min(tdd * 0.3, weight * 0.1),  # Max 30% of TDD or 0.1 units/kg
            'max_basal_per_hour': tdd * 0.5 / 24,  # Max 50% of TDD as basal
            'max_correction': min(tdd * 0.2, 10),  # Max correction dose
            'total_daily_dose': tdd
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        import joblib
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'training_metadata': self.training_metadata,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data.get('model_name', self.model_name)
        self.model_version = model_data.get('model_version', self.model_version)
        self.training_metadata = model_data.get('training_metadata', {})
        self.is_trained = model_data.get('is_trained', False)

