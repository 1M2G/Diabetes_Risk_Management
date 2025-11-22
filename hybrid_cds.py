"""
Hybrid Clinical Decision Support System
Combines ML predictions, rule-based overrides, and explainability
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

from .rule_engine import ClinicalRuleEngine, RuleResult, RiskLevel
from .explainability import ModelExplainer
from .ml_models import LightweightMLModel


@dataclass
class CDSOutput:
    """Output from the Clinical Decision Support System"""
    risk_score: float  # 0-1 scale
    risk_level: str  # low, moderate, high, critical
    ml_prediction: float  # Raw ML prediction
    rule_override: bool  # Whether rule overrode ML
    rule_used: Optional[str] = None
    explanation: str = ""  # Why patient is at risk
    recommendation: str = ""  # What should be done
    contributing_factors: List[Dict] = None  # Top contributing factors
    confidence: float = 1.0  # Confidence in the assessment
    ml_confidence: float = 0.0  # ML model confidence
    rule_confidence: float = 0.0  # Rule confidence
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = asdict(self)
        if self.contributing_factors is None:
            result['contributing_factors'] = []
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class HybridCDS:
    """
    Hybrid Clinical Decision Support System
    Combines ML predictions with rule-based overrides and explainability
    """
    
    def __init__(self, ml_model: Optional[LightweightMLModel] = None,
                 rule_engine: Optional[ClinicalRuleEngine] = None,
                 explainer: Optional[ModelExplainer] = None,
                 rule_override_threshold: float = 0.7,
                 ml_weight: float = 0.6,
                 rule_weight: float = 0.4):
        """
        Initialize Hybrid CDS System
        
        Args:
            ml_model: Trained ML model
            rule_engine: Clinical rule engine
            explainer: Model explainer (SHAP/LIME)
            rule_override_threshold: Risk score threshold for rule override
            ml_weight: Weight for ML prediction in hybrid score
            rule_weight: Weight for rule-based score in hybrid score
        """
        self.ml_model = ml_model or LightweightMLModel()
        self.rule_engine = rule_engine or ClinicalRuleEngine()
        self.explainer = explainer
        self.rule_override_threshold = rule_override_threshold
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        
        # Normalize weights
        total_weight = ml_weight + rule_weight
        if total_weight > 0:
            self.ml_weight = ml_weight / total_weight
            self.rule_weight = rule_weight / total_weight
    
    def predict(self, patient_data: pd.DataFrame, 
                return_explanation: bool = True) -> CDSOutput:
        """
        Generate comprehensive CDS prediction for a patient
        
        Args:
            patient_data: Patient features as DataFrame (single row)
            return_explanation: Whether to generate explanations
            
        Returns:
            CDSOutput with risk score, explanation, and recommendation
        """
        if len(patient_data) != 1:
            raise ValueError("patient_data must contain exactly one row")
        
        # Convert to dict for rule engine
        patient_dict = patient_data.iloc[0].to_dict()
        
        # 1. ML Prediction
        ml_prediction = self._get_ml_prediction(patient_data)
        ml_risk_score = self._normalize_ml_prediction(ml_prediction)
        
        # 2. Rule-based Assessment
        rule_result = self.rule_engine.get_highest_risk_override(patient_dict)
        rule_override = rule_result is not None
        
        # 3. Determine final risk score
        if rule_override and rule_result.override_score is not None:
            # Rule overrides ML if threshold exceeded
            if rule_result.override_score >= self.rule_override_threshold:
                final_risk_score = rule_result.override_score
                risk_level = rule_result.risk_level.value
                explanation = rule_result.explanation
                recommendation = rule_result.recommendation
                rule_used = rule_result.rule_name
                confidence = rule_result.confidence
                rule_confidence = rule_result.confidence
            else:
                # Hybrid score: weighted combination
                final_risk_score = (
                    self.ml_weight * ml_risk_score +
                    self.rule_weight * rule_result.override_score
                )
                risk_level = self._determine_risk_level(final_risk_score)
                explanation = self._combine_explanations(ml_prediction, rule_result)
                recommendation = rule_result.recommendation or self._generate_ml_recommendation(ml_risk_score)
                rule_used = rule_result.rule_name
                confidence = (self.ml_weight * 0.7 + self.rule_weight * rule_result.confidence)
                rule_confidence = rule_result.confidence
        else:
            # Pure ML prediction
            final_risk_score = ml_risk_score
            risk_level = self._determine_risk_level(final_risk_score)
            explanation = self._generate_ml_explanation(ml_risk_score)
            recommendation = self._generate_ml_recommendation(ml_risk_score)
            rule_used = None
            confidence = 0.7  # Default ML confidence
            rule_confidence = 0.0
        
        # 4. Generate explainability insights
        contributing_factors = []
        if return_explanation and self.explainer is not None:
            try:
                explanation_data = self.explainer.explain_combined(patient_data, max_features=10)
                contributing_factors = self._extract_contributing_factors(
                    explanation_data, patient_data
                )
                
                # Enhance explanation with ML insights
                if contributing_factors:
                    ml_explanation_parts = [
                        f"{factor['feature']} ({factor['value']:.2f}) "
                        f"{'increases' if factor['impact'] > 0 else 'decreases'} risk"
                        for factor in contributing_factors[:3]
                    ]
                    if ml_explanation_parts:
                        explanation += "\n\nML Analysis: " + "; ".join(ml_explanation_parts)
            except Exception as e:
                print(f"Warning: Could not generate explanation: {e}")
        
        return CDSOutput(
            risk_score=final_risk_score,
            risk_level=risk_level,
            ml_prediction=float(ml_prediction),
            rule_override=rule_override,
            rule_used=rule_used,
            explanation=explanation,
            recommendation=recommendation,
            contributing_factors=contributing_factors,
            confidence=confidence,
            ml_confidence=0.7,
            rule_confidence=rule_confidence
        )
    
    def _get_ml_prediction(self, patient_data: pd.DataFrame) -> float:
        """Get ML model prediction"""
        if self.ml_model is None or self.ml_model.model is None:
            # Default prediction if no model
            return 0.5
        return self.ml_model.predict(patient_data)[0]
    
    def _normalize_ml_prediction(self, prediction: float) -> float:
        """
        Normalize ML prediction to 0-1 risk score
        This assumes predictions are in some range - adjust based on your target
        """
        # Assuming insulin dose predictions - convert to risk score
        # Higher insulin dose = higher risk
        # Normalize to 0-1 range (adjust based on your data distribution)
        normalized = np.clip((prediction - 0) / 100, 0, 1)  # Adjust denominator as needed
        return float(normalized)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "moderate"
        else:
            return "low"
    
    def _combine_explanations(self, ml_prediction: float, rule_result: RuleResult) -> str:
        """Combine ML and rule explanations"""
        parts = [rule_result.explanation]
        
        ml_part = f"\nML model also indicates elevated risk (prediction: {ml_prediction:.2f})."
        parts.append(ml_part)
        
        return " ".join(parts)
    
    def _generate_ml_explanation(self, risk_score: float) -> str:
        """Generate explanation from ML prediction"""
        if risk_score >= 0.8:
            return f"ML model indicates critical risk (score: {risk_score:.2f}). Multiple risk factors detected."
        elif risk_score >= 0.6:
            return f"ML model indicates high risk (score: {risk_score:.2f}). Several concerning factors present."
        elif risk_score >= 0.4:
            return f"ML model indicates moderate risk (score: {risk_score:.2f}). Some risk factors present."
        else:
            return f"ML model indicates low risk (score: {risk_score:.2f}). Patient appears stable."
    
    def _generate_ml_recommendation(self, risk_score: float) -> str:
        """Generate recommendation from ML prediction"""
        if risk_score >= 0.8:
            return "Immediate clinical review recommended. Consider urgent intervention and close monitoring."
        elif risk_score >= 0.6:
            return "Clinical review recommended within 24 hours. Monitor closely and consider intervention."
        elif risk_score >= 0.4:
            return "Routine follow-up recommended. Continue monitoring and consider preventive measures."
        else:
            return "Continue current management plan. Maintain regular monitoring."
    
    def _extract_contributing_factors(self, explanation_data: Dict, 
                                     patient_data: pd.DataFrame) -> List[Dict]:
        """Extract top contributing factors from explanation"""
        factors = []
        
        # Try combined features first
        if explanation_data.get('top_combined_features'):
            for feat in explanation_data['top_combined_features'][:5]:
                combined_info = explanation_data['combined_features'].get(feat, {})
                if feat in patient_data.columns:
                    factors.append({
                        'feature': feat,
                        'value': float(patient_data[feat].values[0]),
                        'impact': float(combined_info.get('combined_importance', 0)),
                        'method': 'combined'
                    })
        
        # Fallback to SHAP
        elif explanation_data.get('shap', {}).get('available'):
            shap_exp = explanation_data['shap']
            for feat in shap_exp.get('top_features', [])[:5]:
                if feat in patient_data.columns:
                    factors.append({
                        'feature': feat,
                        'value': float(patient_data[feat].values[0]),
                        'impact': float(shap_exp.get('feature_importance', {}).get(feat, 0)),
                        'method': 'shap'
                    })
        
        # Fallback to LIME
        elif explanation_data.get('lime', {}).get('available'):
            lime_exp = explanation_data['lime']
            for feat in lime_exp.get('top_features', [])[:5]:
                if feat in patient_data.columns:
                    factors.append({
                        'feature': feat,
                        'value': float(patient_data[feat].values[0]),
                        'impact': float(lime_exp.get('feature_importance', {}).get(feat, 0)),
                        'method': 'lime'
                    })
        
        # Sort by absolute impact
        factors.sort(key=lambda x: abs(x['impact']), reverse=True)
        return factors
    
    def batch_predict(self, patient_data: pd.DataFrame,
                     return_explanations: bool = True) -> List[CDSOutput]:
        """
        Generate predictions for multiple patients
        
        Args:
            patient_data: DataFrame with multiple patient rows
            return_explanations: Whether to generate explanations
            
        Returns:
            List of CDSOutput objects
        """
        results = []
        for idx in range(len(patient_data)):
            patient_row = patient_data.iloc[[idx]]
            result = self.predict(patient_row, return_explanation=return_explanations)
            results.append(result)
        return results
    
    def train_ml_model(self, X: pd.DataFrame, y: pd.Series,
                      training_data_for_explainer: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the ML model component
        
        Args:
            X: Feature matrix
            y: Target vector
            training_data_for_explainer: Training data for explainer initialization
            
        Returns:
            Training metrics
        """
        metrics = self.ml_model.fit(X, y)
        
        # Initialize explainer if training data provided
        if training_data_for_explainer is not None and self.explainer is None:
            model_type = 'tree' if 'Tree' in metrics.get('model_type', '') or 'LGBM' in metrics.get('model_type', '') else 'generic'
            self.explainer = ModelExplainer(
                self.ml_model.model,
                training_data_for_explainer,
                list(X.columns),
                model_type=model_type
            )
        
        return metrics

