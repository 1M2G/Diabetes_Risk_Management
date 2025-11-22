"""
Explainability Module for Clinical Decision Support
Integrates SHAP and LIME for model interpretability
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")


class ModelExplainer:
    """
    Unified explainability interface for ML models
    Supports both SHAP and LIME explanations
    """
    
    def __init__(self, model: Any, training_data: pd.DataFrame, 
                 feature_names: List[str], model_type: str = 'tree'):
        """
        Initialize explainer
        
        Args:
            model: Trained ML model
            training_data: Training data for background (SHAP) or sampling (LIME)
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', 'neural', 'generic')
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.model_type = model_type
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        # SHAP explainer
        if SHAP_AVAILABLE:
            try:
                if self.model_type == 'tree':
                    # TreeExplainer for tree-based models (XGBoost, LightGBM, RandomForest)
                    self.shap_explainer = shap.TreeExplainer(self.model)
                elif self.model_type == 'linear':
                    # LinearExplainer for linear models
                    self.shap_explainer = shap.LinearExplainer(self.model, self.training_data)
                else:
                    # KernelExplainer as fallback (slower but works for any model)
                    background = shap.sample(self.training_data, min(100, len(self.training_data)))
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict, 
                        background
                    )
            except Exception as e:
                print(f"Warning: Could not initialize SHAP explainer: {e}")
                self.shap_explainer = None
        
        # LIME explainer
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = lime_tabular.LimeTabularExplainer(
                    self.training_data.values,
                    feature_names=self.feature_names,
                    mode='regression' if hasattr(self.model, 'predict') else 'classification',
                    discretize_continuous=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize LIME explainer: {e}")
                self.lime_explainer = None
    
    def explain_shap(self, instance: pd.DataFrame, max_features: int = 10) -> Dict:
        """
        Generate SHAP explanation for an instance
        
        Args:
            instance: Single instance to explain (DataFrame with one row)
            max_features: Maximum number of top features to return
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {
                'available': False,
                'error': 'SHAP not available or explainer not initialized'
            }
        
        try:
            # Get SHAP values
            if self.model_type == 'tree':
                shap_values = self.shap_explainer.shap_values(instance)
            else:
                shap_values = self.shap_explainer.shap_values(instance.values[0])
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature importance
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(shap_values):
                    feature_importance[feature] = float(shap_values[i])
            
            # Sort by absolute importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:max_features]
            
            # Calculate base value (expected value)
            base_value = float(self.shap_explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
            
            # Get prediction
            prediction = self.model.predict(instance)[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0]
            
            return {
                'available': True,
                'base_value': base_value,
                'prediction': float(prediction),
                'feature_importance': dict(sorted_features),
                'top_features': [f[0] for f in sorted_features],
                'top_values': [f[1] for f in sorted_features],
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def explain_lime(self, instance: pd.DataFrame, num_features: int = 10) -> Dict:
        """
        Generate LIME explanation for an instance
        
        Args:
            instance: Single instance to explain (DataFrame with one row)
            num_features: Number of top features to return
            
        Returns:
            Dictionary with LIME explanation
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {
                'available': False,
                'error': 'LIME not available or explainer not initialized'
            }
        
        try:
            # Get LIME explanation
            explanation = self.lime_explainer.explain_instance(
                instance.values[0],
                self.model.predict,
                num_features=num_features
            )
            
            # Extract feature importance
            feature_importance = {}
            for feature, importance in explanation.as_list():
                feature_importance[feature] = importance
            
            # Get prediction
            prediction = self.model.predict(instance)[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0]
            
            return {
                'available': True,
                'prediction': float(prediction),
                'feature_importance': feature_importance,
                'top_features': [f[0] for f in explanation.as_list()],
                'top_values': [f[1] for f in explanation.as_list()],
                'explanation_text': explanation.as_list()
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def explain_combined(self, instance: pd.DataFrame, max_features: int = 10) -> Dict:
        """
        Generate combined SHAP and LIME explanations
        
        Args:
            instance: Single instance to explain
            max_features: Maximum number of features to include
            
        Returns:
            Dictionary with combined explanations
        """
        shap_explanation = self.explain_shap(instance, max_features)
        lime_explanation = self.explain_lime(instance, max_features)
        
        # Combine explanations
        combined = {
            'shap': shap_explanation,
            'lime': lime_explanation,
            'combined_features': {}
        }
        
        # Merge feature importance from both methods
        if shap_explanation.get('available') and lime_explanation.get('available'):
            # Get features from both
            shap_features = set(shap_explanation.get('top_features', []))
            lime_features = set(lime_explanation.get('top_features', []))
            all_features = shap_features.union(lime_features)
            
            # Average importance scores (normalized)
            for feature in all_features:
                shap_val = abs(shap_explanation.get('feature_importance', {}).get(feature, 0))
                lime_val = abs(lime_explanation.get('feature_importance', {}).get(feature, 0))
                
                # Normalize and average
                combined_score = (shap_val + lime_val) / 2 if (shap_val > 0 or lime_val > 0) else 0
                combined['combined_features'][feature] = {
                    'shap_importance': shap_explanation.get('feature_importance', {}).get(feature, 0),
                    'lime_importance': lime_explanation.get('feature_importance', {}).get(feature, 0),
                    'combined_importance': combined_score
                }
            
            # Sort by combined importance
            sorted_combined = sorted(
                combined['combined_features'].items(),
                key=lambda x: abs(x[1]['combined_importance']),
                reverse=True
            )[:max_features]
            
            combined['top_combined_features'] = [f[0] for f in sorted_combined]
        
        return combined
    
    def generate_explanation_text(self, explanation: Dict, instance: pd.DataFrame) -> str:
        """
        Generate human-readable explanation text
        
        Args:
            explanation: Explanation dictionary from explain_combined
            instance: Original instance data
            
        Returns:
            Human-readable explanation text
        """
        texts = []
        
        # SHAP explanation
        if explanation.get('shap', {}).get('available'):
            shap_exp = explanation['shap']
            texts.append("SHAP Analysis:")
            texts.append(f"  Base risk: {shap_exp.get('base_value', 0):.3f}")
            texts.append(f"  Predicted risk: {shap_exp.get('prediction', 0):.3f}")
            
            top_shap = shap_exp.get('top_features', [])[:5]
            if top_shap:
                texts.append("  Top contributing factors:")
                for feat in top_shap:
                    importance = shap_exp.get('feature_importance', {}).get(feat, 0)
                    direction = "increases" if importance > 0 else "decreases"
                    texts.append(f"    - {feat}: {direction} risk by {abs(importance):.3f}")
        
        # LIME explanation
        if explanation.get('lime', {}).get('available'):
            lime_exp = explanation['lime']
            texts.append("\nLIME Analysis:")
            top_lime = lime_exp.get('top_features', [])[:5]
            if top_lime:
                texts.append("  Key factors:")
                for feat in top_lime:
                    importance = lime_exp.get('feature_importance', {}).get(feat, 0)
                    direction = "increases" if importance > 0 else "decreases"
                    texts.append(f"    - {feat}: {direction} risk by {abs(importance):.3f}")
        
        # Combined explanation
        if explanation.get('top_combined_features'):
            texts.append("\nCombined Analysis (Consensus):")
            for feat in explanation['top_combined_features'][:5]:
                combined_info = explanation['combined_features'].get(feat, {})
                importance = combined_info.get('combined_importance', 0)
                direction = "increases" if importance > 0 else "decreases"
                instance_val = instance[feat].values[0] if feat in instance.columns else "N/A"
                texts.append(f"    - {feat} (value: {instance_val:.2f}): {direction} risk")
        
        return "\n".join(texts)

