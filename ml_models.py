"""
Lightweight ML Models for Low-Resource Clinical Decision Support
Optimized for deployment in resource-constrained environments
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class LightweightMLModel:
    """
    Lightweight ML model wrapper optimized for low-resource deployment
    Supports multiple model types with automatic selection
    """
    
    def __init__(self, model_type: str = 'auto', model_params: Optional[Dict] = None):
        """
        Initialize lightweight ML model
        
        Args:
            model_type: Type of model ('auto', 'lightgbm', 'xgboost', 'rf', 'ridge', 'tree')
            model_params: Optional parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.model_type_used = None
        
    def _select_best_model(self, X: pd.DataFrame, y: pd.Series, 
                          sample_size: int = 1000) -> Any:
        """
        Automatically select the best lightweight model based on data size
        
        Args:
            X: Feature matrix
            y: Target vector
            sample_size: Sample size threshold for model selection
            
        Returns:
            Best model instance
        """
        n_samples = len(X)
        
        # For very small datasets, use simple models
        if n_samples < 100:
            print("Small dataset detected. Using Ridge regression.")
            return Ridge(alpha=1.0, random_state=42)
        
        # For small-medium datasets, use LightGBM if available
        elif n_samples < sample_size and LIGHTGBM_AVAILABLE:
            print("Medium dataset detected. Using LightGBM (lightweight).")
            params = {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42,
                'verbosity': -1,
                'force_col_wise': True
            }
            params.update(self.model_params)
            return lgb.LGBMRegressor(**params)
        
        # For larger datasets, prefer LightGBM or XGBoost
        elif LIGHTGBM_AVAILABLE:
            print("Large dataset detected. Using LightGBM (optimized).")
            params = {
                'n_estimators': 100,
                'max_depth': 7,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'random_state': 42,
                'verbosity': -1,
                'force_col_wise': True
            }
            params.update(self.model_params)
            return lgb.LGBMRegressor(**params)
        
        # Fallback to Random Forest
        elif SKLEARN_AVAILABLE:
            print("Using Random Forest (fallback).")
            params = {
                'n_estimators': 50,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': 1  # Low resource
            }
            params.update(self.model_params)
            return RandomForestRegressor(**params)
        
        else:
            raise ValueError("No suitable ML library available")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_split: float = 0.2) -> Dict:
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        if self.model_type == 'auto':
            self.model = self._select_best_model(X, y)
            self.model_type_used = type(self.model).__name__
        else:
            self.model = self._create_model(self.model_type)
            self.model_type_used = self.model_type
        
        self.feature_names = list(X.columns)
        
        # Split data if validation requested
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_val = X, X
            y_train, y_val = y, y
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred),
            'model_type': self.model_type_used
        }
        
        return metrics
    
    def _create_model(self, model_type: str) -> Any:
        """Create model instance based on type"""
        if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            params = {
                'n_estimators': 100,
                'max_depth': 7,
                'learning_rate': 0.05,
                'random_state': 42,
                'verbosity': -1
            }
            params.update(self.model_params)
            return lgb.LGBMRegressor(**params)
        
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            params = {
                'n_estimators': 100,
                'max_depth': 7,
                'learning_rate': 0.05,
                'random_state': 42
            }
            params.update(self.model_params)
            return xgb.XGBRegressor(**params)
        
        elif model_type == 'rf' and SKLEARN_AVAILABLE:
            params = {
                'n_estimators': 50,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': 1
            }
            params.update(self.model_params)
            return RandomForestRegressor(**params)
        
        elif model_type == 'ridge' and SKLEARN_AVAILABLE:
            params = {'alpha': 1.0, 'random_state': 42}
            params.update(self.model_params)
            return Ridge(**params)
        
        elif model_type == 'tree' and SKLEARN_AVAILABLE:
            params = {'max_depth': 10, 'random_state': 42}
            params.update(self.model_params)
            return DecisionTreeRegressor(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict probabilities (if supported)"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            return {}
        
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        return dict(zip(self.feature_names, importances))
    
    def save(self, filepath: Path):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type_used,
            'model_params': self.model_params
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'LightweightMLModel':
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_type=model_data.get('model_type', 'auto'),
                      model_params=model_data.get('model_params'))
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.model_type_used = model_data.get('model_type')
        
        return instance
    
    def get_model_info(self) -> Dict:
        """Get information about the model"""
        return {
            'model_type': self.model_type_used,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'has_feature_importance': hasattr(self.model, 'feature_importances_') if self.model else False
        }

