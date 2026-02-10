"""
Model Registry - Manages all registered ML models
"""

from typing import Dict, Type, Optional
from app.ml_framework.base_model import BaseInsulinModel
import importlib
import os

class ModelRegistry:
    """
    Registry for managing ML models
    Allows dynamic loading and switching between models
    """
    
    _instance = None
    _models: Dict[str, Type[BaseInsulinModel]] = {}
    _active_model: Optional[BaseInsulinModel] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_model(self, model_class: Type[BaseInsulinModel], model_id: str):
        """
        Register a model class
        
        Args:
            model_class: Class that inherits from BaseInsulinModel
            model_id: Unique identifier for the model
        """
        if not issubclass(model_class, BaseInsulinModel):
            raise ValueError(f"Model {model_id} must inherit from BaseInsulinModel")
        
        self._models[model_id] = model_class
        print(f"Registered model: {model_id}")
    
    def get_model(self, model_id: str, **kwargs) -> Optional[BaseInsulinModel]:
        """
        Get an instance of a registered model
        
        Args:
            model_id: Identifier of the model
            **kwargs: Arguments to pass to model constructor
        
        Returns:
            Model instance or None if not found
        """
        if model_id not in self._models:
            return None
        
        model_class = self._models[model_id]
        # Use model_id as model_name if model_name not provided
        if 'model_name' not in kwargs:
            kwargs['model_name'] = model_id.replace('_', ' ').title()
        return model_class(**kwargs)
    
    def list_models(self) -> Dict[str, Dict]:
        """
        List all registered models with their info
        
        Returns:
            Dictionary of model_id -> model_info
        """
        models_info = {}
        for model_id, model_class in self._models.items():
            # Create a temporary instance to get info
            try:
                temp_instance = model_class()
                models_info[model_id] = temp_instance.get_model_info()
            except Exception as e:
                models_info[model_id] = {
                    'model_id': model_id,
                    'error': str(e)
                }
        return models_info
    
    def set_active_model(self, model_id: str, **kwargs) -> bool:
        """
        Set the active model to use for predictions
        
        Args:
            model_id: Identifier of the model
            **kwargs: Arguments for model initialization
        
        Returns:
            True if successful, False otherwise
        """
        model = self.get_model(model_id, **kwargs)
        if model is None:
            return False
        
        self._active_model = model
        return True
    
    def get_active_model(self) -> Optional[BaseInsulinModel]:
        """Get the currently active model"""
        return self._active_model
    
    def load_models_from_directory(self, directory: str):
        """
        Automatically load models from a directory
        
        Args:
            directory: Path to directory containing model files
        """
        if not os.path.exists(directory):
            return
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]  # Remove .py
                try:
                    module = importlib.import_module(f'app.ml_framework.models.{module_name}')
                    # Look for classes that inherit from BaseInsulinModel
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseInsulinModel) and 
                            attr != BaseInsulinModel):
                            model_id = getattr(attr, 'MODEL_ID', module_name)
                            self.register_model(attr, model_id)
                except Exception as e:
                    print(f"Error loading model from {filename}: {e}")

# Global registry instance
model_registry = ModelRegistry()

