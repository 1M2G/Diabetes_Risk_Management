"""
Custom ML Models for Insulin Management
"""

from app.ml_framework.model_registry import model_registry

# Import and register default models
from app.ml_framework.models.default_model import DefaultInsulinModel
from app.ml_framework.models.advanced_model import AdvancedInsulinModel

# Register default models
model_registry.register_model(DefaultInsulinModel, 'default')
model_registry.register_model(AdvancedInsulinModel, 'advanced')

__all__ = ['model_registry', 'DefaultInsulinModel', 'AdvancedInsulinModel']

