"""
Model persistence: save and load the best model with full preprocessing pipeline for inference.
"""
from .bundle import InferenceBundle, save_best_model, load_best_model

__all__ = ["InferenceBundle", "save_best_model", "load_best_model"]
