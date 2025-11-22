"""
Training script for the Hybrid Clinical Decision Support System
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cds import HybridCDS, LightweightMLModel, ModelExplainer
from src.utils.config import (
    FINAL_FEATURES, TARGET, MODELS_DIR, DATA_CLEAN
)


def train_cds_model(
    data_path: Optional[Path] = None,
    model_save_path: Optional[Path] = None,
    model_type: str = 'auto',
    validation_split: float = 0.2
) -> HybridCDS:
    """
    Train the complete Hybrid CDS system
    
    Args:
        data_path: Path to training data (default: uses config)
        model_save_path: Path to save trained model
        model_type: Type of ML model ('auto', 'lightgbm', 'rf', etc.)
        validation_split: Fraction of data for validation
        
    Returns:
        Trained HybridCDS instance
    """
    # Load data
    if data_path is None:
        data_path = FINAL_FEATURES if FINAL_FEATURES.exists() else DATA_CLEAN
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data")
    
    # Get feature columns (exclude target and ID columns)
    exclude_cols = [TARGET, 'Patient_ID', 'timestamp_sorted', 'Timestamp']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    
    # Remove any rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"Training data shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {TARGET}")
    
    # Initialize CDS system
    ml_model = LightweightMLModel(model_type=model_type)
    cds = HybridCDS(ml_model=ml_model)
    
    # Train ML model
    print("\nTraining ML model...")
    metrics = cds.train_ml_model(X, y, training_data_for_explainer=X)
    
    print("\nTraining Metrics:")
    print(f"  Model Type: {metrics['model_type']}")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Train MAE: {metrics['train_mae']:.4f}")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Val RMSE: {metrics['val_rmse']:.4f}")
    print(f"  Val MAE: {metrics['val_mae']:.4f}")
    print(f"  Val R²: {metrics['val_r2']:.4f}")
    
    # Save model (without explainer - it will be recreated on load if needed)
    if model_save_path is None:
        model_save_path = MODELS_DIR / "cds_model.pkl"
    
    import pickle
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save without explainer (LIME has lambda functions that can't be pickled)
    # Explainer will be recreated on inference if needed
    cds_without_explainer = HybridCDS(
        ml_model=cds.ml_model,
        rule_engine=cds.rule_engine,
        explainer=None,  # Don't save explainer
        rule_override_threshold=cds.rule_override_threshold,
        ml_weight=cds.ml_weight,
        rule_weight=cds.rule_weight
    )
    
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'cds': cds_without_explainer,
            'feature_names': feature_cols,
            'target': TARGET,
            'metrics': metrics,
            'training_data_sample': X.sample(min(1000, len(X))).copy()  # Sample for explainer recreation
        }, f)
    
    print(f"\nModel saved to: {model_save_path}")
    
    return cds


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Hybrid CDS System")
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--output", type=str, help="Path to save model")
    parser.add_argument("--model-type", type=str, default="auto",
                       choices=['auto', 'lightgbm', 'xgboost', 'rf', 'ridge', 'tree'],
                       help="Type of ML model")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio")
    
    args = parser.parse_args()
    
    data_path = Path(args.data) if args.data else None
    output_path = Path(args.output) if args.output else None
    
    cds = train_cds_model(
        data_path=data_path,
        model_save_path=output_path,
        model_type=args.model_type,
        validation_split=args.val_split
    )
    
    print("\nTraining completed successfully!")

