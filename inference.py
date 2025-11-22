"""
Inference script for the Hybrid Clinical Decision Support System
Provides easy-to-use interface for making predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, Optional, Union

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cds import HybridCDS, CDSOutput
from src.utils.config import MODELS_DIR


def load_cds_model(model_path: Optional[Path] = None) -> HybridCDS:
    """
    Load trained CDS model from disk
    
    Args:
        model_path: Path to saved model (default: uses config)
        
    Returns:
        Loaded HybridCDS instance
    """
    if model_path is None:
        model_path = MODELS_DIR / "cds_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train the model first using train_cds.py"
        )
    
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    cds = model_data['cds']
    print(f"Model loaded from: {model_path}")
    print(f"Features: {len(model_data.get('feature_names', []))}")
    
    return cds, model_data.get('feature_names', [])


def predict_patient(
    patient_data: Union[Dict, pd.DataFrame, pd.Series],
    cds: Optional[HybridCDS] = None,
    model_path: Optional[Path] = None,
    return_json: bool = False
) -> Union[CDSOutput, str, Dict]:
    """
    Make prediction for a single patient
    
    Args:
        patient_data: Patient data as dict, Series, or DataFrame
        cds: Pre-loaded CDS instance (optional)
        model_path: Path to saved model (if cds not provided)
        return_json: Whether to return JSON string
        
    Returns:
        CDSOutput object, dict, or JSON string
    """
    # Load model if not provided
    if cds is None:
        cds, feature_names = load_cds_model(model_path)
    else:
        feature_names = None
    
    # Convert input to DataFrame
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.DataFrame):
        df = patient_data.copy()
    else:
        raise ValueError("patient_data must be dict, Series, or DataFrame")
    
    # Ensure single row
    if len(df) != 1:
        raise ValueError("patient_data must contain exactly one patient")
    
    # Make prediction
    result = cds.predict(df, return_explanation=True)
    
    # Format output
    if return_json:
        return result.to_json()
    else:
        return result


def predict_batch(
    patient_data: pd.DataFrame,
    cds: Optional[HybridCDS] = None,
    model_path: Optional[Path] = None,
    return_json: bool = False
) -> Union[list, str]:
    """
    Make predictions for multiple patients
    
    Args:
        patient_data: DataFrame with multiple patient rows
        cds: Pre-loaded CDS instance (optional)
        model_path: Path to saved model (if cds not provided)
        return_json: Whether to return JSON string
        
    Returns:
        List of CDSOutput objects or JSON string
    """
    # Load model if not provided
    if cds is None:
        cds, feature_names = load_cds_model(model_path)
    
    # Make predictions
    results = cds.batch_predict(patient_data, return_explanations=True)
    
    # Format output
    if return_json:
        return json.dumps([r.to_dict() for r in results], indent=2)
    else:
        return results


def format_output_for_display(result: CDSOutput) -> str:
    """
    Format CDS output for human-readable display
    
    Args:
        result: CDSOutput object
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "CLINICAL DECISION SUPPORT SYSTEM - ASSESSMENT",
        "=" * 60,
        "",
        f"RISK SCORE: {result.risk_score:.3f} ({result.risk_level.upper()})",
        f"Confidence: {result.confidence:.2%}",
        "",
        "--- PREDICTION SOURCE ---",
        f"ML Prediction: {result.ml_prediction:.2f}",
        f"Rule Override: {'Yes' if result.rule_override else 'No'}",
        f"Rule Used: {result.rule_used or 'None'}",
        "",
        "--- WHY PATIENT IS AT RISK ---",
        result.explanation,
        "",
        "--- RECOMMENDATION ---",
        result.recommendation,
    ]
    
    if result.contributing_factors:
        lines.extend([
            "",
            "--- TOP CONTRIBUTING FACTORS ---"
        ])
        for i, factor in enumerate(result.contributing_factors[:5], 1):
            impact_direction = "increases" if factor['impact'] > 0 else "decreases"
            lines.append(
                f"{i}. {factor['feature']}: {factor['value']:.2f} "
                f"({impact_direction} risk by {abs(factor['impact']):.3f})"
            )
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CDS Inference")
    parser.add_argument("--model", type=str, help="Path to saved model")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to patient data (CSV)")
    parser.add_argument("--output", type=str, help="Path to save output")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    parser.add_argument("--batch", action="store_true",
                       help="Process multiple patients")
    
    args = parser.parse_args()
    
    # Load patient data
    patient_df = pd.read_csv(args.data)
    
    # Make predictions
    if args.batch:
        results = predict_batch(
            patient_df,
            model_path=Path(args.model) if args.model else None,
            return_json=args.json
        )
    else:
        if len(patient_df) > 1:
            print("Warning: Multiple rows detected. Using first row only.")
            patient_df = patient_df.iloc[[0]]
        
        results = predict_patient(
            patient_df,
            model_path=Path(args.model) if args.model else None,
            return_json=args.json
        )
    
    # Output results
    if args.json:
        output = results if isinstance(results, str) else json.dumps(
            [r.to_dict() for r in results] if isinstance(results, list) else results.to_dict(),
            indent=2
        )
    else:
        if isinstance(results, list):
            output = "\n\n".join([format_output_for_display(r) for r in results])
        else:
            output = format_output_for_display(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Output saved to: {args.output}")
    else:
        print(output)

