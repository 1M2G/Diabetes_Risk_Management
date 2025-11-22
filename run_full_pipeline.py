"""
Complete pipeline runner for the Hybrid Clinical Decision Support System
Runs the entire pipeline from preprocessing to model training
"""
import sys
from pathlib import Path
import traceback
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Change to project root directory
import os
os.chdir(project_root)

def main():
    print("=" * 70)
    print("HYBRID CLINICAL DECISION SUPPORT SYSTEM - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Step 1: Check for processed data
    print("\n" + "=" * 70)
    print("[STEP 1/2] CHECKING DATA")
    print("=" * 70)
    
    cleaned_path = Path("data/processed/cleaned_no_outliers.csv")
    selected_path = Path("data/selected/selected_features.csv")
    
    if cleaned_path.exists():
        print(f"[OK] Found cleaned data: {cleaned_path}")
        print(f"  Shape: {pd.read_csv(cleaned_path).shape}")
    elif selected_path.exists():
        print(f"[OK] Found selected features: {selected_path}")
        print(f"  Shape: {pd.read_csv(selected_path).shape}")
    else:
        print(f"\n⚠ Warning: No processed data found!")
        print(f"   Expected: {cleaned_path} or {selected_path}")
        print(f"   The system will try to use raw data if available.")
        print(f"   For best results, run the preprocessing notebooks first.")
    
    # Step 2: Train CDS
    print("\n" + "=" * 70)
    print("[STEP 2/2] TRAINING CDS SYSTEM")
    print("=" * 70)
    try:
        from src.cds.train_cds import train_cds_model
        
        # Check if processed data exists - use cleaned data directly
        cleaned_path = Path("data/processed/cleaned_no_outliers.csv")
        selected_path = Path("data/selected/selected_features.csv")
        models_dir = Path("models")
        
        if cleaned_path.exists():
            print(f"   Using cleaned data: {cleaned_path}")
            data_path = cleaned_path
        elif selected_path.exists():
            print(f"   Using selected features: {selected_path}")
            data_path = selected_path
        else:
            # Try to use raw data directly
            raw_path = Path("data/raw_data/SmartSensor_DiabetesMonitoring.csv")
            if raw_path.exists():
                print(f"\n⚠ No processed data found. Using raw data directly.")
                print(f"   For better results, run preprocessing first.")
                print(f"   Using: {raw_path}")
                data_path = raw_path
            else:
                print(f"\n[ERROR] No data found!")
                print(f"   Expected one of:")
                print(f"   - {cleaned_path}")
                print(f"   - {selected_path}")
                print(f"   - {raw_path}")
                return False
        
        train_cds_model(
            data_path=data_path,
            model_save_path=models_dir / "cds_model.pkl",
            model_type='auto'
        )
        print("\n[OK] CDS training completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] CDS training failed!")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False
    
    # Success summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n[OK] Model saved to: models/cds_model.pkl")
    print(f"\nNext steps:")
    print(f"   1. Test the system: python examples/cds_example.py")
    print(f"   2. Make predictions: python src/cds/inference.py --data <patient_data.csv>")
    print(f"   3. Review notebooks for detailed analysis")
    print("\n" + "=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

