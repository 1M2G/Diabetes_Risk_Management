"""
Quick system test to verify everything is working
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print("  ✓ Core libraries")
        
        try:
            import lightgbm
            print("  ✓ LightGBM")
        except ImportError:
            print("  ⚠ LightGBM not available (optional)")
        
        try:
            import xgboost
            print("  ✓ XGBoost")
        except ImportError:
            print("  ⚠ XGBoost not available (optional)")
        
        try:
            import shap
            print("  ✓ SHAP")
        except ImportError:
            print("  ⚠ SHAP not available (optional)")
        
        try:
            import lime
            print("  ✓ LIME")
        except ImportError:
            print("  ⚠ LIME not available (optional)")
        
        # Test project modules
        sys.path.insert(0, str(Path(__file__).parent))
        from src.cds import HybridCDS, ClinicalRuleEngine
        print("  ✓ CDS modules")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_data_files():
    """Test if required data files exist"""
    print("\nTesting data files...")
    data_path = Path("data/raw_data/SmartSensor_DiabetesMonitoring.csv")
    if data_path.exists():
        print(f"  ✓ Raw data found: {data_path}")
        return True
    else:
        print(f"  ⚠ Raw data not found: {data_path}")
        print("     Please ensure your data file is in the correct location")
        return False

def test_models():
    """Test if models exist"""
    print("\nTesting models...")
    model_path = Path("models/cds_model.pkl")
    if model_path.exists():
        print(f"  ✓ CDS model found: {model_path}")
        return True
    else:
        print(f"  ⚠ CDS model not found: {model_path}")
        print("     Run 'python run_full_pipeline.py' to train the model")
        return False

def test_cds_system():
    """Test if CDS system works"""
    print("\nTesting CDS system...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.cds import HybridCDS, ClinicalRuleEngine
        import pandas as pd
        
        # Create a simple CDS instance
        rule_engine = ClinicalRuleEngine()
        cds = HybridCDS(rule_engine=rule_engine)
        
        # Test with sample data
        patient_data = {
            'Glucose_Level': 280.0,
            'Heart_Rate': 85.0,
            'Activity_Level': 3.0,
            'Calories_Burned': 1200.0,
            'Sleep_Duration': 6.5,
            'Step_Count': 5000,
            'Medication_Intake': 40.0,
            'Diet_Quality_Score': 65.0,
            'Stress_Level': 75.0,
            'BMI': 32.0,
            'HbA1c': 9.5,
            'Blood_Pressure_Systolic': 145.0,
            'Blood_Pressure_Diastolic': 95.0
        }
        
        patient_df = pd.DataFrame([patient_data])
        result = cds.predict(patient_df, return_explanation=True)
        
        print(f"  ✓ CDS system working")
        print(f"     Risk score: {result.risk_score:.3f}")
        print(f"     Risk level: {result.risk_level}")
        return True
    except Exception as e:
        print(f"  ✗ CDS system error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("SYSTEM TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Models", test_models()))
    results.append(("CDS System", test_cds_system()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! System is ready to use.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("\nNext steps:")
        if not results[0][1]:  # Imports failed
            print("  1. Install dependencies: pip install -r requirements.txt")
        if not results[1][1]:  # Data files missing
            print("  2. Ensure data file is in data/raw_data/")
        if not results[2][1]:  # Models missing
            print("  3. Train models: python run_full_pipeline.py")
    
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

