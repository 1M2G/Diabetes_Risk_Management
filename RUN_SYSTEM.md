# How to Run the Entire System

This guide explains how to run the complete Hybrid Clinical Decision Support System from start to finish.

## 📋 Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Data**
   - Ensure raw data is in `data/raw_data/SmartSensor_DiabetesMonitoring.csv`
   - If not, place your data file there

## 🚀 Complete Workflow

### Option 1: Run Notebooks (Recommended for Exploration)

Run the notebooks in sequence:

#### Step 1: Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
- Opens the EDA notebook
- Run all cells (Cell → Run All)
- Review data insights

#### Step 2: Data Cleaning & Preprocessing
```bash
jupyter notebook notebooks/02_Data_Cleaning_Preprocessing.ipynb
```
- Cleans the raw data
- Applies physiological bounds
- Handles outliers and missing values
- Saves cleaned data to `data/processed/cleaned_no_outliers.csv`

#### Step 3: Feature Engineering
```bash
jupyter notebook notebooks/03_Feature_Engineering.ipynb
```
- Creates interaction features
- Applies feature scaling
- Performs feature selection
- Saves processed data to `data/selected/selected_features.csv`

#### Step 4: Modeling & Evaluation
```bash
jupyter notebook notebooks/04_Modeling_Evaluation.ipynb
```
- Trains multiple ML models
- Evaluates and compares models
- Trains Hybrid CDS system
- Saves all models to `models/` directory

### Option 2: Run Python Scripts (Automated)

#### Quick Start Script
```bash
python run_full_pipeline.py
```
This will run the entire pipeline automatically.

#### Individual Steps

1. **Preprocess Data**
   ```bash
   python -m src.preprocessing_pipeline
   ```

2. **Train CDS System**
   ```bash
   python src/cds/train_cds.py --data data/selected/selected_features.csv --output models/cds_model.pkl
   ```

3. **Make Predictions**
   ```bash
   python src/cds/inference.py --data examples/patient_data.csv --model models/cds_model.pkl
   ```

## 📊 Using the System

### Making Predictions

#### Python API
```python
from src.cds.inference import load_cds_model, predict_patient, format_output_for_display
import pandas as pd

# Load trained model
cds, feature_names = load_cds_model()

# Prepare patient data (must include all features from training)
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
    'Blood_Pressure_Diastolic': 95.0,
    'Predicted_Progression': 0.7,
    'Timestamp_hour': 10.0,
    'Timestamp_dow': 2.0,
    'Timestamp_month': 5.0,
    'Timestamp_is_weekend': 0.0,
    'Timestamp_ts': 1748736000.0
}

# Make prediction
patient_df = pd.DataFrame([patient_data])
result = cds.predict(patient_df, return_explanation=True)

# Display results
print(format_output_for_display(result))
```

#### Command Line
```bash
# Single patient
python src/cds/inference.py --data patient_data.csv --model models/cds_model.pkl

# Batch processing
python src/cds/inference.py --data patients_batch.csv --model models/cds_model.pkl --batch

# JSON output
python src/cds/inference.py --data patient_data.csv --model models/cds_model.pkl --json --output results.json
```

### Run Examples
```bash
python examples/cds_example.py
```

## 🔄 Complete Pipeline Script

Create and run `run_full_pipeline.py`:

```python
"""
Complete pipeline runner for the Hybrid CDS System
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing_pipeline import run_full_preprocessing
from src.cds.train_cds import train_cds_model
from src.utils.config import FINAL_FEATURES, MODELS_DIR

def main():
    print("=" * 60)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 60)
    
    # Step 1: Preprocessing
    print("\n[1/2] Running preprocessing pipeline...")
    try:
        run_full_preprocessing()
        print("✓ Preprocessing completed!")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return
    
    # Step 2: Train CDS
    print("\n[2/2] Training CDS system...")
    try:
        train_cds_model(
            data_path=FINAL_FEATURES,
            model_save_path=MODELS_DIR / "cds_model.pkl",
            model_type='auto'
        )
        print("✓ CDS training completed!")
    except Exception as e:
        print(f"✗ CDS training failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nModel saved to: {MODELS_DIR / 'cds_model.pkl'}")
    print("\nYou can now make predictions using:")
    print("  python src/cds/inference.py --data <patient_data.csv> --model models/cds_model.pkl")

if __name__ == "__main__":
    main()
```

## 📁 Expected File Structure After Running

```
Diabetes_Risk_Isulin_Management_System/
├── data/
│   ├── raw_data/
│   │   └── SmartSensor_DiabetesMonitoring.csv
│   ├── processed/
│   │   └── cleaned_no_outliers.csv
│   ├── scaled_data/
│   │   └── scaled_features.csv
│   ├── engineered/
│   │   └── engineered_features.csv
│   └── selected/
│       └── selected_features.csv
├── models/
│   ├── cds_model.pkl
│   ├── best_model.pkl
│   ├── lightgbm_model.pkl
│   ├── random_forest_model.pkl
│   └── evaluation_results.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Cleaning_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Modeling_Evaluation.ipynb
└── src/
    └── cds/
        ├── train_cds.py
        └── inference.py
```

## ⚡ Quick Start (Fastest Way)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete pipeline:**
   ```bash
   python run_full_pipeline.py
   ```

3. **Test with example:**
   ```bash
   python examples/cds_example.py
   ```

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Make sure you're in the project root directory and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: FileNotFoundError for data
**Solution:** Ensure raw data file exists:
```bash
# Check if file exists
ls data/raw_data/SmartSensor_DiabetesMonitoring.csv

# If not, place your data file there
```

### Issue: Model not found during inference
**Solution:** Train the model first:
```bash
python src/cds/train_cds.py --data data/selected/selected_features.csv
```

### Issue: Feature mismatch during prediction
**Solution:** Ensure patient data has all required features. Check feature list:
```python
import pickle
with open('models/cds_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    print("Required features:", model_data['feature_names'])
```

## 📝 Next Steps

After running the pipeline:

1. **Review Results:**
   - Check `models/evaluation_results.csv` for model performance
   - Review notebook outputs for insights

2. **Make Predictions:**
   - Use the inference script or Python API
   - Test with sample patient data

3. **Customize:**
   - Modify clinical rules in `src/cds/rule_engine.py`
   - Adjust model parameters in training scripts
   - Add new features in feature engineering notebook

4. **Deploy:**
   - Integrate into your clinical workflow
   - Set up API endpoints if needed
   - Monitor model performance

## 🆘 Getting Help

- Check `README.md` for detailed documentation
- Review notebook outputs for errors
- Verify data format matches expected structure
- Ensure all dependencies are installed

