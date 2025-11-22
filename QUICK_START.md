# Quick Start Guide

## 🚀 Fastest Way to Run the System

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python run_full_pipeline.py
```

This will:
- ✅ Preprocess your data
- ✅ Engineer features
- ✅ Train the CDS system
- ✅ Save the model

### 3. Test the System
```bash
python examples/cds_example.py
```

### 4. Make Predictions

**Using Python:**
```python
from src.cds.inference import load_cds_model, predict_patient, format_output_for_display
import pandas as pd

# Load model
cds, _ = load_cds_model()

# Patient data (example)
patient_data = {
    'Glucose_Level': 280.0,
    'Heart_Rate': 85.0,
    # ... (all required features)
}

# Predict
result = cds.predict(pd.DataFrame([patient_data]), return_explanation=True)
print(format_output_for_display(result))
```

**Using Command Line:**
```bash
python src/cds/inference.py --data patient_data.csv --model models/cds_model.pkl
```

## 📚 Detailed Workflow

For step-by-step exploration, run the notebooks in order:

1. `notebooks/01_EDA.ipynb` - Explore data
2. `notebooks/02_Data_Cleaning_Preprocessing.ipynb` - Clean data
3. `notebooks/03_Feature_Engineering.ipynb` - Engineer features
4. `notebooks/04_Modeling_Evaluation.ipynb` - Train models

## 📖 Full Documentation

See `RUN_SYSTEM.md` for complete instructions.

