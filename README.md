# Hybrid Clinical Decision Support System (CDS)

An integrated clinical decision support system that combines **Machine Learning predictions**, **Rule-based overrides**, and **Explainability** (SHAP, LIME) for diabetes risk assessment and insulin management.

## 🎯 System Overview

This system provides intelligent clinical decision support that outputs:
1. **Risk Score** (0-1 scale)
2. **Why the patient is at high risk** (explainable insights)
3. **What should be done** (actionable recommendations)

### Key Features

- **Hybrid Prediction Engine**: Combines ML predictions with evidence-based clinical rules
- **Rule-Based Overrides**: Clinical rules can override ML predictions when critical conditions are detected
- **Explainability**: SHAP and LIME integration for model interpretability
- **Low-Resource Models**: Optimized for deployment in resource-constrained environments
- **Real Clinical Analysis**: Incorporates real predictions and rule-based overrides

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Hybrid Clinical Decision Support System         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   ML Model   │    │ Rule Engine  │    │Explain-  │ │
│  │ (Lightweight)│    │  (Clinical   │    │ability   │ │
│  │              │    │    Rules)    │    │(SHAP/LIME)│ │
│  └──────┬───────┘    └──────┬───────┘    └────┬─────┘ │
│         │                   │                  │        │
│         └───────────────────┼──────────────────┘        │
│                             │                           │
│                    ┌────────▼────────┐                  │
│                    │  Hybrid Fusion  │                  │
│                    │     Engine      │                  │
│                    └────────┬────────┘                  │
│                             │                           │
│                    ┌────────▼────────┐                  │
│                    │   CDS Output    │                  │
│                    │  - Risk Score   │                  │
│                    │  - Explanation  │                  │
│                    │  - Recommendation│                  │
│                    └─────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 🌐 Interactive Web Interface (Recommended)

**Start the interactive web interface:**
```bash
python run_app.py
```
Or: `streamlit run app.py`

The interface will open at **http://localhost:8501**

**Features:**
- 👤 Single patient risk assessment
- 📊 Batch patient processing
- 📈 Visual results and explanations
- 💾 Export results (JSON/CSV)

**See [START_INTERFACE.md](START_INTERFACE.md) for quick start guide**

### ⚙️ Command Line Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete pipeline:**
   ```bash
   python run_full_pipeline.py
   ```

3. **Test the system:**
   ```bash
   python examples/cds_example.py
   ```

**For detailed instructions, see [RUN_SYSTEM.md](RUN_SYSTEM.md) or [QUICK_START.md](QUICK_START.md)**

### Installation

1. Clone the repository:
```bash
cd Diabetes_Risk_Isulin_Management_System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

Train the complete CDS system:

```bash
python src/cds/train_cds.py --data data/selected/selected_features.csv --output models/cds_model.pkl
```

Options:
- `--data`: Path to training data (CSV)
- `--output`: Path to save trained model
- `--model-type`: Type of ML model (`auto`, `lightgbm`, `xgboost`, `rf`, `ridge`, `tree`)
- `--val-split`: Validation split ratio (default: 0.2)

### Making Predictions

#### Single Patient

```bash
python src/cds/inference.py --data examples/patient_data.csv --model models/cds_model.pkl
```

#### Batch Prediction

```bash
python src/cds/inference.py --data examples/patients_batch.csv --model models/cds_model.pkl --batch
```

#### JSON Output

```bash
python src/cds/inference.py --data examples/patient_data.csv --model models/cds_model.pkl --json
```

## 💻 Python API Usage

### Basic Usage

```python
from src.cds import HybridCDS, ClinicalRuleEngine
from src.cds.inference import predict_patient, format_output_for_display
import pandas as pd

# Initialize CDS system
rule_engine = ClinicalRuleEngine()
cds = HybridCDS(rule_engine=rule_engine)

# Patient data
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

# Make prediction
patient_df = pd.DataFrame([patient_data])
result = cds.predict(patient_df, return_explanation=True)

# Display results
print(format_output_for_display(result))
```

### With Trained ML Model

```python
from src.cds import HybridCDS, LightweightMLModel, ModelExplainer
import pandas as pd

# Load your training data
X_train = pd.read_csv('data/selected/selected_features.csv')
# ... prepare features and target ...

# Train model
ml_model = LightweightMLModel(model_type='lightgbm')
ml_model.fit(X_train, y_train)

# Initialize explainer
explainer = ModelExplainer(ml_model.model, X_train, feature_names)

# Create CDS system
cds = HybridCDS(ml_model=ml_model, explainer=explainer)

# Make prediction
result = cds.predict(patient_df, return_explanation=True)
```

## 📊 Output Format

The system outputs a comprehensive `CDSOutput` object containing:

```python
{
    "risk_score": 0.85,              # 0-1 risk score
    "risk_level": "high",            # low, moderate, high, critical
    "ml_prediction": 75.5,           # Raw ML prediction
    "rule_override": true,           # Whether rule overrode ML
    "rule_used": "critical_glucose", # Rule that was triggered
    "explanation": "...",            # Why patient is at risk
    "recommendation": "...",         # What should be done
    "contributing_factors": [        # Top contributing factors
        {
            "feature": "Glucose_Level",
            "value": 280.0,
            "impact": 0.45,
            "method": "shap"
        }
    ],
    "confidence": 0.95,              # Overall confidence
    "ml_confidence": 0.7,            # ML model confidence
    "rule_confidence": 0.95          # Rule confidence
}
```

## 🔧 Clinical Rules

The system includes evidence-based clinical rules for:

1. **Critical Glucose Levels**: Severe hypoglycemia (<54 mg/dL) or hyperglycemia (>400 mg/dL)
2. **Hypoglycemia Risk**: Moderate hypoglycemia (54-70 mg/dL)
3. **Hyperglycemia Risk**: Elevated glucose (250-400 mg/dL)
4. **HbA1c Critical**: Poor long-term control (≥8.5%)
5. **Blood Pressure Crisis**: Hypertensive crisis or hypotension
6. **Ketoacidosis Risk**: DKA risk factors
7. **Rapid Glucose Changes**: Unstable glycemic control
8. **Multiple Risk Factors**: Compounding risk factors
9. **Medication Non-compliance**: Low medication intake with high glucose

Rules can override ML predictions when critical conditions are detected.

## 🧠 Explainability

The system provides explainability through:

- **SHAP (SHapley Additive exPlanations)**: Model-agnostic feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local model explanations
- **Combined Analysis**: Consensus from both methods

## 📁 Project Structure

```
Diabetes_Risk_Isulin_Management_System/
├── src/
│   ├── cds/                    # Clinical Decision Support Module
│   │   ├── __init__.py
│   │   ├── hybrid_cds.py      # Main hybrid CDS system
│   │   ├── rule_engine.py     # Clinical rule engine
│   │   ├── explainability.py  # SHAP/LIME integration
│   │   ├── ml_models.py       # Lightweight ML models
│   │   ├── train_cds.py       # Training script
│   │   └── inference.py       # Inference script
│   ├── utils/
│   │   └── config.py          # Configuration
│   └── ...                    # Other modules
├── examples/
│   └── cds_example.py         # Usage examples
├── models/                    # Trained models
├── data/                      # Data files
└── requirements.txt
```

## 🔬 Model Types

The system supports multiple lightweight model types:

- **LightGBM** (recommended): Fast, memory-efficient gradient boosting
- **XGBoost**: High-performance gradient boosting
- **Random Forest**: Ensemble of decision trees
- **Ridge Regression**: Linear model with L2 regularization
- **Decision Tree**: Simple interpretable model

Model selection is automatic based on data size, or can be manually specified.

## 📝 Example Output

```
============================================================
CLINICAL DECISION SUPPORT SYSTEM - ASSESSMENT
============================================================

RISK SCORE: 0.900 (CRITICAL)
Confidence: 95.00%

--- PREDICTION SOURCE ---
ML Prediction: 85.50
Rule Override: Yes
Rule Used: critical_glucose

--- WHY PATIENT IS AT RISK ---
CRITICAL: Severe hyperglycemia detected (Glucose: 420.0 mg/dL). 
Risk of diabetic ketoacidosis (DKA) or hyperosmolar hyperglycemic state (HHS).

--- RECOMMENDATION ---
IMMEDIATE ACTION REQUIRED: Check ketones, administer rapid-acting insulin 
as per protocol, ensure adequate hydration, monitor closely. 
Consider emergency care if ketones elevated or patient symptomatic.

--- TOP CONTRIBUTING FACTORS ---
1. Glucose_Level: 420.00 (increases risk by 0.450)
2. HbA1c: 11.00 (increases risk by 0.320)
3. Stress_Level: 90.00 (increases risk by 0.180)
4. Medication_Intake: 10.00 (increases risk by 0.150)
5. BMI: 38.00 (increases risk by 0.120)

============================================================
```

## 🛠️ Development

### Running Examples

```bash
python examples/cds_example.py
```

### Testing Components

```python
# Test rule engine
from src.cds import ClinicalRuleEngine
rule_engine = ClinicalRuleEngine()
result = rule_engine.evaluate_all_rules(patient_data)

# Test explainability
from src.cds import ModelExplainer
explainer = ModelExplainer(model, training_data, feature_names)
explanation = explainer.explain_combined(instance)
```

## 📚 Dependencies

- `pandas>=1.5.0`
- `numpy>=1.23.0`
- `scikit-learn>=1.2.0`
- `lightgbm>=3.3.0`
- `xgboost>=1.7.0`
- `shap>=0.41.0`
- `lime>=0.2.0`

## ⚠️ Important Notes

- This system is for **research and educational purposes**
- Clinical decisions should always involve healthcare professionals
- Model predictions should be validated with clinical expertise
- Rules are based on general guidelines and may need customization

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]

