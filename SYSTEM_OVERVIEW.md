# Insulin Management System - Type 1 Diabetes Focus

## System Overview

This is an **API-first** ML-powered insulin management system specifically designed for **Type 1 Diabetes**. The system provides:

1. **ML Framework** for developers to create custom models
2. **API Endpoints** for integration with external applications
3. **Doctor Portal** for healthcare providers to monitor and manage patients
4. **Type 1 Diabetes Specific Metrics** (Time in Range, Glucose Variability, Stability)

## Key Features

### 1. ML Framework for Custom Models

- **Base Model Interface**: All models inherit from `BaseInsulinModel`
- **Model Registry**: Dynamic model loading and switching
- **Plugin System**: Easy to add new models
- **Explainability**: Built-in SHAP explanations
- **Safety First**: Automatic safety limit enforcement

**Location**: `backend/app/ml_framework/`

**Documentation**: See `ML_WORKFLOW.md`

### 2. API-First Architecture

The system is designed to support **external applications** (mobile apps, CGM integrations, etc.) rather than direct patient interaction.

**Key API Endpoints**:
- `POST /api/ml/predict` - Get insulin dose recommendation
- `POST /api/ml/explain` - Get detailed explanation
- `POST /api/ml/analyze-patterns` - Analyze historical patterns
- `POST /api/type1/metrics` - Calculate Type 1 diabetes metrics
- `POST /api/type1/safety-check` - Safety validation
- `POST /api/data/submit` - Submit patient data

**Location**: `backend/app/routes_api.py`

### 3. Type 1 Diabetes Specific Features

#### Metrics Calculated:
- **Time in Range (TIR)**: Percentage of time glucose is 70-180 mg/dL (target: ≥70%)
- **Time Below Range (TBR)**: Percentage <70 mg/dL (target: <4%)
- **Time Above Range (TAR)**: Percentage >180 mg/dL (target: <25%)
- **Glucose Variability (CV)**: Coefficient of variation (target: <36%)
- **Stability Score**: Overall glucose control score (0-100)
- **Glucose Management Indicator (GMI)**: Estimated A1C

#### Safety Features:
- Automatic safety limit enforcement
- Hypoglycemia risk detection
- Hyperglycemia alerts
- Dose validation before recommendation

### 4. Enhanced Doctor Portal

**Features**:
- Real-time patient monitoring
- Interactive dashboards with Type 1 diabetes metrics
- ML-powered insights with explanations
- Pattern analysis and trend detection
- Alert management system
- Patient assignment and management

**Location**: `frontend/src/pages/DoctorDashboard.js`, `DoctorPatientView.js`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  External Applications                   │
│  (Mobile Apps, CGM Systems, Other Diabetes Apps)       │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ REST API
                     │
┌────────────────────▼────────────────────────────────────┐
│              API Layer (routes_api.py)                   │
│  - /api/ml/predict                                       │
│  - /api/ml/explain                                       │
│  - /api/type1/metrics                                    │
│  - /api/data/submit                                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     │
┌────────────────────▼────────────────────────────────────┐
│            ML Service (ml_service.py)                    │
│  - Model Registry Management                             │
│  - Prediction Orchestration                              │
│  - Pattern Analysis                                      │
│  - Type 1 Metrics Calculation                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     │
┌────────────────────▼────────────────────────────────────┐
│         ML Framework (ml_framework/)                    │
│  ┌──────────────────────────────────────┐               │
│  │  Base Model Interface               │               │
│  │  - BaseInsulinModel (ABC)           │               │
│  └──────────────────────────────────────┘               │
│  ┌──────────────────────────────────────┐               │
│  │  Model Registry                     │               │
│  │  - Dynamic Model Loading             │               │
│  │  - Model Switching                   │               │
│  └──────────────────────────────────────┘               │
│  ┌──────────────────────────────────────┐               │
│  │  Custom Models                       │               │
│  │  - DefaultInsulinModel               │               │
│  │  - AdvancedInsulinModel              │               │
│  │  - Your Custom Models                │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
                     │
                     │
┌────────────────────▼────────────────────────────────────┐
│              Doctor Portal (Frontend)                    │
│  - Patient Monitoring                                    │
│  - ML Insights Review                                   │
│  - Alert Management                                     │
│  - Type 1 Metrics Visualization                        │
└─────────────────────────────────────────────────────────┘
```

## ML Model Development

### Creating a Custom Model

1. **Inherit from BaseInsulinModel**
   ```python
   from app.ml_framework.base_model import BaseInsulinModel
   
   class MyModel(BaseInsulinModel):
       MODEL_ID = 'my_model'
       ...
   ```

2. **Implement Required Methods**
   - `train()` - Train your model
   - `predict()` - Make predictions
   - `explain()` - Provide explanations
   - `get_supported_features()` - List features

3. **Register Your Model**
   ```python
   model_registry.register_model(MyModel, 'my_model')
   ```

4. **Train and Use**
   ```python
   model = MyModel()
   model.train(training_data)
   result = model.predict(patient_data, patient_profile)
   ```

See `ML_WORKFLOW.md` for detailed instructions.

## API Usage Examples

### Get Insulin Dose Recommendation

```bash
POST /api/ml/predict
{
  "patient_data": {
    "glucose_level": 150,
    "food_intake": 60,
    "physical_activity": 30
  },
  "patient_profile": {
    "diabetes_type": "Type 1",
    "age": 30,
    "weight_kg": 70,
    "carb_ratio": 15,
    "insulin_sensitivity_factor": 50
  }
}
```

**Response**:
```json
{
  "recommended_dose": 4.5,
  "confidence": 0.85,
  "prediction_type": "bolus",
  "explanation": "...",
  "safety_flags": [],
  "reasoning_steps": [...],
  "feature_importance": {...}
}
```

### Calculate Type 1 Metrics

```bash
POST /api/type1/metrics
{
  "glucose_readings": [120, 150, 140, 130, 145],
  "timestamps": ["2024-01-01T08:00:00Z", ...]
}
```

**Response**:
```json
{
  "time_in_range": 0.75,
  "time_below_range": 0.05,
  "time_above_range": 0.20,
  "mean_glucose": 145.5,
  "glucose_variability_cv": 28.3,
  "stability_score": 82.5,
  "assessment": {
    "level": "good",
    "message": "...",
    "recommendations": [...]
  }
}
```

## Safety & Compliance

### Safety Features:
- ✅ Automatic safety limit enforcement
- ✅ Hypoglycemia risk detection
- ✅ Dose validation
- ✅ Explainable AI (all recommendations include explanations)
- ✅ Doctor oversight required for critical decisions

### Type 1 Diabetes Specific:
- ✅ Designed specifically for Type 1 Diabetes
- ✅ No endogenous insulin assumptions
- ✅ Focus on stability and control
- ✅ Time in Range optimization
- ✅ Glucose variability monitoring

## Development Workflow

1. **Backend Development**
   ```bash
   cd backend
   pip install -r requirements.txt
   python run.py
   ```

2. **Frontend Development** (Doctor Portal)
   ```bash
   cd frontend
   npm install
   npm start
   ```

3. **ML Model Development**
   - Create model in `backend/app/ml_framework/models/`
   - Register in `models/__init__.py`
   - Train with your data
   - Test via API endpoints

## Key Differences from Previous Version

1. **API-First**: No direct patient UI, designed for external app integration
2. **ML Framework**: Extensible plugin system for custom models
3. **Type 1 Focus**: Specific metrics and features for Type 1 diabetes
4. **Enhanced Doctor Portal**: Better visualizations, real-time updates, interactive dashboards
5. **Improved Explainability**: Detailed reasoning steps and feature importance
6. **Stability Metrics**: Time in Range, Glucose Variability, Stability Score

## Next Steps

1. **Integrate with External Apps**: Use API endpoints to connect mobile apps, CGM systems
2. **Develop Custom Models**: Create models specific to your patient population
3. **Train on Real Data**: Replace synthetic data with real patient data
4. **Enhance Doctor Portal**: Add more visualizations and interactive features
5. **Deploy**: Use Docker or cloud deployment (see DEPLOYMENT.md)

## Support

- **ML Workflow**: See `ML_WORKFLOW.md`
- **API Documentation**: `GET /api/docs`
- **Deployment**: See `DEPLOYMENT.md`

