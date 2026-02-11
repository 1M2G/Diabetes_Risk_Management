# ML Model Development Workflow

## Overview

This system provides a flexible framework for developing custom ML models for Type 1 Diabetes insulin management. The framework ensures safety, explainability, and consistency across all models.

## ML Procedure Flow

### Phase 1: Model Design

1. **Define Model Purpose**
   - What type of predictions? (basal, bolus, correction)
   - Target population? (Type 1 Diabetes specific)
   - Performance goals? (accuracy, safety)

2. **Identify Required Features**
   - Core: glucose_level, insulin_dosage
   - Patient profile: age, weight, hba1c, diabetes_type
   - Context: food_intake, physical_activity, time_of_day
   - Historical: glucose_trend, time_since_last_meal

3. **Choose Algorithm**
   - Regression: RandomForest, GradientBoosting, XGBoost
   - Time-series: LSTM, GRU (for historical patterns)
   - Ensemble: Combine multiple models

### Phase 2: Model Implementation

1. **Create Model Class**
   ```python
   from app.ml_framework.base_model import BaseInsulinModel
   
   class MyCustomModel(BaseInsulinModel):
       MODEL_ID = 'my_custom_model'
       
       def __init__(self, model_name="My Model", model_version="1.0.0"):
           super().__init__(model_name, model_version)
           # Initialize your model components
       
       def get_supported_features(self):
           return ['glucose_level', 'food_intake', ...]
       
       def train(self, training_data, **kwargs):
           # Your training logic
           pass
       
       def predict(self, patient_data, patient_profile):
           # Your prediction logic
           pass
       
       def explain(self, patient_data, patient_profile):
           # Your explanation logic
           pass
   ```

2. **Implement Required Methods**
   - `train()`: Train the model on data
   - `predict()`: Make dose recommendations
   - `explain()`: Provide explanations
   - `get_supported_features()`: List supported features

3. **Add Safety Checks**
   - Validate inputs
   - Apply safety limits
   - Check for edge cases (hypoglycemia, hyperglycemia)

### Phase 3: Training

1. **Prepare Training Data**
   ```python
   import pandas as pd
   
   # Load your dataset
   data = pd.read_csv('your_training_data.csv')
   
   # Required columns:
   # - glucose_level (mg/dL)
   # - insulin_dosage (units) - target variable
   # - food_intake (grams)
   # - age, bmi, hba1c, weight_kg
   # - insulin_sensitivity_factor, carb_ratio, basal_rate
   ```

2. **Train the Model**
   ```python
   from app.ml_framework.models.my_custom_model import MyCustomModel
   
   model = MyCustomModel()
   metrics = model.train(data)
   
   # Save the model
   model.save_model('ml_models/my_custom_model.pkl')
   ```

3. **Evaluate Performance**
   - Check training metrics (R², MAE, RMSE)
   - Validate safety limits are respected
   - Test on holdout set
   - Ensure explainability works

### Phase 4: Registration

1. **Register Your Model**
   ```python
   from app.ml_framework.model_registry import model_registry
   from app.ml_framework.models.my_custom_model import MyCustomModel
   
   model_registry.register_model(MyCustomModel, 'my_custom_model')
   ```

2. **Set as Active Model**
   ```python
   model_registry.set_active_model('my_custom_model')
   ```

### Phase 5: Testing

1. **Unit Tests**
   - Test prediction with various inputs
   - Test safety limits
   - Test explainability

2. **Integration Tests**
   - Test API endpoints
   - Test with real patient data
   - Validate outputs

3. **Safety Tests**
   - Test edge cases (very high/low glucose)
   - Test missing data handling
   - Test boundary conditions

## Model Requirements

### Safety Requirements

1. **Input Validation**
   - All models must validate inputs
   - Check glucose level ranges (20-600 mg/dL)
   - Verify diabetes type is Type 1

2. **Safety Limits**
   - Maximum bolus dose (typically 30% of TDD)
   - Maximum correction dose
   - No dose if glucose < 70 mg/dL (unless treating hyperglycemia)

3. **Error Handling**
   - Graceful fallback if model fails
   - Clear error messages
   - Log all errors

### Explainability Requirements

1. **Feature Importance**
   - Show which features influenced the decision
   - Provide SHAP values or equivalent

2. **Reasoning Steps**
   - Step-by-step explanation
   - Clear rationale for dose recommendation

3. **Risk Assessment**
   - Identify safety concerns
   - Provide risk level (low, moderate, high, critical)

### Performance Requirements

1. **Accuracy**
   - Target: MAE < 2 units for bolus doses
   - Target: R² > 0.7

2. **Stability**
   - Consistent predictions for similar inputs
   - Low variance in recommendations

3. **Speed**
   - Prediction time < 100ms
   - Explanation generation < 500ms

## Example: Creating a Custom Model

### Step 1: Create Model File

Create `backend/app/ml_framework/models/my_model.py`:

```python
from app.ml_framework.base_model import BaseInsulinModel
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

class MyInsulinModel(BaseInsulinModel):
    MODEL_ID = 'my_model'
    
    def __init__(self):
        super().__init__("My Insulin Model", "1.0.0")
        self.scaler = StandardScaler()
        self.model = None
    
    def get_supported_features(self):
        return ['glucose_level', 'food_intake', 'age', 'weight_kg']
    
    def train(self, training_data, **kwargs):
        # Prepare features
        X = training_data[['glucose_level', 'food_intake', 'age', 'weight_kg']]
        y = training_data['insulin_dosage']
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model = RandomForestRegressor(n_estimators=100)
        self.model.fit(X_scaled, y)
        
        # Evaluate
        score = self.model.score(X_scaled, y)
        
        self.is_trained = True
        self.training_metadata = {'score': score}
        
        return self.training_metadata
    
    def predict(self, patient_data, patient_profile):
        # Validate
        is_valid, error = self.validate_input(patient_data, patient_profile)
        if not is_valid:
            return {'error': error, 'recommended_dose': 0.0}
        
        # Prepare features
        features = pd.DataFrame([{
            'glucose_level': patient_data['glucose_level'],
            'food_intake': patient_data.get('food_intake', 0),
            'age': patient_profile.get('age', 45),
            'weight_kg': patient_profile.get('weight_kg', 70)
        }])
        
        # Predict
        X_scaled = self.scaler.transform(features)
        dose = self.model.predict(X_scaled)[0]
        
        # Apply safety limits
        safety_limits = self.calculate_safety_limits(patient_profile)
        dose = min(dose, safety_limits['max_bolus'])
        
        return {
            'recommended_dose': round(dose, 2),
            'confidence': 0.8,
            'prediction_type': 'bolus',
            'explanation': f'Recommended dose: {dose:.2f} units',
            'safety_flags': []
        }
    
    def explain(self, patient_data, patient_profile):
        return {
            'summary': 'Model explanation',
            'reasoning_steps': [],
            'feature_importance': {}
        }
```

### Step 2: Register Model

In `backend/app/ml_framework/models/__init__.py`:

```python
from app.ml_framework.models.my_model import MyInsulinModel
model_registry.register_model(MyInsulinModel, 'my_model')
```

### Step 3: Train Model

```python
import pandas as pd
from app.ml_framework.models.my_model import MyInsulinModel

# Load data
data = pd.read_csv('training_data.csv')

# Train
model = MyInsulinModel()
metrics = model.train(data)
print(f"Training complete: {metrics}")

# Save
model.save_model('ml_models/my_model.pkl')
```

### Step 4: Use Model

```python
from app.ml_framework.model_registry import model_registry

# Set active model
model_registry.set_active_model('my_model')

# Get model and use
model = model_registry.get_active_model()
result = model.predict(
    patient_data={'glucose_level': 150, 'food_intake': 60},
    patient_profile={'age': 30, 'weight_kg': 70, 'diabetes_type': 'Type 1'}
)
```

## Best Practices

1. **Always validate inputs** - Check ranges and required fields
2. **Apply safety limits** - Never exceed maximum safe doses
3. **Provide explanations** - Users need to understand recommendations
4. **Handle errors gracefully** - Fallback to rule-based if model fails
5. **Test thoroughly** - Test edge cases and boundary conditions
6. **Document your model** - Explain assumptions and limitations
7. **Version your models** - Track changes and improvements
8. **Monitor performance** - Track accuracy and safety in production

## Type 1 Diabetes Specific Considerations

1. **No endogenous insulin** - Patients rely entirely on exogenous insulin
2. **Rapid glucose changes** - Monitor trends, not just current value
3. **Meal timing critical** - Bolus timing affects glucose response
4. **Activity effects** - Exercise can cause delayed hypoglycemia
5. **Stability focus** - Minimize glucose variability
6. **Time in range** - Target 70-180 mg/dL for 70%+ of time

## Resources

- Base Model Interface: `backend/app/ml_framework/base_model.py`
- Model Registry: `backend/app/ml_framework/model_registry.py`
- Example Models: `backend/app/ml_framework/models/`
- API Endpoints: `backend/app/routes.py` (ML endpoints)

