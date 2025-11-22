# Interactive Web Interface Guide

## 🚀 Quick Start

### 1. Install Streamlit
```bash
pip install streamlit
```

Or install all requirements:
```bash
pip install -r requirements_app.txt
```

### 2. Run the Interface
```bash
python run_app.py
```

Or directly:
```bash
streamlit run app.py
```

### 3. Access the Interface
The interface will automatically open in your browser at:
- **URL**: http://localhost:8501

## 📋 Features

### Single Patient Assessment
- Input patient vital signs and lifestyle data
- Get instant risk assessment
- View detailed explanations
- See actionable recommendations
- Export results as JSON

### Batch Assessment
- Upload CSV file with multiple patients
- Process all patients at once
- View summary statistics
- Download results (JSON and CSV)

### System Information
- View system architecture
- See clinical rules
- Check model status
- Understand risk levels

## 🎯 Usage

### Single Patient Assessment

1. Navigate to "Single Patient Assessment"
2. Fill in patient data:
   - Vital signs (Glucose, HbA1c, BMI, etc.)
   - Blood pressure
   - Lifestyle factors (Activity, Sleep, etc.)
   - Optional advanced features
3. Click "Assess Patient Risk"
4. View results:
   - Risk score and level
   - Explanation
   - Recommendation
   - Contributing factors
5. Download results if needed

### Batch Assessment

1. Navigate to "Batch Assessment"
2. Prepare CSV file with patient data
3. Upload the CSV file
4. Click "Assess All Patients"
5. View summary and statistics
6. Download results

### CSV Format for Batch Assessment

Your CSV file should include columns like:
- Glucose_Level
- HbA1c
- BMI
- Heart_Rate
- Blood_Pressure_Systolic
- Blood_Pressure_Diastolic
- Activity_Level
- Calories_Burned
- Sleep_Duration
- Step_Count
- Medication_Intake
- Diet_Quality_Score
- Stress_Level
- (and other features as required)

## 🎨 Interface Features

- **Color-coded Risk Levels**:
  - 🔴 Critical (Red)
  - 🟠 High (Orange)
  - 🟡 Moderate (Yellow)
  - 🟢 Low (Green)

- **Real-time Assessment**: Instant results
- **Export Options**: Download results in JSON or CSV
- **Responsive Design**: Works on desktop and tablet

## 🔧 Troubleshooting

### Interface won't start
```bash
# Check if Streamlit is installed
pip install streamlit

# Try running directly
streamlit run app.py
```

### Model not found
- The interface will work with rule engine only
- Train the model first: `python run_full_pipeline.py`
- Model should be at: `models/cds_model.pkl`

### Port already in use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

## 📱 Access from Other Devices

To access from other devices on your network:

1. Find your local IP address
2. Run with:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```
3. Access from other device: `http://YOUR_IP:8501`

## 🎓 Example Workflow

1. **Start the interface**: `python run_app.py`
2. **Select "Single Patient Assessment"**
3. **Enter patient data**:
   - Glucose: 280 mg/dL
   - HbA1c: 9.5%
   - Other vital signs
4. **Click "Assess Patient Risk"**
5. **Review results**:
   - Risk Score: 0.700 (HIGH)
   - Explanation: Hyperglycemia detected
   - Recommendation: Check ketones, administer insulin
6. **Download results** for medical records

## 💡 Tips

- Use the sidebar to navigate between pages
- Check system status in the sidebar
- Export results for documentation
- Use batch assessment for multiple patients
- Review system information to understand the rules

## 🔒 Security Note

This interface is for local use. For production deployment:
- Add authentication
- Use HTTPS
- Secure the server
- Follow healthcare data regulations (HIPAA, etc.)

