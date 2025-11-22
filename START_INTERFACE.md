# 🚀 Start Interactive Interface

## Quick Start

### Option 1: Using the launcher (Recommended)
```bash
python run_app.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run app.py
```

### Option 3: With custom port
```bash
streamlit run app.py --server.port 8502
```

## What You'll See

The interface will open in your browser at **http://localhost:8501** with:

### 📋 Navigation Pages:
1. **Single Patient Assessment** - Input patient data and get instant risk assessment
2. **Batch Assessment** - Upload CSV and process multiple patients
3. **System Information** - View system details and rules
4. **About** - System documentation

### 🎯 Key Features:
- ✅ Interactive forms for patient data input
- ✅ Real-time risk assessment
- ✅ Color-coded risk levels (Low/Moderate/High/Critical)
- ✅ Detailed explanations and recommendations
- ✅ Export results (JSON/CSV)
- ✅ Batch processing support

## Example Usage

1. **Start the interface**: `python run_app.py`
2. **Select "Single Patient Assessment"**
3. **Enter patient data** (e.g., Glucose: 280, HbA1c: 9.5)
4. **Click "Assess Patient Risk"**
5. **View results** with risk score, explanation, and recommendation
6. **Download results** if needed

## Troubleshooting

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Interface not loading?**
- Check if Streamlit is installed: `pip install streamlit`
- Check if model exists: `models/cds_model.pkl`
- System will work with rule engine only if model not found

**Need help?**
- See `INTERACTIVE_INTERFACE.md` for detailed guide
- Check `README.md` for system documentation

