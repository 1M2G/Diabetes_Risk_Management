"""
Interactive Web Interface for Hybrid Clinical Decision Support System
Built with Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Try to import plotly for better visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cds import HybridCDS, ClinicalRuleEngine
from src.cds.inference import load_cds_model, format_output_for_display

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk & Insulin Management CDS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Hybrid Clinical Decision Support System for Diabetes Risk Assessment"
    }
)

# Custom CSS for better clarity
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .risk-critical {
        background-color: #dc3545;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        border: 3px solid #c82333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-high {
        background-color: #fd7e14;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        border: 3px solid #e8650e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-moderate {
        background-color: #ffc107;
        color: #000;
        padding: 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        border: 3px solid #e0a800;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background-color: #28a745;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        border: 3px solid #218838;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #dee2e6;
        text-align: center;
    }
    .approval-section {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 3px solid #1f77b4;
        margin: 2rem 0;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.1rem;
        padding: 0.75rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cds_system():
    """Load CDS system with caching"""
    try:
        model_path = Path("models/cds_model.pkl")
        if model_path.exists():
            cds, feature_names = load_cds_model(model_path)
            return cds, feature_names, None
        else:
            # Create CDS without trained model (rules only)
            rule_engine = ClinicalRuleEngine()
            cds = HybridCDS(rule_engine=rule_engine)
            return cds, [], "Model not found. Using rule engine only."
    except Exception as e:
        rule_engine = ClinicalRuleEngine()
        cds = HybridCDS(rule_engine=rule_engine)
        return cds, [], f"Error loading model: {str(e)}. Using rule engine only."

def validate_patient_data(patient_data):
    """Validate patient data in real-time"""
    warnings = []
    errors = []
    
    # Critical validations
    if patient_data.get('Glucose_Level', 0) < 40 or patient_data.get('Glucose_Level', 0) > 400:
        errors.append(f"⚠️ CRITICAL: Glucose Level ({patient_data.get('Glucose_Level', 0)}) is outside safe range (40-400 mg/dL)")
    elif patient_data.get('Glucose_Level', 0) < 54:
        warnings.append("⚠️ WARNING: Severe hypoglycemia detected!")
    elif patient_data.get('Glucose_Level', 0) > 400:
        warnings.append("⚠️ WARNING: Severe hyperglycemia detected!")
    
    if patient_data.get('HbA1c', 0) > 10.0:
        warnings.append(f"⚠️ WARNING: Critically high HbA1c ({patient_data.get('HbA1c', 0)}%)")
    
    if patient_data.get('Blood_Pressure_Systolic', 0) >= 180:
        warnings.append("⚠️ WARNING: Hypertensive crisis detected!")
    
    if patient_data.get('Blood_Pressure_Systolic', 0) < 90:
        warnings.append("⚠️ WARNING: Hypotension detected!")
    
    return errors, warnings

def get_quick_risk_estimate(patient_data, cds):
    """Get quick risk estimate for real-time preview"""
    try:
        # Create minimal patient data
        patient_df = pd.DataFrame([patient_data])
        result = cds.predict(patient_df, return_explanation=False)
        return result.risk_score, result.risk_level
    except:
        return None, None

def get_risk_color_class(risk_level):
    """Get CSS class for risk level"""
    risk_classes = {
        'critical': 'risk-critical',
        'high': 'risk-high',
        'moderate': 'risk-moderate',
        'low': 'risk-low'
    }
    return risk_classes.get(risk_level.lower(), 'risk-low')

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Hybrid Clinical Decision Support System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Diabetes Risk & Insulin Management</h2>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Select Page",
            ["Single Patient Assessment", "Batch Assessment", "Medical Review & Approval", "System Information", "About"]
        )
        
        st.markdown("---")
        st.header("ℹ️ System Status")
        
        # Load CDS system
        cds, feature_names, model_status = load_cds_system()
        
        if model_status:
            st.warning(model_status)
        else:
            st.success("✓ System Ready")
            st.info(f"Features: {len(feature_names)}")
    
    # Main content based on selected page
    if page == "Single Patient Assessment":
        single_patient_page(cds, feature_names)
    elif page == "Batch Assessment":
        batch_assessment_page(cds, feature_names)
    elif page == "Medical Review & Approval":
        medical_review_page(cds, feature_names)
    elif page == "System Information":
        system_info_page()
    elif page == "About":
        about_page()

def single_patient_page(cds, feature_names):
    """Single patient assessment interface"""
    st.header("👤 Single Patient Risk Assessment")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vital Signs")
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=120.0, step=1.0)
        hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, value=72.0, step=1.0)
        
        st.subheader("Blood Pressure")
        bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=250.0, value=120.0, step=1.0)
        bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=80.0, step=1.0)
    
    with col2:
        st.subheader("Lifestyle & Activity")
        activity_level = st.slider("Activity Level (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        calories_burned = st.number_input("Calories Burned", min_value=0.0, max_value=10000.0, value=2000.0, step=100.0)
        sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=18.0, value=8.0, step=0.5)
        step_count = st.number_input("Step Count", min_value=0, max_value=100000, value=10000, step=100)
        
        st.subheader("Other Factors")
        medication_intake = st.slider("Medication Intake (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        diet_quality = st.slider("Diet Quality Score (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        stress_level = st.slider("Stress Level (0-100)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    
    # Additional features if model requires them
    with st.expander("⚙️ Advanced Features (Optional - Auto-filled if not provided)"):
        col3, col4 = st.columns(2)
        with col3:
            predicted_progression = st.number_input("Predicted Progression", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="prog_input")
            timestamp_hour = st.number_input("Hour of Day", min_value=0, max_value=23, value=datetime.now().hour, step=1, key="hour_input")
        with col4:
            timestamp_dow = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=datetime.now().weekday(), step=1, key="dow_input")
            timestamp_month = st.number_input("Month", min_value=1, max_value=12, value=datetime.now().month, step=1, key="month_input")
            timestamp_is_weekend = st.checkbox("Is Weekend", value=datetime.now().weekday() >= 5, key="weekend_input")
            timestamp_ts = st.number_input("Timestamp (Unix)", min_value=0, value=int(datetime.now().timestamp()), step=1, key="ts_input")
    
    # Real-time validation and preview
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 REAL-TIME VALIDATION & PREVIEW</div>', unsafe_allow_html=True)
    
    # Prepare current patient data for validation
    current_patient_data = {
        'Glucose_Level': glucose,
        'HbA1c': hba1c,
        'BMI': bmi,
        'Heart_Rate': heart_rate,
        'Blood_Pressure_Systolic': bp_systolic,
        'Blood_Pressure_Diastolic': bp_diastolic,
        'Activity_Level': activity_level,
        'Calories_Burned': calories_burned,
        'Sleep_Duration': sleep_duration,
        'Step_Count': step_count,
        'Medication_Intake': medication_intake,
        'Diet_Quality_Score': diet_quality,
        'Stress_Level': stress_level
    }
    
    # Real-time validation
    errors, warnings = validate_patient_data(current_patient_data)
    
    if errors:
        for error in errors:
            st.error(error)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    if not errors and not warnings:
        st.success("✅ All vital signs within acceptable ranges")
    
    # Quick risk preview
    col_preview1, col_preview2 = st.columns(2)
    with col_preview1:
        if st.button("🔍 Quick Risk Preview", use_container_width=True, help="Get instant risk estimate without full assessment"):
            with st.spinner("Calculating..."):
                risk_score, risk_level = get_quick_risk_estimate(current_patient_data, cds)
                if risk_score is not None:
                    risk_class = get_risk_color_class(risk_level)
                    st.markdown(f'<div class="{risk_class}">QUICK PREVIEW<br/>Risk: {risk_level.upper()}<br/>Score: {risk_score:.3f}</div>', unsafe_allow_html=True)
                else:
                    st.info("Quick preview not available. Please run full assessment.")
    
    with col_preview2:
        # Show key indicators
        st.markdown("**Key Indicators:**")
        glucose_status = "🔴 Critical" if glucose < 54 or glucose > 400 else "🟠 High" if glucose > 250 or glucose < 70 else "🟢 Normal"
        hba1c_status = "🔴 Critical" if hba1c >= 10.0 else "🟠 High" if hba1c >= 8.5 else "🟢 Good" if hba1c < 7.0 else "🟡 Elevated"
        bp_status = "🔴 Critical" if bp_systolic >= 180 or bp_systolic < 90 else "🟠 High" if bp_systolic >= 130 else "🟢 Normal"
        
        st.write(f"Glucose: {glucose_status} ({glucose:.1f} mg/dL)")
        st.write(f"HbA1c: {hba1c_status} ({hba1c:.1f}%)")
        st.write(f"BP: {bp_status} ({bp_systolic:.0f}/{bp_diastolic:.0f} mmHg)")
    
    st.markdown("---")
    
    # Predict button with auto-refresh option
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        auto_refresh = st.checkbox("🔄 Auto-refresh on data change", value=False, help="Automatically update assessment when data changes")
    with col_btn2:
        assess_clicked = st.button("🔍 Assess Patient Risk", type="primary", use_container_width=True)
    
    # Auto-refresh logic
    if auto_refresh:
        # Store current values in session state
        current_values = {
            'glucose': glucose, 'hba1c': hba1c, 'bmi': bmi, 'heart_rate': heart_rate,
            'bp_systolic': bp_systolic, 'bp_diastolic': bp_diastolic,
            'activity': activity_level, 'calories': calories_burned,
            'sleep': sleep_duration, 'steps': step_count,
            'medication': medication_intake, 'diet': diet_quality, 'stress': stress_level
        }
        
        if 'last_values' not in st.session_state:
            st.session_state.last_values = {}
        
        # Check if values changed
        values_changed = current_values != st.session_state.last_values
        
        if values_changed and any(st.session_state.last_values):
            st.session_state.last_values = current_values
            assess_clicked = True  # Trigger assessment
        else:
            st.session_state.last_values = current_values
    
    # Perform assessment
    if assess_clicked or (auto_refresh and 'values_changed' in locals() and values_changed):
        with st.spinner("🔄 Analyzing patient data in real-time..."):
            # Prepare patient data
            patient_data = {
                'Glucose_Level': glucose,
                'Heart_Rate': heart_rate,
                'Activity_Level': activity_level,
                'Calories_Burned': calories_burned,
                'Sleep_Duration': sleep_duration,
                'Step_Count': step_count,
                'Medication_Intake': medication_intake,
                'Diet_Quality_Score': diet_quality,
                'Stress_Level': stress_level,
                'BMI': bmi,
                'HbA1c': hba1c,
                'Blood_Pressure_Systolic': bp_systolic,
                'Blood_Pressure_Diastolic': bp_diastolic,
                'Predicted_Progression': predicted_progression,
                'Timestamp_hour': float(timestamp_hour),
                'Timestamp_dow': float(timestamp_dow),
                'Timestamp_month': float(timestamp_month),
                'Timestamp_is_weekend': 1.0 if timestamp_is_weekend else 0.0,
                'Timestamp_ts': float(timestamp_ts)
            }
            
            # Add any missing features with default values
            if feature_names:
                for feat in feature_names:
                    if feat not in patient_data:
                        patient_data[feat] = 0.0
            
            try:
                patient_df = pd.DataFrame([patient_data])
                result = cds.predict(patient_df, return_explanation=True)
                
                # Display results with approval section
                display_results(result, show_approval=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check that all required features are provided.")

def display_results(result, show_approval=True):
    """Display CDS results in a formatted way"""
    st.markdown("---")
    st.markdown('<div class="section-header">📊 ASSESSMENT RESULTS</div>', unsafe_allow_html=True)
    
    # Risk score and level - larger, clearer display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Risk Score", f"{result.risk_score:.3f}", help="Risk score on a scale of 0.0 (low) to 1.0 (critical)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        risk_class = get_risk_color_class(result.risk_level)
        st.markdown(f'<div class="{risk_class}">RISK LEVEL<br/>{result.risk_level.upper()}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Confidence", f"{result.confidence:.1%}", help="System confidence in this assessment")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        ml_pred = result.ml_prediction
        st.metric("ML Prediction", f"{ml_pred:.2f}", help="Raw machine learning model prediction")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction source - clearer display
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 PREDICTION SOURCE</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**Machine Learning Prediction:** {result.ml_prediction:.2f}")
        st.markdown("This is the raw prediction from the trained ML model.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if result.rule_override:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown(f"**⚠️ Rule Override: YES**")
            st.markdown(f"**Rule Triggered:** {result.rule_used}")
            st.markdown("Clinical rule has overridden ML prediction due to critical condition detected.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("**✓ Rule Override: NO**")
            st.markdown("ML prediction is being used. No critical rules triggered.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Explanation - clearer display
    st.markdown("---")
    st.markdown('<div class="section-header">💡 WHY PATIENT IS AT RISK</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><strong>Explanation:</strong><br/>{result.explanation}</div>', unsafe_allow_html=True)
    
    # Recommendation - clearer display
    st.markdown("---")
    st.markdown('<div class="section-header">📋 CLINICAL RECOMMENDATION</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><strong>Recommended Action:</strong><br/>{result.recommendation}</div>', unsafe_allow_html=True)
    
    # Contributing factors - clearer table display
    if result.contributing_factors:
        st.markdown("---")
        st.markdown('<div class="section-header">📈 TOP CONTRIBUTING FACTORS</div>', unsafe_allow_html=True)
        factors_df = pd.DataFrame(result.contributing_factors[:10])
        if not factors_df.empty:
            factors_df['Impact Direction'] = factors_df['impact'].apply(lambda x: '🔴 Increases Risk' if x > 0 else '🟢 Decreases Risk')
            factors_df['Impact Magnitude'] = factors_df['impact'].abs()
            factors_df = factors_df.sort_values('Impact Magnitude', ascending=False)
            
            # Display as a clear table
            display_df = pd.DataFrame({
                'Feature': factors_df['feature'],
                'Current Value': factors_df['value'].round(2),
                'Impact': factors_df['Impact Direction'],
                'Magnitude': factors_df['Impact Magnitude'].round(3)
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Visual bar chart if plotly available
            if PLOTLY_AVAILABLE:
                try:
                    top_factors = factors_df.head(8).copy()
                    fig = px.bar(
                        top_factors,
                        x='Impact Magnitude',
                        y='feature',
                        orientation='h',
                        color=(top_factors['impact'] > 0),
                        color_discrete_map={True: '#dc3545', False: '#28a745'},
                        title="Feature Impact on Risk Score (Top 8)",
                        labels={'Impact Magnitude': 'Impact Magnitude', 'feature': 'Feature'}
                    )
                    fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass  # Skip if error
    
    # Medical worker approval section
    if show_approval:
        st.markdown("---")
        st.markdown('<div class="approval-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">👨‍⚕️ MEDICAL WORKER REVIEW & APPROVAL</div>', unsafe_allow_html=True)
        
        # Initialize session state for approvals
        if 'assessments' not in st.session_state:
            st.session_state.assessments = []
        
        assessment_id = f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        col1, col2 = st.columns(2)
        with col1:
            medical_worker_name = st.text_input("👤 Medical Worker Name *", placeholder="Enter your name", key="mw_name")
            medical_worker_role = st.selectbox("🏥 Role *", ["Physician", "Nurse", "Endocrinologist", "Clinical Specialist", "Other"], key="mw_role")
        with col2:
            review_date = st.date_input("📅 Review Date", value=datetime.now().date(), key="review_date")
            review_time = st.time_input("⏰ Review Time", value=datetime.now().time(), key="review_time")
        
        st.markdown("---")
        st.subheader("Review Assessment")
        
        # Medical worker can override risk level
        override_risk = st.radio(
            "Do you agree with the system's risk assessment? *",
            ["✅ Agree with System Assessment", "⚠️ Modify Risk Level", "❌ Disagree - Requires Manual Review"],
            horizontal=False,
            key="override_risk"
        )
        
        modified_risk_level = None
        modification_reason = None
        disagreement_reason = None
        
        if override_risk == "⚠️ Modify Risk Level":
            risk_index = ["low", "moderate", "high", "critical"].index(result.risk_level) if result.risk_level in ["low", "moderate", "high", "critical"] else 0
            modified_risk_level = st.selectbox(
                "Select Modified Risk Level *",
                ["low", "moderate", "high", "critical"],
                index=risk_index,
                key="modified_risk"
            )
            modification_reason = st.text_area("Reason for Modification *", placeholder="Explain why you are modifying the risk level...", key="mod_reason")
        elif override_risk == "❌ Disagree - Requires Manual Review":
            disagreement_reason = st.text_area("Reason for Disagreement *", placeholder="Explain why you disagree with the assessment...", key="disagree_reason")
        
        # Additional notes
        clinical_notes = st.text_area(
            "📝 Clinical Notes / Additional Comments",
            placeholder="Add any additional clinical observations, notes, or comments...",
            height=100,
            key="clinical_notes"
        )
        
        # Action taken
        action_taken = st.selectbox(
            "✅ Action Taken *",
            ["Pending Review", "Recommendation Accepted", "Recommendation Modified", "Recommendation Rejected", "Referred to Specialist", "Emergency Protocol Initiated"],
            key="action_taken"
        )
        
        # Approval buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve Assessment", type="primary", use_container_width=True, key="btn_approve"):
                if medical_worker_name:
                    approval_data = {
                        'assessment_id': assessment_id,
                        'timestamp': datetime.now().isoformat(),
                        'medical_worker': medical_worker_name,
                        'role': medical_worker_role,
                        'review_date': str(review_date),
                        'review_time': str(review_time),
                        'system_risk_score': result.risk_score,
                        'system_risk_level': result.risk_level,
                        'override_risk': override_risk,
                        'modified_risk_level': modified_risk_level if override_risk == "⚠️ Modify Risk Level" else result.risk_level,
                        'modification_reason': modification_reason if override_risk == "⚠️ Modify Risk Level" else None,
                        'disagreement_reason': disagreement_reason if override_risk == "❌ Disagree - Requires Manual Review" else None,
                        'clinical_notes': clinical_notes,
                        'action_taken': action_taken,
                        'status': 'Approved',
                        'original_result': result.to_dict()
                    }
                    st.session_state.assessments.append(approval_data)
                    st.success(f"✅ Assessment approved by {medical_worker_name} ({medical_worker_role})")
                    st.balloons()
                else:
                    st.warning("⚠️ Please enter your name before approving.")
        
        with col2:
            if st.button("📋 Save for Review", use_container_width=True, key="btn_save"):
                if medical_worker_name:
                    approval_data = {
                        'assessment_id': assessment_id,
                        'timestamp': datetime.now().isoformat(),
                        'medical_worker': medical_worker_name,
                        'role': medical_worker_role,
                        'review_date': str(review_date),
                        'review_time': str(review_time),
                        'system_risk_score': result.risk_score,
                        'system_risk_level': result.risk_level,
                        'override_risk': override_risk,
                        'modified_risk_level': modified_risk_level if override_risk == "⚠️ Modify Risk Level" else result.risk_level,
                        'modification_reason': modification_reason if override_risk == "⚠️ Modify Risk Level" else None,
                        'disagreement_reason': disagreement_reason if override_risk == "❌ Disagree - Requires Manual Review" else None,
                        'clinical_notes': clinical_notes,
                        'action_taken': action_taken,
                        'status': 'Saved for Review',
                        'original_result': result.to_dict()
                    }
                    st.session_state.assessments.append(approval_data)
                    st.info(f"📋 Assessment saved for review by {medical_worker_name}")
                    # Auto-refresh to show updated list
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter your name before saving.")
        
        with col3:
            if st.button("🔄 Reset Form", use_container_width=True, key="btn_reset"):
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download results
    st.markdown("---")
    st.markdown('<div class="section-header">💾 EXPORT RESULTS</div>', unsafe_allow_html=True)
    results_dict = result.to_dict()
    results_json = json.dumps(results_dict, indent=2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Assessment (JSON)",
            data=results_json,
            file_name=f"cds_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    with col2:
        # Create a formatted text report
        report_text = f"""CLINICAL DECISION SUPPORT SYSTEM - ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RISK ASSESSMENT:
- Risk Score: {result.risk_score:.3f}
- Risk Level: {result.risk_level.upper()}
- Confidence: {result.confidence:.1%}

PREDICTION SOURCE:
- ML Prediction: {result.ml_prediction:.2f}
- Rule Override: {'Yes' if result.rule_override else 'No'}
- Rule Used: {result.rule_used or 'None'}

EXPLANATION:
{result.explanation}

RECOMMENDATION:
{result.recommendation}

CONTRIBUTING FACTORS:
"""
        if result.contributing_factors:
            for i, factor in enumerate(result.contributing_factors[:5], 1):
                report_text += f"{i}. {factor['feature']}: {factor['value']:.2f} ({'increases' if factor['impact'] > 0 else 'decreases'} risk by {abs(factor['impact']):.3f})\n"
        
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report_text,
            file_name=f"cds_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def batch_assessment_page(cds, feature_names):
    """Batch assessment interface"""
    st.header("📊 Batch Patient Assessment")
    
    st.info("Upload a CSV file with patient data. Each row should represent one patient.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File loaded: {df.shape[0]} patients, {df.shape[1]} features")
            
            # Display preview
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            if st.button("🔍 Assess All Patients", type="primary", use_container_width=True):
                with st.spinner("Processing patients..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        try:
                            # Ensure all required features are present
                            patient_data = row.to_dict()
                            if feature_names:
                                for feat in feature_names:
                                    if feat not in patient_data:
                                        patient_data[feat] = 0.0
                            
                            patient_df = pd.DataFrame([patient_data])
                            result = cds.predict(patient_df, return_explanation=True)
                            results.append(result.to_dict())
                        except Exception as e:
                            st.warning(f"Error processing patient {idx+1}: {str(e)}")
                            results.append(None)
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Display results
                    st.markdown("---")
                    st.header("📊 Batch Assessment Results")
                    
                    # Create summary dataframe
                    summary_data = []
                    for i, res in enumerate(results):
                        if res:
                            summary_data.append({
                                'Patient': i + 1,
                                'Risk Score': res['risk_score'],
                                'Risk Level': res['risk_level'],
                                'ML Prediction': res['ml_prediction'],
                                'Rule Override': 'Yes' if res['rule_override'] else 'No',
                                'Rule Used': res.get('rule_used', 'None'),
                                'Confidence': res['confidence']
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Patients", len(summary_df))
                        with col2:
                            critical_count = (summary_df['Risk Level'] == 'critical').sum()
                            st.metric("Critical", critical_count, delta=None)
                        with col3:
                            high_count = (summary_df['Risk Level'] == 'high').sum()
                            st.metric("High Risk", high_count)
                        with col4:
                            avg_risk = summary_df['Risk Score'].mean()
                            st.metric("Avg Risk Score", f"{avg_risk:.3f}")
                        
                        # Download results
                        st.markdown("---")
                        results_json = json.dumps(results, indent=2)
                        st.download_button(
                            label="Download All Results (JSON)",
                            data=results_json,
                            file_name=f"batch_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                        # Download summary CSV
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="Download Summary (CSV)",
                            data=csv,
                            file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to begin batch assessment")

def system_info_page():
    """System information page"""
    st.header("ℹ️ System Information")
    
    st.subheader("System Architecture")
    st.markdown("""
    The Hybrid Clinical Decision Support System combines:
    - **Machine Learning Models**: Lightweight models for risk prediction
    - **Rule Engine**: Evidence-based clinical rules for critical conditions
    - **Explainability**: SHAP and LIME integration for model interpretability
    """)
    
    st.subheader("Clinical Rules")
    st.markdown("""
    The system includes 9 evidence-based clinical rules:
    1. Critical Glucose Levels (hypo/hyperglycemia)
    2. Hypoglycemia Risk
    3. Hyperglycemia Risk
    4. HbA1c Critical Levels
    5. Blood Pressure Crisis
    6. Ketoacidosis Risk
    7. Rapid Glucose Changes
    8. Multiple Risk Factors
    9. Medication Non-compliance
    """)
    
    st.subheader("Risk Levels")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="risk-low">LOW<br/>0.0 - 0.4</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="risk-moderate">MODERATE<br/>0.4 - 0.6</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="risk-high">HIGH<br/>0.6 - 0.8</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="risk-critical">CRITICAL<br/>0.8 - 1.0</div>', unsafe_allow_html=True)
    
    st.subheader("Model Information")
    model_path = Path("models/cds_model.pkl")
    if model_path.exists():
        st.success("✓ Model loaded successfully")
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                if 'metrics' in model_data:
                    st.json(model_data['metrics'])
        except:
            st.info("Model details not available")
    else:
        st.warning("Model not found. System is using rule engine only.")

def medical_review_page(cds, feature_names):
    """Medical worker review and approval page"""
    st.header("👨‍⚕️ Medical Review & Approval Dashboard")
    
    # Initialize session state
    if 'assessments' not in st.session_state:
        st.session_state.assessments = []
    
    st.markdown("""
    <div class="info-box">
    <strong>Medical Review Dashboard</strong><br/>
    This page allows medical workers to review, approve, and manage patient assessments.
    View all assessments, filter by status, and track approval history.
    </div>
    """, unsafe_allow_html=True)
    
    # Filter options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.selectbox("Filter by Status", ["All", "Approved", "Saved for Review", "Pending"], key="filter_status")
    with col2:
        filter_role = st.selectbox("Filter by Role", ["All", "Physician", "Nurse", "Endocrinologist", "Clinical Specialist"], key="filter_role")
    with col3:
        sort_by = st.selectbox("Sort By", ["Date (Newest)", "Date (Oldest)", "Risk Level", "Medical Worker"], key="sort_by")
    
    # Display assessments
    assessments = st.session_state.assessments
    
    if filter_status != "All":
        assessments = [a for a in assessments if a['status'] == filter_status]
    if filter_role != "All":
        assessments = [a for a in assessments if a.get('role') == filter_role]
    
    # Sort assessments
    if sort_by == "Date (Newest)":
        assessments = sorted(assessments, key=lambda x: x['timestamp'], reverse=True)
    elif sort_by == "Date (Oldest)":
        assessments = sorted(assessments, key=lambda x: x['timestamp'])
    elif sort_by == "Risk Level":
        risk_order = {'critical': 4, 'high': 3, 'moderate': 2, 'low': 1}
        assessments = sorted(assessments, key=lambda x: risk_order.get(x.get('system_risk_level', 'low'), 0), reverse=True)
    elif sort_by == "Medical Worker":
        assessments = sorted(assessments, key=lambda x: x.get('medical_worker', ''))
    
    if assessments:
        st.markdown(f"### Found {len(assessments)} assessment(s)")
        
        for idx, assessment in enumerate(assessments):
            with st.expander(f"Assessment #{idx+1} - {assessment.get('medical_worker', 'Unknown')} ({assessment.get('role', 'Unknown')}) - {assessment.get('status', 'Unknown')}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**System Assessment:**")
                    st.write(f"- Risk Score: {assessment.get('system_risk_score', 0):.3f}")
                    st.write(f"- Risk Level: {assessment.get('system_risk_level', 'unknown').upper()}")
                    st.write(f"- Date: {assessment.get('review_date', 'N/A')} {assessment.get('review_time', 'N/A')}")
                
                with col2:
                    st.markdown("**Medical Worker Review:**")
                    st.write(f"- Name: {assessment.get('medical_worker', 'N/A')}")
                    st.write(f"- Role: {assessment.get('role', 'N/A')}")
                    st.write(f"- Status: {assessment.get('status', 'N/A')}")
                    st.write(f"- Action: {assessment.get('action_taken', 'N/A')}")
                
                if assessment.get('override_risk') == "⚠️ Modify Risk Level":
                    st.warning(f"⚠️ Risk Level Modified: {assessment.get('modified_risk_level', 'N/A')}")
                    st.write(f"**Reason:** {assessment.get('modification_reason', 'N/A')}")
                elif assessment.get('override_risk') == "❌ Disagree - Requires Manual Review":
                    st.error(f"❌ Disagreement with Assessment")
                    st.write(f"**Reason:** {assessment.get('disagreement_reason', 'N/A')}")
                
                if assessment.get('clinical_notes'):
                    st.markdown("**Clinical Notes:**")
                    st.info(assessment.get('clinical_notes', ''))
                
                # View full assessment
                if st.button(f"View Full Assessment #{idx+1}", key=f"view_{idx}"):
                    st.json(assessment.get('original_result', {}))
                
                # Delete assessment
                if st.button(f"Delete Assessment #{idx+1}", key=f"delete_{idx}"):
                    st.session_state.assessments.remove(assessment)
                    st.rerun()
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(st.session_state.assessments)
        approved = len([a for a in st.session_state.assessments if a.get('status') == 'Approved'])
        saved = len([a for a in st.session_state.assessments if a.get('status') == 'Saved for Review'])
        modified = len([a for a in st.session_state.assessments if a.get('override_risk') == "⚠️ Modify Risk Level"])
        
        with col1:
            st.metric("Total Assessments", total)
        with col2:
            st.metric("Approved", approved)
        with col3:
            st.metric("Saved for Review", saved)
        with col4:
            st.metric("Modified", modified)
        
        # Export all assessments
        st.markdown("---")
        if st.button("📥 Export All Assessments", use_container_width=True):
            export_data = json.dumps(st.session_state.assessments, indent=2)
            st.download_button(
                label="Download All Assessments (JSON)",
                data=export_data,
                file_name=f"all_assessments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Clear all button
        if st.button("🗑️ Clear All Assessments", use_container_width=True):
            st.session_state.assessments = []
            st.rerun()
    
    else:
        st.info("📋 No assessments found. Assessments will appear here after medical workers review and approve them from the Single Patient Assessment page.")

def about_page():
    """About page"""
    st.header("📖 About")
    
    st.markdown("""
    ## Hybrid Clinical Decision Support System
    
    An integrated clinical decision support system for diabetes risk assessment and insulin management.
    
    ### Features
    - **Hybrid Prediction**: Combines ML predictions with rule-based logic
    - **Rule Overrides**: Clinical rules can override ML for critical conditions
    - **Explainability**: Provides explanations for risk assessments
    - **Actionable Recommendations**: Clinical recommendations for patient care
    
    ### Outputs
    1. **Risk Score** (0-1 scale)
    2. **Why patient is at risk** (explanations)
    3. **What should be done** (recommendations)
    
    ### Disclaimer
    This system is for research and educational purposes. Clinical decisions should always involve healthcare professionals.
    """)
    
    st.markdown("---")
    st.subheader("📚 Documentation")
    st.markdown("""
    - See `README.md` for complete documentation
    - See `RUN_SYSTEM.md` for usage instructions
    - See notebooks for detailed analysis
    """)

if __name__ == "__main__":
    main()

