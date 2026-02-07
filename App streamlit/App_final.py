import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from PIL import Image, ImageOps
from fpdf import FPDF
import tempfile
import os
import datetime
import json
import math

# =========================
# CONFIGURATION
# =========================
st.set_page_config(
    page_title="Alzheimer Diagnostic System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # =========================
# CUSTOM CSS - WHITE BACKGROUND VERSION
# =========================
st.markdown("""
<style>
    /* White background for entire app */
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Main containers */
    .main-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Headers with blue color - KEEP ORIGINAL COLORS */
    h1 {
        color: #00FFCC !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 20px;
    }
    
    h2 {
        color: #00FFCC !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    
    h3 {
        color: #00FFCC !important;
        font-weight: 600 !important;
    }
    
    h4 {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    /* Text colors - BLACK TEXT */
    p, label, li, div, span {
        color: #000000 !important;
    }
    
    /* Text in metric labels */
    div[data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    
    /* Input labels and text */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stRadio label, .stSlider label {
        color: #000000 !important;
    }
    
    /* Cards and containers - WHITE BACKGROUND WITH BLACK TEXT */
    .card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    
    /* Metrics and results - KEEP ORIGINAL GREEN COLOR */
    div[data-testid="stMetricValue"] {
        color: #00FFCC !important;
        font-weight: bold !important;
        font-size: 1.5rem !important;
    }
    
    /* Buttons - KEEP ORIGINAL STYLE */
    button {
        border: 1px solid #00FFCC !important;
        color: #00FFCC !important;
        background-color: transparent !important;
    }
    
    button:hover {
        background-color: rgba(0, 255, 204, 0.1) !important;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #00FFCC !important;
        color: #000000 !important;
        border: none !important;
        font-weight: 500 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #00E6B8 !important;
    }
    
    /* File uploader - WHITE BACKGROUND */
    .stFileUploader {
        border: 2px dashed #D1D5DB;
        border-radius: 8px;
        padding: 20px;
        background-color: #FFFFFF;
        color: #000000 !important;
    }
    
    .stFileUploader label {
        color: #000000 !important;
    }
    
    /* Progress bars - KEEP ORIGINAL GREEN */
    .stProgress > div > div > div > div {
        background-color: #00FFCC;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #F3F4F6;
        padding: 5px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 5px;
        padding: 10px 16px;
        font-weight: 500;
        color: #000000 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #000000 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Input widgets - WHITE BACKGROUND */
    .stSlider, .stSelectbox, .stRadio, .stTextInput, .stNumberInput {
        background-color: white;
        color: #000000 !important;
    }
    
    /* Streamlit widget labels */
    .stSlider label, .stSelectbox label, .stRadio label {
        color: #000000 !important;
    }
    
    /* Risk level colors - KEEP ORIGINAL */
    .risk-high {
        color: #FF3333 !important;
        font-weight: bold;
    }
    
    .risk-moderate {
        color: #F59E0B !important;
        font-weight: bold;
    }
    
    .risk-low {
        color: #00FF00 !important;
        font-weight: bold;
    }
    
    /* Feature impact colors - KEEP ORIGINAL */
    .impact-increase {
        color: #FF3333 !important;
        font-weight: bold;
    }
    
    .impact-decrease {
        color: #00FF00 !important;
        font-weight: bold;
    }
    
    /* ML-specific styles - ADJUST FOR WHITE BACKGROUND */
    .main-header {
        font-size: 2.5rem;
        color: #00FFCC !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #00FFCC !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .feature-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .feature-card h4 {
        color: #000000 !important;
        font-weight: 600;
    }
    .feature-card p {
        color: #000000 !important;
    }
    .risk-high {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        color: #000000 !important;
    }
    .risk-moderate {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        color: #000000 !important;
    }
    .risk-low {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        color: #000000 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .impact-positive {
        background-color: #FECACA;
        color: #991B1B !important;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .impact-negative {
        background-color: #BBF7D0;
        color: #166534 !important;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    
    /* Plotly charts background */
    .js-plotly-plot .plotly, .plot-container {
        background-color: white !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        color: #000000 !important;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Success/Error/Warning messages */
    .stAlert {
        color: #000000 !important;
    }
    
    /* All streamlit text elements */
    .stMarkdown, .stText, .stCode {
        color: #000000 !important;
    }
    
    /* Specifically target any remaining black text issues */
    * {
        color: #000000 !important;
    }
    
    /* Override any specific streamlit elements that might still show white text */
    .element-container, .block-container, .main {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# TAB DEFINITION
# =========================
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🤖 ML Prediction", "🖼️ CV Detection"])

# =========================
# TAB 1: HOME
# =========================
with tab1:
    st.markdown("<h1 class='main-header'>🧠 Alzheimer Diagnostic System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Welcome to the Alzheimer Diagnostic System
        
        This comprehensive platform combines two advanced AI technologies for Alzheimer's disease assessment:
        
        ### 🤖 ML Prediction Module
        - Uses machine learning with XGBoost
        - Analyzes 20+ clinical and lifestyle factors
        - Provides risk probability and detailed analysis
        - SHAP explanations for feature importance
        
        ### 🖼️ CV Detection Module
        - Deep learning with TensorFlow/Keras
        - MRI image classification
        - Grad-CAM heatmap visualization
        - 4-class classification system
        
        ### How to Use:
        1. **ML Prediction Tab**: Enter patient clinical data to get risk assessment
        2. **CV Detection Tab**: Upload MRI images for image-based diagnosis
        
        ### Clinical Disclaimer
        ⚠️ This tool provides risk stratification based on statistical models.
        It does not provide medical diagnosis. All clinical decisions should be made
        by qualified healthcare professionals considering the complete clinical context.
        """)
    
    with col2:
        st.image("image.jpg", 
                caption="Brain MRI Analysis")
        
        st.markdown("""
        ### Supported Features
        
        ✅ **ML Prediction:**
        - Demographic factors (Age, Gender, BMI)
        - Medical history (Diabetes, Hypertension)
        - Genetic factors (Family history, APOE-ε4)
        - Lifestyle factors (Smoking, Alcohol, Exercise)
        - Environmental factors (Air pollution, Living area)
        
        ✅ **CV Detection:**
        - MRI image classification
        - 4 classes: Non Demented, Very Mild, Mild, Moderate
        - Heatmap visualization
        - PDF report generation
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 System Performance
   
    - **CV Model**: 4-class classification, Grad-CAM visualization
    - **Training Data**: 74,283 patients, 20 clinical features
    """)


# =========================
# TAB 2: ML PREDICTION (from appML.py) WITH PDF REPORT
# =========================
with tab2:
    # =========================
    # LOAD MODELS
    # =========================
    @st.cache_resource
    def load_models():
        """Load XGBoost model và calibrator từ files"""
        try:
            # Load XGBoost model
            model = xgb.Booster()
            model.load_model('xgb_model.json')
            
            # Load calibrator
            with open('calibrator.pkl', 'rb') as f:
                calibrator = pickle.load(f)
            
            # Load feature info
            with open('feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
            
            # Load feature weights
            with open('feature_weights.pkl', 'rb') as f:
                feature_weights = pickle.load(f)
            
            st.success("✅ Models loaded successfully!")
            return model, calibrator, feature_info, feature_weights
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return None, None, None, None

    # Load models
    model, calibrator, feature_info, feature_weights = load_models()

    # =========================
    # PREDICTION FUNCTION
    # =========================
    def predict_with_xgboost(model, calibrator, X_array):
        """Predict với XGBoost native API và calibration"""
        dmatrix = xgb.DMatrix(X_array)
        y_pred_raw = model.predict(dmatrix)
        y_pred_calibrated = calibrator.predict(y_pred_raw)
        return float(y_pred_calibrated[0])

    # =========================
    # FEATURE MAPPING
    # =========================
    binary_map = {"No": 0, "Yes": 1}
    gender_map = {"Female": 0, "Male": 1}
    urban_map = {"Rural": 0, "Urban": 1}
    alcohol_map = {"Never": 0, "Occasionally": 1, "Regularly": 2}
    sleep_map = {"Poor": 2, "Average": 1, "Good": 0}
    social_map = {"Low": 2, "Medium": 1, "High": 0}
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    physical_map = {"Low": 2, "Medium": 1, "High": 0}
    depression_map = {"Low": 0, "Medium": 1, "High": 2}
    air_pollution_map = {"Low": 0, "Medium": 1, "High": 2}
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    dietary_map = {"Unhealthy": 2, "Average": 1, "Healthy": 0}

    st.markdown("<h1 class='main-header'>🧠 Alzheimer's Disease Risk Assessment System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
        <p>Advanced machine learning system for predicting Alzheimer's disease risk using 20 clinical and lifestyle factors.</p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # INPUT FORM
    # =========================
    st.markdown("<h2 class='sub-header'>📋 Patient Clinical Assessment</h2>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### **Demographic & Medical Profile**")
            
            age = st.slider("**Age** (years)", 40, 100, 65, 
                           help="Patient age in years. Risk increases significantly after 65.")
            bmi = st.slider("**BMI** (kg/m²)", 15.0, 40.0, 25.0, 0.1,
                           help="Body Mass Index. Both low and high BMI may increase risk.")
            education = st.slider("**Education Level** (years)", 0, 20, 10,
                                 help="Years of formal education.")
            cognitive = st.slider("**Cognitive Test Score**", 30, 100, 75,
                                 help="Cognitive assessment score (higher is better).")
            
            gender = st.radio("**Gender**", ["Female", "Male"])
            urban = st.radio("**Living Environment**", ["Rural", "Urban"])
        
        with col2:
            st.markdown("#### **Medical & Genetic Factors**")
            
            diabetes = st.radio("**Diabetes**", ["No", "Yes"])
            hypertension = st.radio("**Hypertension**", ["No", "Yes"])
            cholesterol = st.radio("**High Cholesterol**", ["No", "Yes"])
            
            family_history = st.radio("**Family History of Alzheimer's**", ["No", "Yes"])
            genetic_risk = st.radio("**APOE-ε4 Allele**", ["No", "Yes"])
            
            physical = st.selectbox("**Physical Activity Level**", ["High", "Medium", "Low"])
            smoking = st.selectbox("**Smoking Status**", ["Never", "Former", "Current"])
        
        with col3:
            st.markdown("#### **Lifestyle & Psychosocial Factors**")
            
            alcohol = st.selectbox("**Alcohol Consumption**", ["Never", "Occasionally", "Regularly"])
            sleep = st.selectbox("**Sleep Quality**", ["Good", "Average", "Poor"])
            dietary = st.selectbox("**Dietary Habits**", ["Healthy", "Average", "Unhealthy"])
            
            social = st.selectbox("**Social Engagement**", ["High", "Medium", "Low"])
            stress = st.selectbox("**Stress Levels**", ["Low", "Medium", "High"])
            depression = st.selectbox("**Depression Level**", ["Low", "Medium", "High"])
            
            air_pollution = st.selectbox("**Air Pollution Exposure**", ["Low", "Medium", "High"])
        
        # Submit button
        submit_button = st.form_submit_button("🔍 **Calculate Risk Probability**", 
                                             type="primary",
                                             use_container_width=True)

    # =========================
    # PROCESS PREDICTION
    # =========================
    age_input=age
    if age <66: age=0
    elif age<76: age=1
    else :age=2
    cog_input=cognitive
    if cognitive<50: cognitive=1
    elif cognitive<75: cognitive=2
    else: cognitve=3

    if submit_button and model is not None:
        # Map inputs to numeric values
        input_data = [
            age,
            binary_map[family_history],
            binary_map[genetic_risk],
            bmi,
            education,
            cognitive,
            physical_map[physical],
            smoking_map[smoking],
            alcohol_map[alcohol],
            binary_map[diabetes],
            binary_map[hypertension],
            binary_map[cholesterol],
            depression_map[depression],
            sleep_map[sleep],
            dietary_map[dietary],
            air_pollution_map[air_pollution],
            gender_map[gender],
            social_map[social],
            stress_map[stress],
            urban_map[urban]
        ]
        
        # Convert to array
        X_array = np.array([input_data])
        
        # Make prediction
        with st.spinner("🧠 Analyzing clinical profile..."):
            try:
                probability = predict_with_xgboost(model, calibrator, X_array)
                probability_percent = probability * 100
                
                # Determine risk level
                if probability_percent < 30:
                    risk_level = "Low"
                    risk_color = "#10B981"
                    risk_class = "risk-low"
                elif probability_percent < 60:
                    risk_level = "Moderate"
                    risk_color = "#F59E0B"
                    risk_class = "risk-moderate"
                else:
                    risk_level = "High"
                    risk_color = "#DC2626"
                    risk_class = "risk-high"
                
                # =========================
                # RESULTS DISPLAY
                # =========================
                st.markdown("<h2 class='sub-header'>📊 Risk Assessment Results</h2>", unsafe_allow_html=True)
                
                # Main metrics in cards
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='margin: 0; font-size: 2rem;'>{probability_percent:.1f}%</h3>
                        <p style='margin: 0; opacity: 0.9;'>Risk Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_result2:
                    st.markdown(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, {risk_color} 0%, {risk_color}80 100%);'>
                        <h3 style='margin: 0; font-size: 2rem;'>{risk_level}</h3>
                        <p style='margin: 0; opacity: 0.9;'>Risk Category</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_result3:
                    # Risk interpretation
                    if risk_level == "High":
                        interpretation = "Clinical Intervention"
                    elif risk_level == "Moderate":
                        interpretation = "Enhanced Surveillance"
                    else:
                        interpretation = "Routine Monitoring"
                    
                    st.markdown(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%);'>
                        <h3 style='margin: 0; font-size: 1.5rem;'>{interpretation}</h3>
                        <p style='margin: 0; opacity: 0.9;'>Recommended Action</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # =========================
                # VISUALIZATION 1: GAUGE CHART
                # =========================
                st.markdown("<h3 class='sub-header'>📈 Risk Probability Distribution</h3>", unsafe_allow_html=True)
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability_percent,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={'suffix': '%', 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': risk_color, 'thickness': 0.4},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#D1FAE5'},
                            {'range': [30, 60], 'color': '#FEF3C7'},
                            {'range': [60, 100], 'color': '#FEE2E2'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': probability_percent
                        }
                    },
                    title={'text': "Alzheimer's Risk Probability", 'font': {'size': 24, 'color': '#1E3A8A'}},
                    delta={'reference': 50, 'relative': True, 'position': "top"}
                ))
                
                fig_gauge.update_layout(
                    height=350,
                    margin=dict(l=50, r=50, t=100, b=50),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # =========================
                # VISUALIZATION 2: FEATURE CONTRIBUTIONS
                # =========================
                st.markdown("<h3 class='sub-header'>🔍 Feature Impact Analysis</h3>", unsafe_allow_html=True)
                
                try:
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_array)[0]
                    
                    # Create enhanced DataFrame
                    shap_df = pd.DataFrame({
                        'Feature': feature_info['feature_names'],
                        'SHAP Value': shap_values,
                        'Absolute Impact': np.abs(shap_values),
                        'Direction': ['Increases Risk' if x > 0 else 'Decreases Risk' for x in shap_values],
                        'Contribution %': (np.abs(shap_values) / np.sum(np.abs(shap_values))) * 100
                    }).sort_values('Absolute Impact', ascending=False)
                    
                    # Assign categories
                    def assign_category(feature):
                        feature_lower = str(feature).lower()
                        
                        if any(keyword in feature_lower for keyword in ['age', 'gender', 'bmi', 'education']):
                            return 'Demographic & Medical'
                        elif any(keyword in feature_lower for keyword in ['family', 'genetic', 'apoe']):
                            return 'Genetic & Family History'
                        elif any(keyword in feature_lower for keyword in ['diabetes', 'hypertension', 'cholesterol']):
                            return 'Medical Conditions'
                        elif any(keyword in feature_lower for keyword in ['cognitive', 'depression', 'stress']):
                            return 'Cognitive & Mental'
                        elif any(keyword in feature_lower for keyword in ['alcohol', 'smoking', 'physical', 'sleep', 'dietary']):
                            return 'Lifestyle Behaviors'
                        elif any(keyword in feature_lower for keyword in ['air', 'pollution', 'social', 'urban', 'rural']):
                            return 'Environmental & Social'
                        else:
                            return 'Other'
                    
                    shap_df['Category'] = shap_df['Feature'].apply(assign_category)
                    
                    # Create beautiful visualization
                    col_viz1, col_viz2 = st.columns([2, 1])
                    
                    with col_viz1:
                        # Horizontal bar chart với top 10 features
                        top_features = shap_df.head(10)
                        
                        fig_bars = px.bar(
                            top_features,
                            x='SHAP Value',
                            y='Feature',
                            orientation='h',
                            color='Category',
                            color_discrete_map={
                                'Demographic & Medical': '#3B82F6',
                                'Genetic & Family History': '#EF4444',
                                'Medical Conditions': '#F59E0B',
                                'Cognitive & Mental': '#8B5CF6',
                                'Lifestyle Behaviors': '#10B981',
                                'Environmental & Social': '#EC4899'
                            },
                            title='<b>Top 10 Feature Contributions</b><br><span style="font-size: 0.8em; color: #6B7280">Impact on Alzheimer\'s Risk Prediction</span>',
                            labels={'SHAP Value': 'Impact Score'},
                            hover_data=['Contribution %', 'Direction']
                        )
                        
                        fig_bars.update_layout(
                            height=450,
                            yaxis={'categoryorder': 'total ascending'},
                            xaxis_title="Impact Score (Positive = Increases Risk)",
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.02
                            )
                        )
                        
                        # Add zero line
                        fig_bars.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig_bars, use_container_width=True)
                    
                    with col_viz2:
                        # Donut chart for contribution percentages by category
                        category_contrib = shap_df.groupby('Category')['Absolute Impact'].sum().reset_index()
                        category_contrib['Percentage'] = (category_contrib['Absolute Impact'] / category_contrib['Absolute Impact'].sum()) * 100
                        
                        fig_donut = px.pie(
                            category_contrib,
                            values='Absolute Impact',
                            names='Category',
                            hole=0.5,
                            title='<b>Impact by Category</b>',
                            color='Category',
                            color_discrete_map={
                                'Demographic & Medical': '#3B82F6',
                                'Genetic & Family History': '#EF4444',
                                'Medical Conditions': '#F59E0B',
                                'Cognitive & Mental': '#8B5CF6',
                                'Lifestyle Behaviors': '#10B981',
                                'Environmental & Social': '#EC4899'
                            }
                        )
                        
                        fig_donut.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate="<b>%{label}</b><br>Contribution: %{percent}<br>Total Impact: %{value:.3f}"
                        )
                        
                        fig_donut.update_layout(
                            height=450,
                            showlegend=False,
                            margin=dict(t=50, b=20),
                            annotations=[dict(text='Category<br>Impact', x=0.5, y=0.5, font_size=14, showarrow=False)]
                        )
                        
                        st.plotly_chart(fig_donut, use_container_width=True)
                    
                    # =========================
                    # FEATURE CARDS DETAIL VIEW
                    # =========================
                    st.markdown("#### 📋 Detailed Feature Analysis by Category")
                    
                    # Group by category
                    for category in ['Genetic & Family History', 'Demographic & Medical', 'Cognitive & Mental', 
                                    'Lifestyle Behaviors', 'Medical Conditions', 'Environmental & Social']:
                        category_features = shap_df[shap_df['Category'] == category]
                        
                        if not category_features.empty:
                            st.markdown(f"**{category}**")
                            
                            # Tạo các hàng, mỗi hàng 4 cards
                            num_features = len(category_features)
                            
                            for i in range(0, num_features, 4):
                                # Lấy nhóm 4 features
                                batch = category_features.iloc[i:i+4]
                                cols = st.columns(4)
                                
                                for col_idx, (_, row) in zip(range(4), batch.iterrows()):
                                    with cols[col_idx]:
                                        impact_class = "impact-positive" if row['SHAP Value'] > 0 else "impact-negative"
                                        impact_text = "Increases Risk" if row['SHAP Value'] > 0 else "Decreases Risk"
                                        
                                        st.markdown(f"""
                                        <div class='feature-card'>
                                            <h4 style='margin: 0 0 0.5rem 0;'>{row['Feature']}</h4>
                                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                                <span style='color: #6B7280; font-size: 0.85rem;'>{impact_text}</span>
                                                <span class='{impact_class}'>
                                                    {row['SHAP Value']:+.3f}
                                                </span>
                                            </div>
                                            <p style='color: #6B7280; font-size: 0.8rem; margin: 0;'>
                                                {row['Contribution %']:.1f}% of total impact
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
                    
                    # Data table in expander
                    with st.expander("📊 View Complete Feature Impact Table"):
                        display_df = shap_df.copy()
                        display_df['SHAP Value'] = display_df['SHAP Value'].round(4)
                        display_df['Contribution %'] = display_df['Contribution %'].round(2)
                        
                        # Apply color formatting
                        def color_shap(val):
                            color = '#DC2626' if val > 0 else '#059669'
                            return f'color: {color}; font-weight: bold'
                        
                        styled_df = display_df.style.applymap(color_shap, subset=['SHAP Value'])
                        
                        st.dataframe(
                            styled_df.format({
                                'SHAP Value': '{:+.4f}',
                                'Contribution %': '{:.2f}%'
                            }),
                            use_container_width=True,
                            column_order=['Feature', 'Category', 'SHAP Value', 'Direction', 'Contribution %', 'Absolute Impact'],
                            height=400
                        )
                        
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")
                    # Fallback visualization
                    st.info("Displaying feature importance from model weights")
                    
                    if feature_weights is not None:
                        weights_df = pd.DataFrame({
                            'Feature': feature_info['feature_names'],
                            'Importance': feature_weights
                        }).sort_values('Importance', ascending=False)
                        
                        fig_fallback = px.bar(
                            weights_df.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance Weights',
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        fig_fallback.update_layout(height=400)
                        st.plotly_chart(fig_fallback, use_container_width=True)
                
                # =========================
                # PATIENT SUMMARY
                # =========================
                st.markdown("<h3 class='sub-header'>👤 Patient Clinical Summary</h3>", unsafe_allow_html=True)
                
                # Create summary in cards
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    st.markdown(f"""
                    <div class='feature-card'>
                        <h4>Demographic Profile</h4>
                        <p><strong>Age:</strong> {age_input} years</p>
                        <p><strong>Gender:</strong> {gender}</p>
                        <p><strong>BMI:</strong> {bmi:.1f} kg/m²</p>
                        <p><strong>Education:</strong> {education} years</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with summary_cols[1]:
                    st.markdown(f"""
                    <div class='feature-card'>
                        <h4>Medical & Genetic</h4>
                        <p><strong>Family History:</strong> {family_history}</p>
                        <p><strong>APOE-ε4:</strong> {genetic_risk}</p>
                        <p><strong>Diabetes:</strong> {diabetes}</p>
                        <p><strong>Hypertension:</strong> {hypertension}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with summary_cols[2]:
                    st.markdown(f"""
                    <div class='feature-card'>
                        <h4>Lifestyle Factors</h4>
                        <p><strong>Alcohol:</strong> {alcohol}</p>
                        <p><strong>Smoking:</strong> {smoking}</p>
                        <p><strong>Physical Activity:</strong> {physical}</p>
                        <p><strong>Sleep Quality:</strong> {sleep}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with summary_cols[3]:
                    st.markdown(f"""
                    <div class='feature-card'>
                        <h4>Psychosocial & Cognitive</h4>
                        <p><strong>Cognitive Score:</strong> {cog_input}</p>
                        <p><strong>Social Engagement:</strong> {social}</p>
                        <p><strong>Stress Level:</strong> {stress}</p>
                        <p><strong>Depression:</strong> {depression}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # =========================
                # CLINICAL RECOMMENDATIONS
                # =========================
                st.markdown("<h3 class='sub-header'>💡 Clinical Recommendations</h3>", unsafe_allow_html=True)
                
                if risk_level == "High":
                    rec_color = "#FEF2F2"
                    rec_border = "#DC2626"
                    recommendations = [
                        "Immediate referral to neurologist for comprehensive evaluation",
                        "Schedule cognitive screening (MMSE, MoCA) within 2 weeks",
                        "Consider neuroimaging (MRI/PET) if not recently performed",
                        "Implement comprehensive lifestyle modifications program",
                        "Quarterly follow-up assessments recommended",
                        "Consider pharmacological interventions evaluation"
                    ]
                elif risk_level == "Moderate":
                    rec_color = "#FFFBEB"
                    rec_border = "#F59E0B"
                    recommendations = [
                        "Annual cognitive screening recommended",
                        "Lifestyle counseling on sleep, stress, and nutrition management",
                        "Encourage regular physical and social engagement activities",
                        "Monitor for cognitive changes every 6 months",
                        "Consider baseline neuropsychological testing",
                        "Cardiovascular risk factor optimization"
                    ]
                else:
                    rec_color = "#F0FDF4"
                    rec_border = "#10B981"
                    recommendations = [
                        "Continue healthy lifestyle practices",
                        "Regular physical activity (150 mins/week)",
                        "Cognitive stimulation activities (reading, puzzles)",
                        "Annual health check-ups with cognitive screening",
                        "Maintain social connections and engagement",
                        "Balanced Mediterranean-style diet recommended"
                    ]
                
                st.markdown(f"""
                <div style='
                    background-color: {rec_color};
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    border-left: 6px solid {rec_border};
                '>
                    <h4 style='color: #1F2937; margin-top: 0;'>For <strong>{risk_level} Risk</strong> ({probability_percent:.1f}%)</h4>
                    <ul style='color: #4B5563;'>
                        {''.join([f'<li style="margin-bottom: 0.5rem;">{rec}</li>' for rec in recommendations])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                                # =========================
                # EXPORT ASSESSMENT REPORT - PDF ONLY
                # =========================
                st.markdown("<h3 class='sub-header'>📄 Export Assessment Report</h3>", unsafe_allow_html=True)
                
                try:
                    # ======================================
                    # CREATE PDF REPORT ONLY
                    # ======================================
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Title
                    pdf.set_font("Arial", 'B', 20)
                    pdf.cell(0, 10, "ALZHEIMER'S DISEASE RISK ASSESSMENT", ln=True, align='C')
                    pdf.ln(5)
                    
                    # Date and ID
                    pdf.set_font("Arial", size=10)
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    report_id = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
                    pdf.cell(0, 5, f"Report ID: {report_id}", ln=True)
                    pdf.cell(0, 5, f"Assessment Date: {current_time}", ln=True)
                    pdf.ln(10)
                    
                    # Section 1: Risk Summary
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 8, "1. RISK ASSESSMENT SUMMARY", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.ln(5)
                    
                    pdf.cell(50, 6, "Risk Probability:", 0, 0)
                    pdf.cell(0, 6, f"{probability_percent:.2f}%", 0, 1)
                    
                    pdf.cell(50, 6, "Risk Category:", 0, 0)
                    pdf.cell(0, 6, risk_level, 0, 1)
                    
                    pdf.cell(50, 6, "Recommended Action:", 0, 0)
                    pdf.cell(0, 6, interpretation, 0, 1)
                    pdf.ln(10)
                    
                    # Section 2: Patient Profile
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 8, "2. PATIENT CLINICAL PROFILE", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.ln(5)
                    
                    # Demographics
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 6, "Demographic Factors:", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Age: {age_input} years", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Gender: {gender}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- BMI: {bmi:.1f} kg/m2", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Education: {education} years", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Cognitive Score: {cog_input}", ln=True)
                    pdf.ln(5)
                    
                    # Medical
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 6, "Medical & Genetic Factors:", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Family History: {family_history}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- APOE-e4 Allele: {genetic_risk}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Diabetes: {diabetes}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Hypertension: {hypertension}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Cholesterol: {cholesterol}", ln=True)
                    pdf.ln(5)
                    
                    # Lifestyle
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 6, "Lifestyle Factors:", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Alcohol: {alcohol}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Smoking: {smoking}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Physical Activity: {physical}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Sleep Quality: {sleep}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Dietary Habits: {dietary}", ln=True)
                    pdf.ln(5)
                    
                    # Psychosocial
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 6, "Psychosocial & Environmental:", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Social Engagement: {social}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Stress Level: {stress}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Depression: {depression}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Air Pollution: {air_pollution}", ln=True)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.cell(0, 5, f"- Living Environment: {urban}", ln=True)
                    pdf.ln(10)
                    
                    # Section 3: Recommendations
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 8, "3. CLINICAL RECOMMENDATIONS", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.ln(5)
                    
                    for i, rec in enumerate(recommendations, 1):
                        pdf.multi_cell(0, 5, f"{i}. {rec}")
                        pdf.ln(3)
                    
                    pdf.ln(10)
                    
                    # Section 4: Disclaimer (đơn giản)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "DISCLAIMER", ln=True)
                    pdf.set_font("Arial", 'I', 9)
                    disclaimer_text = "This report is generated by an AI clinical decision support system. It is intended to assist clinical judgment, not replace it. Always combine with clinical evaluation and professional judgment."
                    pdf.multi_cell(0, 4, disclaimer_text)
                    
                    # Generate PDF bytes
                    pdf_bytes = pdf.output(dest="S").encode('latin-1', 'ignore')
                    
                    # Single download button for PDF only
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"alzheimer_assessment_{report_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ PDF report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)[:100]}")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)[:100]}")
# =========================
# TAB 3: CV DETECTION (from appCV.py)
# =========================
with tab3:
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('model.keras')

    def process_image(image):
        image = ImageOps.fit(image, (128, 128), Image.LANCZOS)
        img_array = np.asarray(image).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_activation', pred_index=None):
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            
            if isinstance(preds, list):
                preds = preds[0]
            
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
                
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        if isinstance(last_conv_layer_output, list):
            last_conv_layer_output = last_conv_layer_output[0]

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def create_pdf(image, result, confidence):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 20)
        pdf.cell(0, 10, "ALZHEIMER DIAGNOSIS REPORT", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pdf.cell(0, 10, f"Date: {current_time}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Diagnosis Result: {result}", ln=True)
        pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
        pdf.ln(10)
        
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            pdf.image(tmp_file_path, x=10, y=80, w=100)
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
        return pdf.output(dest="S").encode("latin-1")

    try:
        model = load_model()
    except:
        st.error("Model file not found. Please check my_alzheimer_model.keras")
        st.stop()

    CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

    st.title("Alzheimer Prediction System")

    file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

    if file is not None:
        original_image = Image.open(file).convert('RGB')
        processed_img = process_image(original_image)
        
        predictions = model.predict(processed_img)
        scores = tf.nn.softmax(predictions[0]).numpy()
        
        final_class = CLASS_NAMES[np.argmax(scores)]
        confidence = np.max(scores) * 100

        col1, col2, col3 = st.columns([1.2, 1, 1])

        with col1:
            st.subheader("Diagnostic Result")
            color_code = "#00FF00" if final_class == "Non Demented" else "#FF3333"
            st.markdown(f"<h2 style='color: {color_code} !important;'>{final_class}</h2>", unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.2f}%")
            
            st.write("---")
            st.write("Probability Distribution:")
            for i, name in enumerate(CLASS_NAMES):
                val = int(scores[i] * 100)
                st.write(f"{name}")
                st.progress(val)
                st.caption(f"{val}%")
                
            pdf_bytes = create_pdf(original_image, final_class, confidence)
            st.download_button(label="Download Medical Report (PDF)", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

        with col2:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)

        with col3:
            st.subheader("Heatmap")
            try:
                heatmap = make_gradcam_heatmap(processed_img, model)
                
                heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
                
                heatmap_resized = np.uint8(255 * heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                superimposed = cv2.addWeighted(np.array(original_image), 0.6, heatmap_colored, 0.4, 0)
                st.image(superimposed, caption="Red areas indicate disease features", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9em; padding: 1rem;">
    <p><strong>⚠️ Clinical Disclaimer</strong></p>
    <p style="margin-bottom: 0.5rem;">This tool provides risk stratification based on statistical models.</p>
    <p style="margin-bottom: 0.5rem;">It does not provide medical diagnosis. All clinical decisions should be made</p>
    <p>by qualified healthcare professionals considering the complete clinical context.</p>
    <p style="margin-top: 1rem; color: #9CA3AF; font-size: 0.8em;">
        Alzheimer Diagnostic System v1.0 | AI Clinical Decision Support
    </p>
</div>
""", unsafe_allow_html=True)