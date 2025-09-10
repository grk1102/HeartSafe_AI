import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="HeartSafe AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Tailwind and animations
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(to bottom, #FFFFFF, #E0F2FE); }
        .title { 
            font-size: 2.5rem; font-weight: bold; color: #3B82F6; 
            animation: fadeIn 1s ease-in; 
        }
        .subtitle { font-size: 1.5rem; color: #374151; }
        .card { 
            background: linear-gradient(135deg, #DBEAFE, #F0FDFA); 
            border-radius: 12px; padding: 24px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); 
            margin-bottom: 1rem; 
        }
        .btn-primary { 
            background-color: #3B82F6; color: white; padding: 12px 24px; border-radius: 8px; 
            transition: transform 0.2s, background-color 0.2s; 
        }
        .btn-primary:hover { 
            background-color: #1E40AF; transform: scale(1.05); 
        }
        .input-label { font-weight: 600; color: #1F2937; }
        .tooltip { font-size: 0.9rem; color: #6B7280; margin-top: 0.25rem; }
        .progress-container { 
            background-color: #E5E7EB; border-radius: 8px; height: 24px; overflow: hidden; 
        }
        .progress-fill { 
            background: linear-gradient(90deg, #34D399, #14B8A6); 
            height: 100%; transition: width 0.5s ease-in-out; 
        }
        .invalid-input { border: 2px solid #F87171; border-radius: 4px; }
        @keyframes fadeIn { 
            from { opacity: 0; transform: translateY(-10px); } 
            to { opacity: 1; transform: translateY(0); } 
        }
        .sidebar-logo { width: 100%; max-width: 150px; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Sidebar with logo and collapsible help
st.sidebar.markdown("""
    <div class="card">
        <img src="https://via.placeholder.com/150x50?text=HeartSafe+AI" class="sidebar-logo" alt="HeartSafe AI Logo">
        <h2 class="subtitle">HeartSafe AI</h2>
    </div>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
    Predict heart attack risk with our advanced ML model.  
    - Trained on UCI Heart Disease dataset  
    - ~91.5% accuracy  
    - Input patient data for instant results
""")
with st.sidebar.expander("ℹ️ Help & Guidance"):
    st.markdown("""
        **How to Use**  
        1. Enter patient details in the form below.  
        2. Ensure inputs match the specified ranges (e.g., Age: 20-99).  
        3. Click "Predict Risk" to view results and visualizations.  
        4. Check feature importance to understand key risk factors.  
        **Tips**  
        - Use medical records for accurate inputs.  
        - Contact support@heartsafe.ai for assistance.
    """)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by HeartSafe AI")

# Title and description with animation
st.markdown('<h1 class="title">🩺 HeartSafe AI: Heart Attack Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
    <p class="subtitle">
        Discover your heart health with our AI-powered tool. Enter patient details to assess heart attack risk with ~91.5% accuracy.
    </p>
""", unsafe_allow_html=True)

# Load pre-trained model and scaler
try:
    hybrid_model = load("hybrid_model.joblib")
    scaler = load("scaler.joblib")
except FileNotFoundError:
    st.error("🚨 Model or scaler file not found. Please run `prototype_code.py` to train and save the model.")
    st.stop()

# Input form
st.markdown('<h2 class="subtitle">Enter Patient Health Details</h2>', unsafe_allow_html=True)
with st.form("patient_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Input fields with validation
    st.markdown('<p class="input-label">Age</p>', unsafe_allow_html=True)
    age = st.number_input("Enter age (20-99)", min_value=20, max_value=99, value=45, step=1, key="age")
    st.markdown('<p class="tooltip">Patient age in years</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Sex</p>', unsafe_allow_html=True)
    sex = st.selectbox("Select sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0], key="sex")
    st.markdown('<p class="tooltip">Biological sex of the patient</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Chest Pain Type</p>', unsafe_allow_html=True)
    cp = st.selectbox("Select chest pain type",
                      options=[("Typical Angina", 0), ("Atypical", 1), ("Non-anginal", 2), ("Asymptomatic", 3)],
                      format_func=lambda x: x[0], key="cp")
    st.markdown('<p class="tooltip">Type of chest pain experienced</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Resting Blood Pressure</p>', unsafe_allow_html=True)
    trestbps = st.number_input("Enter resting blood pressure (mm Hg, 94-200)", min_value=94, max_value=200, value=120, step=1, key="trestbps")
    st.markdown('<p class="tooltip">Blood pressure at rest (mm Hg)</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Serum Cholesterol</p>', unsafe_allow_html=True)
    chol = st.number_input("Enter serum cholesterol (mg/dL, 126-564)", min_value=126, max_value=564, value=200, step=1, key="chol")
    st.markdown('<p class="tooltip">Cholesterol level in blood (mg/dL)</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Fasting Blood Sugar</p>', unsafe_allow_html=True)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dL", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="fbs")
    st.markdown('<p class="tooltip">Blood sugar level after fasting</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Resting ECG Results</p>', unsafe_allow_html=True)
    restecg = st.selectbox("Select resting ECG results",
                           options=[("Normal", 0), ("ST-T Abnormality", 1), ("LVH", 2)],
                           format_func=lambda x: x[0], key="restecg")
    st.markdown('<p class="tooltip">Electrocardiogram results at rest</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Maximum Heart Rate</p>', unsafe_allow_html=True)
    thalach = st.number_input("Enter maximum heart rate (71-202)", min_value=71, max_value=202, value=150, step=1, key="thalach")
    st.markdown('<p class="tooltip">Heart rate achieved during exercise</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Exercise Induced Angina</p>', unsafe_allow_html=True)
    exang = st.selectbox("Exercise induced angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="exang")
    st.markdown('<p class="tooltip">Chest pain triggered by exercise</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">ST Depression</p>', unsafe_allow_html=True)
    oldpeak = st.number_input("Enter ST depression (0-6.2)", min_value=0.0, max_value=6.2, value=1.4, step=0.1, key="oldpeak")
    st.markdown('<p class="tooltip">ST depression induced by exercise relative to rest</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Slope of ST Segment</p>', unsafe_allow_html=True)
    slope = st.selectbox("Select slope of ST segment",
                         options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                         format_func=lambda x: x[0], key="slope")
    st.markdown('<p class="tooltip">Slope of the peak exercise ST segment</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Major Vessels Colored</p>', unsafe_allow_html=True)
    ca = st.number_input("Enter major vessels colored (0-3)", min_value=0, max_value=3, value=0, step=1, key="ca")
    st.markdown('<p class="tooltip">Number of major vessels colored by fluoroscopy</p>', unsafe_allow_html=True)

    st.markdown('<p class="input-label">Thalassemia</p>', unsafe_allow_html=True)
    thal = st.selectbox("Select thalassemia",
                        options=[("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)],
                        format_func=lambda x: x[0], key="thal")
    st.markdown('<p class="tooltip">Thalassemia blood disorder status</p>', unsafe_allow_html=True)

    st.markdown('<button type="submit" class="btn-primary">Predict Risk</button>', unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict Risk", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Process inputs and make prediction
if submitted:
    # Validate inputs
    input_valid = True
    if not (20 <= age <= 99):
        st.error("⚠️ Age must be between 20 and 99.")
        input_valid = False
    if not (94 <= trestbps <= 200):
        st.error("⚠️ Resting Blood Pressure must be between 94 and 200 mm Hg.")
        input_valid = False
    if not (126 <= chol <= 564):
        st.error("⚠️ Serum Cholesterol must be between 126 and 564 mg/dL.")
        input_valid = False
    if not (71 <= thalach <= 202):
        st.error("⚠️ Maximum Heart Rate must be between 71 and 202.")
        input_valid = False
    if not (0 <= oldpeak <= 6.2):
        st.error("⚠️ ST Depression must be between 0 and 6.2.")
        input_valid = False
    if not (0 <= ca <= 3):
        st.error("⚠️ Major Vessels Colored must be between 0 and 3.")
        input_valid = False

    if input_valid:
        # Create DataFrame from inputs
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex[1]], 'cp': [cp[1]], 'trestbps': [trestbps], 'chol': [chol],
            'fbs': [fbs[1]], 'restecg': [restecg[1]], 'thalach': [thalach], 'exang': [exang[1]],
            'oldpeak': [oldpeak], 'slope': [slope[1]], 'ca': [ca], 'thal': [thal[1]]
        })
        
        # Scale inputs
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = hybrid_model.predict(input_scaled)[0]
        probability = hybrid_model.predict_proba(input_scaled)[0][1]
        
        # Display results
        st.markdown('<h2 class="subtitle">Prediction Results</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        result = "💔 High Risk of Heart Disease" if prediction == 1 else "❤️ Low Risk of Heart Disease"
        st.markdown(f"<h3 style='color: {'#F87171' if prediction == 1 else '#34D399'};'>{result}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Risk Probability:</strong> {probability:.2f}</p>", unsafe_allow_html=True)
        
        # Custom progress bar
        st.markdown('<h3>Risk Visualization</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="progress-container"><div class="progress-fill" style="width: {probability*100}%"></div></div>', unsafe_allow_html=True)
        st.markdown('<p class="tooltip">The bar shows the probability of high risk (closer to 100% = higher risk).</p>', unsafe_allow_html=True)
        
        # Plotly bar chart for probability
        st.markdown('<h3>Risk Probability Chart</h3>', unsafe_allow_html=True)
        fig = go.Figure(data=[
            go.Bar(
                x=["Low Risk", "High Risk"],
                y=[1 - probability, probability],
                marker=dict(color=["#34D399", "#F87171"], line=dict(color=["#059669", "#DC2626"], width=1)),
                text=[f"{1 - probability:.2f}", f"{probability:.2f}"],
                textposition="auto"
            )
        ])
        fig.update_layout(
            title="Heart Attack Risk Probability",
            xaxis_title="Risk Category",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            plot_bgcolor="#F3F4F6",
            paper_bgcolor="#F3F4F6",
            font=dict(color="#1F2937")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance chart
        st.markdown('<h3>Feature Importance</h3>', unsafe_allow_html=True)
        importance = hybrid_model.named_estimators_['rf'].feature_importances_
        feature_names = input_data.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=True)
        fig_importance = go.Figure(data=[
            go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker=dict(color="#3B82F6", line=dict(color="#1E40AF", width=1)),
                text=importance_df['Importance'].round(3),
                textposition="auto"
            )
        ])
        fig_importance.update_layout(
            title="Key Factors in Prediction",
            xaxis_title="Importance",
            yaxis_title="Feature",
            plot_bgcolor="#F3F4F6",
            paper_bgcolor="#F3F4F6",
            font=dict(color="#1F2937")
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown('<p class="tooltip">Shows which health factors most influence the risk prediction.</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p style="color: #14B8A6; font-weight: bold;">Built with ❤️ by HeartSafe AI | Powered by Streamlit</p>
        <p style="color: #3B82F6;">For You, For Always</p>
    </div>
""", unsafe_allow_html=True)