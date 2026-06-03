import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap       
import matplotlib.pyplot as plt 
import os
import gdown
import sklearn

# Force matplotlib to use a dark theme so text is visible!
plt.style.use('dark_background')

# --- 1. PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Clinical Diabetes Screening", page_icon="🏥", layout="wide")

# CSS to make all fonts larger and bolder, including the sidebar!
st.markdown("""
    <style>
    label p { font-size: 17px !important; font-weight: 600 !important; }
    div[data-baseweb="select"] div, input { font-size: 16px !important; }
    div[data-testid="stMarkdownContainer"] p { font-size: 17px !important; }
    div[role="radiogroup"] label p { font-weight: 400 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD THE CHAMPION MODEL & SHAP BACKGROUND ---
@st.cache_resource
def load_model():
    # The system will look for the file locally first. If it's not there, it fetches it from your Drive!
    model_path = "voting_champion_model_compressed.pkl"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading optimized clinical model from secure cloud storage (This takes ~10 seconds on first load)..."):
            # 🚨 REPLACE THE STRING BELOW WITH YOUR EXACT GOOGLE DRIVE FILE ID
            file_id = '1d5mcQNOdu11dFEg5u76gWEf-lJOx26tL'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
            
    return joblib.load(model_path)

@st.cache_data
def load_background_data():
    return pd.read_csv("shap_background.csv")

try:
    # We call the functions safely
    background_data = load_background_data()
    model = load_model()
    system_ready = True
except Exception as e:
    st.error(f"⚠️ System initialization failed: {e}")
    system_ready = False
# --- HEADER ---
st.title("🏥 Clinical Diabetes Risk Screening System")
st.markdown("Powered by an optimized Voting Classifier (XGBoost, Gradient Boosting, AdaBoost)")
st.divider()

# ==========================================
# SIDEBAR: PATIENT INTAKE FORM
# ==========================================
with st.sidebar:
    st.header("👤 Patient Feature Input")
    st.markdown("Enter the epidemiological indicators below:")
    
    genhlth = st.selectbox("General Health (1: Excellent, 5: Poor)", options=[1, 2, 3, 4, 5], index=2)

    st.markdown("**BMI (Body Mass Index)**")
    
    # Auto-Calculator stacked vertically for the narrow sidebar
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    bmi = weight / ((height / 100) ** 2)
    st.info(f"🧮 **Calculated BMI: {bmi:.1f}**")
    
    age = st.selectbox("Age Category (1: 18-24, 13: 80+)", options=list(range(1, 14)), index=7)
    highbp = st.radio("High Blood Pressure?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    highchol = st.radio("High Cholesterol?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    income = st.selectbox("Income Bracket (1: Low, 8: High)", options=list(range(1, 9)), index=5)
    education = st.selectbox("Education Level (1: Low, 6: High)", options=list(range(1, 7)), index=3)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_diagnostics = st.button("Run Diagnostics", type="primary", use_container_width=True)

    # --- THE MEDICAL DISCLAIMER ---
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This system is a prototype developed for academic research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Final clinical decisions must be verified by a certified healthcare provider.")

# ==========================================
# MAIN BODY TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["👤 Single Patient Analytics", "📊 Batch Mass Screening", "📈 Model Architecture"])

# ==========================================
# TAB 1: SINGLE PATIENT ANALYTICS
# ==========================================
with tab1:
    if run_diagnostics and system_ready:
        single_patient_data = pd.DataFrame({
            'GenHlth': [genhlth],
            'BMI': [bmi],
            'Age': [age],
            'HighBP': [highbp],
            'HighChol': [highchol],
            'Income': [income],
            'Education': [education]
        })
        
        def ensemble_predict_proba(X_data):
            if isinstance(X_data, np.ndarray):
                X_data = pd.DataFrame(X_data, columns=single_patient_data.columns)
            return model.predict_proba(X_data)[:, 1]

        with st.spinner("Analyzing patient profile & generating visuals..."):
            prediction = model.predict(single_patient_data)[0]
            probability = model.predict_proba(single_patient_data)[0][1] * 100 
            
            # --- ROW 1: DIAGNOSTIC RESULT ---
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown("### 📋 Diagnostic Result")
                if prediction == 1:
                    st.error("🚨 **HIGH RISK DETECTED**\n\nPatient profile strongly matches diabetic indicators. Secondary clinical screening is strictly recommended.")
                else:
                    st.success("✅ **HEALTHY PROFILE**\n\nNo significant diabetic indicators detected based on current inputs.")
            
            with res_col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability,
                    title = {'text': "Diabetic Risk Probability (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "lightcoral"}],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.divider()
            
            # --- ROW 2: SHAP VISUAL EXPLANATION ---
            st.markdown("### 🔍 Clinical Risk Factors (SHAP Analysis)")
            st.info("⏳ **Executing Full Ensemble Explainable AI...**\n\nThe KernelExplainer is performing brute-force perturbation across all three algorithms to guarantee maximum transparency. This process takes 15-45 seconds to complete.")
            st.caption("This waterfall plot explains which specific factors pushed the algorithm's risk higher (red) or lower (blue).")
            
            try:
                explainer = shap.KernelExplainer(ensemble_predict_proba, background_data)
                shap_values = explainer.shap_values(single_patient_data)
                exp_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                
                shap_explanation = shap.Explanation(
                    values=shap_values[0], base_values=exp_val, 
                    data=single_patient_data.values[0], feature_names=single_patient_data.columns.tolist()
                )
                
                fig_shap, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_explanation, show=False)
                plt.tight_layout()
                
                fig_shap.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                st.pyplot(fig_shap, bbox_inches='tight', transparent=True)
            except Exception as e:
                st.error(f"SHAP explanation failed. (Error: {e})")
            
            st.divider()

            # --- ROW 3: PATIENT HEALTH TOPOGRAPHY ---
            st.markdown("### 🕸️ Patient Health Topography")
            st.caption("A multi-axis radar view of the patient's physical and socioeconomic profile.")
            
            categories = ['BMI (Scaled)', 'Gen Health (x2)', 'Age Bracket', 'Income', 'Education']
            radar_values = [
                min(bmi / 4, 10), 
                genhlth * 2,      
                age * (10/13),    
                income * (10/8),  
                education * (10/6)
            ]
            radar_values.append(radar_values[0])
            categories.append(categories[0])
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=categories,
                fill='toself',
                name='Patient Profile',
                line_color='red' if prediction == 1 else 'green'
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                height=450, 
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("👈 Please enter patient details in the sidebar and click 'Run Diagnostics' to view analytics.")

# ==========================================
# TAB 2: BATCH MASS SCREENING (SESSION STATE FIXED)
# ==========================================
with tab2:
    st.subheader("Mass Screening File Upload")
    st.markdown("Upload a CSV file containing patient data. The system will automatically extract the required clinical indicators.")
    
    template_df = pd.DataFrame(columns=['Patient_ID', 'Patient_Name', 'GenHlth', 'BMI', 'Age', 'HighBP', 'HighChol', 'Income', 'Education'])
    st.download_button(
        label="📄 Download Blank CSV Template",
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name='patient_screening_template.csv',
        mime='text/csv',
    )
    
    st.divider()

    uploaded_file = st.file_uploader("Upload Patient Records (CSV format)", type=["csv"])

    # Clear memory if the user removes the file
    if uploaded_file is None:
        if 'batch_results' in st.session_state:
            del st.session_state['batch_results']

    if uploaded_file is not None and system_ready:
        try:
            patient_df = pd.read_csv(uploaded_file)
            required_columns = ['GenHlth', 'BMI', 'Age', 'HighBP', 'HighChol', 'Income', 'Education']
            missing_cols = [col for col in required_columns if col not in patient_df.columns]
            
            if missing_cols:
                st.error(f"❌ **Upload Failed:** The CSV is missing the following required columns: {missing_cols}")
            else:
                # 1. THE BUTTON LOGIC (Calculates and saves to memory)
                if st.button("Execute Mass Screening", type="primary"):
                    with st.spinner("Processing records through the ensemble model..."):
                        X_batch = patient_df[required_columns]
                        predictions = model.predict(X_batch)
                        
                        results_df = patient_df.copy()
                        results_df['Diagnostic_Result'] = ['High Risk' if p == 1 else 'Healthy' for p in predictions]
                        
                        cols = ['Diagnostic_Result'] + [col for col in results_df.columns if col != 'Diagnostic_Result']
                        
                        # SAVE TO STREAMLIT'S MEMORY
                        st.session_state.batch_results = results_df[cols]

                # 2. THE DISPLAY LOGIC (Reads from memory, survives dropdown clicks!)
                if 'batch_results' in st.session_state:
                    results_df = st.session_state.batch_results
                    
                    st.success(f"Screening Complete! Processed {len(results_df)} patients.")

                    # --- DYNAMIC COHORT FEATURE ANALYTICS ---
                    st.markdown("### 📈 Dynamic Cohort Trend Analysis")
                    st.caption("Select a specific clinical indicator below to instantly visualize how it correlates with diabetes risk in this patient batch.")
                    
                    analysis_features = [col for col in required_columns if col != 'Diagnostic_Result']
                    
                    # UI UPGRADE: Use columns to constrain the dropdown width!
                    dd_col1, dd_col2 = st.columns([1, 2])
                    
                    with dd_col1:
                        selected_feature = st.selectbox(
                            "🔍 Select Clinical Indicator:", 
                            options=analysis_features, 
                            index=analysis_features.index('Age') 
                        )
                    
                    # Create the Dynamic Visual
                    fig_hist = px.histogram(
                        results_df, 
                        x=selected_feature, 
                        color="Diagnostic_Result",
                        barmode="group", 
                        color_discrete_map={'High Risk': '#d62728', 'Healthy': '#2ca02c'}
                    )
                    
                    # UI UPGRADE: Polish the axes, fonts, and legend!
                    fig_hist.update_layout(
                        title=dict(text=f"Risk Distribution by {selected_feature}", font=dict(size=18, weight='bold')),
                        height=420, 
                        plot_bgcolor="rgba(0,0,0,0)", 
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(
                            title=dict(text=selected_feature, font=dict(size=15, weight='bold')),
                            tickfont=dict(size=14)
                        ),
                        yaxis=dict(
                            title=dict(text="Number of Patients", font=dict(size=15, weight='bold')),
                            tickfont=dict(size=14),
                            gridcolor='rgba(255, 255, 255, 0.1)' 
                        ),
                        legend=dict(
                            title=dict(text="Clinical Diagnosis", font=dict(weight='bold')),
                            font=dict(size=14),
                            bgcolor="rgba(0,0,0,0)",
                            bordercolor="rgba(255,255,255,0.2)",
                            borderwidth=1
                        )
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    st.divider()

                    # --- RESULTS VISUALIZATION ---
                    st.markdown("### 📊 Population Risk Distribution")
                    risk_counts = results_df['Diagnostic_Result'].value_counts().reset_index()
                    risk_counts.columns = ['Status', 'Count']
                    
                    fig_bar = px.bar(
                        risk_counts, 
                        x='Status', 
                        y='Count', 
                        color='Status', 
                        color_discrete_map={'High Risk': '#d62728', 'Healthy': '#2ca02c'},
                        text='Count'
                    )
                    
                    # UI UPGRADE: Push the numbers outside the bars and make them bold
                    fig_bar.update_traces(
                        textposition='outside',
                        textfont=dict(size=16, weight='bold'),
                        cliponaxis=False 
                    )
                    max_count = risk_counts['Count'].max()
                    
                    # 2. UI UPGRADE: Polish axes, hide legend, add dynamic buffer
                    fig_bar.update_layout(
                        height=380, 
                        plot_bgcolor="rgba(0,0,0,0)", 
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False, 
                        xaxis=dict(
                            title=None, 
                            tickfont=dict(size=16, weight='bold') 
                        ),
                        yaxis=dict(
                            title=dict(text="Number of Patients", font=dict(size=15, weight='bold')),
                            tickfont=dict(size=14),
                            gridcolor='rgba(255, 255, 255, 0.1)',
                            range=[0, max_count * 1.2] # <-- NEW: Always adds a 20% empty space above the tallest bar
                        ),
                        margin=dict(t=50) # Increased top margin slightly for extra breathing room
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.markdown("### 📋 Final Screening Report")
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Clinical Report",
                        data=csv_data,
                        file_name='mass_screening_results.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"❌ Error reading the file. Please ensure it is a valid CSV. Error Details: {e}")

# ==========================================
# TAB 3: MODEL ARCHITECTURE & VALIDATION
# ==========================================
with tab3:
    st.subheader("Model Architecture & Clinical Validation")
    st.markdown("This system is powered by a **Soft-Voting Ensemble Classifier** consisting of optimized XGBoost, Gradient Boosting, and AdaBoost algorithms. To address severe epidemiological class imbalances, the training pipeline utilized **SMOTE-ENN** (Synthetic Minority Over-sampling Technique combined with Edited Nearest Neighbors).")

    st.divider()
    
    st.markdown("### 🎯 Final Performance Evaluation Metrics")
    
    # Define your exact metrics from Chapter 4
    metrics_names = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score', 'AUC']
    metrics_scores = [80.09, 66.74, 69.38, 67.81, 80.07]
    
    # Create a beautiful, interactive Plotly Bar Chart
    fig_metrics = go.Figure(data=[
        go.Bar(
            x=metrics_names,
            y=metrics_scores,
            text=[f"{val}%" for val in metrics_scores],
            textposition='outside', 
            textfont=dict(size=16, color='white', weight='bold'), 
            marker_color=['#1f77b4', '#17becf', '#d62728', '#ff7f0e', '#9467bd'],
            hovertemplate="<b>%{x}</b>: %{y}%<extra></extra>",
            cliponaxis=False 
        )
    ])
    
    # Upgraded Layout: Better margins, subtle gridlines, and bolder axes
    fig_metrics.update_layout(
        yaxis=dict(
            range=[0, 110], 
            title=dict(text="Score (%)", font=dict(size=16, weight='bold')),
            gridcolor='rgba(255, 255, 255, 0.1)', 
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        ),
        xaxis=dict(
            tickfont=dict(size=15, weight='bold') 
        ),
        height=450, 
        margin=dict(l=60, r=20, t=50, b=50), 
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig_metrics, use_container_width=True)

    # THE CONFUSION MATRIX
    st.markdown("### 🧩 Model Evaluation: Confusion Matrix")
    st.markdown("The matrix below illustrates the model's true predictive power on unseen clinical data.")
    
    # Create 3 columns. The [1, 2, 1] ratio makes the middle column twice as big as the sides.
    # This centers the image and stops it from taking up the whole screen!
    cm_col1, cm_col2, cm_col3 = st.columns([1, 2, 1])
    
    with cm_col2:
        st.image("(1)confusion_matrix.png", use_container_width=True, caption="Confusion Matrix: Soft-Voting Ensemble Classifier")
