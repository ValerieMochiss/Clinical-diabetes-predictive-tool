import streamlit as st
import pandas as pd
import joblib

# --- 1. PAGE CONFIGURATION ---
# This ensures the dashboard scales perfectly on laptops, iPads, and phones
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🏥", layout="wide")

# --- 2. LOAD THE CHAMPION MODEL ---
# st.cache_resource prevents the app from reloading the model every time a button is clicked
@st.cache_resource
def load_model():
    # IMPORTANT: Ensure your exported model file is in the same folder as this script
    return joblib.load("voting_champion_model.pkl")

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ Model file 'voting_champion_model.pkl' not found. Please ensure it is in the same directory.")
    model_loaded = False

# --- 3. SIDEBAR: SINGLE PATIENT SCREENING ---
st.sidebar.title("🏥 Single Patient Screening")
st.sidebar.markdown("Input patient health indicators below:")


bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=99.0, value=25.0, step=0.1)
age = st.sidebar.selectbox("Age Category (1: 18-24, 13: 80+)", options=list(range(1, 14)), index=7)
genhlth = st.sidebar.selectbox("General Health (1: Excellent, 5: Poor)", options=[1, 2, 3, 4, 5], index=2)
highbp = st.sidebar.radio("High Blood Pressure?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
highchol = st.sidebar.radio("High Cholesterol?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
income = st.sidebar.selectbox("Income Bracket (1: Low, 8: High)", options=list(range(1, 9)), index=5)
education = st.sidebar.selectbox("Education Level (1: Low, 6: High)", options=list(range(1, 7)), index=3)


if st.sidebar.button("Run Diagnostics") and model_loaded:
    # Compile the inputs into a DataFrame that matches X_test
    # The column names MUST match the exact names used during model training
    single_patient_data = pd.DataFrame({
        'BMI': [bmi],
        'Age': [age],
        'GenHlth': [genhlth],
        'HighBP': [highbp],
        'HighChol': [highchol],
        'Income': [income],
        'Education': [education]
    })
    
    with st.sidebar:
        with st.spinner("Analyzing profile..."):
            # Use the standard .predict() for the baseline 0.50 threshold
            prediction = model.predict(single_patient_data)[0]
            
            st.markdown("---")
            st.markdown("### 📋 Diagnostic Result")
            if prediction == 1:
                st.error("🚨 **HIGH RISK DETECTED**\n\nPatient profile matches diabetic indicators. Secondary clinical screening recommended.")
            else:
                st.success("✅ **HEALTHY PROFILE**\n\nNo significant diabetic indicators detected.")

# --- 4. MAIN PAGE: BATCH PREDICTION (FILE UPLOAD) ---
st.title("📊 Clinical Diabetes Mass Screening System")
st.markdown("""
This system utilizes an ensemble Voting Classifier (XGBoost, Gradient Boosting, AdaBoost) to process 
epidemiological data from the BRFSS. Upload a CSV of patient records to conduct a mass screening.
""")

st.divider()

uploaded_file = st.file_uploader("Upload Patient Records (CSV format)", type=["csv"])

if uploaded_file is not None and model_loaded:
    try:
        # Read the uploaded dataset
        patient_df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(patient_df.head(), use_container_width=True)
        
        if st.button("Execute Mass Screening", type="primary"):
            with st.spinner("Processing records through the ensemble model..."):
                # Run the standard prediction
                predictions = model.predict(patient_df)
                
                # Append the results to a copy of the dataframe
                results_df = patient_df.copy()
                results_df['Diagnostic_Result'] = ['High Risk (1)' if p == 1 else 'Healthy (0)' for p in predictions]
                
                # Reorder columns to put the result first for easy reading
                cols = ['Diagnostic_Result'] + [col for col in results_df.columns if col != 'Diagnostic_Result']
                results_df = results_df[cols]
                
                st.success("Screening Complete!")
                st.subheader("Final Screening Report")
                st.dataframe(results_df, use_container_width=True)
                
                # Generate a downloadable CSV report
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Clinical Report",
                    data=csv_data,
                    file_name='mass_screening_results.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"❌ Error processing file. Please ensure the CSV columns exactly match the model's training data features. Error Details: {e}")