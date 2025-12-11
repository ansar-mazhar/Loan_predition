import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
# Use the optimized threshold determined from your testing (e.g., 0.35)
OPTIMIZED_THRESHOLD = 0.35 

# --- 1. Load All Artifacts (Cached for Performance) ---
# @st.cache_resource is used to load heavy artifacts only once
@st.cache_resource
def load_artifacts():
    try:
        # Load Model, Scaler, and Feature Names
        # **** FIX: Added "./" prefix to all file paths to ensure correct loading on Streamlit Cloud ****
        model = joblib.load("./loan_model.pkl") 
        scaler = joblib.load("./standard_scaler.pkl")
        feature_names = joblib.load("./feature_names.pkl")

        # Load Mapping Dictionaries (from your joblib.dump() calls)
        home_ownership_mapping = joblib.load("./person_home_ownership_mapping.pkl")
        defaults_mapping = joblib.load("./previous_loan_defaults_on_file_mapping.pkl")

        # Combine mappings for easy access
        mappings = {
            'person_home_ownership': home_ownership_mapping,
            'previous_loan_defaults_on_file': defaults_mapping
        }
        
        # List of numerical columns that need scaling (must match train_model.py)
        numerical_cols = [
            'person_age', 'person_income', 'person_emp_exp', 
            'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
            'cb_person_cred_hist_length', 'credit_score'
        ]

        return model, scaler, feature_names, mappings, numerical_cols

    except FileNotFoundError as e:
        st.error(f"Error loading artifact: {e}. Please ensure all .pkl files are in the repository.")
        return None, None, None, None, None

model, scaler, feature_names, mappings, numerical_cols = load_artifacts()

if not model:
    st.stop()


# --- 2. Streamlit UI and Input Collection ---

st.set_page_config(page_title="Loan Risk Prediction", layout="wide")
st.title("🏦 Loan Risk Prediction Application")
st.markdown("Enter the applicant's details to assess the probability of loan default.")

# Define input fields based on your features
with st.form("loan_input_form"):
    st.subheader("Applicant Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income", min_value=1000.0, value=50000.0, step=1000.0)
        emp_exp = st.number_input("Employment Experience (Years)", min_value=0, max_value=50, value=5)
    
    with col2:
        credit_score = st.number_input("Credit Score (FICO)", min_value=300, max_value=850, value=680)
        cred_hist = st.number_input("Credit History Length (Years)", min_value=0, value=5)
        
        home_ownership = st.selectbox(
            "Home Ownership", 
            options=list(mappings['person_home_ownership'].keys()) # RENT, OWN, MORTGAGE, OTHER
        )

    with col3:
        loan_amnt = st.number_input("Loan Amount Requested", min_value=100.0, value=15000.0)
        int_rate = st.number_input("Loan Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.1)
        defaults = st.selectbox(
            "Previous Loan Defaults on File?", 
            options=list(mappings['previous_loan_defaults_on_file'].keys()) # Yes, No
        )
        
    submitted = st.form_submit_button("Predict Loan Status")

# --- 3. Prediction Logic ---

if submitted:
    # 3.1. Create Raw Input DataFrame
    loan_percent_income = loan_amnt / income if income > 0 else 0
    
    raw_input = {
        'person_age': age,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home_ownership,
        'loan_amnt': loan_amnt,
        'loan_int_rate': int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cred_hist,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': defaults
    }
    df_input = pd.DataFrame([raw_input])
    
    # 3.2. Preprocessing (Mapping & Scaling - MUST MATCH TRAINING)
    
    # Apply Mappings (Categorical Encoding)
    for col, mapping in mappings.items():
        df_input[col] = df_input[col].map(mapping)
        
    # Apply Scaling (Numerical Normalization)
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
    
    # 3.3. Align Column Order (CRUCIAL STEP)
    # This uses the saved feature_names list to ensure the input order is correct
    try:
        X_predict = df_input[feature_names]
    except KeyError as e:
        st.error(f"Feature mismatch: Ensure all required columns are present. Missing: {e}")
        st.stop()
        
    # 3.4. Generate Prediction
    probabilities = model.predict_proba(X_predict)[0]
    confidence_default = probabilities[1] # Probability of Class 1 (Default)

    # 3.5. Apply Optimized Threshold
    if confidence_default >= OPTIMIZED_THRESHOLD:
        status = "REJECTED"
        risk_level = "HIGH RISK OF DEFAULT"
        color = "red"
    else:
        status = "APPROVED"
        risk_level = "LOW RISK"
        color = "green"

    # --- 4. Display Results ---
    st.markdown("---")
    st.subheader("Prediction Result")
    
    st.metric(label="Decision Status", 
              value=status, 
              delta=risk_level, 
              delta_color=color)

    st.markdown(f"**Probability of Default:** **`{confidence_default:.2f}`**")
    st.caption(f"Decision Threshold used: {OPTIMIZED_THRESHOLD} (Trained with SMOTE for better risk detection)")