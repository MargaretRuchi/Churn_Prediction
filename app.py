import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model (your 30-feature trained model)
model = joblib.load("model_30_features.pkl")

st.title("Customer Churn Prediction App")

# Sidebar - User Inputs
st.sidebar.header("Enter Customer Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
    TotalCharges = st.sidebar.slider("Total Charges", 0, 9000, 1500)
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                            "Bank transfer (automatic)", "Credit card (automatic)"])

    # Collect into a dict
    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'PaperlessBilling': PaperlessBilling,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaymentMethod': PaymentMethod
    }

    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Customer Information Preview")
st.write(input_df)

# Expected features after one-hot encoding (30 features)
expected_features = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling',
                     'MonthlyCharges', 'TotalCharges', 'gender_Male', 'MultipleLines_No phone service', 
                     'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
                     'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
                     'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
                     'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
                     'TechSupport_No internet service', 'TechSupport_Yes', 
                     'StreamingTV_No internet service', 'StreamingTV_Yes', 
                     'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
                     'Contract_One year', 'Contract_Two year', 
                     'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
                     'PaymentMethod_Mailed check']


# One-hot encode the user input
input_encoded = pd.get_dummies(input_df)

# Ensure all expected columns from training are in the right order
input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)
    prediction_label = "Churn" if prediction[0] == 1 else "No Churn"
    st.success(f"Prediction: {prediction_label}")

