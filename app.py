import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('churn_model.pkl')

st.title("📊 Customer Churn Prediction App")
st.markdown("Fill in the customer details below:")

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    senior = st.selectbox("Is the customer a Senior Citizen?", ["No", "Yes"])
    partner = st.selectbox("Has a Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    phone_service = st.selectbox("Has Phone Service?", ["No", "Yes"])
    paperless = st.selectbox("Uses Paperless Billing?", ["No", "Yes"])

with col2:
    tenure = st.number_input("Tenure (months)", min_value=0)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)

# Dummy encoded features
gender_male = st.selectbox("Gender", ["Female", "Male"]) == "Male"
multiple_lines_yes = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"]) == "Yes"
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check"
])

# Map inputs to the format your model expects
input_data = [
    1 if senior == "Yes" else 0,
    1 if partner == "Yes" else 0,
    1 if dependents == "Yes" else 0,
    tenure,
    1 if phone_service == "Yes" else 0,
    1 if paperless == "Yes" else 0,
    monthly_charges,
    total_charges,
    int(gender_male),
    1 if multiple_lines_yes else 0,
    # Add the rest of your dummy variables according to your model input order
]

if st.button("Predict"):
    prediction = model.predict([input_data])[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
