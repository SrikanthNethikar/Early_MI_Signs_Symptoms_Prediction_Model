import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sklearn # Ensure scikit-learn is available for model compatibility
import streamlit.components.v1 as components

# Load the trained model
model = joblib.load("early_mi_model.pkl")

st.set_page_config(page_title="Early MI Signs Prediction", layout="centered")
st.title("🫀 Early Myocardial Infarction (MI) Risk Prediction")
st.markdown("Enter patient health details below to assess early risk of heart attack.")

# Numerical inputs
age = st.number_input("Age", 18, 100, 45)
resting_bp = st.number_input("Resting BP", 80, 200, 120)
chol_total = st.number_input("Total Cholesterol", 100, 400, 180)
ldl = st.number_input("LDL Cholesterol", 30, 300, 100)
hdl = st.number_input("HDL Cholesterol", 10, 100, 50)
trig = st.number_input("Triglycerides", 50, 600, 150)
bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
fbs = st.number_input("Fasting Blood Sugar", 70, 200, 90)
hr_rest = st.number_input("Resting Heart Rate", 40, 120, 70)
troponin = st.number_input("Sensitive Troponin", 0.0, 5.0, 0.01)
activity_mins = st.number_input("Physical Activity (min/week)", 0, 1000, 150)
oxygen_sat = st.number_input("Oxygen Saturation at Rest (%)", 70, 100, 98)
artery_block = st.number_input("Percent Artery Blockage", 0, 100, 20)

# Binary features
gender = st.radio("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0

chest_discomfort = st.checkbox("Mild Chest Discomfort")
sob_exertion = st.checkbox("Shortness of Breath on Exertion")
fatigue = st.checkbox("Unexplained Fatigue")
jaw_pain = st.checkbox("Jaw/Neck/Left Arm Pain")
cold_sweat = st.checkbox("Cold Sweat at Night")
diabetes = st.checkbox("Diabetes")
family_history = st.checkbox("Family History of Heart Disease")
ecg_abnormal = st.checkbox("Minor ECG Abnormality")
sleep_disruption = st.checkbox("Sleep Disruption")

# One-hot categorical selections
smoking_status = st.selectbox("Smoking Status", ["Current", "Never", "Past"])
smoke_never = 1 if smoking_status == "Never" else 0
smoke_past = 1 if smoking_status == "Past" else 0

alcohol_use = st.selectbox("Alcohol Use", ["None", "Regular", "Unknown"])
alc_regular = 1 if alcohol_use == "Regular" else 0
alc_unknown = 1 if alcohol_use == "Unknown" else 0

stress_level = st.selectbox("Stress Level", ["High", "Moderate", "Low"])
stress_low = 1 if stress_level == "Low" else 0
stress_moderate = 1 if stress_level == "Moderate" else 0

diet = st.selectbox("Diet Type", ["Balanced", "High-fat", "Low-fat"])
diet_high = 1 if diet == "High-fat" else 0
diet_low = 1 if diet == "Low-fat" else 0

job_type = st.selectbox("Job Type", ["Other", "Physically Demanding", "Sedentary"])
job_physical = 1 if job_type == "Physically Demanding" else 0
job_sedentary = 1 if job_type == "Sedentary" else 0

menopause_status = st.selectbox("Menopause Status", ["None", "Postmenopausal", "Premenopausal"])
meno_post = 1 if menopause_status == "Postmenopausal" else 0
meno_pre = 1 if menopause_status == "Premenopausal" else 0

# Input data array
input_data = np.array([[
    age, resting_bp, chol_total, ldl, hdl, trig, bmi, fbs, hr_rest, troponin,
    activity_mins, oxygen_sat, artery_block,
    gender_val, chest_discomfort, sob_exertion, fatigue, jaw_pain, cold_sweat,
    diabetes, family_history, ecg_abnormal, sleep_disruption,
    smoke_never, smoke_past, alc_regular, alc_unknown,
    stress_low, stress_moderate,
    diet_high, diet_low,
    job_physical, job_sedentary,
    meno_post, meno_pre
]])

if st.button("🩺 Predict MI Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction] * 100

    st.subheader("🔍 Prediction Result:")
    st.success("High Risk of MI 💔" if prediction == 1 else "Low Risk of MI ❤️")
    st.info(f"Prediction Confidence: {prob:.2f}%")

# --- SHAP Section ---
st.markdown("---")
st.subheader("🧠 Feature Importance (SHAP Summary)")

with st.expander("📊 See SHAP Feature Importance"):
    st.image("shap_summary_early_MI.png", caption="Top features impacting MI prediction", use_column_width=True)    