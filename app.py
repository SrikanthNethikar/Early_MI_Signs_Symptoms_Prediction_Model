import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt # Still needed for other potential plots if added later, but not for static image
import sklearn # Ensure scikit-learn is available for model compatibility
import streamlit.components.v1 as components

# --- Configuration and Model Loading ---
st.set_page_config(page_title="Early MI Signs Prediction", layout="centered")
st.title("🫀 Early Myocardial Infarction (MI) Risk Prediction")
st.markdown("Enter patient health details below to assess early risk of heart attack.")

# Load the trained model
# Ensure 'early_mi_model.pkl' is in the same directory as your app.py
try:
    model = joblib.load("early_mi_model.pkl")
except FileNotFoundError:
    st.error("Error: 'early_mi_model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop() # Stop the app if model is not found

# --- Define Feature Names (Crucial for SHAP) ---
# This list MUST match the order of features in your input_data array
feature_names = [
    'age', 'resting_bp', 'chol_total', 'ldl', 'hdl', 'trig', 'bmi', 'fbs', 'hr_rest', 'troponin',
    'activity_mins', 'oxygen_sat', 'artery_block',
    'gender_val', 'chest_discomfort', 'sob_exertion', 'fatigue', 'jaw_pain', 'cold_sweat',
    'diabetes', 'family_history', 'ecg_abnormal', 'sleep_disruption',
    'smoke_never', 'smoke_past', # Assuming 'Current' is reference for Smoking Status
    'alc_regular', 'alc_unknown', # Assuming 'None' is reference for Alcohol Use
    'stress_low', 'stress_moderate', # Assuming 'High' is reference for Stress Level
    'diet_high', 'diet_low', # Assuming 'Balanced' is reference for Diet Type
    'job_physical', 'job_sedentary', # Assuming 'Other' is reference for Job Type
    'meno_post', 'meno_pre' # Assuming 'None' is reference for Menopause Status
]

# --- User Inputs ---
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

# Input data array for prediction
input_data_array = np.array([[
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

# Convert input data to DataFrame for SHAP
input_df_for_shap = pd.DataFrame(input_data_array, columns=feature_names)

# --- Dummy Background Data for SHAP Explainer (IMPORTANT: Replace with your actual training data) ---
# For accurate SHAP values, replace this with a representative sample of your model's training data.
@st.cache_resource # Cache this to avoid re-creating on every rerun
def create_dummy_background_data(feature_names_list):
    # Create a dummy DataFrame with random values for demonstration
    num_dummy_samples = 500 # Increased number of samples for better variance
    dummy_data = {}
    for feature in feature_names_list:
        if feature in ['age', 'resting_bp', 'chol_total', 'ldl', 'hdl', 'trig', 'bmi', 'fbs', 'hr_rest', 'troponin',
                       'activity_mins', 'oxygen_sat', 'artery_block']:
            dummy_data[feature] = np.random.rand(num_dummy_samples).astype(float) * 100 # Scale for numerical
        else: # Binary/One-hot encoded features
            dummy_data[feature] = np.random.choice([0.0, 1.0], size=num_dummy_samples, p=[0.5, 0.5]).astype(float)
    return pd.DataFrame(dummy_data, columns=feature_names_list)

X_background_for_shap = create_dummy_background_data(feature_names)

# Initialize SHAP Explainer (cached for performance)
@st.cache_resource
def get_shap_explainer(_model_obj, background_data):
    return shap.TreeExplainer(_model_obj, background_data)

explainer = get_shap_explainer(model, X_background_for_shap)
expected_value = explainer.expected_value # This will be used for waterfall/force plots

# --- Prediction Logic ---
if st.button("🩺 Predict MI Risk"):
    prediction = model.predict(input_data_array)[0]
    # For binary classification, predict_proba returns [prob_class_0, prob_class_1]
    prob = model.predict_proba(input_data_array)[0][prediction] * 100

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error("High Risk of MI 💔")
    else:
        st.success("Low Risk of MI ❤️")
    st.info(f"Prediction Confidence: {prob:.2f}%")

    # --- SHAP Section for Individual Prediction (Waterfall & Force) ---
    st.markdown("---")
    st.subheader("🧠 Understanding This Prediction with SHAP")

    # Compute SHAP values for the single input instance
    shap_values_instance = explainer.shap_values(input_df_for_shap)

    # Determine the SHAP values and expected value for the predicted class for plotting
    if isinstance(shap_values_instance, list):
        # Multi-class model: select SHAP values for the predicted class and flatten to 1D
        shap_values_for_plot = np.array(shap_values_instance[prediction]).flatten()
        expected_value_for_plot = expected_value[prediction]
    else:
        # Binary or Regression model: flatten to 1D
        shap_values_for_plot = np.array(shap_values_instance).flatten()
        expected_value_for_plot = expected_value

    # --- SHAP Waterfall Plot for the current prediction ---
    with st.expander("🌊 See Waterfall Plot for This Prediction"):
        st.write(
            "This plot shows how each feature contributes to the current prediction, "
            "moving from the model's base value to the final predicted value."
        )
        explanation_object = shap.Explanation(
            values=shap_values_for_plot,
            base_values=expected_value_for_plot,
            data=input_df_for_shap.values[0],
            feature_names=feature_names
        )
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation_object, show=False)
        st.pyplot(fig_waterfall)
        st.caption(
            "Red bars increase the prediction, blue bars decrease it. "
            "The gray text shows the feature's value for this instance."
        )

    # --- SHAP Force Plot for the current prediction ---
    with st.expander("💪 See Interactive Force Plot for This Prediction"):
        st.write(
            "This interactive plot provides another view of feature contributions for the current prediction. "
            "Red features push the prediction higher, blue features push it lower."
        )
        shap.initjs() # Initialize JavaScript for interactive plots
        force_plot_html = shap.force_plot(
            expected_value_for_plot,
            shap_values_for_plot,
            input_df_for_shap,
            feature_names=feature_names,
            matplotlib=False
        )
        components.html(force_plot_html.html, height=300, scrolling=True)
        st.caption(
            "Drag the plot to explore feature contributions. Features in red increase the prediction, blue decrease it."
        )

# --- Global SHAP Insights (Static Image) ---
st.markdown("---")
st.subheader("📊 Global Model Insights with SHAP")

# Display the static SHAP summary image
try:
    st.image("shap_summary_early_MI.png", caption="Global SHAP Summary Plot (Static Image)", use_column_width=True)
except FileNotFoundError:
    st.error("Error: 'shap_summary_early_MI.png' not found. Please ensure the image file is in the same directory.")
    st.info("You can generate this plot using `shap.summary_plot` with your training data and save it as 'shap_summary_early_MI.png'.")

st.markdown("---")
st.write("This app uses `shap` library for model interpretability.")
st.write("Remember to install necessary libraries: `pip install streamlit joblib numpy pandas shap matplotlib scikit-learn`")
