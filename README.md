Early Myocardial Infarction (MI) Signs & Symptoms Prediction Model
Detecting early heart attack (MI) risk from symptoms, lab results, and lifestyle indicators
Trained on curated clinical and behavioral data for 10–30% arterial occlusion cases.

🧠 Overview
This project uses machine learning to predict early signs of myocardial infarction (heart attack), especially in patients showing mild symptoms or partial occlusion (10%–30%).

It includes:

A trained Random Forest model with 95%+ accuracy

A Streamlit app for real-time prediction

SHAP-based visual explanations to ensure model transparency

📂 What’s Inside
File	Purpose
app.py	Streamlit web application for user interaction
early_mi_model.pkl	Trained Random Forest classifier
requirements.txt	List of all Python dependencies
shap_summary_early_MI.png	Feature importance summary using SHAP

📊 Features Used in Prediction
Demographic: Age, gender, menopause status

Lifestyle: Smoking, alcohol use, physical activity, diet

Clinical: Troponin levels, blood pressure, oxygen saturation

Symptoms: Chest discomfort, shortness of breath, fatigue, jaw/neck/arm pain

⚙️ Technologies
Python, Pandas, NumPy

Scikit-learn (Random Forest)

SHAP (Model Explainability)

Streamlit (Web App Deployment)

Git, GitHub (Version Control & Hosting)

🧪 Model Performance
Metric	Value
Accuracy	95.9%
ROC AUC Score	1.00
Confusion Matrix	TN: 707, FP: 0, FN: 41, TP: 252

🧠 Explainability with SHAP
To enhance trust in healthcare AI, this app includes a SHAP feature importance chart to explain how different features contribute to each prediction.



💻 Try It Yourself
Coming Soon: Live App Link

Clone the repo and run locally:

bash
Copy
Edit
git clone https://github.com/SrikanthNethikar/Early_MI_Signs_Symptoms_Prediction_Model.git
cd Early_MI_Signs_Symptoms_Prediction_Model
pip install -r requirements.txt
streamlit run app.py
👨‍💻 Author
Srikanth Nethikar
AI Engineer | Healthcare Specialist | Streamlit Developer
📧 nethikarsrikanth@gmail.com
🔗 LinkedIn | GitHub

🩺 Why This Project Matters
Early intervention in heart attack cases saves lives

Accessible predictions via web interface (no installations needed)

Transparent AI using SHAP makes the model explainable to patients and doctors
