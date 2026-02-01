import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# 1. Load the model and the scaler
model = tf.keras.models.load_model('heart_disease_model.h5')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("Heart Disease Detection AI 🩺")
st.write(
    "This tool uses clinical data to estimate heart disease risk. Patients should refer to their latest medical reports for these values.")

# Use columns to make the 13 inputs look better on screen
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                      help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal pain, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120,
                               help="Your blood pressure while at rest (mm Hg on admission to the hospital)")
    chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200,
                           help="Cholesterol level from your blood test")

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0],
                       format_func=lambda x: "True" if x == 1 else "False",
                       help="Is your blood sugar level higher than 120 mg/dl after fasting?")
    restecg = st.selectbox("Resting ECG results", [0, 1, 2],
                           help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150,
                              help="Maximum heart rate achieved during a stress test")
    exang = st.selectbox("Exercise Induced Angina", options=[1, 0],
                         format_func=lambda x: "Yes" if x == 1 else "No",
                         help="Do you experience chest pain during physical exercise?")

with col3:
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0,
                              help="ST depression induced by exercise relative to rest (from ECG)")
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                         help="The slope of the peak exercise ST segment (from ECG)")
    ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3],
                      help="Number of major vessels colored by flourosopy")
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                        help="A blood disorder: 0: NULL, 1: Normal, 2: Fixed defect, 3: Reversible defect")

st.divider()

if st.button("Predict Heart Disease Risk", type="primary"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prob = prediction[0][0]

    st.subheader("Results")
    st.write(f"**AI Confidence Score:** {prob:.4f}")

    if prob > 0.5:
        st.error(
            f"**Result: High Risk.** The model indicates a {prob * 100:.2f}% probability of heart disease. Please consult a cardiologist.")
    else:
        st.success(f"**Result: Low Risk.** The model indicates a {(1 - prob) * 100:.2f}% probability of being healthy.")

st.info(
    "Disclaimer: This is an AI prototype for educational purposes and should not be used as a substitute for professional medical advice.")