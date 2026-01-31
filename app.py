import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the saved model and scaler
model = tf.keras.models.load_model('heart_disease_model.h5')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Detection AI 🩺")
st.write("Enter patient details to check the risk.")

# Create input fields for the user
age = st.slider("Age", 1, 100, 25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
chol = st.number_input("Cholesterol", value=200)
# ... add other 10 inputs here ...

if st.button("Predict"):
    # Organize inputs into the correct format (13 features)
    features = np.array([[age, sex, 3, 145, chol, 1, 0, 150, 0, 2.3, 0, 0, 1]])  # Example values
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.write(f"Raw Model Output: {prediction[0][0]}")
    # Get the raw decimal (e.g., 0.85)
    prediction = model.predict(features_scaled)
    probability = prediction[0][0]

    # Convert decimal to Yes/No
    if probability > 0.5:
        st.error(f"⚠️ Positive: Heart Disease Detected ({probability * 100:.2f}%)")
    else:
        st.success(f"✅ Negative: No Heart Disease Detected ({(1 - probability) * 100:.2f}%)")
    # See the actual decimal