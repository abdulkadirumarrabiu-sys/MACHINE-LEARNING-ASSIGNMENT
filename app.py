import streamlit as st
import joblib
import numpy as np


# 1. Load Model and Assets
@st.cache_resource
def load_assets():
    # Ensure these filenames match your saved files
    model = joblib.load('model.joblib')
    encoder = joblib.load('encoder.joblib')
    return model, encoder


try:
    model, encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 2. User Input Interface
st.set_page_config(page_title="Disease Prediction System", layout="centered")
st.title("🩺 Medical Disease Prediction")
st.markdown("Enter the patient details below to predict the likely condition.")

with st.form("prediction_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        blood_pressure = st.number_input("Blood Pressure (systolic)", value=120)
        cholesterol = st.number_input("Cholesterol Level", value=200)
        glucose = st.number_input("Glucose Level", value=100)
        bmi = st.number_input("BMI", value=22.5)

    with col2:
        # Binary Symptom/History Inputs
        smoking = st.toggle("Smoking History")
        alcohol = st.toggle("Alcohol Consumption")
        exercise = st.toggle("Regular Exercise")
        family_history = st.toggle("Family History of Disease")

        st.write("Existing Conditions:")
        alzheimers = st.checkbox("Alzheimer's Disease")
        heart_disease = st.checkbox("Heart Disease")
        stroke = st.checkbox("Stroke")
        kidney_disease = st.checkbox("Kidney Disease")
        cancer = st.checkbox("Cancer")
        copd = st.checkbox("COPD")
        liver_disease = st.checkbox("Liver Disease")
        parkinsons = st.checkbox("Parkinson's Disease")
        tuberculosis = st.checkbox("Tuberculosis")

    submit = st.form_submit_button("Predict Disease")

# 3. Prediction System & 4. Output Display
if submit:
    # Arrange inputs in the exact order the model expects
    features = np.array([[
        age, gender, int(alzheimers), blood_pressure, cholesterol,
        glucose, int(smoking), int(alcohol), int(exercise), bmi,
        int(family_history), int(heart_disease), int(stroke),
        int(kidney_disease), int(cancer), int(copd),
        int(liver_disease), int(parkinsons), int(tuberculosis)
    ]])

    with st.spinner("Analyzing data..."):
        # The pipeline automatically handles the StandardScaler
        prediction_encoded = model.predict(features)

        # Decode the result to show the disease name
        prediction_name = encoder.inverse_transform(prediction_encoded)[0]

    st.success("Analysis Complete")
    st.metric(label="Predicted Condition", value=f"{prediction_name}")

    # Optional: Probability visualization
    probs = model.predict_proba(features)[0]
    st.info(f"Prediction Confidence: {max(probs) * 100:.2f}%")