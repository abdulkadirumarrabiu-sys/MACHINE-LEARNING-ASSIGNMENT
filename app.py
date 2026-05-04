import streamlit as st
import joblib
import numpy as np


# ======     ===================
# 1. LOAD MODEL + ENCODER
# =========================
model = joblib.load("model.pkl")
encoder = joblib.load("label_encoder.pkl")

# 2. Application UI
st.set_page_config(page_title="Diabetes Predictor", layout="wide")
st.title("🩺 Health Diagnostic System")

if model:
    # Everything inside this 'with' block belongs to the form
    with st.form("main_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", 1, 120, 30)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

        with col2:
            st.subheader("Lifestyle")
            smoking = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            alcohol = st.selectbox("Alcohol Consumption", [0, 1],
                                   format_func=lambda x: "High/Moderate" if x == 1 else "Low/None")
            exercise = st.selectbox("Regular Exercise", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        with col3:
            st.subheader("Clinical Vitals")
            bp = st.number_input("Systolic Blood Pressure", 80, 200, 120)
            chol = st.number_input("Cholesterol Level", 100, 400, 200)
            glucose = st.number_input("Glucose Level", 50, 300, 100)

        st.divider()
        st.subheader("Medical History")
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            alz = st.checkbox("Alzheimer's History")
            heart_dis = st.checkbox("Heart Disease History")
        with h_col2:
            stroke = st.checkbox("Previous Stroke")
            chronic = st.checkbox("Other Chronic Conditions")
        with h_col3:
            meds = st.checkbox("On Regular Medication")
            fam_hist = st.checkbox("Family History of Diabetes")
            fatigue = st.checkbox("Frequent Fatigue/Weakness")

        # This button MUST be the last thing inside the 'with st.form' block
        submit = st.form_submit_button("Analyze Health Data")

    # 3. Prediction Logic (Triggers after the button is clicked)
    if submit:
        # We must maintain the 19-feature shape for the model[cite: 5]
        # Features removed from UI (Heart Rate, Oxygen, Temp) are set to 0
        features = np.array([[
            age, gender, int(alz), bp, chol, glucose,
            smoking, alcohol, exercise, bmi,
            0,  # Heart Rate (Removed)
            int(heart_dis), int(stroke),
            0,  # Oxygen Saturation (Removed)
            0,  # Body Temperature (Removed)
            int(chronic), int(meds), int(fam_hist), int(fatigue)
        ]])

        # Generate the numerical prediction[cite: 5]
        prediction = model.predict(features)[0]

        st.divider()
        if prediction == 1:
            st.error("### Result: You have Diabetes")
        else:
            st.success("### Result: You do not have Diabetes")
else:
    st.warning("Please ensure 'model.pkl' is in the project folder.")
    # streamlit run app.py