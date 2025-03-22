import joblib
import pandas as pd
import streamlit as st
from sklearn.tree import export_text

# Load the trained model
model = joblib.load('decision_tree_model.joblib')

# Set up Streamlit GUI
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title(" Heart Disease Prediction")

# Input form
st.sidebar.header("Enter Patient Data")

age = st.sidebar.number_input("Age", 1, 120, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate", 50, 250, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

# Encode sex
sex_encoded = 1 if sex == "Male" else 0

# Prediction button
if st.sidebar.button("Predict"):
    input_data = [[age, sex_encoded, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error(" High Risk of Heart Disease")
    else:
        st.success(" Low Risk of Heart Disease")

    # Show decision tree rules
    st.subheader(" Decision Tree Rules")
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                     'restecg', 'thalach', 'exang', 'oldpeak',
                     'slope', 'ca', 'thal']

    tree_rules = export_text(model, feature_names=feature_names)
    st.text(tree_rules)

# Footer
st.markdown("---")
st.caption("Developed by Your Name | Expert Systems Project")
