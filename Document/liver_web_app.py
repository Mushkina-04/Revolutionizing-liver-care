# liver_web_app.py

import streamlit as st
import pandas as pd
import joblib

model = joblib.load('liver_model.pkl')

st.title("Liver Disease Prediction System")

# Input fields
age = st.number_input("Age")
gender = st.selectbox("Gender", ["Male", "Female"])
tb = st.number_input("Total Bilirubin")
db = st.number_input("Direct Bilirubin")
alp = st.number_input("Alkaline Phosphotase")
alt = st.number_input("Alamine Aminotransferase")
ast = st.number_input("Aspartate Aminotransferase")
tp = st.number_input("Total Proteins")
alb = st.number_input("Albumin")
ag_ratio = st.number_input("Albumin and Globulin Ratio")

if st.button("Predict"):
    gender = 1 if gender == "Male" else 0
    features = [[age, gender, tb, db, alp, alt, ast, tp, alb, ag_ratio]]
    prediction = model.predict(features)
    result = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease"
    st.success(result)
