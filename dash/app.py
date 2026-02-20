import streamlit as st
import joblib
import json
import pandas as pd
from features import extract_features

# Load model
model = joblib.load("model/ai_detector_model.pkl")

with open("model/feature_columns.json", "r") as f:
    feature_cols = json.load(f)

st.title("AI vs Human Text Detector")
st.write("Paste text below to classify whether it is AI-generated or human-written.")

user_input = st.text_area("Enter text here:")

if st.button("Predict"):

    if len(user_input.strip()) == 0:
        st.warning("Please enter some text.")
    else:
        features = extract_features(user_input)

        df = pd.DataFrame([features])[feature_cols]

        prob = model.predict_proba(df)[0][1]
        pred = model.predict(df)[0]

        if pred == 1:
            st.error(f"Prediction: AI-generated (Confidence: {prob:.2f})")
        else:
            st.success(f"Prediction: Human-written (Confidence: {1 - prob:.2f})")