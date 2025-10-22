import streamlit as st
import numpy as np
import joblib

# 🎯 Load model and scaler
model = joblib.load("model.pkl")
try:
    scaler = joblib.load("scaler.pkl")  # Optional: only if you saved a scaler
except FileNotFoundError:
    scaler = None
    st.warning("scaler.pkl not found — input will not be scaled.")
except Exception as e:
    scaler = None
    st.warning(f"Failed to load scaler.pkl: {e}")

# 🏷️ App title
st.title("🚢 Titanic Survivor Predictor")
st.write("Fill in passenger details to check survival prediction:")

# 📥 Input fields — interactive style
pclass = st.radio("🛏️ Passenger Class", [1, 2, 3], index=2)
sex = st.selectbox("🧍 Sex", ["male", "female"])
age = st.slider("🎂 Age", min_value=0, max_value=100, value=25)
# 🔘 Predict button
if st.button("🔍 Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    # Apply scaler if available, otherwise use raw inputs
    try:
        features = scaler.transform(input_data) if scaler is not None else input_data
    except Exception as e:
        st.warning(f"Scaling failed, using raw inputs: {e}")
        features = input_data
    prediction = model.predict(features)[0]
    result = "🟢 Survived" if int(prediction) == 1 else "🔴 Did Not Survive"
    st.header(f"Prediction: {result}")
# 🔘 Predict button
if st.button("🔍 Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    input_scaled = scaler.transform(input_data)  # Skip if no scaler used
    prediction = model.predict(input_scaled)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.header(f"Prediction: {result}")
