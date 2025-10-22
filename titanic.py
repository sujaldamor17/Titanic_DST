import streamlit as st
import numpy as np
import joblib

# 🎯 Load model and scaler
model = joblib.load("model.pkl")
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    scaler = None
    st.warning("⚠️ scaler.pkl not found — input will not be scaled.")
except Exception as e:
    scaler = None
    st.warning(f"⚠️ Failed to load scaler.pkl: {e}")

# 🏷️ App title
st.markdown("<h1 style='text-align: center;'>🚢 Titanic Survivor Predictor</h1>", unsafe_allow_html=True)

# 📥 Input layout — 2 columns for clean UI
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🛏️ Passenger Class", options=["1st Class", "2nd Class", "3rd Class"], index=2,
                          help="Higher class had better survival chances")
    pclass_val = {"1st Class": 1, "2nd Class": 2, "3rd Class": 3}[pclass]

    sex = st.radio("🧍 Sex", options=["Male 🚹", "Female 🚺"], index=0,
                   help="Females had higher survival rate")
    sex_encoded = 1 if "Male" in sex else 0

    age_group = st.selectbox("🎂 Age Group", options=["Child (0-12)", "Teen (13-19)", "Adult (20-59)", "Senior (60+)"], index=2,
                             help="Age affects survival probability")
    age_val = {"Child (0-12)": 6, "Teen (13-19)": 16, "Adult (20-59)": 35, "Senior (60+)": 65}[age_group]

with col2:
    sibsp = st.selectbox("👫 Siblings/Spouses Aboard", options=list(range(0, 6)), index=0,
                         help="Number of siblings/spouses aboard")
    parch = st.selectbox("👨‍👩‍👧‍👦 Parents/Children Aboard", options=list(range(0, 6)), index=0,
                         help="Number of parents/children aboard")

    fare_range = st.selectbox("💰 Fare Paid", options=["Low (<₹500)", "Medium (₹500–₹2000)", "High (₹2000+)"], index=1,
                              help="Fare often correlates with class")
    fare_val = {"Low (<₹500)": 300, "Medium (₹500–₹2000)": 1000, "High (₹2000+)": 2500}[fare_range]

# 🔘 Predict button
if st.button("🔍 Predict Survival", key="predict_button"):
    input_data = np.array([[pclass_val, sex_encoded, age_val, sibsp, parch, fare_val]])

    if scaler:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data

    prediction = model.predict(input_scaled)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"

    st.success(f"🎯 Prediction: {result}")
    st.balloons()
