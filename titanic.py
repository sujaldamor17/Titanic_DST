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
    pclass = st.selectbox("🛏️ Passenger Class", options=[1, 2, 3], index=2,
                          help="Higher class had better survival chances")

    sex = st.radio("🧍 Sex", options=["Male 🚹", "Female 🚺"], index=0,
                   help="Females had higher survival rate")
    sex_encoded = 1 if "Male" in sex else 0

    age = st.slider("🎂 Age", min_value=0.0, max_value=80.0, value=30.0, step=1.0,
                    help="Age affects survival probability")

with col2:
    sibsp = st.selectbox("👫 Siblings/Spouses Aboard", options=list(range(0, 6)), index=0,
                         help="Number of siblings/spouses aboard")
    parch = st.selectbox("👨‍👩‍👧‍👦 Parents/Children Aboard", options=list(range(0, 6)), index=0,
                         help="Number of parents/children aboard")

    fare = st.slider("💰 Fare Paid (₹)", min_value=0.0, max_value=550.0, value=50.0, step=1.0,
                     help="Fare often correlates with class")

# 🔘 Predict button
if st.button("🔍 Predict Survival", key="predict_button"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])

    if scaler:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data

    prediction = model.predict(input_scaled)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"

    st.success(f"🎯 Prediction: {result}")
    st.balloons()
