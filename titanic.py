import streamlit as st
import numpy as np
import joblib

# 🎯 Load model and scaler
model = joblib.load(r"C:\Users\Lenovo\Desktop\Robotronix Python ML\Machine Learning practice\Decision Tree practice\model.pkl")
scaler = joblib.load(r"C:\Users\Lenovo\Desktop\Robotronix Python ML\Machine Learning practice\Decision Tree practice\scaler.pkl")  # If used

# 🏷️ App title
st.title("🚢 Titanic Survivor Predictor")
st.write("Fill in passenger details to check survival prediction:")

# 📥 Input fields — interactive style
pclass = st.radio("🛏️ Passenger Class", [1, 2, 3], index=2)
sex = st.selectbox("🧍 Sex", ["male", "female"])
age = st.slider("🎂 Age", min_value=0, max_value=100, value=25)
sibsp = st.slider("👫 Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.slider("👨‍👩‍👧‍👦 Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("💰 Fare Paid", min_value=0.0, max_value=600.0, value=30.0)

# 🔄 Encode categorical feature
sex_encoded = 1 if sex == "male" else 0

# 🔘 Predict button
if st.button("🔍 Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    input_scaled = scaler.transform(input_data)  # Skip if no scaler used
    prediction = model.predict(input_scaled)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.header(f"Prediction: {result}")
