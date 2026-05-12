import streamlit as st
import pickle
import numpy as np

# Load saved files
model = pickle.load(open("../models/student_model.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))
label_encoder = pickle.load(open("../models/label_encoder.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Performance Prediction System")

st.markdown(
    """
    Predict whether a student is likely to perform:
    - Weak
    - Average
    - Excellent
    
    using machine learning techniques.
    """
)

# Sidebar
st.sidebar.header("Enter Student Details")

age = st.sidebar.slider("Age", 15, 22, 18)

studytime = st.sidebar.slider(
    "Study Time",
    1,
    4,
    2
)

failures = st.sidebar.slider(
    "Past Failures",
    0,
    4,
    0
)

absences = st.sidebar.slider(
    "Absences",
    0,
    50,
    5
)

G1 = st.sidebar.slider(
    "First Period Grade (G1)",
    0,
    20,
    10
)

G2 = st.sidebar.slider(
    "Second Period Grade (G2)",
    0,
    20,
    10
)

# Create input array
input_data = np.array([
    G1,
    G2,
    absences,
    failures,
    studytime,
    age
]).reshape(1, -1)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

# Probabilities
prediction_proba = model.predict_proba(input_scaled)

# Decode label
result = label_encoder.inverse_transform(prediction)

# Prediction button
if st.button("Predict Performance"):

    confidence = np.max(prediction_proba) * 100

    if result[0] == "Excellent":
        st.success(
            f"🎉 Predicted Performance: {result[0]}"
        )

    elif result[0] == "Average":
        st.warning(
            f"📘 Predicted Performance: {result[0]}"
        )

    else:
        st.error(
            f"⚠️ Predicted Performance: {result[0]}"
        )

    st.info(
        f"Confidence Score: {confidence:.2f}%"
    )

# Footer
st.markdown("---")

st.caption(
    "Machine Learning Project | Student Performance Prediction"
)