import streamlit as st
import pickle
import numpy as np
import os

# =========================
# LOAD SAVED FILES
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "student_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "scaler.pkl"
)

encoder_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "label_encoder.pkl"
)

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
label_encoder = pickle.load(open(encoder_path, "rb"))

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# =========================
# TITLE
# =========================

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

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("Enter Student Details")

age = st.sidebar.slider(
    "Age",
    15,
    22,
    18
)

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

# =========================
# CREATE INPUT ARRAY
# =========================

input_data = np.array([
    G1,
    G2,
    absences,
    age,
    studytime,
    failures
]).reshape(1, -1)

# =========================
# SCALE INPUT
# =========================

input_scaled = scaler.transform(input_data)

# =========================
# PREDICTION
# =========================

prediction = model.predict(input_scaled)

prediction_proba = model.predict_proba(input_scaled)

result = label_encoder.inverse_transform(prediction)

# =========================
# BUTTON
# =========================

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

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Machine Learning Project | Student Performance Prediction"
)