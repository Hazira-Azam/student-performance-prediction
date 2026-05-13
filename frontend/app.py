import streamlit as st
import requests

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

st.markdown("""
Predict student academic performance using Machine Learning.

Possible categories:
- Weak
- Average
- Excellent
""")

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
# BUTTON
# =========================

if st.button("Predict Performance"):

    url = "http://127.0.0.1:8000/predict"

    payload = {
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": G1,
        "G2": G2
    }

    response = requests.post(url, json=payload)

    result = response.json()

    prediction = result["prediction"]
    confidence = result["confidence_score"]

    if prediction == "Excellent":

        st.success(
            f"🎉 Predicted Performance: {prediction}"
        )

    elif prediction == "Average":

        st.warning(
            f"📘 Predicted Performance: {prediction}"
        )

    else:

        st.error(
            f"⚠️ Predicted Performance: {prediction}"
        )

    st.info(
        f"Confidence Score: {confidence}%"
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Student Performance Prediction using ML + FastAPI + Docker + AWS"
)